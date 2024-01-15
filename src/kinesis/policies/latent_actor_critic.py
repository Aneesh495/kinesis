"""Latent space actor-critic policy trained on imagined trajectories from world model."""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from kinesis.contracts import ActionSpec, ModelVersion, ObservationSpec
from kinesis.models.rssm import RSSMState
from kinesis.models.world_model import WorldModel
from kinesis.policies.base import BasePolicy


class LatentActor(nn.Module):
    """Maps latent features [h_t, z_t] to continuous bounded action distributions."""

    def __init__(self, feature_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
        )
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(features)
        mean = self.mean_layer(h)
        log_std = torch.clamp(self.log_std_layer(h), min=-4.0, max=1.0)
        std = torch.exp(log_std)
        return mean, std

    def sample_action(
        self, features: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, std = self(features)
        if deterministic:
            raw_action = mean
            log_prob = torch.zeros_like(mean).sum(dim=-1)
        else:
            normal = torch.distributions.Normal(mean, std)
            raw_action = normal.rsample()  # Reparameterized sample
            log_prob = normal.log_prob(raw_action).sum(dim=-1)

        # Bounded tanh squashing
        action = torch.tanh(raw_action)
        # Correction for tanh squashing in log prob
        log_prob -= torch.log(1.0 - action.pow(2) + 1e-6).sum(dim=-1)
        return action, log_prob


class LatentCritic(nn.Module):
    """Estimates expected return V(s_t) from latent belief features."""

    def __init__(self, feature_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


class ImaginedActorCritic(BasePolicy, nn.Module):
    """Actor-Critic policy optimized strictly on imagined trajectories within learned latent dynamics."""

    def __init__(
        self,
        world_model: WorldModel,
        action_spec: ActionSpec,
        observation_spec: ObservationSpec,
        horizon: int = 12,
        gamma: float = 0.99,
        lambda_: float = 0.95,
        tau: float = 0.02,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-4,
        version: str = "latent_ac_v1.0",
    ) -> None:
        super().__init__()
        self._world_model = world_model
        self._action_spec = action_spec
        self._obs_spec = observation_spec
        self._horizon = horizon
        self._gamma = gamma
        self._lambda = lambda_
        self._tau = tau
        self._version = ModelVersion(version)

        feat_dim = world_model.feature_dim
        act_dim = action_spec.dim

        self.actor = LatentActor(feature_dim=feat_dim, action_dim=act_dim)
        self.critic = LatentCritic(feature_dim=feat_dim)
        self.target_critic = copy.deepcopy(self.critic)
        for p in self.target_critic.parameters():
            p.requires_grad = False

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Internal belief state tracking for online closed-loop execution
        self._belief_state: RSSMState | None = None

    @property
    def version(self) -> ModelVersion:
        return self._version

    @property
    def action_spec(self) -> ActionSpec:
        return self._action_spec

    def reset(self) -> None:
        self._belief_state = None

    def update_belief(self, observation: dict[str, np.ndarray], last_action: np.ndarray | None = None) -> RSSMState:
        """Update recursive belief state given incoming observation."""
        flat_obs = self._obs_spec.flatten_public(observation)
        obs_t = torch.tensor(flat_obs, dtype=torch.float32).unsqueeze(0)
        embed = self._world_model.encoder(obs_t)

        if self._belief_state is None:
            init_s = self._world_model.rssm.initial_state(1, obs_t.device)
            zero_act = torch.zeros(1, self._action_spec.dim, device=obs_t.device)
            _, post_s = self._world_model.rssm.observe_step(init_s, zero_act, embed)
        else:
            act_t = torch.tensor(
                last_action if last_action is not None else self._action_spec.zero_action(),
                dtype=torch.float32,
            ).unsqueeze(0)
            _, post_s = self._world_model.rssm.observe_step(self._belief_state, act_t, embed)

        self._belief_state = post_s
        return post_s

    def act(self, observation: dict[str, np.ndarray], deterministic: bool = False) -> np.ndarray:
        post_s = self.update_belief(observation)
        with torch.no_grad():
            action, _ = self.actor.sample_action(post_s.features, deterministic=deterministic)
        action_np = action.squeeze(0).cpu().numpy()
        return self._action_spec.clip(action_np)

    def compute_lambda_targets(
        self,
        rewards: torch.Tensor,     # [B, H]
        discounts: torch.Tensor,   # [B, H]
        values: torch.Tensor,      # [B, H+1]
    ) -> torch.Tensor:
        """Calculate TD(lambda) targets backwards:

        R_t^lambda = r_t + gamma_t * ((1 - lambda) * V_{t+1} + lambda * R_{t+1}^lambda)
        """
        B, H = rewards.shape
        targets = torch.zeros(B, H, device=rewards.device)
        last_return = values[:, -1]

        for t in reversed(range(H)):
            disc = discounts[:, t]
            v_next = values[:, t + 1]
            last_return = rewards[:, t] + disc * ((1.0 - self._lambda) * v_next + self._lambda * last_return)
            targets[:, t] = last_return

        return targets

    def train_on_imagined_trajectories(self, init_states: RSSMState) -> dict[str, float]:
        """Unroll imagined trajectories from init_states and update actor and critic."""
        B = init_states.deter.shape[0]
        device = init_states.deter.device

        # Step 1: Unroll imagined rollout under current policy
        imagined_states: list[RSSMState] = [init_states]
        actions_list: list[torch.Tensor] = []
        log_probs_list: list[torch.Tensor] = []

        cur_s = init_states
        for _ in range(self._horizon):
            act, log_prob = self.actor.sample_action(cur_s.features)
            next_s = self._world_model.rssm.imagine_step(cur_s, act)
            imagined_states.append(next_s)
            actions_list.append(act)
            log_probs_list.append(log_prob)
            cur_s = next_s

        # Features of trajectory: [B, H, feat_dim]
        traj_features = torch.stack([s.features for s in imagined_states[1:]], dim=1)
        all_features = torch.stack([s.features for s in imagined_states], dim=1)  # [B, H+1, feat_dim]

        # Predicted rewards and discounts
        pred_rewards = self._world_model.reward_head(traj_features).squeeze(-1)
        pred_discounts = self._world_model.discount_head(traj_features).squeeze(-1) * self._gamma

        # Target critic evaluation on all states [B, H+1]
        with torch.no_grad():
            target_values = self.target_critic(all_features)
            lambda_targets = self.compute_lambda_targets(pred_rewards, pred_discounts, target_values)

        # Step 2: Critic update
        pred_values = self.critic(traj_features.detach())
        critic_loss = F.mse_loss(pred_values, lambda_targets.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Step 3: Actor update (maximize expected value estimated by critic)
        actor_values = self.critic(traj_features)
        actor_loss = -actor_values.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Step 4: Polyak target network update
        with torch.no_grad():
            for p, p_targ in zip(self.critic.parameters(), self.target_critic.parameters()):
                p_targ.data.mul_(1.0 - self._tau)
                p_targ.data.add_(self._tau * p.data)

        return {
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "mean_imagined_reward": float(pred_rewards.mean().item()),
        }
