"""Model-free Actor-Critic (PPO) baseline policy and GAE estimation."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from kinesis.contracts import ActionSpec, ModelVersion, ObservationSpec
from kinesis.policies.base import BasePolicy


class GaussianActor(nn.Module):
    """Diagonal Gaussian policy network outputting bounded continuous actions."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.net(obs)
        mean = torch.tanh(self.mean_layer(features))
        std = torch.exp(torch.clamp(self.log_std, min=-2.0, max=0.5))
        return mean, std


class ValueCritic(nn.Module):
    """State value network estimating V(s)."""

    def __init__(self, obs_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


class ModelFreeActorCritic(BasePolicy, nn.Module):
    """PPO-style model-free actor-critic controller."""

    def __init__(
        self,
        observation_spec: ObservationSpec,
        action_spec: ActionSpec,
        hidden_dim: int = 128,
        lr: float = 3e-4,
        clip_ratio: float = 0.2,
        version: str = "mf_ppo_v1.0",
    ) -> None:
        super().__init__()
        self._obs_spec = observation_spec
        self._action_spec = action_spec
        self._version = ModelVersion(version)
        self._clip_ratio = clip_ratio

        obs_dim = observation_spec.total_flattened_dim
        action_dim = action_spec.dim

        self.actor = GaussianActor(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim)
        self.critic = ValueCritic(obs_dim=obs_dim, hidden_dim=hidden_dim)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    @property
    def version(self) -> ModelVersion:
        return self._version

    @property
    def action_spec(self) -> ActionSpec:
        return self._action_spec

    def act(self, observation: dict[str, np.ndarray], deterministic: bool = False) -> np.ndarray:
        flat_obs = self._obs_spec.flatten_public(observation)
        obs_t = torch.tensor(flat_obs, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            mean, std = self.actor(obs_t)
            if deterministic:
                action = mean
            else:
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()

        action_np = action.squeeze(0).numpy()
        return self._action_spec.clip(action_np)

    def compute_gae(
        self,
        rewards: torch.Tensor,     # [T]
        values: torch.Tensor,      # [T]
        next_value: torch.Tensor,  # scalar
        dones: torch.Tensor,       # [T]
        gamma: float = 0.99,
        lam: float = 0.95,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation: A_t = delta_t + (gamma * lam) * A_{t+1}."""
        T = len(rewards)
        advantages = torch.zeros(T, dtype=torch.float32)
        returns = torch.zeros(T, dtype=torch.float32)
        gae = 0.0

        for t in reversed(range(T)):
            v_next = next_value if t == T - 1 else values[t + 1]
            non_terminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * v_next * non_terminal - values[t]
            gae = delta + gamma * lam * non_terminal * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]

        return advantages, returns

    def update_step(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
    ) -> dict[str, float]:
        """Execute one PPO policy and value clipping update."""
        mean, std = self.actor(obs)
        dist = torch.distributions.Normal(mean, std)
        new_log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1).mean()

        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self._clip_ratio, 1.0 + self._clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        values = self.critic(obs)
        value_loss = F.mse_loss(values, returns)

        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
        self.optimizer.step()

        return {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy.item()),
        }
