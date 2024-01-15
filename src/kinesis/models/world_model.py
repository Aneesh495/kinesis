"""Unified WorldModel combining encoder, RSSM dynamics, decoder, and prediction heads."""

from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from kinesis.contracts import ObservationSpec
from kinesis.models.encoder import ProprioceptiveDecoder, ProprioceptiveEncoder
from kinesis.models.heads import ConstraintHead, DiscountHead, RewardHead
from kinesis.models.losses import WorldModelLossResult, compute_balanced_kl_loss
from kinesis.models.rssm import RecurrentStateSpaceModel, RSSMState


class WorldModel(nn.Module):
    """Integrated world model system for representation, dynamics, and trajectory evaluation."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        embed_dim: int = 256,
        deter_dim: int = 256,
        stoch_dim: int = 32,
        hidden_dim: int = 256,
        kl_scale: float = 1.0,
        reward_scale: float = 1.0,
        discount_scale: float = 1.0,
        constraint_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.feature_dim = deter_dim + stoch_dim

        self.kl_scale = kl_scale
        self.reward_scale = reward_scale
        self.discount_scale = discount_scale
        self.constraint_scale = constraint_scale

        # Core submodules
        self.encoder = ProprioceptiveEncoder(input_dim=obs_dim, embed_dim=embed_dim, hidden_dim=hidden_dim)
        self.rssm = RecurrentStateSpaceModel(
            action_dim=action_dim,
            embed_dim=embed_dim,
            deter_dim=deter_dim,
            stoch_dim=stoch_dim,
            hidden_dim=hidden_dim,
        )
        self.decoder = ProprioceptiveDecoder(feature_dim=self.feature_dim, output_dim=obs_dim, hidden_dim=hidden_dim)
        self.reward_head = RewardHead(feature_dim=self.feature_dim, hidden_dim=hidden_dim)
        self.discount_head = DiscountHead(feature_dim=self.feature_dim, hidden_dim=hidden_dim)
        self.constraint_head = ConstraintHead(feature_dim=self.feature_dim, hidden_dim=hidden_dim)

    def compute_losses(
        self,
        observations: torch.Tensor,  # [B, T, obs_dim]
        actions: torch.Tensor,       # [B, T, action_dim]
        rewards: torch.Tensor,       # [B, T]
        discounts: torch.Tensor,     # [B, T]
        constraint_costs: torch.Tensor,  # [B, T]
        is_first: torch.Tensor,      # [B, T]
    ) -> WorldModelLossResult:
        """Compute full world model multi-task loss over a sequence batch."""
        B, T, _ = observations.shape

        # Step 1: Encode observations into embeddings [B, T, embed_dim]
        embeds = self.encoder(observations)

        # Step 2: Unroll recurrent belief state dynamics
        prior_states, post_states = self.rssm.observe_sequence(embeds, actions, is_first)

        # Collect features across time: [B, T, feature_dim]
        post_features = torch.stack([s.features for s in post_states], dim=1)

        # Step 3: Reconstruction loss
        recon_preds = self.decoder(post_features)
        recon_loss = F.mse_loss(recon_preds, observations)

        # Step 4: Balanced KL divergence loss
        kl_loss, p_ent, q_ent, usage_ratio = compute_balanced_kl_loss(
            prior_states=prior_states,
            post_states=post_states,
            alpha=0.8,
            free_bits=1.0,
        )

        # Step 5: Reward prediction loss
        reward_preds = self.reward_head(post_features).squeeze(-1)
        reward_loss = F.mse_loss(reward_preds, rewards)

        # Step 6: Discount continuation loss
        discount_loss = self.discount_head.compute_loss(post_features, discounts)

        # Step 7: Constraint cost prediction loss
        cost_preds = self.constraint_head(post_features).squeeze(-1)
        cost_loss = F.mse_loss(cost_preds, constraint_costs)

        total_loss = (
            recon_loss
            + self.kl_scale * kl_loss
            + self.reward_scale * reward_loss
            + self.discount_scale * discount_loss
            + self.constraint_scale * cost_loss
        )

        return WorldModelLossResult(
            total_loss=total_loss,
            recon_loss=recon_loss,
            kl_loss=kl_loss,
            reward_loss=reward_loss,
            discount_loss=discount_loss,
            constraint_loss=cost_loss,
            prior_entropy=p_ent,
            post_entropy=q_ent,
            latent_usage_ratio=usage_ratio,
        )

    def imagine_rollout(
        self,
        init_state: RSSMState,
        action_sequence: torch.Tensor,  # [B, H, action_dim]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Roll out imagined latent trajectory and score rewards, discounts, and constraint costs."""
        imagined_states = self.rssm.imagine_trajectory(init_state, action_sequence)
        features = torch.stack([s.features for s in imagined_states], dim=1)  # [B, H, feat_dim]

        rewards = self.reward_head(features).squeeze(-1)
        discounts = self.discount_head(features).squeeze(-1)
        costs = self.constraint_head(features).squeeze(-1)

        return features, rewards, discounts, costs

    def check_action_sensitivity(self, init_obs: torch.Tensor) -> float:
        """Verify that varying action sequences produces distinct latent trajectories."""
        with torch.no_grad():
            B = 1
            device = init_obs.device
            embed = self.encoder(init_obs.unsqueeze(0))  # [1, embed_dim]
            init_s = self.rssm.initial_state(B, device)
            zero_act = torch.zeros(B, self.action_dim, device=device)
            _, post_s = self.rssm.observe_step(init_s, zero_act, embed)

            H = 5
            act_seq1 = torch.ones(B, H, self.action_dim, device=device) * 0.8
            act_seq2 = torch.ones(B, H, self.action_dim, device=device) * -0.8

            feat1, rew1, _, _ = self.imagine_rollout(post_s, act_seq1)
            feat2, rew2, _, _ = self.imagine_rollout(post_s, act_seq2)

            latent_diff = torch.norm(feat1 - feat2, dim=-1).mean().item()
            return float(latent_diff)
