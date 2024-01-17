"""World model ensemble and epistemic uncertainty disagreement scoring."""

from typing import Any
import torch
import torch.nn as nn

from kinesis.models.rssm import RSSMState
from kinesis.models.world_model import WorldModel


class WorldModelEnsemble(nn.Module):
    """Committee of independently initialized world models for epistemic uncertainty quantification."""

    def __init__(
        self,
        models: list[WorldModel],
    ) -> None:
        super().__init__()
        if len(models) < 3:
            raise ValueError(f"Ensemble requires at least 3 models, got {len(models)}")
        self.models = nn.ModuleList(models)
        self.num_models = len(models)

    def observe_step_all(
        self,
        prev_states: list[RSSMState],
        action: torch.Tensor,
        embeds: list[torch.Tensor],
    ) -> tuple[list[RSSMState], list[RSSMState]]:
        """Step all ensemble members forward with current observation."""
        priors = []
        posts = []
        for i, m in enumerate(self.models):
            pr, po = m.rssm.observe_step(prev_states[i], action, embeds[i])
            priors.append(pr)
            posts.append(po)
        return priors, posts

    def imagine_rollout_with_disagreement(
        self,
        init_states: list[RSSMState],
        action_sequence: torch.Tensor,  # [B, H, action_dim]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Roll out imagined candidate paths across all ensemble members and quantify disagreement.

        Returns:
          mean_rewards: [B, H]
          mean_discounts: [B, H]
          mean_costs: [B, H]
          disagreement_penalty: [B, H] (variance of predicted latent features across committee)
          mean_features: [B, H, feat_dim]
        """
        B, H, _ = action_sequence.shape
        M = self.num_models

        all_features: list[torch.Tensor] = []
        all_rewards: list[torch.Tensor] = []
        all_discounts: list[torch.Tensor] = []
        all_costs: list[torch.Tensor] = []

        for m_idx, model in enumerate(self.models):
            feat_m, rew_m, disc_m, cost_m = model.imagine_rollout(init_states[m_idx], action_sequence)
            all_features.append(feat_m)
            all_rewards.append(rew_m)
            all_discounts.append(disc_m)
            all_costs.append(cost_m)

        # Stack over ensemble dimension: [M, B, H, ...]
        stacked_features = torch.stack(all_features, dim=0)  # [M, B, H, feat_dim]
        stacked_rewards = torch.stack(all_rewards, dim=0)    # [M, B, H]
        stacked_discounts = torch.stack(all_discounts, dim=0) # [M, B, H]
        stacked_costs = torch.stack(all_costs, dim=0)        # [M, B, H]

        mean_features = stacked_features.mean(dim=0)
        mean_rewards = stacked_rewards.mean(dim=0)
        mean_discounts = stacked_discounts.mean(dim=0)
        mean_costs = stacked_costs.mean(dim=0)

        # Epistemic disagreement: feature variance across committee members
        feat_var = torch.var(stacked_features, dim=0, unbiased=True).sum(dim=-1)  # [B, H]
        reward_var = torch.var(stacked_rewards, dim=0, unbiased=True)             # [B, H]

        # Total epistemic penalty combining state divergence and reward prediction ambiguity
        disagreement_penalty = feat_var + reward_var

        return mean_rewards, mean_discounts, mean_costs, disagreement_penalty, mean_features
