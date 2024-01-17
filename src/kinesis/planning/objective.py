"""Trajectory objective scoring with epistemic uncertainty penalization and constraint costs."""

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class PlanningObjectiveWeights:
    """Hyperparameters balancing reward, uncertainty risk, constraints, and control effort."""
    uncertainty_beta: float = 1.0
    constraint_weight: float = 10.0
    action_effort_weight: float = 0.01
    gamma: float = 0.99


class TrajectoryObjective:
    """Evaluates cumulative discounted trajectory return penalized by epistemic risk and constraint costs."""

    def __init__(self, weights: PlanningObjectiveWeights | None = None) -> None:
        self.weights = weights if weights is not None else PlanningObjectiveWeights()

    def score_trajectories(
        self,
        rewards: torch.Tensor,       # [B, H]
        discounts: torch.Tensor,     # [B, H]
        costs: torch.Tensor,         # [B, H]
        disagreement: torch.Tensor,  # [B, H]
        actions: torch.Tensor,       # [B, H, act_dim]
    ) -> torch.Tensor:
        """Compute cumulative penalized return for each candidate action sequence."""
        B, H, _ = actions.shape
        scores = torch.zeros(B, device=rewards.device)
        cumulative_discount = torch.ones(B, device=rewards.device)

        # Action regularization effort: sum_i (a_i^2)
        action_penalties = actions.pow(2).sum(dim=-1)  # [B, H]

        for t in range(H):
            # Step return penalized by epistemic uncertainty and constraints
            step_return = (
                rewards[:, t]
                - self.weights.uncertainty_beta * disagreement[:, t]
                - self.weights.constraint_weight * costs[:, t]
                - self.weights.action_effort_weight * action_penalties[:, t]
            )

            scores += cumulative_discount * step_return
            cumulative_discount *= discounts[:, t] * self.weights.gamma

        return scores
