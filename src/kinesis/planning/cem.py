"""Cross-Entropy Method (CEM) latent trajectory optimizer with warm-starting and action bounds."""

from dataclasses import dataclass
from typing import Any
import numpy as np
import torch

from kinesis.contracts import ActionSpec
from kinesis.models.rssm import RSSMState
from kinesis.models.world_model import WorldModel
from kinesis.planning.objective import PlanningObjectiveWeights, TrajectoryObjective
from kinesis.uncertainty.ensemble import WorldModelEnsemble


@dataclass(frozen=True)
class PlanningResult:
    """Detailed record of a completed latent trajectory optimization."""
    selected_action: np.ndarray
    best_action_sequence: np.ndarray
    predicted_return: float
    elapsed_time_sec: float
    iterations_completed: int
    mean_disagreement: float
    fallback_used: bool = False


class CrossEntropyMethodPlanner:
    """Original Cross-Entropy Method optimizer in learned latent space."""

    def __init__(
        self,
        ensemble: WorldModelEnsemble | WorldModel,
        action_spec: ActionSpec,
        horizon: int = 10,
        num_candidates: int = 50,
        num_iterations: int = 4,
        elite_ratio: float = 0.1,
        momentum: float = 0.2,
        min_std: float = 0.05,
        objective_weights: PlanningObjectiveWeights | None = None,
    ) -> None:
        self.ensemble = ensemble
        self.action_spec = action_spec
        self.horizon = horizon
        self.num_candidates = num_candidates
        self.num_iterations = num_iterations
        self.num_elites = max(2, int(num_candidates * elite_ratio))
        self.momentum = momentum
        self.min_std = min_std
        self.objective = TrajectoryObjective(weights=objective_weights)

        self.action_dim = action_spec.dim
        # Warm start mean and std: [H, action_dim]
        self._warm_mean = np.zeros((self.horizon, self.action_dim), dtype=np.float32)
        self._warm_std = np.ones((self.horizon, self.action_dim), dtype=np.float32) * 0.5

    def reset_warm_start(self) -> None:
        self._warm_mean = np.zeros((self.horizon, self.action_dim), dtype=np.float32)
        self._warm_std = np.ones((self.horizon, self.action_dim), dtype=np.float32) * 0.5

    def shift_warm_start(self, best_sequence: np.ndarray) -> None:
        """Advance optimized sequence by 1 step for receded-horizon execution."""
        self._warm_mean[:-1] = best_sequence[1:]
        self._warm_mean[-1] = self.action_spec.zero_action()
        self._warm_std = np.ones((self.horizon, self.action_dim), dtype=np.float32) * 0.5

    def plan(
        self,
        belief_state: RSSMState | list[RSSMState],
        device: torch.device | None = None,
    ) -> PlanningResult:
        """Execute CEM search over candidate action trajectories."""
        import time

        t_start = time.perf_counter()

        if device is None:
            if isinstance(belief_state, list):
                device = belief_state[0].deter.device
            else:
                device = belief_state.deter.device

        # If a single model was provided, wrap as single-state list
        if not isinstance(belief_state, list):
            belief_states = [belief_state]
        else:
            belief_states = belief_state

        min_bound = torch.tensor(self.action_spec.minimum, dtype=torch.float32, device=device)
        max_bound = torch.tensor(self.action_spec.maximum, dtype=torch.float32, device=device)

        mean = torch.tensor(self._warm_mean, dtype=torch.float32, device=device)
        std = torch.tensor(self._warm_std, dtype=torch.float32, device=device)

        best_score = -float("inf")
        best_sequence = mean.clone()
        mean_disagreement = 0.0

        with torch.no_grad():
            for it in range(self.num_iterations):
                # Sample candidate batch: [num_candidates, horizon, action_dim]
                eps = torch.randn(self.num_candidates, self.horizon, self.action_dim, device=device)
                candidates = mean.unsqueeze(0) + std.unsqueeze(0) * eps
                candidates = torch.clamp(candidates, min=min_bound, max=max_bound)

                # Evaluate candidates through ensemble dynamics
                if isinstance(self.ensemble, WorldModelEnsemble):
                    expanded_states = []
                    for s in belief_states:
                        exp_deter = s.deter.repeat(self.num_candidates, 1)
                        exp_stoch = s.stoch.repeat(self.num_candidates, 1)
                        exp_mean = s.mean.repeat(self.num_candidates, 1)
                        exp_std = s.std.repeat(self.num_candidates, 1)
                        expanded_states.append(
                            RSSMState(deter=exp_deter, stoch=exp_stoch, mean=exp_mean, std=exp_std)
                        )

                    rews, discs, costs, disags, _ = self.ensemble.imagine_rollout_with_disagreement(
                        expanded_states, candidates
                    )
                    mean_disagreement = float(disags.mean().item())
                else:
                    # Single model fallback (no ensemble disagreement)
                    s0 = belief_states[0]
                    exp_s = RSSMState(
                        deter=s0.deter.repeat(self.num_candidates, 1),
                        stoch=s0.stoch.repeat(self.num_candidates, 1),
                        mean=s0.mean.repeat(self.num_candidates, 1),
                        std=s0.std.repeat(self.num_candidates, 1),
                    )
                    _, rews, discs, costs = self.ensemble.imagine_rollout(exp_s, candidates)
                    disags = torch.zeros_like(rews)

                # Score candidates
                scores = self.objective.score_trajectories(rews, discs, costs, disags, candidates)

                # Select elite candidates
                top_scores, top_indices = torch.topk(scores, k=self.num_elites)
                elites = candidates[top_indices]  # [num_elites, horizon, action_dim]

                elite_mean = elites.mean(dim=0)
                elite_std = elites.std(dim=0, unbiased=True)
                elite_std = torch.clamp(elite_std, min=self.min_std)

                # Smooth updates with momentum
                mean = self.momentum * mean + (1.0 - self.momentum) * elite_mean
                std = self.momentum * std + (1.0 - self.momentum) * elite_std

                if top_scores[0].item() > best_score:
                    best_score = float(top_scores[0].item())
                    best_sequence = elites[0].clone()

            # Extract selected first action
            selected_act_t = best_sequence[0].detach().cpu().numpy()
            best_seq_np = best_sequence.detach().cpu().numpy()

        # Update warm start for next step
        self.shift_warm_start(best_seq_np)
        elapsed = time.perf_counter() - t_start

        return PlanningResult(
            selected_action=self.action_spec.clip(selected_act_t),
            best_action_sequence=best_seq_np,
            predicted_return=best_score,
            elapsed_time_sec=elapsed,
            iterations_completed=self.num_iterations,
            mean_disagreement=mean_disagreement,
            fallback_used=False,
        )
