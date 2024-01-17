"""Model Predictive Path Integral (MPPI) trajectory optimization."""

import time
from typing import Any
import numpy as np
import torch

from kinesis.contracts import ActionSpec
from kinesis.models.rssm import RSSMState
from kinesis.models.world_model import WorldModel
from kinesis.planning.cem import PlanningResult
from kinesis.planning.objective import PlanningObjectiveWeights, TrajectoryObjective
from kinesis.uncertainty.ensemble import WorldModelEnsemble


class ModelPredictivePathIntegralPlanner:
    """Original MPPI optimizer over learned latent dynamics with temperature weighting."""

    def __init__(
        self,
        ensemble: WorldModelEnsemble | WorldModel,
        action_spec: ActionSpec,
        horizon: int = 10,
        num_candidates: int = 50,
        temperature: float = 0.5,
        noise_std: float = 0.3,
        objective_weights: PlanningObjectiveWeights | None = None,
    ) -> None:
        self.ensemble = ensemble
        self.action_spec = action_spec
        self.horizon = horizon
        self.num_candidates = num_candidates
        self.temperature = max(1e-4, temperature)
        self.noise_std = noise_std
        self.objective = TrajectoryObjective(weights=objective_weights)

        self.action_dim = action_spec.dim
        self._nominal_actions = np.zeros((self.horizon, self.action_dim), dtype=np.float32)

    def reset_warm_start(self) -> None:
        self._nominal_actions = np.zeros((self.horizon, self.action_dim), dtype=np.float32)

    def shift_warm_start(self, sequence: np.ndarray) -> None:
        self._nominal_actions[:-1] = sequence[1:]
        self._nominal_actions[-1] = self.action_spec.zero_action()

    def plan(
        self,
        belief_state: RSSMState | list[RSSMState],
        device: torch.device | None = None,
    ) -> PlanningResult:
        t_start = time.perf_counter()

        if device is None:
            if isinstance(belief_state, list):
                device = belief_state[0].deter.device
            else:
                device = belief_state.deter.device

        belief_states = [belief_state] if not isinstance(belief_state, list) else belief_state

        min_bound = torch.tensor(self.action_spec.minimum, dtype=torch.float32, device=device)
        max_bound = torch.tensor(self.action_spec.maximum, dtype=torch.float32, device=device)
        u_nom = torch.tensor(self._nominal_actions, dtype=torch.float32, device=device)

        # Sample Gaussian perturbations: [num_candidates, horizon, action_dim]
        eps = torch.randn(self.num_candidates, self.horizon, self.action_dim, device=device) * self.noise_std
        candidates = u_nom.unsqueeze(0) + eps
        candidates = torch.clamp(candidates, min=min_bound, max=max_bound)
        actual_eps = candidates - u_nom.unsqueeze(0)

        # Rollout through dynamics
        if isinstance(self.ensemble, WorldModelEnsemble):
            expanded_states = []
            for s in belief_states:
                exp_deter = s.deter.repeat(self.num_candidates, 1)
                exp_stoch = s.stoch.repeat(self.num_candidates, 1)
                exp_mean = s.mean.repeat(self.num_candidates, 1)
                exp_std = s.std.repeat(self.num_candidates, 1)
                expanded_states.append(RSSMState(deter=exp_deter, stoch=exp_stoch, mean=exp_mean, std=exp_std))

            rews, discs, costs, disags, _ = self.ensemble.imagine_rollout_with_disagreement(
                expanded_states, candidates
            )
            mean_disag = float(disags.mean().item())
        else:
            s0 = belief_states[0]
            exp_s = RSSMState(
                deter=s0.deter.repeat(self.num_candidates, 1),
                stoch=s0.stoch.repeat(self.num_candidates, 1),
                mean=s0.mean.repeat(self.num_candidates, 1),
                std=s0.std.repeat(self.num_candidates, 1),
            )
            _, rews, discs, costs = self.ensemble.imagine_rollout(exp_s, candidates)
            disags = torch.zeros_like(rews)
            mean_disag = 0.0

        with torch.no_grad():
            scores = self.objective.score_trajectories(rews, discs, costs, disags, candidates)

            # Softmax path integral weighting
            max_score = torch.max(scores)
            scaled_scores = (scores - max_score) / self.temperature
            weights = torch.softmax(scaled_scores, dim=0)  # [num_candidates]

            # Updated action sequence: u* = u_nom + sum_k w_k eps_k
            weighted_eps = torch.sum(weights.view(-1, 1, 1) * actual_eps, dim=0)
            updated_sequence = torch.clamp(u_nom + weighted_eps, min=min_bound, max=max_bound)

            selected_action = updated_sequence[0].detach().cpu().numpy()
            updated_seq_np = updated_sequence.detach().cpu().numpy()

        self.shift_warm_start(updated_seq_np)
        elapsed = time.perf_counter() - t_start

        return PlanningResult(
            selected_action=self.action_spec.clip(selected_action),
            best_action_sequence=updated_seq_np,
            predicted_return=float(max_score.item()),
            elapsed_time_sec=elapsed,
            iterations_completed=1,
            mean_disagreement=mean_disag,
            fallback_used=False,
        )
