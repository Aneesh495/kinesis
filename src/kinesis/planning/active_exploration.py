"""Active information-seeking curiosity planner maximizing anticipated ensemble disagreement."""

from dataclasses import dataclass
import numpy as np
import torch

from kinesis.contracts.action import ActionSpec
from kinesis.models.rssm import RSSMState
from kinesis.models.world_model import WorldModel
from kinesis.uncertainty.ensemble import WorldModelEnsemble


@dataclass
class CuriosityPlanResult:
    """Telemetry from active exploration planner."""
    best_action_sequence: np.ndarray
    expected_information_gain: float
    iterations: int
    mean_gain_history: list[float]


class ActiveExplorationPlanner:
    """Plans action sequences that maximize epistemic ensemble disagreement to explore novel dynamics."""

    def __init__(
        self,
        ensemble: WorldModelEnsemble,
        action_spec: ActionSpec,
        horizon: int = 10,
        num_candidates: int = 24,
        num_elites: int = 6,
        num_iterations: int = 4,
        gamma: float = 0.95,
    ) -> None:
        self.ensemble = ensemble
        self.action_spec = action_spec
        self.horizon = horizon
        self.num_candidates = num_candidates
        self.num_elites = num_elites
        self.num_iters = num_iterations
        self.gamma = gamma

    def plan(self, initial_states: list[RSSMState]) -> CuriosityPlanResult:
        """Optimize trajectory maximizing discounted ensemble disagreement."""
        act_dim = self.action_spec.dim
        device = initial_states[0].deter.device

        mean = np.zeros((self.horizon, act_dim), dtype=np.float64)
        std = np.ones((self.horizon, act_dim), dtype=np.float64) * 0.6

        best_gain = -float("inf")
        best_sequence = mean.copy()
        history = []

        with torch.no_grad():
            for it in range(self.num_iters):
                # Sample candidate actions: [num_candidates, horizon, act_dim]
                eps = np.random.randn(self.num_candidates, self.horizon, act_dim)
                candidates = mean[None, :, :] + std[None, :, :] * eps
                candidates = np.clip(candidates, self.action_spec.minimum, self.action_spec.maximum)
                cand_tensor = torch.from_numpy(candidates).float().to(device)

                # Evaluate candidate rollouts through all models in ensemble
                info_gains = torch.zeros(self.num_candidates, device=device)

                # Initialize states for each candidate in committee
                comm_states = [
                    RSSMState(
                        deter=s.deter.repeat(self.num_candidates, 1),
                        stoch=s.stoch.repeat(self.num_candidates, 1),
                        mean=s.mean.repeat(self.num_candidates, 1),
                        std=s.std.repeat(self.num_candidates, 1),
                    )
                    for s in initial_states
                ]

                discount = 1.0
                for t in range(self.horizon):
                    act_t = cand_tensor[:, t, :]
                    # Step each model in committee
                    next_states = []
                    features_list = []
                    for m_idx, wm in enumerate(self.ensemble.models):
                        next_s = wm.rssm.imagine_step(comm_states[m_idx], act_t, sample=False)
                        next_states.append(next_s)
                        feat = torch.cat([next_s.deter, next_s.stoch], dim=-1)
                        features_list.append(feat)

                    comm_states = next_states

                    # Compute disagreement across features: variance across models [num_models, B, D]
                    feat_stack = torch.stack(features_list, dim=0)
                    disagreement_t = torch.var(feat_stack, dim=0, unbiased=True).sum(dim=-1)

                    info_gains += discount * disagreement_t
                    discount *= self.gamma

                scores = info_gains.cpu().numpy()
                idx_sorted = np.argsort(scores)[::-1]

                if scores[idx_sorted[0]] > best_gain:
                    best_gain = float(scores[idx_sorted[0]])
                    best_sequence = candidates[idx_sorted[0]]

                history.append(float(np.mean(scores)))

                # Refit Gaussian to top elites
                elites = candidates[idx_sorted[:self.num_elites]]
                mean = np.mean(elites, axis=0)
                std = np.std(elites, axis=0) + 1e-4

        return CuriosityPlanResult(
            best_action_sequence=best_sequence,
            expected_information_gain=best_gain,
            iterations=self.num_iters,
            mean_gain_history=history,
        )
