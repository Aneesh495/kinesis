"""Covariance Matrix Adaptation Evolution Strategy (CMA-ES) latent trajectory planner."""

from dataclasses import dataclass
from typing import Callable
import numpy as np
import torch

from kinesis.contracts.action import ActionSpec
from kinesis.models.rssm import RSSMState
from kinesis.models.world_model import WorldModel
from kinesis.planning.objective import PlanningObjectiveWeights, TrajectoryObjective


@dataclass
class CMAESResult:
    """Output telemetry from CMA-ES optimization."""
    best_action_sequence: np.ndarray  # [H, action_dim]
    best_return: float
    iterations: int
    mean_history: list[float]


class CMAESPlanner:
    """CMA-ES stochastic optimizer for latent action sequence planning."""

    def __init__(
        self,
        world_model: WorldModel,
        action_spec: ActionSpec,
        horizon: int = 12,
        population_size: int = 24,
        num_elites: int = 6,
        max_iterations: int = 5,
        initial_std: float = 0.5,
        uncertainty_penalty_coeff: float = 0.0,
    ) -> None:
        self.world_model = world_model
        self.action_spec = action_spec
        self.horizon = horizon
        self.pop_size = max(population_size, 4)
        self.num_elites = min(num_elites, self.pop_size // 2)
        self.max_iters = max_iterations
        self.init_std = initial_std
        self.objective = TrajectoryObjective(PlanningObjectiveWeights(uncertainty_beta=uncertainty_penalty_coeff))

        self.dim = self.horizon * self.action_spec.dim
        # Weights for recombination
        raw_weights = np.log(self.num_elites + 0.5) - np.log(np.arange(1, self.num_elites + 1))
        self.weights = raw_weights / np.sum(raw_weights)
        self.mueff = 1.0 / np.sum(self.weights**2)

        # Learning rates
        self.cc = (4.0 + self.mueff / self.dim) / (self.dim + 4.0 + 2.0 * self.mueff / self.dim)
        self.cs = (self.mueff + 2.0) / (self.dim + self.mueff + 5.0)
        self.c1 = 2.0 / ((self.dim + 1.3)**2 + self.mueff)
        self.cmu = min(1.0 - self.c1, 2.0 * (self.mueff - 2.0 + 1.0 / self.mueff) / ((self.dim + 2.0)**2 + self.mueff))
        self.damps = 1.0 + 2.0 * max(0.0, np.sqrt((self.mueff - 1.0) / (self.dim + 1.0)) - 1.0) + self.cs
        self.chiN = np.sqrt(self.dim) * (1.0 - 1.0 / (4.0 * self.dim) + 1.0 / (21.0 * self.dim**2))

    def plan(
        self,
        initial_h: torch.Tensor,
        initial_z: torch.Tensor,
        nominal_actions: np.ndarray | None = None,
    ) -> CMAESResult:
        """Execute CMA-ES optimization in world model latent space."""
        act_dim = self.action_spec.dim
        if nominal_actions is not None:
            mean = nominal_actions.flatten().copy()
        else:
            mean = np.zeros(self.dim, dtype=np.float64)

        sigma = float(self.init_std)
        C = np.eye(self.dim, dtype=np.float64)
        pc = np.zeros(self.dim, dtype=np.float64)
        ps = np.zeros(self.dim, dtype=np.float64)

        best_score = -float("inf")
        best_candidate = mean.copy()
        history = []

        device = initial_h.device

        with torch.no_grad():
            for it in range(self.max_iters):
                # Eigendecomposition of C: C = B * D^2 * B^T
                try:
                    w, B_mat = np.linalg.eigh(C)
                    w = np.maximum(w, 1e-10)
                    D_mat = np.diag(np.sqrt(w))
                    BD = B_mat @ D_mat
                except np.linalg.LinAlgError:
                    BD = np.eye(self.dim)

                # Sample population: z_k = randn; x_k = mean + sigma * BD * z_k
                z_k = np.random.randn(self.pop_size, self.dim)
                x_k = np.zeros((self.pop_size, self.dim))
                for k in range(self.pop_size):
                    x_k[k] = mean + sigma * (BD @ z_k[k])

                # Evaluate rollouts
                # Reshape to [pop_size, H, act_dim]
                actions_pop = x_k.reshape(self.pop_size, self.horizon, act_dim)
                actions_pop = np.clip(actions_pop, self.action_spec.minimum, self.action_spec.maximum)
                act_tensor = torch.from_numpy(actions_pop).float().to(device)

                # Vectorized latent rollout
                state_curr = RSSMState(
                    deter=initial_h.repeat(self.pop_size, 1),
                    stoch=initial_z.repeat(self.pop_size, 1),
                    mean=initial_z.repeat(self.pop_size, 1),
                    std=torch.ones_like(initial_z).repeat(self.pop_size, 1),
                )
                cumulative_rewards = torch.zeros(self.pop_size, device=device)

                for t in range(self.horizon):
                    act_t = act_tensor[:, t, :]
                    state_curr = self.world_model.rssm.imagine_step(state_curr, act_t, sample=False)
                    feat = torch.cat([state_curr.deter, state_curr.stoch], dim=-1)
                    pred_r = self.world_model.reward_head(feat).squeeze(-1)
                    cumulative_rewards += pred_r

                scores = cumulative_rewards.cpu().numpy()
                idx_sorted = np.argsort(scores)[::-1]

                if scores[idx_sorted[0]] > best_score:
                    best_score = float(scores[idx_sorted[0]])
                    best_candidate = actions_pop[idx_sorted[0]]

                history.append(float(np.mean(scores)))

                # Elite selection and mean update
                elite_idx = idx_sorted[:self.num_elites]
                x_elites = x_k[elite_idx]
                z_elites = z_k[elite_idx]

                old_mean = mean.copy()
                mean = np.sum(self.weights[:, None] * x_elites, axis=0)
                z_mean = np.sum(self.weights[:, None] * z_elites, axis=0)

                # Path updates
                ps = (1.0 - self.cs) * ps + np.sqrt(self.cs * (2.0 - self.cs) * self.mueff) * (B_mat @ z_mean)
                hsig = float(np.linalg.norm(ps) / np.sqrt(1.0 - (1.0 - self.cs)**(2.0 * (it + 1))) / self.chiN < (1.4 + 2.0 / (self.dim + 1.0)))
                pc = (1.0 - self.cc) * pc + hsig * np.sqrt(self.cc * (2.0 - self.cc) * self.mueff) * (BD @ z_mean)

                # Covariance update
                c1a = self.c1 * (1.0 - (1.0 - hsig**2) * self.cc * (2.0 - self.cc))
                C = (1.0 - self.c1 - self.cmu) * C + self.c1 * (np.outer(pc, pc) + (1.0 - hsig) * self.cc * (2.0 - self.cc) * C)
                for i in range(self.num_elites):
                    diff_i = BD @ z_elites[i]
                    C += self.cmu * self.weights[i] * np.outer(diff_i, diff_i)

                # Step-size update
                sigma = sigma * np.exp((self.cs / self.damps) * (np.linalg.norm(ps) / self.chiN - 1.0))
                sigma = float(np.clip(sigma, 1e-4, 2.0))

        if isinstance(best_candidate, np.ndarray) and best_candidate.ndim == 1:
            best_seq = best_candidate.reshape(self.horizon, act_dim)
        else:
            best_seq = best_candidate

        return CMAESResult(
            best_action_sequence=best_seq,
            best_return=best_score,
            iterations=self.max_iters,
            mean_history=history,
        )
