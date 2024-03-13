"""Risk-sensitive trajectory optimization optimizing CVaR and entropic risk measures."""

from dataclasses import dataclass
from enum import Enum
import numpy as np
import torch

from kinesis.contracts.action import ActionSpec
from kinesis.models.rssm import RSSMState
from kinesis.models.world_model import WorldModel


class RiskMeasure(str, Enum):
    """Supported risk-averse objective metrics."""
    EXPECTED_VALUE = "expected_value"
    CVAR = "cvar"
    ENTROPIC = "entropic"


@dataclass
class RiskSensitiveResult:
    """Telemetry from risk-sensitive trajectory planner."""
    best_action_sequence: np.ndarray
    risk_adjusted_score: float
    cvar_value: float
    mean_return: float
    risk_measure: RiskMeasure


class RiskSensitivePlanner:
    """Stochastic latent trajectory planner optimizing CVaR and entropic risk."""

    def __init__(
        self,
        world_model: WorldModel,
        action_spec: ActionSpec,
        horizon: int = 12,
        num_candidates: int = 30,
        rollouts_per_candidate: int = 5,
        risk_measure: RiskMeasure = RiskMeasure.CVAR,
        alpha_cvar: float = 0.2,       # Worst 20% quantile
        beta_entropic: float = 1.0,    # Risk aversion coefficient
        num_iterations: int = 4,
        elite_ratio: float = 0.2,
    ) -> None:
        self.world_model = world_model
        self.action_spec = action_spec
        self.horizon = horizon
        self.num_candidates = num_candidates
        self.num_rollouts = rollouts_per_candidate
        self.risk_measure = risk_measure
        self.alpha = float(alpha_cvar)
        self.beta = float(beta_entropic)
        self.num_iters = num_iterations
        self.num_elites = max(2, int(num_candidates * elite_ratio))

    def evaluate_risk(self, returns_per_candidate: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute risk-adjusted scores across rollouts [num_candidates, num_rollouts]."""
        K, R = returns_per_candidate.shape
        mean_returns = np.mean(returns_per_candidate, axis=-1)
        cvar_values = np.zeros(K, dtype=np.float64)
        scores = np.zeros(K, dtype=np.float64)

        for k in range(K):
            rets = returns_per_candidate[k]
            # Empirical CVaR at alpha level
            var_val = np.percentile(rets, self.alpha * 100.0)
            tail = rets[rets <= var_val]
            cvar = float(np.mean(tail)) if len(tail) > 0 else float(var_val)
            cvar_values[k] = cvar

            if self.risk_measure == RiskMeasure.CVAR:
                scores[k] = cvar
            elif self.risk_measure == RiskMeasure.ENTROPIC:
                # -1 / beta * ln(mean(exp(-beta * R)))
                # Numerically stable log-mean-exp
                scaled = -self.beta * rets
                max_s = np.max(scaled)
                lme = max_s + np.log(np.mean(np.exp(scaled - max_s)))
                scores[k] = float(-1.0 / self.beta * lme)
            else:
                scores[k] = float(mean_returns[k])

        return scores, cvar_values, mean_returns

    def plan(self, initial_state: RSSMState) -> RiskSensitiveResult:
        """Optimize candidate action distributions under risk-averse objective."""
        act_dim = self.action_spec.dim
        device = initial_state.deter.device

        mean = np.zeros((self.horizon, act_dim), dtype=np.float64)
        std = np.ones((self.horizon, act_dim), dtype=np.float64) * 0.5

        best_score = -float("inf")
        best_sequence = mean.copy()
        best_cvar = -float("inf")
        best_mean_ret = -float("inf")

        with torch.no_grad():
            for it in range(self.num_iters):
                # Sample candidate actions
                eps = np.random.randn(self.num_candidates, self.horizon, act_dim)
                candidates = mean[None, :, :] + std[None, :, :] * eps
                candidates = np.clip(candidates, self.action_spec.minimum, self.action_spec.maximum)

                # Evaluate rollouts for each candidate
                cand_returns = np.zeros((self.num_candidates, self.num_rollouts), dtype=np.float64)

                for k in range(self.num_candidates):
                    cand_act = torch.from_numpy(candidates[k]).float().to(device)

                    # Multi-rollout evaluation
                    for r_idx in range(self.num_rollouts):
                        state_curr = initial_state
                        cum_rew = 0.0

                        for t in range(self.horizon):
                            act_t = cand_act[t : t + 1]
                            state_curr = self.world_model.rssm.imagine_step(state_curr, act_t, sample=True)
                            feat = torch.cat([state_curr.deter, state_curr.stoch], dim=-1)
                            rew = float(self.world_model.reward_head(feat).item())
                            cum_rew += rew

                        cand_returns[k, r_idx] = cum_rew

                scores, cvars, mean_rets = self.evaluate_risk(cand_returns)
                idx_sorted = np.argsort(scores)[::-1]

                if scores[idx_sorted[0]] > best_score:
                    best_score = float(scores[idx_sorted[0]])
                    best_sequence = candidates[idx_sorted[0]]
                    best_cvar = float(cvars[idx_sorted[0]])
                    best_mean_ret = float(mean_rets[idx_sorted[0]])

                # Refit Gaussian to elites
                elites = candidates[idx_sorted[:self.num_elites]]
                mean = np.mean(elites, axis=0)
                std = np.std(elites, axis=0) + 1e-4

        return RiskSensitiveResult(
            best_action_sequence=best_sequence,
            risk_adjusted_score=best_score,
            cvar_value=best_cvar,
            mean_return=best_mean_ret,
            risk_measure=self.risk_measure,
        )
