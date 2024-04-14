"""Twin-world counterfactual intervention engine for do-calculus causal analysis."""

from dataclasses import dataclass
from typing import Any
import numpy as np

from kinesis.contracts import Transition
from kinesis.shifts.profiles import build_shifted_environment


@dataclass
class TwinWorldStep:
    """Synchronized step across factual and counterfactual simulations."""
    step_idx: int
    factual_action: np.ndarray
    counterfactual_action: np.ndarray
    factual_reward: float
    counterfactual_reward: float
    observation_divergence: float
    reward_gap: float


@dataclass
class TwinWorldResult:
    """Summary metrics of a paired twin-world counterfactual execution."""
    task_name: str
    seed: int
    factual_profile: str
    counterfactual_profile: str
    factual_return: float
    counterfactual_return: float
    return_gap: float
    mean_observation_divergence: float
    max_observation_divergence: float
    estimated_lyapunov_exponent: float
    trajectory_steps: list[TwinWorldStep]


def _flatten_observation(env: Any, obs: Any) -> np.ndarray:
    """Helper to extract flat 1D observation array."""
    if hasattr(env, "observation_spec") and hasattr(env.observation_spec, "flatten_public") and isinstance(obs, dict):
        return env.observation_spec.flatten_public(obs)
    if isinstance(obs, dict):
        return np.concatenate([np.asarray(v).ravel() for v in obs.values()]).astype(np.float32)
    return np.asarray(obs).ravel().astype(np.float32)


class TwinWorldEngine:
    """Executes aligned twin simulations to quantify causal impact of interventions."""

    def __init__(self, task_name: str = "cartpole_swingup") -> None:
        self.task_name = task_name

    def run_counterfactual_rollout(
        self,
        factual_profile: str = "nominal",
        counterfactual_profile: str = "mass_increase",
        seed: int = 42,
        policy: Any | None = None,
        max_steps: int = 50,
        action_intervention: dict[int, np.ndarray] | None = None,
    ) -> TwinWorldResult:
        """Execute twin environments from identical seed and track divergence."""
        env_f = build_shifted_environment(self.task_name, factual_profile, seed=seed)
        env_cf = build_shifted_environment(self.task_name, counterfactual_profile, seed=seed)

        t_f = env_f.reset(seed=seed)
        t_cf = env_cf.reset(seed=seed)

        rng = np.random.default_rng(seed)
        steps: list[TwinWorldStep] = []
        f_return = 0.0
        cf_return = 0.0
        raw_divergences: list[float] = []

        obs_f_flat = _flatten_observation(env_f, t_f.observation)
        obs_cf_flat = _flatten_observation(env_cf, t_cf.observation)

        initial_diff = float(np.linalg.norm(obs_f_flat - obs_cf_flat))

        for step_idx in range(max_steps):
            # Determine factual action
            if policy is not None:
                act_out = policy.act(t_f.observation)
                act_f = act_out.action if hasattr(act_out, "action") else act_out
            else:
                act_f = env_f.action_spec.sample_uniform(rng)

            # Determine counterfactual action: check if intervened
            if action_intervention is not None and step_idx in action_intervention:
                act_cf = action_intervention[step_idx]
            else:
                act_cf = act_f.copy()

            t_f = env_f.step(act_f)
            t_cf = env_cf.step(act_cf)

            f_reward = float(t_f.reward)
            cf_reward = float(t_cf.reward)
            f_return += f_reward
            cf_return += cf_reward

            obs_f_step = _flatten_observation(env_f, t_f.observation)
            obs_cf_step = _flatten_observation(env_cf, t_cf.observation)
            div = float(np.linalg.norm(obs_f_step - obs_cf_step))
            raw_divergences.append(div)

            steps.append(
                TwinWorldStep(
                    step_idx=step_idx,
                    factual_action=act_f,
                    counterfactual_action=act_cf,
                    factual_reward=f_reward,
                    counterfactual_reward=cf_reward,
                    observation_divergence=div,
                    reward_gap=f_reward - cf_reward,
                )
            )

            if t_f.step_type.is_terminal() or t_cf.step_type.is_terminal():
                break

        env_f.close()
        env_cf.close()

        # Finite-time Lyapunov exponent estimate: lambda = (1 / T) * ln(||d_T|| / ||d_0||)
        T = len(raw_divergences)
        if T > 0 and raw_divergences[-1] > 1e-7:
            d0 = max(initial_diff, 1e-6)
            lyap = float((1.0 / T) * np.log(raw_divergences[-1] / d0))
        else:
            lyap = 0.0

        mean_div = float(np.mean(raw_divergences)) if len(raw_divergences) > 0 else 0.0
        max_div = float(np.max(raw_divergences)) if len(raw_divergences) > 0 else 0.0

        return TwinWorldResult(
            task_name=self.task_name,
            seed=seed,
            factual_profile=factual_profile,
            counterfactual_profile=counterfactual_profile,
            factual_return=f_return,
            counterfactual_return=cf_return,
            return_gap=f_return - cf_return,
            mean_observation_divergence=mean_div,
            max_observation_divergence=max_div,
            estimated_lyapunov_exponent=lyap,
            trajectory_steps=steps,
        )
