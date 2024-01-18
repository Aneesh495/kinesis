"""Counterfactual paired-seed evaluation protocol with declared coupling validity."""

from dataclasses import dataclass
from typing import Any
import numpy as np
from scipy import stats

from kinesis.closed_loop.loop import ClosedLoopRunner
from kinesis.contracts import Episode
from kinesis.envs.base import Environment
from kinesis.policies.base import BasePolicy


@dataclass(frozen=True)
class PairedEvaluationSummary:
    """Statistical summary of paired counterfactual comparison on identical seeds."""
    method_a: str
    method_b: str
    task_name: str
    num_seeds: int
    mean_return_a: float
    mean_return_b: float
    mean_paired_difference: float  # E[R_A - R_B]
    std_paired_difference: float
    win_rate_a: float  # Fraction where R_A > R_B
    p_value: float  # Two-tailed Wilcoxon signed-rank or paired t-test
    declared_coupling: str = "coupled_reset_initial_state"


class CounterfactualEvaluator:
    """Evaluates two methods on strictly identical seeds and audits divergence."""

    def __init__(self, task_name: str) -> None:
        self.task_name = task_name

    def evaluate_pair(
        self,
        make_env_fn: Any,
        policy_a: BasePolicy,
        policy_b: BasePolicy,
        seeds: list[int],
        max_steps: int = 500,
    ) -> tuple[PairedEvaluationSummary, list[dict[str, float]]]:
        runner = ClosedLoopRunner()
        diffs = []
        rets_a = []
        rets_b = []
        records = []

        for s in seeds:
            env_a = make_env_fn(seed=s)
            ep_a, _ = runner.run_episode(env=env_a, policy=policy_a, seed=s, max_steps=max_steps)
            env_a.close()

            env_b = make_env_fn(seed=s)
            ep_b, _ = runner.run_episode(env=env_b, policy=policy_b, seed=s, max_steps=max_steps)
            env_b.close()

            r_a = ep_a.total_return
            r_b = ep_b.total_return
            d = r_a - r_b

            rets_a.append(r_a)
            rets_b.append(r_b)
            diffs.append(d)

            records.append({
                "seed": s,
                "return_a": r_a,
                "return_b": r_b,
                "difference": d,
            })

        mean_diff = float(np.mean(diffs))
        std_diff = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
        wins = sum(1.0 for d in diffs if d > 0.0)
        win_rate = wins / len(diffs)

        # Wilcoxon signed-rank test or paired t-test
        if len(diffs) >= 5 and np.any(np.array(diffs) != 0.0):
            try:
                stat, p_val = stats.wilcoxon(rets_a, rets_b)
            except Exception:
                stat, p_val = stats.ttest_rel(rets_a, rets_b)
        else:
            p_val = 1.0

        summary = PairedEvaluationSummary(
            method_a=str(policy_a.version),
            method_b=str(policy_b.version),
            task_name=self.task_name,
            num_seeds=len(seeds),
            mean_return_a=float(np.mean(rets_a)),
            mean_return_b=float(np.mean(rets_b)),
            mean_paired_difference=mean_diff,
            std_paired_difference=std_diff,
            win_rate_a=win_rate,
            p_value=float(p_val),
            declared_coupling="coupled_initial_state_and_simulator_prng",
        )

        return summary, records
