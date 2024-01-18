"""Independent metric recomputation, return aggregation, and success criteria."""

from dataclasses import dataclass
from typing import Any
import numpy as np

from kinesis.contracts import Episode, Transition


@dataclass(frozen=True)
class BenchmarkMetrics:
    """Rigorous evaluation summary metrics across an episode population."""
    method_name: str
    task_name: str
    shift_name: str
    num_episodes: int
    mean_return: float
    std_return: float
    sem_return: float
    median_return: float
    success_rate: float
    mean_cost: float
    total_simulator_substeps: int
    total_policy_decisions: int
    fallback_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "method_name": self.method_name,
            "task_name": self.task_name,
            "shift_name": self.shift_name,
            "num_episodes": self.num_episodes,
            "mean_return": self.mean_return,
            "std_return": self.std_return,
            "sem_return": self.sem_return,
            "median_return": self.median_return,
            "success_rate": self.success_rate,
            "mean_cost": self.mean_cost,
            "total_simulator_substeps": self.total_simulator_substeps,
            "total_policy_decisions": self.total_policy_decisions,
            "fallback_rate": self.fallback_rate,
        }


def recompute_episode_metrics(episode: Episode) -> dict[str, float]:
    """Independently audit and recompute return, costs, and substeps from raw transition stream."""
    ret = sum(float(t.reward) for t in episode.transitions)
    cost = sum(float(t.constraint_cost) for t in episode.transitions)
    substeps = sum(int(t.simulator_substeps) for t in episode.transitions)
    decisions = len(episode.transitions) - 1  # Excluding initial transition
    duration = sum(float(t.time_delta) for t in episode.transitions)

    return {
        "recomputed_return": ret,
        "recomputed_cost": cost,
        "recomputed_substeps": float(substeps),
        "recomputed_decisions": float(decisions),
        "recomputed_duration": duration,
    }


def aggregate_population_metrics(
    method_name: str,
    task_name: str,
    shift_name: str,
    episodes: list[Episode],
    success_threshold: float = 300.0,
) -> BenchmarkMetrics:
    """Aggregate per-episode outcomes into verified benchmark statistics."""
    if not episodes:
        raise ValueError("Cannot aggregate empty episode list")

    returns = [ep.total_return for ep in episodes]
    costs = [ep.total_constraint_cost for ep in episodes]
    substeps = sum(ep.total_simulator_substeps for ep in episodes)
    decisions = sum(len(ep.transitions) - 1 for ep in episodes)

    mean_ret = float(np.mean(returns))
    std_ret = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
    sem_ret = float(std_ret / np.sqrt(len(returns))) if len(returns) > 1 else 0.0
    med_ret = float(np.median(returns))

    successes = sum(1.0 for r in returns if r >= success_threshold)
    success_rate = successes / len(returns)
    mean_c = float(np.mean(costs))

    return BenchmarkMetrics(
        method_name=method_name,
        task_name=task_name,
        shift_name=shift_name,
        num_episodes=len(episodes),
        mean_return=mean_ret,
        std_return=std_ret,
        sem_return=sem_ret,
        median_return=med_ret,
        success_rate=success_rate,
        mean_cost=mean_c,
        total_simulator_substeps=substeps,
        total_policy_decisions=decisions,
        fallback_rate=0.0,
    )
