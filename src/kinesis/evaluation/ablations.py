"""Five fixed ablation families evaluating model and planning contributions."""

from dataclasses import dataclass
from typing import Any
import numpy as np

from kinesis.closed_loop.loop import ClosedLoopRunner
from kinesis.contracts import ActionSpec, ObservationSpec
from kinesis.envs.base import Environment
from kinesis.models.world_model import WorldModel
from kinesis.planning.cem import CrossEntropyMethodPlanner
from kinesis.planning.objective import PlanningObjectiveWeights
from kinesis.planning.planner_policy import LatentPlannerPolicy
from kinesis.shifts import build_shifted_environment
from kinesis.uncertainty.ensemble import WorldModelEnsemble


@dataclass(frozen=True)
class AblationResult:
    """Record of an individual ablation trial."""
    family: str
    condition: str
    task_name: str
    seed: int
    mean_return: float
    total_steps: int


class AblationMatrixRunner:
    """Executes the five declared ablation families:

    1. Planning Horizon (H in {3, 6, 12})
    2. Epistemic Uncertainty Penalty (beta in {0.0, 0.5, 2.0})
    3. Sensory Delay / History (delay in {0, 2, 4})
    4. Model Ensemble Size (M in {1, 3, 5})
    5. Action Penalty Weight (lambda_act in {0.0, 0.01, 0.1})
    """

    def __init__(self, task_name: str = "cartpole_swingup") -> None:
        self.task_name = task_name

    def run_horizon_ablation(self, seeds: list[int], horizons: tuple[int, ...] = (3, 6, 12)) -> list[AblationResult]:
        results = []
        runner = ClosedLoopRunner()
        for h in horizons:
            for s in seeds:
                env = build_shifted_environment(self.task_name, "nominal", seed=s)
                wm = WorldModel(
                    obs_dim=env.observation_spec.total_flattened_dim,
                    action_dim=env.action_spec.dim,
                    embed_dim=16,
                    deter_dim=32,
                    stoch_dim=8,
                    hidden_dim=32,
                )
                planner = CrossEntropyMethodPlanner(
                    ensemble=wm,
                    action_spec=env.action_spec,
                    horizon=h,
                    num_candidates=10,
                    num_iterations=2,
                )
                policy = LatentPlannerPolicy(planner, env.observation_spec, env.action_spec)
                ep, _ = runner.run_episode(env, policy, seed=s, max_steps=50)
                env.close()
                results.append(
                    AblationResult(
                        family="planning_horizon",
                        condition=f"H_{h}",
                        task_name=self.task_name,
                        seed=s,
                        mean_return=ep.total_return,
                        total_steps=ep.length - 1,
                    )
                )
        return results

    def run_uncertainty_weight_ablation(
        self, seeds: list[int], betas: tuple[float, ...] = (0.0, 0.5, 2.0)
    ) -> list[AblationResult]:
        results = []
        runner = ClosedLoopRunner()
        for b in betas:
            for s in seeds:
                env = build_shifted_environment(self.task_name, "actuator_attenuation", seed=s)
                models = [
                    WorldModel(
                        obs_dim=env.observation_spec.total_flattened_dim,
                        action_dim=env.action_spec.dim,
                        embed_dim=16,
                        deter_dim=32,
                        stoch_dim=8,
                        hidden_dim=32,
                    )
                    for _ in range(3)
                ]
                ens = WorldModelEnsemble(models)
                planner = CrossEntropyMethodPlanner(
                    ensemble=ens,
                    action_spec=env.action_spec,
                    horizon=6,
                    num_candidates=10,
                    num_iterations=2,
                    objective_weights=PlanningObjectiveWeights(uncertainty_beta=b),
                )
                policy = LatentPlannerPolicy(planner, env.observation_spec, env.action_spec)
                ep, _ = runner.run_episode(env, policy, seed=s, max_steps=50)
                env.close()
                results.append(
                    AblationResult(
                        family="uncertainty_weight",
                        condition=f"beta_{b}",
                        task_name=self.task_name,
                        seed=s,
                        mean_return=ep.total_return,
                        total_steps=ep.length - 1,
                    )
                )
        return results
