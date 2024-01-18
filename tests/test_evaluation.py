"""Tests for evaluation metrics recomputation, paired counterfactual comparisons, error gaps, and ablations."""

import pytest
import numpy as np

from kinesis.contracts import (
    Episode,
    EpisodeID,
    ShiftCategory,
    StepType,
    TerminationReason,
    Transition,
    TransitionID,
)
from kinesis.envs import DMControlAdapter
from kinesis.evaluation import (
    AblationMatrixRunner,
    CounterfactualEvaluator,
    PredictionControlErrorGapAnalyzer,
    aggregate_population_metrics,
    recompute_episode_metrics,
)
from kinesis.models import WorldModel
from kinesis.policies import CartpoleEnergyHeuristic, RandomPolicy


def _make_sample_episode(num_steps: int = 10, reward_per_step: float = 1.0) -> Episode:
    transitions = []
    for i in range(num_steps + 1):
        st = StepType.FIRST if i == 0 else (StepType.LAST if i == num_steps else StepType.MID)
        t = Transition(
            transition_id=TransitionID(f"t_{i}"),
            episode_id=EpisodeID("ep_eval_test"),
            step_index=i,
            step_type=st,
            observation={"q": np.array([float(i) * 0.1, 0.5], dtype=np.float32)},
            action=np.array([0.2], dtype=np.float32),
            reward=0.0 if i == 0 else reward_per_step,
            constraint_cost=0.01,
            discount=0.0 if i == num_steps else 1.0,
            time_delta=0.02,
            simulator_substeps=2,
        )
        transitions.append(t)

    return Episode(
        episode_id=EpisodeID("ep_eval_test"),
        task_name="cartpole_swingup",
        seed=42,
        transitions=tuple(transitions),
        termination_reason=TerminationReason.TASK_SUCCESS,
        shift_category=ShiftCategory.NOMINAL,
    )


def test_independent_metric_recomputation_and_population_aggregation() -> None:
    ep = _make_sample_episode(num_steps=10, reward_per_step=2.0)
    recomp = recompute_episode_metrics(ep)
    assert recomp["recomputed_return"] == 20.0
    assert recomp["recomputed_decisions"] == 10.0
    assert recomp["recomputed_substeps"] == 22.0

    # Population aggregation
    eps = [ep, _make_sample_episode(num_steps=10, reward_per_step=1.0)]
    bench = aggregate_population_metrics(
        method_name="test_method",
        task_name="cartpole_swingup",
        shift_name="nominal",
        episodes=eps,
        success_threshold=15.0,
    )
    assert bench.num_episodes == 2
    assert bench.mean_return == 15.0
    assert bench.success_rate == 0.5


def test_counterfactual_paired_seed_evaluation() -> None:
    def _make_env(seed: int) -> DMControlAdapter:
        return DMControlAdapter("cartpole_swingup", action_repeat=2, seed=seed)

    env_ref = _make_env(seed=42)
    policy_a = CartpoleEnergyHeuristic(env_ref.action_spec)
    policy_b = RandomPolicy(env_ref.action_spec, seed=42)
    env_ref.close()

    evaluator = CounterfactualEvaluator("cartpole_swingup")
    summary, records = evaluator.evaluate_pair(
        make_env_fn=_make_env,
        policy_a=policy_a,
        policy_b=policy_b,
        seeds=[101, 102, 103],
        max_steps=20,
    )

    assert summary.num_seeds == 3
    assert len(records) == 3
    assert summary.declared_coupling == "coupled_initial_state_and_simulator_prng"


def test_prediction_control_error_gap_analysis() -> None:
    ep = _make_sample_episode(num_steps=15, reward_per_step=1.0)
    wm = WorldModel(obs_dim=2, action_dim=1, embed_dim=16, deter_dim=32, stoch_dim=8, hidden_dim=32)

    analyzer = PredictionControlErrorGapAnalyzer(world_model=wm)
    res = analyzer.analyze_episode(ep, shift_name="nominal")

    assert res.task_name == "cartpole_swingup"
    assert res.one_step_recon_mse > 0.0
    assert 1 in res.horizon_errors
    assert 3 in res.horizon_errors


def test_ablation_matrix_runner() -> None:
    runner = AblationMatrixRunner(task_name="cartpole_swingup")
    # Small test on horizon ablation
    res = runner.run_horizon_ablation(seeds=[101], horizons=(3, 6))
    assert len(res) == 2
    assert res[0].family == "planning_horizon"
    assert res[0].condition == "H_3"
    assert res[1].condition == "H_6"
