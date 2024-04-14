"""Unit tests for twin-world counterfactual intervention engine."""

import numpy as np
import pytest

from kinesis.evaluation.twin_world import TwinWorldEngine, TwinWorldResult, TwinWorldStep


def test_twin_world_identical_environments():
    """Verify twin simulation with identical nominal environments yields zero divergence."""
    engine = TwinWorldEngine("cartpole_swingup")
    res = engine.run_counterfactual_rollout(
        factual_profile="nominal",
        counterfactual_profile="nominal",
        seed=42,
        max_steps=10,
    )

    assert isinstance(res, TwinWorldResult)
    assert res.factual_return == res.counterfactual_return
    assert res.return_gap == 0.0
    assert res.mean_observation_divergence == 0.0
    assert len(res.trajectory_steps) == 10


def test_twin_world_shifted_mass():
    """Verify twin simulation with mass shift exhibits positive physical divergence."""
    engine = TwinWorldEngine("cartpole_swingup")
    res = engine.run_counterfactual_rollout(
        factual_profile="nominal",
        counterfactual_profile="mass_increase",
        seed=101,
        max_steps=15,
    )

    assert isinstance(res, TwinWorldResult)
    assert len(res.trajectory_steps) == 15
    # The trajectories should diverge physically over time
    assert res.max_observation_divergence > 0.0


def test_twin_world_action_intervention():
    """Verify do(a) action intervention causes observable divergence."""
    engine = TwinWorldEngine("cartpole_swingup")
    # Intervene at step 2 with saturated control
    interventions = {2: np.array([1.0], dtype=np.float32)}

    res = engine.run_counterfactual_rollout(
        factual_profile="nominal",
        counterfactual_profile="nominal",
        seed=42,
        max_steps=10,
        action_intervention=interventions,
    )

    assert res.max_observation_divergence > 0.0
    assert res.trajectory_steps[2].counterfactual_action[0] == 1.0
