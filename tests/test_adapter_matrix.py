"""Comprehensive validation matrix across tasks and physical shift profiles."""

import numpy as np
import pytest

from kinesis.contracts import StepType
from kinesis.envs.dm_control_adapter import DMControlAdapter
from kinesis.shifts.profiles import ALL_SHIFT_PROFILES, build_shifted_environment


@pytest.mark.parametrize("task_name", ["cartpole_swingup", "walker_walk", "finger_spin"])
def test_task_adapter_base_contracts(task_name: str):
    """Verify standard dm_control task adapters satisfy core Kinesis contracts."""
    env = DMControlAdapter(task_name=task_name, seed=42)

    assert env.action_spec.dim > 0
    assert len(env.observation_spec.channels) > 0
    assert env.observation_spec.total_flattened_dim > 0

    t0 = env.reset(seed=42)
    assert t0.step_type == StepType.FIRST
    assert t0.step_index == 0
    assert isinstance(t0.observation, dict)

    act = env.action_spec.zero_action()
    t1 = env.step(act)
    assert t1.step_type in (StepType.MID, StepType.LAST)
    assert t1.step_index == 1
    assert t1.simulator_substeps >= 1

    env.close()


@pytest.mark.parametrize("profile_name", list(ALL_SHIFT_PROFILES.keys()))
def test_cartpole_shift_profile_matrix(profile_name: str):
    """Verify all 7 declared shift profiles initialize and step correctly on cartpole."""
    env = build_shifted_environment(
        task_name="cartpole_swingup",
        profile_name=profile_name,
        seed=100,
    )

    t0 = env.reset(seed=100)
    assert t0.step_type == StepType.FIRST

    # Step through 5 control cycles
    for i in range(5):
        act = env.action_spec.zero_action()
        step_res = env.step(act)
        assert np.isfinite(step_res.reward)
        assert not step_res.step_type.is_terminal() or (i > 0)

    # Validate physics probe exists and has expected properties
    probe = env.get_physics_probe()
    assert isinstance(probe, dict)
    assert len(probe) > 0

    env.close()


@pytest.mark.parametrize("task_name", ["walker_walk", "finger_spin"])
def test_multitask_stress_profiles(task_name: str):
    """Verify combined stress and degradation profiles on walker and finger tasks."""
    for prof in ["combined_stress", "sensor_degradation"]:
        env = build_shifted_environment(
            task_name=task_name,
            profile_name=prof,
            seed=200,
        )
        t0 = env.reset(seed=200)
        assert t0.step_type == StepType.FIRST
        t1 = env.step(env.action_spec.zero_action())
        assert t1.step_index == 1
        env.close()
