"""Tests for DeepMind Control Suite environment adapter, observation wrapper, and shift profiles."""

import pytest
import numpy as np

from kinesis.envs import DMControlAdapter
from kinesis.observations import PartialObservationWrapper, RunningNormalizationStats
from kinesis.shifts import (
    ALL_SHIFT_PROFILES,
    build_shifted_environment,
    make_standard_manifest,
    probe_dynamics_divergence,
)


@pytest.mark.parametrize("task_name", ["cartpole_swingup", "walker_walk", "finger_spin"])
def test_pinned_dm_control_adapter(task_name: str) -> None:
    env = DMControlAdapter(task_name=task_name, action_repeat=2, seed=42)
    assert env.task_name == task_name
    assert env.action_spec.action_repeat == 2

    # Model hash check
    m_hash = env.get_model_hash()
    assert len(m_hash) == 64

    # Reset
    t0 = env.reset(seed=42)
    assert t0.step_index == 0
    assert t0.step_type.is_first()
    assert t0.simulator_substeps == 2
    assert env.observation_spec.validate(t0.observation)

    # Step
    act = env.action_spec.zero_action()
    t1 = env.step(act)
    assert t1.step_index == 1
    assert t1.simulator_substeps == 2

    # Physics probe
    probe = env.get_physics_probe()
    assert "total_mass" in probe
    assert probe["total_mass"] > 0.0
    env.close()


def test_partial_observation_delay_and_noise() -> None:
    raw_env = DMControlAdapter(task_name="cartpole_swingup", action_repeat=2, seed=42)
    wrapped = PartialObservationWrapper(
        env=raw_env,
        observation_delay=2,
        noise_std=0.05,
        dropout_prob=0.0,
        seed=42,
    )

    t0 = wrapped.reset(seed=42)
    assert t0.step_index == 0

    # Step through
    for _ in range(5):
        act = wrapped.action_spec.zero_action()
        t = wrapped.step(act)
        assert wrapped.observation_spec.validate(t.observation)

    wrapped.close()


def test_running_normalization_stats() -> None:
    stats = RunningNormalizationStats()
    rng = np.random.default_rng(42)
    data = rng.normal(loc=5.0, scale=2.0, size=(100, 4))
    for row in data:
        stats.update(row)

    assert stats.count == 100
    assert stats.mean is not None
    assert np.allclose(stats.mean, 5.0, atol=0.5)
    assert np.allclose(stats.std, 2.0, atol=0.5)

    sample = np.array([5.0, 5.0, 5.0, 5.0])
    normed = stats.normalize(sample)
    assert np.allclose(normed, 0.0, atol=0.25)


@pytest.mark.parametrize("profile_name", list(ALL_SHIFT_PROFILES.keys()))
def test_all_shift_profiles_and_probe_divergence(profile_name: str) -> None:
    """Verify that all declared profiles construct cleanly and that non-nominal profiles diverge."""
    env = build_shifted_environment("cartpole_swingup", profile_name=profile_name, seed=42)
    t = env.reset(seed=42)
    assert t.step_type.is_first()
    env.close()

    if profile_name != "nominal":
        # walker_walk exercises contact friction, actuation delay, mass, and sensory channels
        div = probe_dynamics_divergence("walker_walk", profile_name, num_steps=10, seed=42)
        # Verify that either observations or rewards diverge quantitatively from nominal
        divergence_detected = div["mean_obs_divergence"] > 1e-4 or div["mean_reward_difference"] > 1e-4
        assert divergence_detected, f"Shift profile {profile_name} failed to produce dynamics divergence"


def test_split_manifest_disjoint_seeds() -> None:
    manifest = make_standard_manifest()
    assert len(manifest.train_seeds) == 100
    assert len(manifest.val_seeds) == 20
    assert len(manifest.test_seeds) == 30
    assert manifest.is_train_seed(150)
    assert not manifest.is_train_seed(210)
    assert manifest.is_val_seed(210)
    assert manifest.is_test_seed(230)
