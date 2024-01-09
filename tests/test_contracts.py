"""Tests for versioned data contracts, actions, observations, transitions, and episodes."""

import pytest
import numpy as np

from kinesis.contracts import (
    ActionSpec,
    Episode,
    EpisodeID,
    Modality,
    ModelVersion,
    ObservationChannel,
    ObservationSpec,
    ShiftCategory,
    StepType,
    TerminationReason,
    Transition,
    TransitionID,
)


def test_action_spec_bounds_and_clipping() -> None:
    spec = ActionSpec(
        shape=(2,),
        minimum=np.array([-1.0, -0.5], dtype=np.float32),
        maximum=np.array([1.0, 0.5], dtype=np.float32),
        action_repeat=2,
    )
    assert spec.dim == 2
    assert spec.action_repeat == 2

    # Clipping
    clipped = spec.clip(np.array([2.0, -1.0], dtype=np.float32))
    assert np.allclose(clipped, [1.0, -0.5])

    # Validation
    assert spec.validate(np.array([0.5, 0.0], dtype=np.float32))
    assert not spec.validate(np.array([1.5, 0.0], dtype=np.float32))

    # Control cost
    cost = spec.compute_control_cost(np.array([1.0, 1.0], dtype=np.float32))
    assert cost == 0.0  # default weight is 0.0

    # Serialization roundtrip
    d = spec.serialize()
    restored = ActionSpec.from_dict(d)
    assert restored.shape == spec.shape
    assert np.allclose(restored.minimum, spec.minimum)
    assert np.allclose(restored.maximum, spec.maximum)


def test_observation_spec_flattening_and_privacy() -> None:
    ch1 = ObservationChannel(name="pos", shape=(3,), modality=Modality.PROPRIOCEPTION)
    ch2 = ObservationChannel(name="vel", shape=(2,), modality=Modality.PROPRIOCEPTION)
    ch_priv = ObservationChannel(
        name="true_friction", shape=(1,), modality=Modality.PRIVILEGED_PHYSICS
    )

    spec = ObservationSpec(channels=(ch1, ch2, ch_priv))
    assert spec.total_flattened_dim == 5
    assert spec.privileged_dim == 1

    sample_obs = {
        "pos": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "vel": np.array([0.1, -0.2], dtype=np.float32),
        "true_friction": np.array([0.8], dtype=np.float32),
    }

    flat = spec.flatten_public(sample_obs)
    assert flat.shape == (5,)
    assert np.allclose(flat, [1.0, 2.0, 3.0, 0.1, -0.2])

    priv = spec.extract_privileged(sample_obs)
    assert "true_friction" in priv
    assert "pos" not in priv


def test_transition_immutability_and_hashing() -> None:
    t = Transition(
        transition_id=TransitionID("t_001"),
        episode_id=EpisodeID("ep_001"),
        step_index=0,
        step_type=StepType.FIRST,
        observation={"q": np.array([0.0, 1.0], dtype=np.float32)},
        action=np.array([0.5], dtype=np.float32),
        reward=1.0,
        constraint_cost=0.0,
        discount=1.0,
        time_delta=0.01,
        simulator_substeps=4,
    )
    h1 = t.compute_sha256()
    assert len(h1) == 64

    # Check immutability
    with pytest.raises(Exception):
        t.reward = 2.0  # type: ignore

    # Serialization roundtrip
    d = t.to_dict()
    restored = Transition.from_dict(d)
    assert restored.transition_id == t.transition_id
    assert restored.compute_sha256() == h1


def test_episode_sequence_integrity_and_sampling() -> None:
    transitions = []
    for i in range(5):
        st = StepType.FIRST if i == 0 else (StepType.LAST if i == 4 else StepType.MID)
        t = Transition(
            transition_id=TransitionID(f"t_{i}"),
            episode_id=EpisodeID("ep_test"),
            step_index=i,
            step_type=st,
            observation={"x": np.array([float(i)], dtype=np.float32)},
            action=np.array([0.1 * i], dtype=np.float32),
            reward=float(i * 10),
            constraint_cost=0.0,
            discount=1.0 if i < 4 else 0.0,
            time_delta=0.02,
        )
        transitions.append(t)

    ep = Episode(
        episode_id=EpisodeID("ep_test"),
        task_name="cartpole_swingup",
        seed=123,
        transitions=tuple(transitions),
        termination_reason=TerminationReason.TASK_SUCCESS,
        shift_category=ShiftCategory.NOMINAL,
    )

    assert ep.length == 5
    assert ep.total_return == 100.0  # 0 + 10 + 20 + 30 + 40
    assert ep.is_terminal

    # Sampling slices
    sub_slice = ep.sample_slice(1, 3)
    assert len(sub_slice) == 3
    assert sub_slice[0].step_index == 1
    assert sub_slice[-1].step_index == 3

    with pytest.raises(IndexError):
        ep.sample_slice(3, 4)

    # Episode serialization roundtrip
    ep_dict = ep.to_dict()
    restored_ep = Episode.from_dict(ep_dict)
    assert restored_ep.episode_id == ep.episode_id
    assert restored_ep.total_return == ep.total_return
    assert restored_ep.compute_sha256() == ep.compute_sha256()
