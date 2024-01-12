"""Tests for bounded immutable replay buffer, sequence sampling, and atomic checkpoints."""

import pytest
import numpy as np
import torch
from pathlib import Path

from kinesis.contracts import (
    Episode,
    EpisodeID,
    ShiftCategory,
    StepType,
    TerminationReason,
    Transition,
    TransitionID,
)
from kinesis.replay import (
    AtomicCheckpointManager,
    BoundedReplayBuffer,
    ReplayChunk,
)


def _make_dummy_transition(ep_id: str, step: int, is_last: bool = False) -> Transition:
    return Transition(
        transition_id=TransitionID(f"{ep_id}_s{step}"),
        episode_id=EpisodeID(ep_id),
        step_index=step,
        step_type=StepType.FIRST if step == 0 else (StepType.LAST if is_last else StepType.MID),
        observation={"obs": np.array([float(step), 1.0], dtype=np.float32)},
        action=np.array([0.5], dtype=np.float32),
        reward=1.0,
        constraint_cost=0.0,
        discount=0.0 if is_last else 1.0,
        time_delta=0.01,
        simulator_substeps=2,
    )


def test_replay_chunk_integrity_and_tamper_detection(tmp_path: Path) -> None:
    transitions = [_make_dummy_transition("ep1", i, i == 4) for i in range(5)]
    chunk = ReplayChunk(chunk_id="chunk_test", transitions=transitions)
    assert chunk.num_transitions == 5
    assert len(chunk.sha256) == 64

    # Save atomically
    saved_path = chunk.save_atomic(tmp_path)
    assert saved_path.exists()

    # Load and verify
    loaded = ReplayChunk.load_and_verify(saved_path)
    assert loaded.chunk_id == chunk.chunk_id
    assert loaded.sha256 == chunk.sha256

    # Tamper with chunk file
    with open(saved_path, "r", encoding="utf-8") as f:
        content = f.read()
    tampered = content.replace("chunk_test", "chunk_tampered")
    with open(saved_path, "w", encoding="utf-8") as f:
        f.write(tampered)

    with pytest.raises(ValueError, match="Chunk corrupted"):
        ReplayChunk.load_and_verify(saved_path)


def test_bounded_replay_buffer_sequence_sampling_no_boundary_crossing() -> None:
    buf = BoundedReplayBuffer(capacity_transitions=500, chunk_size=10)

    # Add 3 distinct episodes of varying lengths
    for ep_idx, ep_len in enumerate([15, 20, 25]):
        ep_id = f"ep_{ep_idx}"
        for step in range(ep_len):
            t = _make_dummy_transition(ep_id, step, step == ep_len - 1)
            buf.add_transition(t)

    assert buf.total_transitions == 60
    assert buf.total_episodes == 3

    # Sample batch of sequences of length 8
    batch = buf.sample_sequences(batch_size=4, sequence_length=8)
    assert batch["actions"].shape == (4, 8, 1)
    assert batch["rewards"].shape == (4, 8)
    assert batch["discounts"].shape == (4, 8)
    assert batch["observations"]["obs"].shape == (4, 8, 2)

    # Verify that transitions in each sampled sequence are consecutive within their episode
    for b in range(4):
        obs_seq = batch["observations"]["obs"][b, :, 0].numpy()
        steps_diff = np.diff(obs_seq)
        assert np.allclose(steps_diff, 1.0), "Sampled sequence crossed an episode boundary or had gap!"


def test_bounded_replay_buffer_capacity_eviction() -> None:
    buf = BoundedReplayBuffer(capacity_transitions=50, chunk_size=10)

    # Add 10 episodes of 10 steps each = 100 transitions
    for ep_idx in range(10):
        ep_id = f"ep_{ep_idx}"
        for step in range(10):
            t = _make_dummy_transition(ep_id, step, step == 9)
            buf.add_transition(t)

    stats = buf.get_stats()
    assert stats.evicted_chunks_count > 0
    assert stats.total_transitions <= 60  # Within bounded capacity bounds


def test_atomic_checkpoint_save_and_recovery(tmp_path: Path) -> None:
    mgr = AtomicCheckpointManager(tmp_path)

    state = {
        "step": 1000,
        "weights": torch.tensor([1.0, 2.0, 3.0]),
    }

    path = mgr.save_checkpoint("ckpt_001", state, step_count=1000)
    assert path.exists()

    loaded = mgr.load_checkpoint("ckpt_001")
    assert loaded["step"] == 1000
    assert torch.allclose(loaded["weights"], state["weights"])

    # Load latest
    latest = mgr.load_latest()
    assert latest is not None
    assert latest["step"] == 1000
