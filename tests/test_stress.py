"""Stress, fault injection, corrupted replay, and memory stability test suite."""

import os
from pathlib import Path
import numpy as np
import pytest
import torch

from kinesis.contracts import (
    ActionSpec,
    EpisodeID,
    StepType,
    Transition,
    TransitionID,
)
from kinesis.envs.dm_control_adapter import DMControlAdapter
from kinesis.replay.buffer import BoundedReplayBuffer
from kinesis.replay.checkpoint import AtomicCheckpointManager
from kinesis.policies.random import RandomPolicy
from kinesis.safety.shield import RuntimeSafetyShield, ShieldDecision


def test_checkpoint_tamper_detection(tmp_path: Path):
    """Verify cryptographic SHA-256 integrity checker rejects tampered checkpoint files."""
    ckpt_mgr = AtomicCheckpointManager(tmp_path)
    state = {"weights": torch.tensor([1.0, 2.0, 3.0])}
    ckpt_file = ckpt_mgr.save_checkpoint("ckpt_001", state, step_count=100)

    # Corrupt one byte of the checkpoint file
    with open(ckpt_file, "r+b") as f:
        f.seek(10)
        f.write(b"\xff")

    # Loading corrupted checkpoint must raise ValueError
    with pytest.raises(ValueError, match="corrupted"):
        ckpt_mgr.load_checkpoint("ckpt_001")


def test_replay_buffer_capacity_stress():
    """Verify replay buffer strictly bounds memory and evicts oldest chunks under heavy load."""
    buf = BoundedReplayBuffer(capacity_transitions=50, chunk_size=10)

    for ep_idx in range(10):
        ep_id = EpisodeID(f"ep_stress_{ep_idx}")
        for step_idx in range(10):
            t = Transition(
                transition_id=TransitionID(f"{ep_id}_s{step_idx}"),
                episode_id=ep_id,
                step_index=step_idx,
                step_type=StepType.FIRST if step_idx == 0 else StepType.MID,
                observation={"obs": np.ones((4,), dtype=np.float32) * step_idx},
                action=np.zeros((1,), dtype=np.float32),
                reward=1.0,
                constraint_cost=0.0,
                discount=0.99,
                time_delta=0.05,
                simulator_substeps=2,
                privileged_diagnostics=np.zeros((3,), dtype=np.float32),
            )
            buf.add_transition(t)

    # Buffer must have evicted older chunks to stay within bounded capacity
    assert buf.total_transitions <= 60
    assert buf._evicted_chunks_count > 0


def test_safety_shield_extreme_input_handling():
    """Verify runtime safety shield safely handles extreme, out-of-bound action proposals."""
    action_spec = ActionSpec(
        shape=(1,),
        minimum=np.array([-1.0], dtype=np.float32),
        maximum=np.array([1.0], dtype=np.float32),
    )
    fallback = RandomPolicy(action_spec=action_spec, seed=42)

    shield = RuntimeSafetyShield(
        fallback_policy=fallback,
        action_spec=action_spec,
        max_allowed_uncertainty=1.0,
        min_safety_barrier_margin=0.1,
    )

    # Proposed wild action
    wild_action = np.array([1000.0], dtype=np.float32)
    # High uncertainty triggers fallback intervention
    decision = shield.evaluate_and_filter(
        observation={"obs": np.array([0.0], dtype=np.float32)},
        proposed_action=wild_action,
        estimated_uncertainty=2.5,
    )

    assert isinstance(decision, ShieldDecision)
    assert decision.intervened is True
    assert action_spec.validate(decision.executed_action) is True


def test_environment_rapid_reset_stress():
    """Verify rapid reset cycles across distinct seeds execute cleanly without state corruption."""
    env = DMControlAdapter("cartpole_swingup", seed=10)

    for s in range(25):
        t0 = env.reset(seed=s)
        assert t0.step_type == StepType.FIRST
        assert t0.step_index == 0
        t1 = env.step(np.array([0.0], dtype=np.float32))
        assert t1.step_index == 1

    env.close()
