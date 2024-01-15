"""Tests for parallel environment collection, worker synchronization, and stale-response rejection."""

import pytest
import numpy as np

from kinesis.contracts import ModelVersion
from kinesis.collection import (
    CollectionStats,
    EnvironmentWorker,
    ParallelCollector,
    PolicyProtocol,
    WorkerToken,
)
from kinesis.envs import DMControlAdapter
from kinesis.replay import BoundedReplayBuffer


class DummyRandomPolicy:
    """Simple random policy conforming to PolicyProtocol."""
    def __init__(self, action_dim: int, version: str = "v1.0") -> None:
        self._action_dim = action_dim
        self._version = ModelVersion(version)

    @property
    def version(self) -> ModelVersion:
        return self._version

    def act(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        return np.zeros((self._action_dim,), dtype=np.float32)


def test_worker_token_matching() -> None:
    t1 = WorkerToken(worker_id=0, generation=1, request_id=10)
    t2 = WorkerToken(worker_id=0, generation=1, request_id=10)
    t3 = WorkerToken(worker_id=0, generation=2, request_id=10)

    assert t1.matches(t2)
    assert not t1.matches(t3)


def test_environment_worker_stale_response_detection() -> None:
    env = DMControlAdapter("cartpole_swingup", action_repeat=2, seed=42)
    worker = EnvironmentWorker(worker_id=0, env=env)

    # Valid step
    resp = worker.execute_step(
        action=np.array([0.0], dtype=np.float32),
        expected_generation=worker.generation,
        policy_version=ModelVersion("v0.0"),
    )
    assert not resp.is_stale
    assert resp.transition.step_index == 1

    # Invalidate generation
    worker.reset_generation()

    # Stale step with previous generation
    resp_stale = worker.execute_step(
        action=np.array([0.0], dtype=np.float32),
        expected_generation=0,  # Old generation
        policy_version=ModelVersion("v0.0"),
    )
    assert resp_stale.is_stale
    worker.close()


def test_parallel_collector_budget_and_replay_delivery() -> None:
    envs = [
        DMControlAdapter("cartpole_swingup", action_repeat=2, seed=101),
        DMControlAdapter("cartpole_swingup", action_repeat=2, seed=102),
    ]
    buf = BoundedReplayBuffer(capacity_transitions=500, chunk_size=20)
    policy = DummyRandomPolicy(action_dim=1, version="v1.0")

    collector = ParallelCollector(envs=envs, replay_buffer=buf, initial_policy=policy)
    assert collector.num_workers == 2

    stats = collector.collect_interaction_budget(target_steps=30)
    assert stats.total_steps_collected >= 30
    assert buf.total_transitions >= 30

    # Policy switch test
    new_policy = DummyRandomPolicy(action_dim=1, version="v2.0")
    collector.set_policy(new_policy)

    more_stats = collector.collect_interaction_budget(target_steps=10)
    assert more_stats.total_steps_collected >= 40

    collector.close()


def test_gate_ra10_stale_response_stress() -> None:
    """Stress test executing 100 rapid worker generation shifts and verifying 100 percent stale handling."""
    env = DMControlAdapter("cartpole_swingup", action_repeat=2, seed=42)
    worker = EnvironmentWorker(worker_id=0, env=env)

    stale_count = 0
    for i in range(100):
        # Deliberately desynchronize generation
        current_gen = worker.generation
        worker.reset_generation()

        resp = worker.execute_step(
            action=np.array([0.0], dtype=np.float32),
            expected_generation=current_gen,  # Purposely outdated token
            policy_version=ModelVersion("v0.0"),
        )
        if resp.is_stale:
            stale_count += 1

    assert stale_count == 100, f"Expected 100 detected stale responses, got {stale_count}"
    worker.close()
