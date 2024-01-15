"""Bounded concurrent environment collection runtime with worker synchronization and budget tracking."""

from dataclasses import dataclass
from typing import Any, Protocol
import numpy as np

from kinesis.contracts import ModelVersion, Transition
from kinesis.collection.worker import EnvironmentWorker, WorkerStepResponse, WorkerToken
from kinesis.envs.base import Environment
from kinesis.replay.buffer import BoundedReplayBuffer


class PolicyProtocol(Protocol):
    """Protocol for any decision-making policy or planner."""
    @property
    def version(self) -> ModelVersion: ...
    def act(self, observation: dict[str, np.ndarray]) -> np.ndarray: ...


@dataclass(frozen=True)
class CollectionStats:
    """Summary of collection runtime metrics."""
    total_steps_collected: int
    total_episodes_completed: int
    stale_responses_rejected: int
    worker_cancellations: int


class ParallelCollector:
    """Bounded parallel actor and collector runtime with backpressure."""

    def __init__(
        self,
        envs: list[Environment],
        replay_buffer: BoundedReplayBuffer,
        initial_policy: PolicyProtocol,
    ) -> None:
        if len(envs) == 0:
            raise ValueError("ParallelCollector requires at least one environment")

        self._replay_buffer = replay_buffer
        self._policy = initial_policy
        self._workers: list[EnvironmentWorker] = []
        for i, env in enumerate(envs):
            w = EnvironmentWorker(worker_id=i, env=env)
            w.set_policy_version(initial_policy.version)
            self._workers.append(w)

        self._total_steps = 0
        self._total_episodes = 0
        self._stale_rejected = 0
        self._worker_cancellations = 0

        # Current state per worker: latest transition
        self._latest_transitions: list[Transition] = []
        for w in self._workers:
            t = w.reset()
            self._latest_transitions.append(t)
            self._replay_buffer.add_transition(t)

    @property
    def num_workers(self) -> int:
        return len(self._workers)

    def set_policy(self, policy: PolicyProtocol) -> None:
        """Update active policy across all workers, invalidating pending generations."""
        self._policy = policy
        for w in self._workers:
            w.set_policy_version(policy.version)

    def step_all(self) -> list[Transition]:
        """Execute one coordinated step across all workers with stale-response rejection."""
        new_transitions: list[Transition] = []

        for i, w in enumerate(self._workers):
            prev_t = self._latest_transitions[i]
            expected_gen = w.generation

            # Query policy for action using permitted observation only
            action = self._policy.act(prev_t.observation)

            # Step worker
            resp: WorkerStepResponse = w.execute_step(
                action=action,
                expected_generation=expected_gen,
                policy_version=self._policy.version,
            )

            if resp.is_stale:
                self._stale_rejected += 1
                # Worker was resynchronized; save new initial transition
                self._latest_transitions[i] = resp.transition
                self._replay_buffer.add_transition(resp.transition)
                continue

            trans = resp.transition
            self._total_steps += 1
            new_transitions.append(trans)
            self._replay_buffer.add_transition(trans)

            if trans.step_type.is_terminal():
                self._total_episodes += 1
                # Reset worker for next episode
                new_t = w.reset()
                self._latest_transitions[i] = new_t
                self._replay_buffer.add_transition(new_t)
            else:
                self._latest_transitions[i] = trans

        return new_transitions

    def collect_interaction_budget(self, target_steps: int) -> CollectionStats:
        """Collect real environment transitions until target_steps budget is fulfilled."""
        steps_at_start = self._total_steps
        while (self._total_steps - steps_at_start) < target_steps:
            self.step_all()

        return CollectionStats(
            total_steps_collected=self._total_steps,
            total_episodes_completed=self._total_episodes,
            stale_responses_rejected=self._stale_rejected,
            worker_cancellations=self._worker_cancellations,
        )

    def close(self) -> None:
        for w in self._workers:
            w.close()
