"""Closed-loop execution runner with information boundary checks and metric accounting."""

from dataclasses import dataclass
import time
from typing import Any
import numpy as np

from kinesis.contracts import (
    Episode,
    EpisodeID,
    Modality,
    ShiftCategory,
    StepType,
    TerminationReason,
    Transition,
)
from kinesis.envs.base import Environment
from kinesis.policies.base import BasePolicy
from kinesis.replay.buffer import BoundedReplayBuffer


@dataclass(frozen=True)
class EpisodeExecutionResult:
    """Summary of a completed closed-loop evaluation episode."""
    episode_id: str
    task_name: str
    seed: int
    steps: int
    total_return: float
    total_cost: float
    duration_sec: float
    termination_reason: TerminationReason
    fallback_invocations: int
    privileged_leakage_detected: bool


class ClosedLoopRunner:
    """Coordinates online interaction between policy and environment under strict information boundaries."""

    def __init__(self, replay_buffer: BoundedReplayBuffer | None = None) -> None:
        self.replay_buffer = replay_buffer

    def run_episode(
        self,
        env: Environment,
        policy: BasePolicy,
        seed: int = 42,
        max_steps: int = 1000,
        shift_category: ShiftCategory = ShiftCategory.NOMINAL,
    ) -> tuple[Episode, EpisodeExecutionResult]:
        """Execute one complete closed-loop episode."""
        policy.reset()
        t_start = time.perf_counter()

        transitions: list[Transition] = []
        privileged_leakage = False

        # Reset environment
        t0 = env.reset(seed=seed)
        transitions.append(t0)
        if self.replay_buffer is not None:
            self.replay_buffer.add_transition(t0)

        cur_t = t0
        fallback_count = 0

        for step in range(1, max_steps + 1):
            # Information boundary check: policy observation MUST NOT contain privileged keys
            obs_to_policy = cur_t.observation
            for ch in env.observation_spec.channels:
                if ch.modality == Modality.PRIVILEGED_PHYSICS and ch.name in obs_to_policy:
                    privileged_leakage = True
                    raise SecurityError(f"Privileged channel {ch.name} leaked into policy observation!")

            # Policy decides action
            action = policy.act(obs_to_policy)
            clipped_action = env.action_spec.clip(action)

            # Check if policy used fallback
            if hasattr(policy, "last_planning_result") and policy.last_planning_result is not None:  # type: ignore
                if policy.last_planning_result.fallback_used:  # type: ignore
                    fallback_count += 1

            # Step real environment
            next_t = env.step(clipped_action)
            transitions.append(next_t)
            if self.replay_buffer is not None:
                self.replay_buffer.add_transition(next_t)

            cur_t = next_t
            if cur_t.step_type.is_terminal():
                break

        duration = time.perf_counter() - t_start

        # Determine termination reason
        if cur_t.step_type == StepType.LAST:
            term_reason = TerminationReason.ENVIRONMENT_LIMIT
        elif len(transitions) > max_steps:
            term_reason = TerminationReason.TIMEOUT
        else:
            term_reason = TerminationReason.TASK_SUCCESS

        episode = Episode(
            episode_id=t0.episode_id,
            task_name=env.task_name,
            seed=seed,
            transitions=tuple(transitions),
            termination_reason=term_reason,
            shift_category=shift_category,
        )

        exec_res = EpisodeExecutionResult(
            episode_id=str(t0.episode_id),
            task_name=env.task_name,
            seed=seed,
            steps=len(transitions) - 1,
            total_return=episode.total_return,
            total_cost=episode.total_constraint_cost,
            duration_sec=duration,
            termination_reason=term_reason,
            fallback_invocations=fallback_count,
            privileged_leakage_detected=privileged_leakage,
        )

        return episode, exec_res
