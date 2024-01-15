"""Environment worker with generation tokens, stale response rejection, and policy tracking."""

from dataclasses import dataclass
from typing import Any
import numpy as np

from kinesis.contracts import ModelVersion, Transition
from kinesis.contracts.types import WorkerID
from kinesis.envs.base import Environment


@dataclass(frozen=True)
class WorkerToken:
    """Generation and request identifier preventing stale response acceptance."""
    worker_id: WorkerID
    generation: int
    request_id: int

    def matches(self, other: "WorkerToken") -> bool:
        return (
            self.worker_id == other.worker_id
            and self.generation == other.generation
            and self.request_id == other.request_id
        )


@dataclass(frozen=True)
class WorkerStepResponse:
    """Packaged response from an environment worker."""
    token: WorkerToken
    transition: Transition
    policy_version: ModelVersion
    is_stale: bool = False


class EnvironmentWorker:
    """Single environment worker process or thread with state tracking."""

    def __init__(self, worker_id: int, env: Environment) -> None:
        self._worker_id = WorkerID(worker_id)
        self._env = env
        self._generation = 0
        self._request_counter = 0
        self._active_policy_version = ModelVersion("v0.0")
        self._is_closed = False

    @property
    def worker_id(self) -> WorkerID:
        return self._worker_id

    @property
    def generation(self) -> int:
        return self._generation

    def reset_generation(self) -> int:
        """Increment generation to invalidate in-flight requests."""
        self._generation += 1
        return self._generation

    def set_policy_version(self, version: ModelVersion) -> None:
        """Update active policy version."""
        if version != self._active_policy_version:
            self._active_policy_version = version
            self.reset_generation()

    def reset(self, seed: int | None = None) -> Transition:
        """Reset environment and increment generation to flush pending step responses."""
        self.reset_generation()
        return self._env.reset(seed=seed)

    def execute_step(
        self,
        action: np.ndarray,
        expected_generation: int,
        policy_version: ModelVersion,
    ) -> WorkerStepResponse:
        """Execute step and stamp response with token."""
        self._request_counter += 1
        token = WorkerToken(
            worker_id=self._worker_id,
            generation=expected_generation,
            request_id=self._request_counter,
        )

        # Check for generation mismatch (stale request)
        if expected_generation != self._generation or policy_version != self._active_policy_version:
            # Stale request detected
            t = self._env.reset()  # Resynchronize
            return WorkerStepResponse(
                token=token,
                transition=t,
                policy_version=policy_version,
                is_stale=True,
            )

        t = self._env.step(action)
        return WorkerStepResponse(
            token=token,
            transition=t,
            policy_version=policy_version,
            is_stale=False,
        )

    def close(self) -> None:
        if not self._is_closed:
            self._env.close()
            self._is_closed = True
