"""Explicit safe-within-simulation fallback policy and invocation tracking."""

from dataclasses import dataclass
from typing import Any
import numpy as np

from kinesis.contracts import ActionSpec, ModelVersion
from kinesis.policies.base import BasePolicy


@dataclass
class FallbackMetrics:
    """Metrics tracking fallback usage and triggers."""
    total_invocations: int = 0
    timeout_triggers: int = 0
    nonfinite_triggers: int = 0
    constraint_violation_triggers: int = 0
    model_failure_triggers: int = 0


class SafeFallbackPolicy(BasePolicy):
    """Declared fallback policy providing bounded neutral actions during planner exceptions."""

    def __init__(self, action_spec: ActionSpec, version: str = "fallback_safe_v1.0") -> None:
        self._action_spec = action_spec
        self._version = ModelVersion(version)
        self.metrics = FallbackMetrics()

    @property
    def version(self) -> ModelVersion:
        return self._version

    @property
    def action_spec(self) -> ActionSpec:
        return self._action_spec

    def record_trigger(self, reason: str) -> None:
        self.metrics.total_invocations += 1
        if reason == "timeout":
            self.metrics.timeout_triggers += 1
        elif reason == "nonfinite":
            self.metrics.nonfinite_triggers += 1
        elif reason == "constraint_violation":
            self.metrics.constraint_violation_triggers += 1
        elif reason == "model_failure":
            self.metrics.model_failure_triggers += 1

    def act(self, observation: dict[str, np.ndarray], deterministic: bool = False) -> np.ndarray:
        # Zero-torque / neutral damping action safely bounded
        self.metrics.total_invocations += 1
        return self._action_spec.zero_action()

    def reset(self) -> None:
        pass
