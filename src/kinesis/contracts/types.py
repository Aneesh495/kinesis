"""Fundamental type definitions, enums, and identifiers for Kinesis."""

from enum import Enum, auto
from typing import NamedTuple, NewType, TypeAlias
import numpy as np


class StepType(Enum):
    """Lifecycle step type within an environment trajectory."""
    FIRST = 0
    MID = 1
    LAST = 2
    TRUNCATED = 3

    def is_first(self) -> bool:
        return self == StepType.FIRST

    def is_terminal(self) -> bool:
        return self in (StepType.LAST, StepType.TRUNCATED)

    def is_last(self) -> bool:
        return self == StepType.LAST

    def is_truncated(self) -> bool:
        return self == StepType.TRUNCATED


class Modality(Enum):
    """Observation modality category."""
    PROPRIOCEPTION = auto()
    PIXEL = auto()
    FORCE_TORQUE = auto()
    PRIVILEGED_PHYSICS = auto()  # Strictly diagnostic, prohibited from policy inputs


class TerminationReason(Enum):
    """Reason for episode termination or truncation."""
    RUNNING = "running"
    TASK_SUCCESS = "task_success"
    ENVIRONMENT_LIMIT = "environment_limit"
    TIMEOUT = "timeout"
    SAFETY_VIOLATION = "safety_violation"
    SIMULATOR_DIVERGENCE = "simulator_divergence"
    FALLBACK_TRIGGERED = "fallback_triggered"


class ShiftCategory(Enum):
    """Physical and observational shift classification."""
    NOMINAL = "nominal"
    ACTUATOR_DELAY = "actuator_delay"
    ACTUATOR_DEADBAND = "actuator_deadband"
    FRICTION_PERTURBATION = "friction_perturbation"
    MASS_INERTIA = "mass_inertia"
    SENSOR_DROPOUT = "sensor_dropout"
    COMBINED_STRESS = "combined_stress"


# Distinct typed identifiers
EpisodeID = NewType("EpisodeID", str)
TransitionID = NewType("TransitionID", str)
RunID = NewType("RunID", str)
WorkerID = NewType("WorkerID", int)
ModelVersion = NewType("ModelVersion", str)

# Common numerical type aliases
NDArrayFloat: TypeAlias = np.ndarray
NDArrayInt: TypeAlias = np.ndarray


class LatentShape(NamedTuple):
    """Dimensionality specification for RSSM latent space."""
    deterministic_dim: int  # GRU recurrent state (h_t)
    stochastic_dim: int     # Gaussian distribution dimension (z_t)
    discrete_classes: int = 0  # Optional categorical dimension (if discrete)

    @property
    def total_feature_dim(self) -> int:
        if self.discrete_classes > 0:
            return self.deterministic_dim + (self.stochastic_dim * self.discrete_classes)
        return self.deterministic_dim + self.stochastic_dim
