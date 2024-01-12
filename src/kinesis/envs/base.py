"""Base environment interfaces, observation boundary protocols, and reset probe contracts."""

from abc import ABC, abstractmethod
from typing import Any
import numpy as np

from kinesis.contracts import ActionSpec, ObservationSpec, Transition


class Environment(ABC):
    """Abstract base class for all physical and simulated control environments."""

    @property
    @abstractmethod
    def task_name(self) -> str:
        """Declared identifier of the underlying task."""
        pass

    @property
    @abstractmethod
    def action_spec(self) -> ActionSpec:
        """Legal action space specification."""
        pass

    @property
    @abstractmethod
    def observation_spec(self) -> ObservationSpec:
        """Permitted observation space specification."""
        pass

    @abstractmethod
    def reset(self, seed: int | None = None) -> Transition:
        """Reset environment to initial state, returning first transition record."""
        pass

    @abstractmethod
    def step(self, action: np.ndarray) -> Transition:
        """Step environment dynamics with given action, returning transition record."""
        pass

    @abstractmethod
    def get_physics_probe(self) -> dict[str, float]:
        """Return diagnostic physical parameters (friction, mass, actuator gain).

        NOTE: Diagnostic parameters are strictly for independent audit and verification.
        They must NEVER be exposed to the policy or world model decision boundary.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Release underlying simulator or device resources."""
        pass
