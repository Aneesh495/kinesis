"""Base policy interfaces and action selection contracts."""

from abc import ABC, abstractmethod
from typing import Any
import numpy as np

from kinesis.contracts import ActionSpec, ModelVersion


class BasePolicy(ABC):
    """Abstract base class for all control policies and planners."""

    @property
    @abstractmethod
    def version(self) -> ModelVersion:
        """Version identifier of policy parameters or planner configuration."""
        pass

    @property
    @abstractmethod
    def action_spec(self) -> ActionSpec:
        """Declared legal action specification."""
        pass

    @abstractmethod
    def act(self, observation: dict[str, np.ndarray], deterministic: bool = False) -> np.ndarray:
        """Select bounded action given permitted public observation dictionary."""
        pass

    def reset(self) -> None:
        """Reset internal recurrent or temporal state at episode boundary."""
        pass
