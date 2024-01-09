"""Action specifications, bounding, validation, and repeat accounting."""

from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass(frozen=True)
class ActionSpec:
    """Rigorous contract defining legal action space and conversion."""
    shape: tuple[int, ...]
    minimum: np.ndarray
    maximum: np.ndarray
    action_repeat: int = 1
    units: str = "normalized_torque"
    control_cost_weight: float = 0.0
    dtype: np.dtype = field(default_factory=lambda: np.dtype(np.float32))

    def __post_init__(self) -> None:
        if self.action_repeat < 1:
            raise ValueError(f"action_repeat must be >= 1, got {self.action_repeat}")
        min_arr = np.asarray(self.minimum, dtype=self.dtype)
        max_arr = np.asarray(self.maximum, dtype=self.dtype)
        if min_arr.shape != self.shape:
            raise ValueError(f"minimum shape {min_arr.shape} does not match {self.shape}")
        if max_arr.shape != self.shape:
            raise ValueError(f"maximum shape {max_arr.shape} does not match {self.shape}")
        if np.any(min_arr > max_arr):
            raise ValueError("Found minimum values strictly greater than maximum values")
        object.__setattr__(self, "minimum", min_arr)
        object.__setattr__(self, "maximum", max_arr)

    @property
    def dim(self) -> int:
        return int(np.prod(self.shape))

    def clip(self, action: np.ndarray | list[float] | float) -> np.ndarray:
        """Clip proposed action strictly within legal physical limits."""
        act_arr = np.asarray(action, dtype=self.dtype)
        if act_arr.shape != self.shape:
            raise ValueError(f"Action shape {act_arr.shape} does not match spec {self.shape}")
        return np.clip(act_arr, self.minimum, self.maximum)

    def validate(self, action: np.ndarray) -> bool:
        """Validate whether an action satisfies bounds and finite conditions."""
        if not np.all(np.isfinite(action)):
            return False
        if action.shape != self.shape:
            return False
        return bool(np.all(action >= self.minimum) and np.all(action <= self.maximum))

    def sample_uniform(self, rng: np.random.Generator) -> np.ndarray:
        """Draw an unbiased uniform sample within bounds."""
        return rng.uniform(self.minimum, self.maximum).astype(self.dtype)

    def zero_action(self) -> np.ndarray:
        """Return neutral zero-torque action clipped to bounds."""
        zeros = np.zeros(self.shape, dtype=self.dtype)
        return np.clip(zeros, self.minimum, self.maximum)

    def compute_control_cost(self, action: np.ndarray) -> float:
        """Quadratic control penalty term for regularization."""
        if self.control_cost_weight <= 0.0:
            return 0.0
        return float(self.control_cost_weight * np.sum(np.square(action)))

    def serialize(self) -> dict[str, Any]:
        """Convert specification to a portable JSON-compatible dictionary."""
        return {
            "shape": list(self.shape),
            "minimum": self.minimum.tolist(),
            "maximum": self.maximum.tolist(),
            "action_repeat": self.action_repeat,
            "units": self.units,
            "control_cost_weight": self.control_cost_weight,
            "dtype": str(self.dtype)
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ActionSpec":
        """Reconstruct spec from dictionary."""
        return cls(
            shape=tuple(data["shape"]),
            minimum=np.array(data["minimum"], dtype=np.float32),
            maximum=np.array(data["maximum"], dtype=np.float32),
            action_repeat=int(data.get("action_repeat", 1)),
            units=str(data.get("units", "normalized_torque")),
            control_cost_weight=float(data.get("control_cost_weight", 0.0)),
            dtype=np.dtype(data.get("dtype", "float32"))
        )
