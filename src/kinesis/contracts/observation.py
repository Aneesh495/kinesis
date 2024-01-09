"""Observation specifications, channel definitions, and feature flattening."""

from dataclasses import dataclass, field
from typing import Any
import numpy as np

from kinesis.contracts.types import Modality


@dataclass(frozen=True)
class ObservationChannel:
    """Definition of an individual sensor channel."""
    name: str
    shape: tuple[int, ...]
    dtype: np.dtype = field(default_factory=lambda: np.dtype(np.float32))
    modality: Modality = Modality.PROPRIOCEPTION
    units: str = "radians_or_meters"
    low: float = -np.inf
    high: float = np.inf

    @property
    def dim(self) -> int:
        return int(np.prod(self.shape))

    def serialize(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": str(self.dtype),
            "modality": self.modality.name,
            "units": self.units,
            "low": float(self.low) if np.isfinite(self.low) else None,
            "high": float(self.high) if np.isfinite(self.high) else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ObservationChannel":
        low_val = -np.inf if data.get("low") is None else float(data["low"])
        high_val = np.inf if data.get("high") is None else float(data["high"])
        return cls(
            name=data["name"],
            shape=tuple(data["shape"]),
            dtype=np.dtype(data.get("dtype", "float32")),
            modality=Modality[data.get("modality", "PROPRIOCEPTION")],
            units=data.get("units", "radians_or_meters"),
            low=low_val,
            high=high_val,
        )


@dataclass(frozen=True)
class ObservationSpec:
    """Rigorous contract defining legal observation space and feature layouts."""
    channels: tuple[ObservationChannel, ...]

    def __post_init__(self) -> None:
        names = set()
        for ch in self.channels:
            if ch.name in names:
                raise ValueError(f"Duplicate observation channel name: {ch.name}")
            names.add(ch.name)

    @property
    def channel_names(self) -> list[str]:
        return [ch.name for ch in self.channels]

    @property
    def total_flattened_dim(self) -> int:
        return sum(ch.dim for ch in self.channels if ch.modality != Modality.PRIVILEGED_PHYSICS)

    @property
    def privileged_dim(self) -> int:
        return sum(ch.dim for ch in self.channels if ch.modality == Modality.PRIVILEGED_PHYSICS)

    def flatten_public(self, obs_dict: dict[str, np.ndarray]) -> np.ndarray:
        """Flatten only permitted public observation channels into a 1D vector."""
        parts = []
        for ch in self.channels:
            if ch.modality == Modality.PRIVILEGED_PHYSICS:
                continue
            if ch.name not in obs_dict:
                raise KeyError(f"Missing declared public channel {ch.name} in observation")
            arr = np.asarray(obs_dict[ch.name], dtype=np.float32).reshape(-1)
            parts.append(arr)
        if not parts:
            return np.empty((0,), dtype=np.float32)
        return np.concatenate(parts, axis=0)

    def extract_privileged(self, obs_dict: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Extract privileged channels strictly separated from public policy state."""
        return {
            ch.name: obs_dict[ch.name]
            for ch in self.channels
            if ch.modality == Modality.PRIVILEGED_PHYSICS and ch.name in obs_dict
        }

    def validate(self, obs_dict: dict[str, np.ndarray]) -> bool:
        """Verify that observation contains valid shape and finite entries."""
        for ch in self.channels:
            if ch.name not in obs_dict:
                return False
            arr = obs_dict[ch.name]
            if arr.shape != ch.shape:
                return False
            if not np.all(np.isfinite(arr)):
                return False
        return True

    def serialize(self) -> dict[str, Any]:
        return {
            "channels": [ch.serialize() for ch in self.channels],
            "total_flattened_dim": self.total_flattened_dim
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ObservationSpec":
        channels = tuple(ObservationChannel.from_dict(c) for c in data["channels"])
        return cls(channels=channels)
