"""Versioned transition contract with immutability, hashing, and substep accounting."""

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any
import numpy as np

from kinesis.contracts.types import EpisodeID, ModelVersion, StepType, TransitionID


@dataclass(frozen=True)
class Transition:
    """Rigorous, immutable record of an individual step in the environment."""
    transition_id: TransitionID
    episode_id: EpisodeID
    step_index: int
    step_type: StepType
    observation: dict[str, np.ndarray]
    action: np.ndarray
    reward: float
    constraint_cost: float
    discount: float
    time_delta: float
    simulator_substeps: int = 1
    model_version: ModelVersion = field(default_factory=lambda: ModelVersion("v0.0"))
    is_imagined: bool = False
    privileged_diagnostics: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.step_index < 0:
            raise ValueError(f"step_index cannot be negative: {self.step_index}")
        if self.time_delta <= 0:
            raise ValueError(f"time_delta must be strictly positive: {self.time_delta}")
        if self.simulator_substeps < 1:
            raise ValueError(f"simulator_substeps must be >= 1: {self.simulator_substeps}")
        if not (0.0 <= self.discount <= 1.0):
            raise ValueError(f"discount must be in [0, 1], got {self.discount}")

    def compute_sha256(self) -> str:
        """Compute deterministic SHA-256 checksum over transition contents."""
        hasher = hashlib.sha256()
        hasher.update(str(self.transition_id).encode("utf-8"))
        hasher.update(str(self.episode_id).encode("utf-8"))
        hasher.update(self.step_index.to_bytes(8, "big", signed=False))
        hasher.update(self.step_type.name.encode("utf-8"))
        hasher.update(np.ascontiguousarray(self.action, dtype=np.float32).tobytes())
        hasher.update(np.float64(self.reward).tobytes())
        hasher.update(np.float64(self.constraint_cost).tobytes())
        hasher.update(np.float64(self.discount).tobytes())
        hasher.update(np.float64(self.time_delta).tobytes())
        hasher.update(self.simulator_substeps.to_bytes(4, "big", signed=False))
        hasher.update(b"1" if self.is_imagined else b"0")

        # Hash sorted observation keys and bytes
        for k in sorted(self.observation.keys()):
            hasher.update(k.encode("utf-8"))
            hasher.update(np.ascontiguousarray(self.observation[k], dtype=np.float32).tobytes())

        return hasher.hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Convert transition to a serializable dictionary."""
        return {
            "transition_id": str(self.transition_id),
            "episode_id": str(self.episode_id),
            "step_index": self.step_index,
            "step_type": self.step_type.name,
            "observation": {k: v.tolist() for k, v in self.observation.items()},
            "action": self.action.tolist(),
            "reward": float(self.reward),
            "constraint_cost": float(self.constraint_cost),
            "discount": float(self.discount),
            "time_delta": float(self.time_delta),
            "simulator_substeps": self.simulator_substeps,
            "model_version": str(self.model_version),
            "is_imagined": self.is_imagined,
            "privileged_diagnostics": {k: v.tolist() for k, v in self.privileged_diagnostics.items()},
            "sha256": self.compute_sha256()
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Transition":
        """Reconstruct transition from dictionary and verify checksum."""
        obs = {k: np.array(v, dtype=np.float32) for k, v in data["observation"].items()}
        act = np.array(data["action"], dtype=np.float32)
        priv = {k: np.array(v, dtype=np.float32) for k, v in data.get("privileged_diagnostics", {}).items()}

        t = cls(
            transition_id=TransitionID(data["transition_id"]),
            episode_id=EpisodeID(data["episode_id"]),
            step_index=int(data["step_index"]),
            step_type=StepType[data["step_type"]],
            observation=obs,
            action=act,
            reward=float(data["reward"]),
            constraint_cost=float(data["constraint_cost"]),
            discount=float(data["discount"]),
            time_delta=float(data["time_delta"]),
            simulator_substeps=int(data.get("simulator_substeps", 1)),
            model_version=ModelVersion(data.get("model_version", "v0.0")),
            is_imagined=bool(data.get("is_imagined", False)),
            privileged_diagnostics=priv
        )
        if "sha256" in data:
            expected_hash = t.compute_sha256()
            if data["sha256"] != expected_hash:
                raise ValueError(f"Checksum mismatch: expected {expected_hash}, got {data['sha256']}")
        return t
