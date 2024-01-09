"""Versioned episode contract, integrity checking, and sequence sampling."""

from dataclasses import dataclass, field
import hashlib
from typing import Any
import numpy as np

from kinesis.contracts.transition import Transition
from kinesis.contracts.types import EpisodeID, ShiftCategory, StepType, TerminationReason


@dataclass(frozen=True)
class Episode:
    """Immutable sequence of transitions forming a complete or ongoing episode."""
    episode_id: EpisodeID
    task_name: str
    seed: int
    transitions: tuple[Transition, ...]
    termination_reason: TerminationReason
    shift_category: ShiftCategory = ShiftCategory.NOMINAL
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.transitions) == 0:
            raise ValueError("Episode must contain at least one transition")

        # Validate sequential step indices and episode ID consistency
        prev_idx = -1
        for i, t in enumerate(self.transitions):
            if t.episode_id != self.episode_id:
                raise ValueError(
                    f"Transition {t.transition_id} episode_id {t.episode_id} does not match {self.episode_id}"
                )
            if t.step_index != prev_idx + 1:
                raise ValueError(
                    f"Step index gap in episode: expected {prev_idx + 1}, got {t.step_index}"
                )
            prev_idx = t.step_index

            if i == 0 and t.step_type != StepType.FIRST:
                raise ValueError(f"First transition must have StepType.FIRST, got {t.step_type}")

    @property
    def length(self) -> int:
        return len(self.transitions)

    @property
    def total_return(self) -> float:
        return sum(t.reward for t in self.transitions)

    @property
    def total_constraint_cost(self) -> float:
        return sum(t.constraint_cost for t in self.transitions)

    @property
    def total_simulator_substeps(self) -> int:
        return sum(t.simulator_substeps for t in self.transitions)

    @property
    def duration_seconds(self) -> float:
        return sum(t.time_delta for t in self.transitions)

    @property
    def is_terminal(self) -> bool:
        return self.transitions[-1].step_type.is_terminal()

    def compute_sha256(self) -> str:
        """Deterministic hash of entire episode sequence."""
        hasher = hashlib.sha256()
        hasher.update(str(self.episode_id).encode("utf-8"))
        hasher.update(self.task_name.encode("utf-8"))
        hasher.update(self.seed.to_bytes(8, "big", signed=True))
        hasher.update(self.shift_category.value.encode("utf-8"))
        hasher.update(self.termination_reason.value.encode("utf-8"))

        for t in self.transitions:
            hasher.update(t.compute_sha256().encode("utf-8"))

        return hasher.hexdigest()

    def sample_slice(self, start_idx: int, sequence_length: int) -> tuple[Transition, ...]:
        """Extract a contiguous slice guaranteed within this episode's boundaries."""
        if start_idx < 0 or start_idx + sequence_length > len(self.transitions):
            raise IndexError(
                f"Invalid slice [{start_idx}:{start_idx + sequence_length}] for length {len(self.transitions)}"
            )
        return self.transitions[start_idx : start_idx + sequence_length]

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": str(self.episode_id),
            "task_name": self.task_name,
            "seed": self.seed,
            "termination_reason": self.termination_reason.value,
            "shift_category": self.shift_category.value,
            "length": self.length,
            "total_return": self.total_return,
            "total_constraint_cost": self.total_constraint_cost,
            "total_simulator_substeps": self.total_simulator_substeps,
            "duration_seconds": self.duration_seconds,
            "transitions": [t.to_dict() for t in self.transitions],
            "sha256": self.compute_sha256()
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Episode":
        transitions = tuple(Transition.from_dict(t) for t in data["transitions"])
        ep = cls(
            episode_id=EpisodeID(data["episode_id"]),
            task_name=str(data["task_name"]),
            seed=int(data["seed"]),
            transitions=transitions,
            termination_reason=TerminationReason(data["termination_reason"]),
            shift_category=ShiftCategory(data.get("shift_category", "nominal")),
            metadata=data.get("metadata", {})
        )
        if "sha256" in data:
            expected = ep.compute_sha256()
            if data["sha256"] != expected:
                raise ValueError(f"Episode hash mismatch: expected {expected}, got {data['sha256']}")
        return ep
