"""Manifest definitions for episode-level train, validation, and test splits."""

from dataclasses import dataclass
from typing import Any
import json
from pathlib import Path


@dataclass(frozen=True)
class SplitManifest:
    """Rigorous split contract ensuring disjoint seeds and frozen evaluation domains."""
    name: str
    train_seeds: tuple[int, ...]
    val_seeds: tuple[int, ...]
    test_seeds: tuple[int, ...]
    nominal_task_names: tuple[str, ...]
    shift_profile_names: tuple[str, ...]

    def __post_init__(self) -> None:
        tr_set = set(self.train_seeds)
        val_set = set(self.val_seeds)
        te_set = set(self.test_seeds)

        if tr_set.intersection(val_set):
            raise ValueError("Overlap detected between train and val seeds")
        if tr_set.intersection(te_set):
            raise ValueError("Overlap detected between train and test seeds")
        if val_set.intersection(te_set):
            raise ValueError("Overlap detected between val and test seeds")

    def is_train_seed(self, seed: int) -> bool:
        return seed in self.train_seeds

    def is_val_seed(self, seed: int) -> bool:
        return seed in self.val_seeds

    def is_test_seed(self, seed: int) -> bool:
        return seed in self.test_seeds

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "train_seeds": list(self.train_seeds),
            "val_seeds": list(self.val_seeds),
            "test_seeds": list(self.test_seeds),
            "nominal_task_names": list(self.nominal_task_names),
            "shift_profile_names": list(self.shift_profile_names),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SplitManifest":
        return cls(
            name=data["name"],
            train_seeds=tuple(data["train_seeds"]),
            val_seeds=tuple(data["val_seeds"]),
            test_seeds=tuple(data["test_seeds"]),
            nominal_task_names=tuple(data["nominal_task_names"]),
            shift_profile_names=tuple(data["shift_profile_names"]),
        )


def make_standard_manifest() -> SplitManifest:
    """Create standard frozen benchmark split with 100 train seeds, 20 val seeds, 30 test seeds."""
    return SplitManifest(
        name="kinesis_standard_v1",
        train_seeds=tuple(range(100, 200)),
        val_seeds=tuple(range(200, 220)),
        test_seeds=tuple(range(220, 250)),
        nominal_task_names=("cartpole_swingup", "walker_walk", "finger_spin"),
        shift_profile_names=(
            "nominal",
            "actuator_attenuation",
            "actuator_delay",
            "friction_reduction",
            "mass_increase",
            "sensor_degradation",
            "combined_stress",
        ),
    )
