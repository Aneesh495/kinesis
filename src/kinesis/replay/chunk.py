"""Immutable replay chunk storage with cryptographic checksumming and atomic persistence."""

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any
import numpy as np

from kinesis.contracts import Episode, Transition


@dataclass(frozen=True)
class ReplayChunkMetadata:
    """Metadata summary and cryptographic checksum of an immutable chunk."""
    chunk_id: str
    num_transitions: int
    num_episodes: int
    episode_ids: tuple[str, ...]
    sha256: str
    byte_size: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "num_transitions": self.num_transitions,
            "num_episodes": self.num_episodes,
            "episode_ids": list(self.episode_ids),
            "sha256": self.sha256,
            "byte_size": self.byte_size,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReplayChunkMetadata":
        return cls(
            chunk_id=data["chunk_id"],
            num_transitions=data["num_transitions"],
            num_episodes=data["num_episodes"],
            episode_ids=tuple(data["episode_ids"]),
            sha256=data["sha256"],
            byte_size=data["byte_size"],
        )


class ReplayChunk:
    """Immutable sequence of transitions stored with cryptographic integrity guarantees."""

    def __init__(self, chunk_id: str, transitions: list[Transition] | tuple[Transition, ...]) -> None:
        if len(transitions) == 0:
            raise ValueError("ReplayChunk must contain at least one transition")

        self._chunk_id = chunk_id
        self._transitions = tuple(transitions)

        # Gather distinct episode IDs
        ep_ids = []
        for t in self._transitions:
            ep_id_str = str(t.episode_id)
            if ep_id_str not in ep_ids:
                ep_ids.append(ep_id_str)
        self._episode_ids = tuple(ep_ids)

        # Compute deterministic checksum
        hasher = hashlib.sha256()
        hasher.update(self._chunk_id.encode("utf-8"))
        for t in self._transitions:
            hasher.update(t.compute_sha256().encode("utf-8"))
        self._sha256 = hasher.hexdigest()

    @property
    def chunk_id(self) -> str:
        return self._chunk_id

    @property
    def transitions(self) -> tuple[Transition, ...]:
        return self._transitions

    @property
    def num_transitions(self) -> int:
        return len(self._transitions)

    @property
    def episode_ids(self) -> tuple[str, ...]:
        return self._episode_ids

    @property
    def sha256(self) -> str:
        return self._sha256

    def get_metadata(self) -> ReplayChunkMetadata:
        # Approximate in-memory footprint
        byte_size = sum(t.action.nbytes + sum(v.nbytes for v in t.observation.values()) for t in self._transitions)
        return ReplayChunkMetadata(
            chunk_id=self._chunk_id,
            num_transitions=self.num_transitions,
            num_episodes=len(self._episode_ids),
            episode_ids=self._episode_ids,
            sha256=self._sha256,
            byte_size=byte_size,
        )

    def save_atomic(self, directory: Path) -> Path:
        """Write chunk atomically to disk using a temporary file and replace."""
        directory.mkdir(parents=True, exist_ok=True)
        final_path = directory / f"{self._chunk_id}.json"
        temp_path = directory / f"{self._chunk_id}.tmp"

        payload = {
            "chunk_id": self._chunk_id,
            "sha256": self._sha256,
            "transitions": [t.to_dict() for t in self._transitions],
        }

        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

        temp_path.replace(final_path)
        return final_path

    @classmethod
    def load_and_verify(cls, path: Path) -> "ReplayChunk":
        """Load chunk from disk and verify SHA-256 checksum."""
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        transitions = [Transition.from_dict(t) for t in payload["transitions"]]
        chunk = cls(chunk_id=payload["chunk_id"], transitions=transitions)

        if chunk.sha256 != payload["sha256"]:
            raise ValueError(
                f"Chunk corrupted! Stored SHA256 {payload['sha256']} != computed {chunk.sha256}"
            )
        return chunk
