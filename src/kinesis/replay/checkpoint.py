"""Atomic checkpoint persistence and verified state recovery."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any
import torch


@dataclass(frozen=True)
class CheckpointMetadata:
    """Metadata for an atomic checkpoint."""
    checkpoint_id: str
    step_count: int
    sha256: str
    timestamp_utc: str


class AtomicCheckpointManager:
    """Manages transactional, atomic saving and verified recovery of training state."""

    def __init__(self, checkpoint_dir: Path | str) -> None:
        self._dir = Path(checkpoint_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self._dir / "checkpoint_manifest.json"
        self._manifest: dict[str, Any] = self._load_manifest()

    def _load_manifest(self) -> dict[str, Any]:
        if self._manifest_path.exists():
            with open(self._manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"checkpoints": [], "latest": None}

    def _save_manifest(self) -> None:
        temp_manifest = self._dir / "checkpoint_manifest.tmp"
        with open(temp_manifest, "w", encoding="utf-8") as f:
            json.dump(self._manifest, f, indent=2)
        temp_manifest.replace(self._manifest_path)

    def save_checkpoint(
        self,
        checkpoint_id: str,
        state_dict: dict[str, Any],
        step_count: int,
    ) -> Path:
        """Atomically persist state dict to disk with SHA-256 verification."""
        final_path = self._dir / f"{checkpoint_id}.pt"
        temp_path = self._dir / f"{checkpoint_id}.tmp"

        # Save to temporary file
        torch.save(state_dict, temp_path)

        # Compute SHA-256 over saved bytes
        hasher = hashlib.sha256()
        with open(temp_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        digest = hasher.hexdigest()

        # Transactional rename
        temp_path.replace(final_path)

        # Record in manifest
        ckpt_meta = {
            "checkpoint_id": checkpoint_id,
            "filename": final_path.name,
            "step_count": step_count,
            "sha256": digest,
        }
        self._manifest["checkpoints"].append(ckpt_meta)
        self._manifest["latest"] = checkpoint_id
        self._save_manifest()

        return final_path

    def load_checkpoint(self, checkpoint_id: str) -> dict[str, Any]:
        """Load checkpoint and verify cryptographic integrity."""
        # Locate in manifest
        meta = None
        for item in self._manifest["checkpoints"]:
            if item["checkpoint_id"] == checkpoint_id:
                meta = item
                break

        target_file = self._dir / f"{checkpoint_id}.pt"
        if not target_file.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {target_file}")

        # Verify hash if manifest entry exists
        if meta is not None:
            hasher = hashlib.sha256()
            with open(target_file, "rb") as f:
                while chunk := f.read(65536):
                    hasher.update(chunk)
            computed_hash = hasher.hexdigest()
            if computed_hash != meta["sha256"]:
                raise ValueError(
                    f"Checkpoint {checkpoint_id} corrupted! Manifest hash {meta['sha256']} != computed {computed_hash}"
                )

        state = torch.load(target_file, map_location="cpu", weights_only=False)
        return state

    def load_latest(self) -> dict[str, Any] | None:
        """Load the most recent verified checkpoint."""
        latest_id = self._manifest.get("latest")
        if not latest_id:
            return None
        return self.load_checkpoint(latest_id)
