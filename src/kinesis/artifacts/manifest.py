"""Cryptographic manifest and artifact specifications for evidence bundles."""

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FileArtifactRecord:
    """Record of an individual file artifact in evidence bundle."""
    relative_path: str
    sha256: str
    byte_size: int


@dataclass
class EvidenceBundleManifest:
    """Comprehensive cryptographic manifest of an executed acceptance bundle."""
    run_id: str
    git_commit: str
    timestamp_utc: str
    python_version: str
    platform: str
    files: dict[str, FileArtifactRecord] = field(default_factory=dict)

    def add_file(self, base_dir: Path, file_path: Path) -> None:
        rel_str = str(file_path.relative_to(base_dir))
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        self.files[rel_str] = FileArtifactRecord(
            relative_path=rel_str,
            sha256=hasher.hexdigest(),
            byte_size=file_path.stat().st_size,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "git_commit": self.git_commit,
            "timestamp_utc": self.timestamp_utc,
            "python_version": self.python_version,
            "platform": self.platform,
            "files": {
                k: {"relative_path": v.relative_path, "sha256": v.sha256, "byte_size": v.byte_size}
                for k, v in self.files.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvidenceBundleManifest":
        manifest = cls(
            run_id=data["run_id"],
            git_commit=data["git_commit"],
            timestamp_utc=data["timestamp_utc"],
            python_version=data["python_version"],
            platform=data["platform"],
        )
        for k, v in data["files"].items():
            manifest.files[k] = FileArtifactRecord(
                relative_path=v["relative_path"],
                sha256=v["sha256"],
                byte_size=v["byte_size"],
            )
        return manifest
