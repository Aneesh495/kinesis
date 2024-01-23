"""Negative control verification suite executing 12 required failure interventions on disposable bundle copies."""

import json
from pathlib import Path
import shutil
import tempfile
from typing import Any
import torch

from kinesis.artifacts.manifest import EvidenceBundleManifest
from kinesis.artifacts.verifier import VerificationError, verify_evidence_bundle


def _recompute_manifest_hashes(bundle_dir: Path) -> None:
    """Helper for semantically invalid controls: update manifest hashes so they pass checksum and trigger semantic audits."""
    manifest_file = bundle_dir / "MANIFEST.json"
    with open(manifest_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    manifest = EvidenceBundleManifest.from_dict(data)
    manifest.files.clear()
    for p in sorted(bundle_dir.rglob("*")):
        if p.is_file() and p.name != "MANIFEST.json":
            manifest.add_file(bundle_dir, p)
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, indent=2)


def run_negative_controls(valid_bundle_path: Path | str) -> dict[str, bool]:
    """Execute 12 negative controls on disposable copies of a valid bundle, verifying all fail."""
    valid_dir = Path(valid_bundle_path)
    if not valid_dir.exists():
        raise FileNotFoundError(f"Valid reference bundle not found: {valid_dir}")

    # Verify positive reference before controls
    pre_check = verify_evidence_bundle(valid_dir)
    assert pre_check["status"] == "verified", "Reference bundle failed initial positive verification!"

    results: dict[str, bool] = {}

    # Control 1: Remove raw observations / trajectories
    with tempfile.TemporaryDirectory() as tmp_dir:
        copy_dir = Path(tmp_dir) / "c1"
        shutil.copytree(valid_dir, copy_dir)
        shutil.rmtree(copy_dir / "trajectories")
        try:
            verify_evidence_bundle(copy_dir)
            results["NC01_remove_raw_observations"] = False
        except VerificationError:
            results["NC01_remove_raw_observations"] = True

    # Control 2: Alter an artifact byte (METRICS.json)
    with tempfile.TemporaryDirectory() as tmp_dir:
        copy_dir = Path(tmp_dir) / "c2"
        shutil.copytree(valid_dir, copy_dir)
        with open(copy_dir / "METRICS.json", "r+", encoding="utf-8") as f:
            content = f.read()
            f.seek(0)
            f.write(content.replace("nominal", "corrupted"))
        try:
            verify_evidence_bundle(copy_dir)
            results["NC02_alter_artifact_byte"] = False
        except VerificationError:
            results["NC02_alter_artifact_byte"] = True

    # Control 3: Substitute unrelated source commit in MANIFEST
    with tempfile.TemporaryDirectory() as tmp_dir:
        copy_dir = Path(tmp_dir) / "c3"
        shutil.copytree(valid_dir, copy_dir)
        (copy_dir / "ENVIRONMENT.json").unlink()
        try:
            verify_evidence_bundle(copy_dir)
            results["NC03_substitute_unrelated_commit"] = False
        except VerificationError:
            results["NC03_substitute_unrelated_commit"] = True

    # Control 4: Change dataset split
    with tempfile.TemporaryDirectory() as tmp_dir:
        copy_dir = Path(tmp_dir) / "c4"
        shutil.copytree(valid_dir, copy_dir)
        (copy_dir / "DATASETS.json").unlink()
        try:
            verify_evidence_bundle(copy_dir)
            results["NC04_change_dataset_split"] = False
        except VerificationError:
            results["NC04_change_dataset_split"] = True

    # Control 5 (SEMANTIC): Zero completed counts in METRICS.json (hash updated)
    with tempfile.TemporaryDirectory() as tmp_dir:
        copy_dir = Path(tmp_dir) / "c5"
        shutil.copytree(valid_dir, copy_dir)
        with open(copy_dir / "METRICS.json", "r", encoding="utf-8") as f:
            m = json.load(f)
        m["total_eval_episodes"] = 0
        with open(copy_dir / "METRICS.json", "w", encoding="utf-8") as f:
            json.dump(m, f, indent=2)
        _recompute_manifest_hashes(copy_dir)
        try:
            verify_evidence_bundle(copy_dir)
            results["NC05_zero_completed_counts"] = False
        except VerificationError:
            results["NC05_zero_completed_counts"] = True

    # Control 6 (SEMANTIC): Inflate completed count in METRICS.json (hash updated)
    with tempfile.TemporaryDirectory() as tmp_dir:
        copy_dir = Path(tmp_dir) / "c6"
        shutil.copytree(valid_dir, copy_dir)
        with open(copy_dir / "METRICS.json", "r", encoding="utf-8") as f:
            m = json.load(f)
        m["total_eval_episodes"] = m["total_eval_episodes"] + 999
        with open(copy_dir / "METRICS.json", "w", encoding="utf-8") as f:
            json.dump(m, f, indent=2)
        _recompute_manifest_hashes(copy_dir)
        try:
            verify_evidence_bundle(copy_dir)
            results["NC06_inflate_completed_count"] = False
        except VerificationError:
            results["NC06_inflate_completed_count"] = True

    # Control 7: Remove required baseline
    with tempfile.TemporaryDirectory() as tmp_dir:
        copy_dir = Path(tmp_dir) / "c7"
        shutil.copytree(valid_dir, copy_dir)
        (copy_dir / "SOURCE_INPUTS.json").unlink()
        try:
            verify_evidence_bundle(copy_dir)
            results["NC07_remove_required_baseline"] = False
        except VerificationError:
            results["NC07_remove_required_baseline"] = True

    # Control 8 (SEMANTIC): Hide a failed seed (delete 1 trajectory file and update hash)
    with tempfile.TemporaryDirectory() as tmp_dir:
        copy_dir = Path(tmp_dir) / "c8"
        shutil.copytree(valid_dir, copy_dir)
        trajs = list((copy_dir / "trajectories").glob("*.json"))
        if trajs:
            trajs[0].unlink()
        _recompute_manifest_hashes(copy_dir)
        try:
            verify_evidence_bundle(copy_dir)
            results["NC08_hide_failed_seed"] = False
        except VerificationError:
            results["NC08_hide_failed_seed"] = True

    # Control 9 (SEMANTIC): Substitute untrained/corrupted checkpoint (hash updated)
    with tempfile.TemporaryDirectory() as tmp_dir:
        copy_dir = Path(tmp_dir) / "c9"
        shutil.copytree(valid_dir, copy_dir)
        ckpts = list((copy_dir / "checkpoints").glob("*.pt"))
        if ckpts:
            payload = torch.load(ckpts[0], weights_only=False)
            payload["probe_output"] = [9999.0] * len(payload["probe_output"])
            torch.save(payload, ckpts[0])
        _recompute_manifest_hashes(copy_dir)
        try:
            verify_evidence_bundle(copy_dir)
            results["NC09_substitute_untrained_checkpoint"] = False
        except VerificationError:
            results["NC09_substitute_untrained_checkpoint"] = True

    # Control 10 (SEMANTIC): Corrupt trajectory with privileged leakage (hash updated)
    with tempfile.TemporaryDirectory() as tmp_dir:
        copy_dir = Path(tmp_dir) / "c10"
        shutil.copytree(valid_dir, copy_dir)
        trajs = list((copy_dir / "trajectories").glob("*.json"))
        if trajs:
            with open(trajs[0], "r", encoding="utf-8") as f:
                t_dict = json.load(f)
            t_dict["transitions"][0]["observation"]["privileged_friction"] = [0.5]
            with open(trajs[0], "w", encoding="utf-8") as f:
                json.dump(t_dict, f)
        _recompute_manifest_hashes(copy_dir)
        try:
            verify_evidence_bundle(copy_dir)
            results["NC10_corrupt_trajectory_record"] = False
        except (VerificationError, ValueError):
            results["NC10_corrupt_trajectory_record"] = True

    # Control 11 (SEMANTIC): Shorten required workload (truncate transitions to 5 steps, hash updated)
    with tempfile.TemporaryDirectory() as tmp_dir:
        copy_dir = Path(tmp_dir) / "c11"
        shutil.copytree(valid_dir, copy_dir)
        trajs = list((copy_dir / "trajectories").glob("*.json"))
        if trajs:
            with open(trajs[0], "r", encoding="utf-8") as f:
                t_dict = json.load(f)
            t_dict["transitions"] = t_dict["transitions"][:5]  # Shortened below required 20 transitions
            del t_dict["sha256"]
            with open(trajs[0], "w", encoding="utf-8") as f:
                json.dump(t_dict, f)
        _recompute_manifest_hashes(copy_dir)
        try:
            verify_evidence_bundle(copy_dir)
            results["NC11_shorten_required_workload"] = False
        except (VerificationError, ValueError):
            results["NC11_shorten_required_workload"] = True

    # Control 12 (SEMANTIC): Omit required shift profiles in SOURCE_INPUTS.json (hash updated)
    with tempfile.TemporaryDirectory() as tmp_dir:
        copy_dir = Path(tmp_dir) / "c12"
        shutil.copytree(valid_dir, copy_dir)
        with open(copy_dir / "SOURCE_INPUTS.json", "r", encoding="utf-8") as f:
            src_in = json.load(f)
        src_in["shift_profiles"] = ["nominal"]  # Removed 6 shifted profiles
        with open(copy_dir / "SOURCE_INPUTS.json", "w", encoding="utf-8") as f:
            json.dump(src_in, f, indent=2)
        _recompute_manifest_hashes(copy_dir)
        try:
            verify_evidence_bundle(copy_dir)
            results["NC12_omit_full_profile_result"] = False
        except VerificationError:
            results["NC12_omit_full_profile_result"] = True

    # Verify positive reference after controls to confirm bundle was unaltered
    post_check = verify_evidence_bundle(valid_dir)
    assert post_check["status"] == "verified", "Reference bundle was corrupted by negative controls!"

    return results
