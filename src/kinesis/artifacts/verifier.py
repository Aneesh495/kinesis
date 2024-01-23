"""Independent evidence bundle verifier enforcing cryptographic, semantic, and boundary integrity."""

import hashlib
import json
from pathlib import Path
import sys
from typing import Any
import numpy as np
import torch

from kinesis.artifacts.manifest import EvidenceBundleManifest
from kinesis.contracts import Episode, Modality
from kinesis.evaluation.metrics import recompute_episode_metrics
from kinesis.models import WorldModel


class VerificationError(Exception):
    """Raised when an acceptance bundle fails an independent verification check."""
    pass


def verify_evidence_bundle(bundle_path: Path | str) -> dict[str, Any]:
    """Independently audit an evidence bundle without regenerating artifacts."""
    b_dir = Path(bundle_path)
    if not b_dir.exists() or not b_dir.is_dir():
        raise VerificationError(f"Evidence bundle directory does not exist: {b_dir}")

    manifest_file = b_dir / "MANIFEST.json"
    if not manifest_file.exists():
        raise VerificationError(f"MANIFEST.json missing from evidence directory: {b_dir}")

    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)
    manifest = EvidenceBundleManifest.from_dict(manifest_data)

    # 1. Cryptographic file integrity check
    for rel_path, record in manifest.files.items():
        target = b_dir / rel_path
        if not target.exists():
            raise VerificationError(f"Manifest referenced file missing: {rel_path}")

        hasher = hashlib.sha256()
        with open(target, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        current_sha = hasher.hexdigest()
        if current_sha != record.sha256:
            raise VerificationError(
                f"Checksum mismatch for {rel_path}: manifest={record.sha256} vs disk={current_sha}"
            )

    # 2. Check required metadata files
    for req in ["ENVIRONMENT.json", "SOURCE_INPUTS.json", "DATASETS.json", "METRICS.json", "ACCEPTANCE.json"]:
        if not (b_dir / req).exists():
            raise VerificationError(f"Required artifact file missing: {req}")

    # 3. Independent audit of raw trajectory records
    traj_dir = b_dir / "trajectories"
    if not traj_dir.exists() or not list(traj_dir.glob("*.json")):
        raise VerificationError("No raw trajectory records found in trajectories/ directory")

    traj_files = list(traj_dir.glob("*.json"))
    audited_episodes = 0
    total_recomputed_return = 0.0

    for tf in traj_files:
        try:
            with open(tf, "r", encoding="utf-8") as f:
                ep_dict = json.load(f)
            ep = Episode.from_dict(ep_dict)
        except Exception as e:
            raise VerificationError(f"Trajectory {tf.name} failed integrity verification: {e}")
        # Workload completeness check
        if len(ep.transitions) < 20:
            raise VerificationError(
                f"Episode {ep.episode_id} shortened below required workload: {len(ep.transitions)} transitions"
            )

        # Check information boundary: public observations cannot have privileged keys
        for t in ep.transitions:
            for k in t.observation.keys():
                if "privileged" in k.lower() or "true_friction" in k.lower():
                    raise VerificationError(f"Privileged leakage detected in transition {t.transition_id}: {k}")

        # Independently recompute return and metrics
        recomp = recompute_episode_metrics(ep)
        if not np.isclose(recomp["recomputed_return"], ep.total_return, atol=1e-4):
            raise VerificationError(
                f"Recomputed return {recomp['recomputed_return']} differs from recorded {ep.total_return}"
            )
        total_recomputed_return += recomp["recomputed_return"]
        audited_episodes += 1

    # 4. Cross-artifact agreement: METRICS.json vs raw trajectories
    with open(b_dir / "METRICS.json", "r", encoding="utf-8") as f:
        metrics_data = json.load(f)

    if metrics_data.get("total_eval_episodes", 0) != audited_episodes:
        raise VerificationError(
            f"Episode count mismatch: METRICS.json reports {metrics_data.get('total_eval_episodes')} "
            f"but found {audited_episodes} raw trajectory files"
        )

    # 5. Checkpoint deterministic probe replay
    ckpt_dir = b_dir / "checkpoints"
    ckpt_files = list(ckpt_dir.glob("*.pt"))
    if not ckpt_files:
        raise VerificationError("No saved tensor checkpoints found in checkpoints/ directory")

    first_ckpt = ckpt_files[0]
    payload = torch.load(first_ckpt, map_location="cpu", weights_only=False)

    if "probe_input" not in payload or "probe_output" not in payload:
        raise VerificationError(f"Checkpoint {first_ckpt.name} missing deterministic probe inputs/outputs")

    probe_in = torch.tensor(payload["probe_input"], dtype=torch.float32)
    expected_out = np.array(payload["probe_output"])

    wm = WorldModel(obs_dim=probe_in.shape[-1], action_dim=1, embed_dim=16, deter_dim=32, stoch_dim=8, hidden_dim=32)
    wm.load_state_dict(payload["model_state"], strict=False)
    with torch.no_grad():
        actual_out = wm.encoder(probe_in).cpu().numpy()

    if not np.allclose(actual_out, expected_out, atol=1e-4):
        raise VerificationError("Checkpoint probe replay failed: model weights do not reproduce reported output!")

    # 6. Check required shift profiles presence
    with open(b_dir / "SOURCE_INPUTS.json", "r", encoding="utf-8") as f:
        source_inputs = json.load(f)
    if len(source_inputs.get("shift_profiles", [])) < 7:
        raise VerificationError("Bundle does not contain nominal plus 6 required shift profiles")

    return {
        "status": "verified",
        "audited_episodes": audited_episodes,
        "total_files_verified": len(manifest.files),
        "mean_recomputed_return": total_recomputed_return / max(1, audited_episodes),
    }


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m kinesis.artifacts.verifier <path_to_evidence_bundle>")
        sys.exit(1)

    target_dir = sys.argv[1]
    try:
        res = verify_evidence_bundle(target_dir)
        print(f"[VERIFIED] Evidence bundle {target_dir} passed independent audit successfully!")
        print(f"           Audited {res['audited_episodes']} episodes across {res['total_files_verified']} files.")
        sys.exit(0)
    except VerificationError as e:
        print(f"[REJECTED] Independent verification failed: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
