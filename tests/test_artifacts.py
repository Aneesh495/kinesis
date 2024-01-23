"""Tests for acceptance evidence generation, independent verifier, and 12 negative controls."""

from pathlib import Path
import pytest

from kinesis.artifacts import (
    build_acceptance_bundle,
    run_negative_controls,
    verify_evidence_bundle,
)


def test_acceptance_bundle_generation_and_positive_verification(tmp_path: Path) -> None:
    bundle_dir = build_acceptance_bundle(run_id="test_run_bundle", output_base=tmp_path)
    assert bundle_dir.exists()
    assert (bundle_dir / "MANIFEST.json").exists()
    assert (bundle_dir / "ACCEPTANCE.json").exists()
    assert (bundle_dir / "ENVIRONMENT.json").exists()
    assert (bundle_dir / "METRICS.json").exists()

    # Positive independent verification
    res = verify_evidence_bundle(bundle_dir)
    assert res["status"] == "verified"
    assert res["audited_episodes"] > 0


def test_gate_ra12_twelve_negative_controls(tmp_path: Path) -> None:
    """Execute all 12 negative controls and verify 100 percent detection failure rate."""
    bundle_dir = build_acceptance_bundle(run_id="test_controls_bundle", output_base=tmp_path)

    control_results = run_negative_controls(bundle_dir)
    assert len(control_results) == 12, f"Expected 12 negative controls, got {len(control_results)}"

    for name, passed in control_results.items():
        assert passed, f"Negative control {name} was expected to fail verification but passed!"
