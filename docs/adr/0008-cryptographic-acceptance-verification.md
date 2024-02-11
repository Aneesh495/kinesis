# ADR-0008: Cryptographic Acceptance Verification and Negative Controls

## Status
Accepted

## Context
Scientific and engineering evaluation packages often suffer from unverified summaries, where reported results diverge from raw trajectory data or where corrupted files go unnoticed. A rigorous experimental laboratory requires automated, independent verification of raw evidence packages.

## Decision
We implement a zero-trust verification pipeline in `kinesis.artifacts`:
1. `Evidence Manifest`: Generates `MANIFEST.json` containing SHA-256 digests and file sizes for all trajectory JSONs, model weights, environment specs, and summary reports.
2. `Independent Verifier`: A standalone verifier (`verify_evidence_bundle`) that re-hashes every file, recomputes episode returns directly from raw per-step transition rewards, and audits observation keys for privileged leakage.
3. `Comprehensive Negative Controls`: Exactly 12 programmatic negative controls mutating checksums, truncating files, modifying rewards, altering seeds, injecting privileged keys, and stripping shift profiles. The verifier must reject all 12 negative controls with explicit errors.

## Consequences
### Positive
- Ensures complete tamper-evidence for benchmark results and acceptance evidence bundles.
- Proves that the verifier is sensitive to corruption through unit-tested negative controls.
- Enables reproducible auditing on air-gapped or independent verification environments.

### Negative
- Requires maintaining negative control mutation fixtures alongside the positive verifier.
