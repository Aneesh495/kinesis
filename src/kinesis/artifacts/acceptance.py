"""Acceptance runner CLI entrypoint generating verified acceptance bundle."""

import argparse
from pathlib import Path
import sys

from kinesis.artifacts.builder import build_acceptance_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Kinesis acceptance evidence bundle generator")
    parser.add_argument("--run-id", type=str, default="acceptance_run_v1", help="Unique run identifier")
    parser.add_argument("--output-base", type=str, default="evidence", help="Base directory for evidence")
    args = parser.parse_args()

    print(f"[ACCEPTANCE] Generating complete acceptance bundle for run '{args.run_id}'...")
    bundle_path = build_acceptance_bundle(run_id=args.run_id, output_base=args.output_base)
    print(f"[ACCEPTANCE] Acceptance bundle successfully generated at: {bundle_path.resolve()}")


if __name__ == "__main__":
    main()
