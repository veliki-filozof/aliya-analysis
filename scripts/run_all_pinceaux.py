#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from unified_analysis import discover_pinceaux_ids, raw_config_path, run_full_pipeline


def main() -> None:
    base_dir = Path(__file__).resolve().parent.parent
    raw_root = base_dir / "Inputs" / "Raw"
    if not raw_root.exists():
        raise FileNotFoundError(f"Raw inputs root not found: {raw_root}")

    ids = discover_pinceaux_ids(raw_root)
    if not ids:
        print("No pinceaux folders with both PNG slices and analysis_config.json found. Nothing to run.")
        return

    print(f"Found {len(ids)} pinceaux to process: {ids}")
    for pinceaux_id in ids:
        print(f"\n=== Running pinceaux_{pinceaux_id} ===")
        run_full_pipeline(base_dir=base_dir, pinceaux_id=pinceaux_id)

    print("\nAll available pinceaux processed.")


if __name__ == "__main__":
    main()
