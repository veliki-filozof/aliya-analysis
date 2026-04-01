#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from full_pipeline import raw_config_path, run_full_pipeline


def discover_pinceaux_ids(raw_root: Path) -> list[int]:
    ids: list[int] = []
    for path in sorted(raw_root.glob("pinceaux_*")):
        if not path.is_dir():
            continue
        suffix = path.name.replace("pinceaux_", "", 1)
        if not suffix.isdigit():
            continue

        pinceaux_id = int(suffix)
        has_png = any(path.glob("*.png"))
        has_cfg = raw_config_path(raw_root.parent.parent, pinceaux_id).exists()
        if has_png and has_cfg:
            ids.append(pinceaux_id)
    return ids


def main() -> None:
    base_dir = Path(__file__).resolve().parent
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
