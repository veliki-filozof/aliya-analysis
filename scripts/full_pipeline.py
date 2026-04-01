#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

# Ensure scripts directory is in path for imports
scripts_dir = Path(__file__).parent
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

from edge_cleanup import load_allowed_colors, process_single_slice as edge_process_single_slice
from mli_only_from_edge_corrected import (
    process_single_slice as mli_process_single_slice,
    write_scale_metadata,
)
from perimeter_area_step4 import analyze_pinceaux


@dataclass
class SliceMapRow:
    index: int
    original_filename: str
    processed_filename: str
    z_coordinate: float


@dataclass
class PipelineConfig:
    scale_um: float
    scale_px: float
    slice_thickness_nm: float
    z_first: float
    z_last: float
    capture_order: str


def raw_config_path(base_dir: Path, pinceaux_id: int) -> Path:
    return base_dir / "Inputs" / "Raw" / f"pinceaux_{pinceaux_id}" / "analysis_config.json"


def load_raw_config(base_dir: Path, pinceaux_id: int) -> Optional[PipelineConfig]:
    config_path = raw_config_path(base_dir, pinceaux_id)
    if not config_path.exists():
        return None

    with config_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    required = ["scale_um", "scale_px", "slice_thickness_nm", "z_first", "z_last", "capture_order"]
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"Missing keys in {config_path}: {missing}")

    config = PipelineConfig(
        scale_um=float(data["scale_um"]),
        scale_px=float(data["scale_px"]),
        slice_thickness_nm=float(data["slice_thickness_nm"]),
        z_first=float(data["z_first"]),
        z_last=float(data["z_last"]),
        capture_order=str(data["capture_order"]),
    )
    return config


def write_raw_config(base_dir: Path, pinceaux_id: int, config: PipelineConfig) -> Path:
    config_path = raw_config_path(base_dir, pinceaux_id)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "scale_um": config.scale_um,
        "scale_px": config.scale_px,
        "slice_thickness_nm": config.slice_thickness_nm,
        "z_first": config.z_first,
        "z_last": config.z_last,
        "capture_order": config.capture_order,
    }
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return config_path


def resolve_config(
    base_dir: Path,
    pinceaux_id: int,
    scale_um: Optional[float],
    scale_px: Optional[float],
    slice_thickness_nm: Optional[float],
    z_first: Optional[float],
    z_last: Optional[float],
    capture_order: Optional[str],
) -> PipelineConfig:
    raw_cfg = load_raw_config(base_dir, pinceaux_id)

    resolved = PipelineConfig(
        scale_um=scale_um if scale_um is not None else (raw_cfg.scale_um if raw_cfg else None),
        scale_px=scale_px if scale_px is not None else (raw_cfg.scale_px if raw_cfg else None),
        slice_thickness_nm=(
            slice_thickness_nm
            if slice_thickness_nm is not None
            else (raw_cfg.slice_thickness_nm if raw_cfg else None)
        ),
        z_first=z_first if z_first is not None else (raw_cfg.z_first if raw_cfg else None),
        z_last=z_last if z_last is not None else (raw_cfg.z_last if raw_cfg else None),
        capture_order=(
            capture_order if capture_order is not None else (raw_cfg.capture_order if raw_cfg else None)
        ),
    )

    missing = []
    if resolved.scale_um is None:
        missing.append("scale_um")
    if resolved.scale_px is None:
        missing.append("scale_px")
    if resolved.slice_thickness_nm is None:
        missing.append("slice_thickness_nm")
    if resolved.z_first is None:
        missing.append("z_first")
    if resolved.z_last is None:
        missing.append("z_last")
    if resolved.capture_order is None:
        missing.append("capture_order")

    if missing:
        cfg_path = raw_config_path(base_dir, pinceaux_id)
        raise ValueError(
            "Missing required inputs and no usable raw config fallback for keys: "
            f"{missing}. Expected config at {cfg_path}"
        )

    return resolved


def build_z_values(n_slices: int, z_first: float, z_last: float) -> List[float]:
    if n_slices <= 0:
        return []
    if n_slices == 1:
        return [float(z_first)]
    return np.linspace(z_first, z_last, n_slices).tolist()


def build_processed_name(index: int, z_value: float) -> str:
    return f"z_{index:04d}_{z_value:.6f}.png"


def write_slice_mapping_csv(output_dir: Path, rows: List[SliceMapRow]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    mapping_path = output_dir / "slice_z_mapping.csv"
    with mapping_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["slice_index", "original_filename", "processed_filename", "z_coordinate"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "slice_index": row.index,
                    "original_filename": row.original_filename,
                    "processed_filename": row.processed_filename,
                    "z_coordinate": f"{row.z_coordinate:.6f}",
                }
            )
    return mapping_path


def write_run_metadata(
    output_dir: Path,
    pinceaux_id: int,
    scale_um: float,
    scale_px: float,
    slice_thickness_nm: float,
    z_first: float,
    z_last: float,
    capture_order: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "run_metadata.json"
    payload = {
        "pinceaux_id": pinceaux_id,
        "scale_um": scale_um,
        "scale_px": scale_px,
        "slice_thickness_nm": slice_thickness_nm,
        "z_first": z_first,
        "z_last": z_last,
        "capture_order": capture_order,
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return metadata_path


def run_full_pipeline(
    base_dir: Path,
    pinceaux_id: int,
    scale_um: Optional[float] = None,
    scale_px: Optional[float] = None,
    slice_thickness_nm: Optional[float] = None,
    z_first: Optional[float] = None,
    z_last: Optional[float] = None,
    capture_order: Optional[str] = None,
    write_config: bool = False,
) -> None:
    raw_dir = base_dir / "Inputs" / "Raw" / f"pinceaux_{pinceaux_id}"
    edge_dir = base_dir / "Inputs" / "Edge_Corrected" / f"pinceaux_{pinceaux_id}"
    mli_dir = base_dir / "Inputs" / "MLI_only" / f"pinceaux_{pinceaux_id}"
    output_dir = base_dir / "Outputs" / f"pinceaux_{pinceaux_id}"

    config = resolve_config(
        base_dir=base_dir,
        pinceaux_id=pinceaux_id,
        scale_um=scale_um,
        scale_px=scale_px,
        slice_thickness_nm=slice_thickness_nm,
        z_first=z_first,
        z_last=z_last,
        capture_order=capture_order,
    )

    if config.capture_order not in {"ascending", "descending"}:
        raise ValueError("capture_order must be 'ascending' or 'descending'")

    if config.scale_um <= 0 or config.scale_px <= 0:
        raise ValueError("scale_um and scale_px must be > 0")

    if config.slice_thickness_nm <= 0:
        raise ValueError("slice_thickness_nm must be > 0")

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")

    raw_files = sorted(raw_dir.glob("*.png"))
    if not raw_files:
        raise FileNotFoundError(f"No PNG files found in: {raw_dir}")

    z_values = build_z_values(len(raw_files), config.z_first, config.z_last)
    allowed_colors = load_allowed_colors(base_dir, pinceaux_id)

    edge_dir.mkdir(parents=True, exist_ok=True)
    mli_dir.mkdir(parents=True, exist_ok=True)

    rows: List[SliceMapRow] = []

    print(f"Stage 1/3: Edge correction for {len(raw_files)} slices...")
    for index, (raw_path, z_value) in enumerate(zip(raw_files, z_values), start=1):
        processed_name = build_processed_name(index, z_value)
        edge_path = edge_dir / processed_name
        edge_process_single_slice(raw_path, edge_path, allowed_colors)
        rows.append(
            SliceMapRow(
                index=index,
                original_filename=raw_path.name,
                processed_filename=processed_name,
                z_coordinate=float(z_value),
            )
        )

    print("Stage 2/3: Scale metadata + MLI-only generation...")
    write_scale_metadata(base_dir, pinceaux_id, config.scale_um, config.scale_px)
    for row in rows:
        edge_path = edge_dir / row.processed_filename
        mli_path = mli_dir / row.processed_filename
        mli_process_single_slice(edge_path, mli_path)

    print("Stage 3/3: Surface area + volume analysis + contours...")
    analyze_pinceaux(
        base_dir=base_dir,
        pinceaux_id=pinceaux_id,
        slice_thickness_nm=config.slice_thickness_nm,
        scale_um=config.scale_um,
        scale_px=config.scale_px,
    )

    mapping_path = write_slice_mapping_csv(output_dir, rows)
    metadata_path = write_run_metadata(
        output_dir=output_dir,
        pinceaux_id=pinceaux_id,
        scale_um=config.scale_um,
        scale_px=config.scale_px,
        slice_thickness_nm=config.slice_thickness_nm,
        z_first=config.z_first,
        z_last=config.z_last,
        capture_order=config.capture_order,
    )

    if write_config:
        cfg_path = write_raw_config(base_dir, pinceaux_id, config)
        print(f"Raw config saved to: {cfg_path}")

    if config.capture_order == "ascending" and config.z_last < config.z_first:
        print("Warning: capture_order is 'ascending' but z_last < z_first.")
    if config.capture_order == "descending" and config.z_last > config.z_first:
        print("Warning: capture_order is 'descending' but z_last > z_first.")

    print(f"Done. Mapping saved to: {mapping_path}")
    print(f"Run metadata saved to: {metadata_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full pinceaux analysis pipeline with z-based naming")
    parser.add_argument("--id", type=int, required=True, help="Pinceaux identifier X")
    parser.add_argument("--scale-um", type=float, required=False, help="Scale bar length in um")
    parser.add_argument("--scale-px", type=float, required=False, help="Scale bar length in pixels")
    parser.add_argument("--slice-thickness-nm", type=float, default=None)
    parser.add_argument("--z-first", type=float, required=False, help="Z of the first screenshot slice")
    parser.add_argument("--z-last", type=float, required=False, help="Z of the last screenshot slice")
    parser.add_argument(
        "--capture-order",
        type=str,
        choices=["ascending", "descending"],
        required=False,
        help="Whether screenshots were taken in ascending or descending z order",
    )
    parser.add_argument(
        "--write-config",
        action="store_true",
        help="Write resolved inputs into Inputs/Raw/pinceaux_X/analysis_config.json",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Repository root directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_full_pipeline(
        base_dir=args.base_dir,
        pinceaux_id=args.id,
        scale_um=args.scale_um,
        scale_px=args.scale_px,
        slice_thickness_nm=args.slice_thickness_nm,
        z_first=args.z_first,
        z_last=args.z_last,
        capture_order=args.capture_order,
        write_config=args.write_config,
    )


if __name__ == "__main__":
    main()
