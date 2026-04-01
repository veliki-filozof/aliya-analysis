#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
from PIL import Image

RGB = Tuple[int, int, int]
WHITE: RGB = (255, 255, 255)
PC_GRAY: RGB = (128, 128, 128)  # #808080
SCALEBAR_GRAY: RGB = (178, 178, 178)  # #b2b2b2
FORCED_ALLOWED_COLORS: Set[RGB] = {WHITE, PC_GRAY, SCALEBAR_GRAY}
NEIGHBORHOOD_RADIUS = 2  # 5x5 mask


class EdgeCleanupError(RuntimeError):
    pass


@dataclass
class SliceSummary:
    filename: str
    allowed_colors_count: int
    reassigned_pixels: int
    output_path: str


def to_rgb_array(img_path: Path) -> np.ndarray:
    with Image.open(img_path) as image:
        rgb = image.convert("RGB")
        return np.array(rgb, dtype=np.uint8)


def save_rgb_array(arr: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="RGB").save(output_path)


def hex_to_rgb(hex_color: str) -> RGB:
    value = hex_color.strip().lower()
    if not value.startswith("#"):
        raise EdgeCleanupError(f"Invalid hex color (missing #): {hex_color}")
    if len(value) != 7:
        raise EdgeCleanupError(f"Invalid hex color length: {hex_color}")

    try:
        return (int(value[1:3], 16), int(value[3:5], 16), int(value[5:7], 16))
    except ValueError as exc:
        raise EdgeCleanupError(f"Invalid hex color value: {hex_color}") from exc


def load_allowed_colors(base_dir: Path, pinceaux_id: int) -> Set[RGB]:
    json_path = base_dir / "Inputs" / f"pinceaux_{pinceaux_id}.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Metadata JSON not found: {json_path}")

    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    layers = data.get("layers")
    if not isinstance(layers, list):
        raise EdgeCleanupError("Invalid JSON structure: 'layers' must be a list")

    hex_colors: Set[str] = set()
    for layer in layers:
        if not isinstance(layer, dict):
            continue
        if layer.get("type") != "segmentation":
            continue

        segment_colors = layer.get("segmentColors")
        if not isinstance(segment_colors, dict):
            continue

        for color in segment_colors.values():
            if isinstance(color, str):
                hex_colors.add(color)

    if not hex_colors:
        raise EdgeCleanupError(
            f"No colors found in segmentColors for pinceaux_{pinceaux_id}"
        )

    allowed_colors = {hex_to_rgb(color) for color in hex_colors}
    allowed_colors.update(FORCED_ALLOWED_COLORS)
    return allowed_colors


def choose_replacement_color(
    arr: np.ndarray,
    y: int,
    x: int,
    allowed_colors: Set[RGB],
) -> RGB:
    h, w, _ = arr.shape
    y0 = max(0, y - NEIGHBORHOOD_RADIUS)
    y1 = min(h, y + NEIGHBORHOOD_RADIUS + 1)
    x0 = max(0, x - NEIGHBORHOOD_RADIUS)
    x1 = min(w, x + NEIGHBORHOOD_RADIUS + 1)

    patch = arr[y0:y1, x0:x1]
    patch_colors = [tuple(map(int, c)) for c in patch.reshape(-1, 3)]

    center_index = (y - y0) * (x1 - x0) + (x - x0)
    patch_colors.pop(center_index)

    counts = Counter(c for c in patch_colors if c in allowed_colors)
    if not counts:
        return WHITE

    max_count = max(counts.values())
    top_colors = [color for color, count in counts.items() if count == max_count]
    if WHITE in top_colors:
        return WHITE
    return top_colors[0]


def reassign_non_allowed_pixels(arr: np.ndarray, allowed_colors: Set[RGB]) -> Tuple[np.ndarray, int]:
    working = arr.copy()

    h, w, _ = working.shape
    assignments: Dict[Tuple[int, int], RGB] = {}

    for y in range(h):
        for x in range(w):
            color = tuple(map(int, working[y, x]))
            if color in allowed_colors:
                continue
            assignments[(y, x)] = choose_replacement_color(working, y, x, allowed_colors)

    for (y, x), new_color in assignments.items():
        working[y, x] = new_color

    return working, len(assignments)


def validate_only_allowed_colors(arr: np.ndarray, allowed_colors: Set[RGB]) -> None:
    unique_colors = {tuple(map(int, c)) for c in np.unique(arr.reshape(-1, 3), axis=0)}
    disallowed = sorted(unique_colors - allowed_colors)
    if disallowed:
        sample = ", ".join(str(c) for c in disallowed[:15])
        raise EdgeCleanupError(
            "Processed image contains colors not in JSON allowed set. "
            f"Found {len(disallowed)} unexpected colors. Sample: {sample}"
        )


def process_single_slice(
    input_path: Path,
    output_path: Path,
    allowed_colors: Set[RGB],
) -> SliceSummary:
    original = to_rgb_array(input_path)
    working = original.copy()

    cleaned, reassigned_pixels = reassign_non_allowed_pixels(working, allowed_colors)

    validate_only_allowed_colors(cleaned, allowed_colors)
    save_rgb_array(cleaned, output_path)

    return SliceSummary(
        filename=input_path.name,
        allowed_colors_count=len(allowed_colors),
        reassigned_pixels=reassigned_pixels,
        output_path=str(output_path),
    )


def process_pinceaux_id(base_dir: Path, pinceaux_id: int) -> List[SliceSummary]:
    input_dir = base_dir / "Inputs" / "Raw" / f"pinceaux_{pinceaux_id}"
    output_dir = base_dir / "Inputs" / "Edge_Corrected" / f"pinceaux_{pinceaux_id}"
    allowed_colors = load_allowed_colors(base_dir, pinceaux_id)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    png_files = sorted(input_dir.glob("*.png"))
    if not png_files:
        raise EdgeCleanupError(f"No PNG files found in {input_dir}")

    summaries: List[SliceSummary] = []
    for png_file in png_files:
        out_file = output_dir / png_file.name
        summary = process_single_slice(png_file, out_file, allowed_colors)
        summaries.append(summary)

    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Edge cleanup for pinceaux slices: removes anti-aliasing gradient artifacts."
    )
    parser.add_argument(
        "--id",
        type=int,
        required=True,
        help="Pinceaux identifier X for Inputs/Raw/pinceaux_X",
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
    summaries = process_pinceaux_id(args.base_dir, args.id)

    total_reassigned = sum(s.reassigned_pixels for s in summaries)
    print(f"Processed {len(summaries)} slices for pinceaux_{args.id}.")
    print(f"Total reassigned edge pixels: {total_reassigned}")


if __name__ == "__main__":
    main()
