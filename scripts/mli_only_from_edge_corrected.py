#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
from PIL import Image

RGB = Tuple[int, int, int]
WHITE: RGB = (255, 255, 255)
PC_GRAY: RGB = (128, 128, 128)  # #808080
SCALEBAR_GRAY: RGB = (178, 178, 178)  # #b2b2b2


class MLIOnlyError(RuntimeError):
    pass


@dataclass
class SliceSummary:
    filename: str
    gray_pixels_whitened: int
    edge_components_removed: int
    edge_component_pixels_removed: int
    output_path: str


def to_rgb_array(img_path: Path) -> np.ndarray:
    with Image.open(img_path) as image:
        rgb = image.convert("RGB")
        return np.array(rgb, dtype=np.uint8)


def save_rgb_array(arr: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="RGB").save(output_path)


def remove_gray_pixels(arr: np.ndarray) -> Tuple[np.ndarray, int]:
    working = arr.copy()
    is_pc_gray = np.all(working == PC_GRAY, axis=2)
    is_scalebar_gray = np.all(working == SCALEBAR_GRAY, axis=2)
    gray_mask = is_pc_gray | is_scalebar_gray
    count = int(np.count_nonzero(gray_mask))
    working[gray_mask] = WHITE
    return working, count


def get_neighbors_8(y: int, x: int, h: int, w: int):
    for ny in range(max(0, y - 1), min(h, y + 2)):
        for nx in range(max(0, x - 1), min(w, x + 2)):
            if ny == y and nx == x:
                continue
            yield ny, nx


def remove_edge_touching_nonwhite_blobs(arr: np.ndarray) -> Tuple[np.ndarray, int, int]:
    working = arr.copy()
    h, w, _ = working.shape

    visited = np.zeros((h, w), dtype=bool)
    removed_components = 0
    removed_pixels = 0

    edge_coords: List[Tuple[int, int]] = []
    if h == 0 or w == 0:
        return working, removed_components, removed_pixels

    for x in range(w):
        edge_coords.append((0, x))
        if h > 1:
            edge_coords.append((h - 1, x))
    for y in range(1, h - 1):
        edge_coords.append((y, 0))
        if w > 1:
            edge_coords.append((y, w - 1))

    for sy, sx in edge_coords:
        if visited[sy, sx]:
            continue

        start_color = tuple(map(int, working[sy, sx]))
        if start_color == WHITE:
            visited[sy, sx] = True
            continue

        stack = [(sy, sx)]
        visited[sy, sx] = True
        component: List[Tuple[int, int]] = []

        while stack:
            y, x = stack.pop()
            component.append((y, x))
            for ny, nx in get_neighbors_8(y, x, h, w):
                if visited[ny, nx]:
                    continue
                color = tuple(map(int, working[ny, nx]))
                if color == start_color:
                    visited[ny, nx] = True
                    stack.append((ny, nx))

        if component:
            removed_components += 1
            removed_pixels += len(component)
            ys, xs = zip(*component)
            working[list(ys), list(xs)] = WHITE

    return working, removed_components, removed_pixels


def process_single_slice(input_path: Path, output_path: Path) -> SliceSummary:
    original = to_rgb_array(input_path)
    no_gray, gray_pixels_whitened = remove_gray_pixels(original)
    cleaned, removed_components, removed_pixels = remove_edge_touching_nonwhite_blobs(no_gray)
    save_rgb_array(cleaned, output_path)

    return SliceSummary(
        filename=input_path.name,
        gray_pixels_whitened=gray_pixels_whitened,
        edge_components_removed=removed_components,
        edge_component_pixels_removed=removed_pixels,
        output_path=str(output_path),
    )


def write_scale_metadata(base_dir: Path, pinceaux_id: int, scale_um: float, scale_px: float) -> None:
    if scale_px <= 0:
        raise MLIOnlyError("scale-px must be > 0")
    if scale_um <= 0:
        raise MLIOnlyError("scale-um must be > 0")

    um_per_px = scale_um / scale_px
    nm_per_px = um_per_px * 1000.0
    metadata = {
        "pinceaux_id": pinceaux_id,
        "scale_um": scale_um,
        "scale_px": scale_px,
        "um_per_px": um_per_px,
        "nm_per_px": nm_per_px,
        "uniform_across_slices": True,
    }

    outputs_dir = base_dir / "Outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = outputs_dir / "scale_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def process_pinceaux_id(base_dir: Path, pinceaux_id: int, scale_um: float, scale_px: float) -> List[SliceSummary]:
    input_dir = base_dir / "Inputs" / "Edge_Corrected" / f"pinceaux_{pinceaux_id}"
    output_dir = base_dir / "Inputs" / "MLI_only" / f"pinceaux_{pinceaux_id}"

    if not input_dir.exists():
        raise FileNotFoundError(f"Edge-corrected input folder not found: {input_dir}")

    png_files = sorted(input_dir.glob("*.png"))
    if not png_files:
        raise MLIOnlyError(f"No PNG files found in {input_dir}")

    write_scale_metadata(base_dir, pinceaux_id, scale_um, scale_px)

    summaries: List[SliceSummary] = []
    for png_file in png_files:
        out_file = output_dir / png_file.name
        summary = process_single_slice(png_file, out_file)
        summaries.append(summary)

    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create MLI_only slices from Edge_Corrected by whitening #808080/#b2b2b2 and "
            "removing edge-touching non-white blobs."
        )
    )
    parser.add_argument("--id", type=int, required=True, help="Pinceaux identifier X")
    parser.add_argument(
        "--scale-um",
        type=float,
        required=True,
        help="Scale bar length in micrometers (uniform for this pinceaux)",
    )
    parser.add_argument(
        "--scale-px",
        type=float,
        required=True,
        help="Scale bar length in pixels (uniform for this pinceaux)",
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
    summaries = process_pinceaux_id(args.base_dir, args.id, args.scale_um, args.scale_px)

    total_gray_whitened = sum(s.gray_pixels_whitened for s in summaries)
    total_components_removed = sum(s.edge_components_removed for s in summaries)
    total_component_pixels_removed = sum(s.edge_component_pixels_removed for s in summaries)

    print(f"Processed {len(summaries)} slices for pinceaux_{args.id}.")
    print(f"Total gray pixels whitened: {total_gray_whitened}")
    print(f"Total edge-touching components removed: {total_components_removed}")
    print(f"Total pixels removed via edge-touching components: {total_component_pixels_removed}")


if __name__ == "__main__":
    main()
