#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

RGB = Tuple[int, int, int]
WHITE: RGB = (255, 255, 255)
NEIGHBORS_8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
]


class Step4Error(RuntimeError):
    pass


@dataclass
class BlobResult:
    color: RGB
    perimeter_px: float
    contour_points: List[Tuple[int, int]]


def to_rgb_array(img_path: Path) -> np.ndarray:
    with Image.open(img_path) as image:
        return np.array(image.convert("RGB"), dtype=np.uint8)


def rgb_to_hex(color: RGB) -> str:
    return f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"


def load_um_per_px(base_dir: Path, pinceaux_id: int, scale_um: float | None, scale_px: float | None) -> float:
    if scale_um is not None or scale_px is not None:
        if scale_um is None or scale_px is None:
            raise Step4Error("Both --scale-um and --scale-px must be provided together")
        if scale_um <= 0 or scale_px <= 0:
            raise Step4Error("--scale-um and --scale-px must be positive")
        return scale_um / scale_px

    metadata_path = base_dir / "Inputs" / "scale_metadata.json"
    if not metadata_path.exists():
        raise Step4Error(
            "No Inputs/scale_metadata.json found. Provide --scale-um and --scale-px explicitly."
        )

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    metadata_id = metadata.get("pinceaux_id")
    if metadata_id != pinceaux_id:
        raise Step4Error(
            f"scale_metadata.json is for pinceaux_{metadata_id}, not pinceaux_{pinceaux_id}. "
            "Provide --scale-um and --scale-px explicitly or update metadata."
        )

    um_per_px = metadata.get("um_per_px")
    if not isinstance(um_per_px, (int, float)) or um_per_px <= 0:
        raise Step4Error("Invalid um_per_px in Inputs/scale_metadata.json")

    return float(um_per_px)


def extract_component(mask: np.ndarray, start_y: int, start_x: int, visited: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    stack = [(start_y, start_x)]
    visited[start_y, start_x] = True
    component_coords: List[Tuple[int, int]] = []

    while stack:
        y, x = stack.pop()
        component_coords.append((y, x))
        for dy, dx in NEIGHBORS_8:
            ny, nx = y + dy, x + dx
            if ny < 0 or nx < 0 or ny >= h or nx >= w:
                continue
            if visited[ny, nx] or not mask[ny, nx]:
                continue
            visited[ny, nx] = True
            stack.append((ny, nx))

    comp = np.zeros_like(mask, dtype=bool)
    ys, xs = zip(*component_coords)
    comp[list(ys), list(xs)] = True
    return comp


def find_components(mask: np.ndarray) -> List[np.ndarray]:
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    components: List[np.ndarray] = []
    ys, xs = np.where(mask)

    for y, x in zip(ys, xs):
        if visited[y, x]:
            continue
        comp = extract_component(mask, int(y), int(x), visited)
        components.append(comp)

    return components


def is_boundary_pixel(mask: np.ndarray, y: int, x: int) -> bool:
    h, w = mask.shape
    if not mask[y, x]:
        return False

    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ny, nx = y + dy, x + dx
        if ny < 0 or nx < 0 or ny >= h or nx >= w or not mask[ny, nx]:
            return True
    return False


def find_start_boundary(mask: np.ndarray) -> Tuple[int, int]:
    ys, xs = np.where(mask)
    ordered = sorted(zip(ys.tolist(), xs.tolist()))
    for y, x in ordered:
        if is_boundary_pixel(mask, y, x):
            return y, x
    raise Step4Error("Boundary start not found for non-empty component")


def trace_boundary(component_mask: np.ndarray) -> List[Tuple[int, int]]:
    # Moore-neighbor tracing on padded mask
    padded = np.pad(component_mask.astype(np.uint8), 1, mode="constant", constant_values=0)
    sy, sx = find_start_boundary(component_mask)
    sy += 1
    sx += 1

    current = (sy, sx)
    backtrack = (sy, sx - 1)  # west
    start = current
    start_backtrack = backtrack

    boundary: List[Tuple[int, int]] = []
    max_steps = int(component_mask.size * 8)

    for _ in range(max_steps):
        cy, cx = current
        boundary.append((cy - 1, cx - 1))

        rel_back = (backtrack[0] - cy, backtrack[1] - cx)
        try:
            start_idx = NEIGHBORS_8.index(rel_back)
        except ValueError:
            start_idx = 7

        found_next = False
        next_pixel = current
        next_backtrack = backtrack

        for step in range(1, 9):
            idx = (start_idx + step) % 8
            dy, dx = NEIGHBORS_8[idx]
            ny, nx = cy + dy, cx + dx
            if padded[ny, nx] == 1:
                prev_idx = (idx - 1) % 8
                pdy, pdx = NEIGHBORS_8[prev_idx]
                next_backtrack = (cy + pdy, cx + pdx)
                next_pixel = (ny, nx)
                found_next = True
                break

        if not found_next:
            break

        backtrack = next_backtrack
        current = next_pixel

        if current == start and backtrack == start_backtrack:
            break

    if len(boundary) < 2:
        by, bx = np.where(component_mask)
        return list(zip(by.tolist(), bx.tolist()))

    return boundary


def contour_length_px(points: List[Tuple[int, int]]) -> float:
    if len(points) < 2:
        return 0.0

    length = 0.0
    for i in range(1, len(points)):
        y0, x0 = points[i - 1]
        y1, x1 = points[i]
        length += math.hypot(x1 - x0, y1 - y0)

    y0, x0 = points[-1]
    y1, x1 = points[0]
    length += math.hypot(x1 - x0, y1 - y0)
    return length


def blob_results_for_color(arr: np.ndarray, color: RGB) -> List[BlobResult]:
    mask = np.all(arr == color, axis=2)
    if not np.any(mask):
        return []

    components = find_components(mask)
    results: List[BlobResult] = []

    for comp in components:
        boundary_points = trace_boundary(comp)
        perimeter = contour_length_px(boundary_points)
        results.append(BlobResult(color=color, perimeter_px=perimeter, contour_points=boundary_points))

    return results


def draw_contours(image_shape: Tuple[int, int, int], blobs: List[BlobResult], output_path: Path) -> None:
    h, w, _ = image_shape
    canvas = Image.new("RGB", (w, h), WHITE)
    draw = ImageDraw.Draw(canvas)

    for blob in blobs:
        if len(blob.contour_points) < 2:
            continue
        xy = [(x, y) for y, x in blob.contour_points]
        xy.append((blob.contour_points[0][1], blob.contour_points[0][0]))
        draw.line(xy, fill=blob.color, width=3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def save_total_area_bar_chart(totals_per_color_area: Dict[RGB, float], output_path: Path) -> None:
    if not totals_per_color_area:
        raise Step4Error("No color totals available for bar chart")

    ordered = sorted(totals_per_color_area.items(), key=lambda item: item[1], reverse=True)
    labels = [rgb_to_hex(color) for color, _ in ordered]
    values = [area for _, area in ordered]
    bar_colors = labels

    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 0.5), 6), dpi=150)
    ax.bar(labels, values, color=bar_colors)
    ax.set_xlabel("MLI Color")
    ax.set_ylabel("Surface Area (um^2)")
    ax.set_title("Total MLI Surface Area by Color")
    ax.tick_params(axis="x", rotation=75)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def save_total_volume_bar_chart(totals_per_color_volume: Dict[RGB, float], output_path: Path) -> None:
    if not totals_per_color_volume:
        raise Step4Error("No color totals available for volume bar chart")

    ordered = sorted(totals_per_color_volume.items(), key=lambda item: item[1], reverse=True)
    labels = [rgb_to_hex(color) for color, _ in ordered]
    values = [volume for _, volume in ordered]
    bar_colors = labels

    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 0.5), 6), dpi=150)
    ax.bar(labels, values, color=bar_colors)
    ax.set_xlabel("MLI Color")
    ax.set_ylabel("Volume (um^3)")
    ax.set_title("Total MLI Volume by Color")
    ax.tick_params(axis="x", rotation=75)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def process_slice(
    arr: np.ndarray,
) -> Tuple[List[BlobResult], Dict[RGB, float], Dict[RGB, int], Dict[RGB, int]]:
    unique_colors = np.unique(arr.reshape(-1, 3), axis=0)
    colors = [tuple(map(int, c)) for c in unique_colors if tuple(map(int, c)) != WHITE]

    all_blobs: List[BlobResult] = []
    per_color_perimeter: Dict[RGB, float] = defaultdict(float)
    per_color_blob_count: Dict[RGB, int] = defaultdict(int)
    per_color_area_px: Dict[RGB, int] = defaultdict(int)

    for color in colors:
        color_blobs = blob_results_for_color(arr, color)
        all_blobs.extend(color_blobs)
        per_color_blob_count[color] += len(color_blobs)
        per_color_area_px[color] = int(np.count_nonzero(np.all(arr == color, axis=2)))
        for blob in color_blobs:
            per_color_perimeter[color] += blob.perimeter_px

    return all_blobs, per_color_perimeter, per_color_blob_count, per_color_area_px


def analyze_pinceaux(
    base_dir: Path,
    pinceaux_id: int,
    slice_thickness_nm: float,
    scale_um: float | None,
    scale_px: float | None,
) -> None:
    if slice_thickness_nm <= 0:
        raise Step4Error("slice thickness must be positive")

    um_per_px = load_um_per_px(base_dir, pinceaux_id, scale_um, scale_px)
    slice_thickness_um = slice_thickness_nm / 1000.0

    input_dir = base_dir / "Inputs" / "MLI_only" / f"pinceaux_{pinceaux_id}"
    output_dir = base_dir / "Outputs" / f"pinceaux_{pinceaux_id}"
    contour_dir = output_dir / "Contours"

    if not input_dir.exists():
        raise FileNotFoundError(f"MLI_only input folder not found: {input_dir}")

    png_files = sorted(input_dir.glob("*.png"))
    if not png_files:
        raise Step4Error(f"No PNG slices found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    contour_dir.mkdir(parents=True, exist_ok=True)

    per_slice_csv = output_dir / "slice_color_perimeter_area.csv"
    totals_csv = output_dir / "color_total_area.csv"
    totals_chart = output_dir / "color_total_area_bar_chart.png"
    per_slice_volume_csv = output_dir / "slice_color_area_volume.csv"
    totals_volume_csv = output_dir / "color_total_volume.csv"
    totals_volume_chart = output_dir / "color_total_volume_bar_chart.png"

    totals_per_color_area: Dict[RGB, float] = defaultdict(float)
    totals_per_color_perimeter_um: Dict[RGB, float] = defaultdict(float)
    totals_per_color_blob_count: Dict[RGB, int] = defaultdict(int)
    totals_per_color_volume_um3: Dict[RGB, float] = defaultdict(float)
    totals_per_color_area_um2_for_volume: Dict[RGB, float] = defaultdict(float)

    with per_slice_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "slice_file",
                "color_hex",
                "blob_count",
                "perimeter_px",
                "perimeter_um",
                "slice_area_um2",
                "slice_thickness_um",
            ],
        )
        writer.writeheader()

        with per_slice_volume_csv.open("w", newline="", encoding="utf-8") as volume_handle:
            volume_writer = csv.DictWriter(
                volume_handle,
                fieldnames=[
                    "slice_file",
                    "color_hex",
                    "area_px",
                    "cross_section_area_um2",
                    "slice_volume_um3",
                    "slice_thickness_um",
                ],
            )
            volume_writer.writeheader()

            for png_file in png_files:
                arr = to_rgb_array(png_file)
                blobs, per_color_perimeter_px, per_color_blob_count, per_color_area_px = process_slice(arr)

                draw_contours(arr.shape, blobs, contour_dir / png_file.name)

                for color, perimeter_px in per_color_perimeter_px.items():
                    perimeter_um = perimeter_px * um_per_px
                    slice_area_um2 = perimeter_um * slice_thickness_um
                    totals_per_color_area[color] += slice_area_um2
                    totals_per_color_perimeter_um[color] += perimeter_um
                    totals_per_color_blob_count[color] += per_color_blob_count[color]

                    writer.writerow(
                        {
                            "slice_file": png_file.name,
                            "color_hex": rgb_to_hex(color),
                            "blob_count": per_color_blob_count[color],
                            "perimeter_px": f"{perimeter_px:.6f}",
                            "perimeter_um": f"{perimeter_um:.6f}",
                            "slice_area_um2": f"{slice_area_um2:.6f}",
                            "slice_thickness_um": f"{slice_thickness_um:.6f}",
                        }
                    )

                for color, area_px in per_color_area_px.items():
                    cross_section_area_um2 = area_px * (um_per_px ** 2)
                    slice_volume_um3 = cross_section_area_um2 * slice_thickness_um
                    totals_per_color_volume_um3[color] += slice_volume_um3
                    totals_per_color_area_um2_for_volume[color] += cross_section_area_um2

                    volume_writer.writerow(
                        {
                            "slice_file": png_file.name,
                            "color_hex": rgb_to_hex(color),
                            "area_px": area_px,
                            "cross_section_area_um2": f"{cross_section_area_um2:.6f}",
                            "slice_volume_um3": f"{slice_volume_um3:.6f}",
                            "slice_thickness_um": f"{slice_thickness_um:.6f}",
                        }
                    )

    with totals_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["color_hex", "total_blob_count", "total_perimeter_um", "total_area_um2"],
        )
        writer.writeheader()
        for color in sorted(totals_per_color_area.keys()):
            writer.writerow(
                {
                    "color_hex": rgb_to_hex(color),
                    "total_blob_count": totals_per_color_blob_count[color],
                    "total_perimeter_um": f"{totals_per_color_perimeter_um[color]:.6f}",
                    "total_area_um2": f"{totals_per_color_area[color]:.6f}",
                }
            )

    save_total_area_bar_chart(totals_per_color_area, totals_chart)

    with totals_volume_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "color_hex",
                "sum_cross_section_area_um2",
                "slice_thickness_um",
                "total_volume_um3",
            ],
        )
        writer.writeheader()
        for color in sorted(totals_per_color_volume_um3.keys()):
            writer.writerow(
                {
                    "color_hex": rgb_to_hex(color),
                    "sum_cross_section_area_um2": f"{totals_per_color_area_um2_for_volume[color]:.6f}",
                    "slice_thickness_um": f"{slice_thickness_um:.6f}",
                    "total_volume_um3": f"{totals_per_color_volume_um3[color]:.6f}",
                }
            )

    save_total_volume_bar_chart(totals_per_color_volume_um3, totals_volume_chart)

    print(f"Processed {len(png_files)} slices for pinceaux_{pinceaux_id}.")
    print(f"Contours saved in: {contour_dir}")
    print(f"Per-slice table: {per_slice_csv}")
    print(f"Color totals table: {totals_csv}")
    print(f"Color totals bar chart: {totals_chart}")
    print(f"Per-slice volume table: {per_slice_volume_csv}")
    print(f"Color volume totals table: {totals_volume_csv}")
    print(f"Color volume bar chart: {totals_volume_chart}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Step 4: detect color blobs per MLI_only slice, estimate contour perimeters, "
            "and compute slice area contributions."
        )
    )
    parser.add_argument("--id", type=int, required=True, help="Pinceaux identifier X")
    parser.add_argument(
        "--slice-thickness-nm",
        type=float,
        default=40.0,
        help="Slice thickness in nanometers (default: 40)",
    )
    parser.add_argument("--scale-um", type=float, default=None, help="Scale bar length in um")
    parser.add_argument("--scale-px", type=float, default=None, help="Scale bar length in pixels")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Repository root directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyze_pinceaux(
        base_dir=args.base_dir,
        pinceaux_id=args.id,
        slice_thickness_nm=args.slice_thickness_nm,
        scale_um=args.scale_um,
        scale_px=args.scale_px,
    )


if __name__ == "__main__":
    main()
