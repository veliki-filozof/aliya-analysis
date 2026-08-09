#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_volume_csv(path: Path) -> Dict[str, float]:
    out: Dict[str, float] = {}
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = row.get("color_hex")
            if not key:
                continue
            try:
                out[key] = float(row.get("total_volume_um3", row.get("total_volume_um3", "0")))
            except Exception:
                out[key] = float(row.get("total_volume_um3", 0))
    return out


def read_area_csv(path: Path) -> Dict[str, float]:
    out: Dict[str, float] = {}
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = row.get("color_hex")
            if not key:
                continue
            try:
                out[key] = float(row.get("total_area_um2", row.get("total_area_um2", "0")))
            except Exception:
                out[key] = float(row.get("total_area_um2", 0))
    return out


def compute_sa_to_v(base_dir: Path, pinceaux_id: int) -> Path:
    base_dir = Path(base_dir)
    output_dir = base_dir / "Outputs" / f"pinceaux_{pinceaux_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    vol_path = output_dir / "color_total_volume.csv"
    area_path = output_dir / "color_total_area.csv"
    if not vol_path.exists() or not area_path.exists():
        raise FileNotFoundError(f"Missing input CSVs in {output_dir}")

    volumes = read_volume_csv(vol_path)
    areas = read_area_csv(area_path)

    colors: List[str] = sorted(set(volumes) | set(areas), key=lambda c: volumes.get(c, 0.0), reverse=True)

    rows: List[Dict[str, str]] = []
    sa_values: List[float] = []
    colors_for_plot: List[str] = []

    for c in colors:
        v = volumes.get(c, 0.0)
        a = areas.get(c, 0.0)
        if v == 0.0:
            sa = float("nan")
        else:
            sa = a / v
        rows.append(
            {
                "color_hex": c,
                "total_volume_um3": f"{v:.6f}",
                "total_area_um2": f"{a:.6f}",
                "surface_area_to_volume_um_inv": f"{sa:.6f}" if not math.isnan(sa) else "",
            }
        )
        sa_values.append(sa)
        colors_for_plot.append(c)

    out_csv = output_dir / "sa_to_v.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "color_hex",
            "total_volume_um3",
            "total_area_um2",
            "surface_area_to_volume_um_inv",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    def write_plot(sorted_rows: List[Dict[str, str]], png_name: str, title: str) -> Path:
        out_png = output_dir / png_name
        n = len(sorted_rows)
        width = max(6, n * 0.25)
        fig, ax = plt.subplots(figsize=(width, 4))
        x = list(range(n))
        y = []
        colors = []
        labels = []
        for row in sorted_rows:
            raw_sa = row["surface_area_to_volume_um_inv"]
            if raw_sa == "":
                y.append(0.0)
            else:
                y.append(float(raw_sa))
            colors.append(row["color_hex"])
            labels.append(row["color_hex"])
        ax.bar(x, y, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_ylabel("Surface area / Volume (um^-1)")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        return out_png

    by_volume = sorted(rows, key=lambda row: float(row["total_volume_um3"]), reverse=True)
    by_area = sorted(rows, key=lambda row: float(row["total_area_um2"]), reverse=True)

    write_plot(by_volume, "sa_to_v.png", f"SA:V by color sorted by total volume for pinceaux_{pinceaux_id}")
    write_plot(
        by_volume,
        "sa_to_v_sorted_by_volume.png",
        f"SA:V by color sorted by total volume for pinceaux_{pinceaux_id}",
    )
    write_plot(
        by_area,
        "sa_to_v_sorted_by_area.png",
        f"SA:V by color sorted by total area for pinceaux_{pinceaux_id}",
    )

    return out_csv


def parse_args():
    parser = argparse.ArgumentParser(description="Compute surface-area-to-volume ratio per color and plot")
    parser.add_argument("--id", "-i", type=int, required=True, help="Pinceaux id")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Repository root",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = compute_sa_to_v(args.base_dir, args.id)
    print(f"SA:V CSV saved to: {out}")


if __name__ == "__main__":
    main()
