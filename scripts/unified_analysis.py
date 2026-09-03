#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from skimage import measure


RGB = Tuple[int, int, int]
WHITE: RGB = (255, 255, 255)
PC_GRAY: RGB = (128, 128, 128)  # #808080
SCALEBAR_GRAY: RGB = (178, 178, 178)  # #b2b2b2
# These colors are always allowed, even if not in the JSON metadata:
FORCED_ALLOWED_COLORS: Set[RGB] = {WHITE, PC_GRAY, SCALEBAR_GRAY} 
NEIGHBORHOOD_RADIUS = 2
# Define pixel neighborhood for edge correction:
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




# =========================================================================
# CLASS DEFINITIONS AND LOADING FUNCTIONS
# =========================================================================





class AnalysisError(RuntimeError):
    pass


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


@dataclass
class BlobResult:
    """BlobResult represents a connected component of a specific color in the image.
    
    This component is a contiguous cross section of an MLI in an image.
    
    A section may have multiple regions of the same color, if it cuts through several branches of the same MLI.

    See below for the usage of this class in the context of the analysis pipeline.
    """
    color: RGB
    perimeter_px: float
    contour_points: np.ndarray


def require_positive(value: float, label: str) -> None:
    if value <= 0:
        raise AnalysisError(f"{label} must be > 0")


def to_rgb_array(img_path: Path) -> np.ndarray:
    with Image.open(img_path) as image:
        return np.array(image.convert("RGB"), dtype=np.uint8)


def save_rgb_array(arr: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="RGB").save(output_path)


def rgb_to_hex(color: RGB) -> str:
    return f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"


def hex_to_rgb(hex_color: str) -> RGB:
    value = hex_color.strip().lower()
    if not value.startswith("#"):
        raise AnalysisError(f"Invalid hex color (missing #): {hex_color}")
    if len(value) != 7:
        raise AnalysisError(f"Invalid hex color length: {hex_color}")

    try:
        return (int(value[1:3], 16), int(value[3:5], 16), int(value[5:7], 16))
    except ValueError as exc:
        raise AnalysisError(f"Invalid hex color value: {hex_color}") from exc


def write_scale_metadata(base_dir: Path, pinceaux_id: int, scale_um: float, scale_px: float) -> Path:
    """Write a JSON file containing scale metadata for a given pinceau ID.
    
    Mostly obsolete artifact of a previous implementation, kept with the pipeline for backward compatibility. 
    
    The scale metadata is now stored in the raw config JSON.
    """
    require_positive(scale_um, "scale_um")
    require_positive(scale_px, "scale_px")

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
    return metadata_path


def raw_config_path(base_dir: Path, pinceaux_id: int) -> Path:
    base_dir = Path(base_dir)
    return base_dir / "Inputs" / "Raw" / f"pinceaux_{pinceaux_id}" / "analysis_config.json"


def load_raw_config(base_dir: Path, pinceaux_id: int) -> Optional[PipelineConfig]:
    """Load the configuration (scale, slice thickness, etc.) from a JSON file for a given pinceau ID.
    
    Args:
        base_dir (Path): The base directory containing the Inputs/Raw folder.
        pinceaux_id (int): The ID of the pinceau to load the configuration for.

    Returns:
        Optional[PipelineConfig]: The loaded pipeline configuration, or None if it doesn't exist.
    """
    config_path = raw_config_path(base_dir, pinceaux_id)
    if not config_path.exists():
        return None

    with config_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    required = ["scale_um", "scale_px", "slice_thickness_nm", "z_first", "z_last", "capture_order"]
    missing = [key for key in required if key not in data]
    if missing:
        raise AnalysisError(f"Missing keys in {config_path}: {missing}")

    return PipelineConfig(
        scale_um=float(data["scale_um"]),
        scale_px=float(data["scale_px"]),
        slice_thickness_nm=float(data["slice_thickness_nm"]),
        z_first=float(data["z_first"]),
        z_last=float(data["z_last"]),
        capture_order=str(data["capture_order"]),
    )


def write_raw_config(base_dir: Path, pinceaux_id: int, config: PipelineConfig) -> Path:
    """Write the configuration (scale, slice thickness, etc.) for a given pinceau ID to a JSON file.
    
    Args:
        base_dir (Path): The base directory containing the Inputs/Raw folder.
        pinceaux_id (int): The ID of the pinceau to write the configuration for.
        config (PipelineConfig): The pipeline configuration to write.

    Returns:
        Path: The path to the written configuration file.
    """
    config_path = raw_config_path(base_dir, pinceaux_id)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "scale_um": config.scale_um,
                "scale_px": config.scale_px,
                "slice_thickness_nm": config.slice_thickness_nm,
                "z_first": config.z_first,
                "z_last": config.z_last,
                "capture_order": config.capture_order,
            },
            handle,
            indent=2,
        )
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
    """Resolve the pipeline configuration for a given pinceau ID, using provided values or falling back to the 
    JSON config if none are provided.

    Args:
        base_dir (Path): The base directory containing the Inputs/Raw folder.
        pinceaux_id (int): The ID of the pinceau to resolve the configuration for.
        scale_um (Optional[float]): The scale bar length in micrometers.
        scale_px (Optional[float]): The scale bar length in pixels.
        slice_thickness_nm (Optional[float]): The section thickness in nanometers.
        z_first (Optional[float]): The z-coordinate of the first section.
        z_last (Optional[float]): The z-coordinate of the last section.
        capture_order (Optional[str]): The screenshot capture order.

    Returns:
        PipelineConfig: The resolved pipeline configuration.
    """
    raw_cfg = load_raw_config(base_dir, pinceaux_id)

    resolved = PipelineConfig(
        scale_um=scale_um if scale_um is not None else (raw_cfg.scale_um if raw_cfg else None),
        scale_px=scale_px if scale_px is not None else (raw_cfg.scale_px if raw_cfg else None),
        slice_thickness_nm=(
            slice_thickness_nm if slice_thickness_nm is not None else (raw_cfg.slice_thickness_nm if raw_cfg else None)
        ),
        z_first=z_first if z_first is not None else (raw_cfg.z_first if raw_cfg else None),
        z_last=z_last if z_last is not None else (raw_cfg.z_last if raw_cfg else None),
        capture_order=capture_order if capture_order is not None else (raw_cfg.capture_order if raw_cfg else None),
    )

    missing = [
        name
        for name, value in (
            ("scale_um", resolved.scale_um),
            ("scale_px", resolved.scale_px),
            ("slice_thickness_nm", resolved.slice_thickness_nm),
            ("z_first", resolved.z_first),
            ("z_last", resolved.z_last),
            ("capture_order", resolved.capture_order),
        )
        if value is None
    ]
    if missing:
        raise AnalysisError(
            "Missing required inputs and no usable raw config fallback for keys: "
            f"{missing}. Expected config at {raw_config_path(base_dir, pinceaux_id)}"
        )

    return resolved


def load_um_per_px(base_dir: Path, pinceaux_id: int, scale_um: Optional[float], scale_px: Optional[float]) -> float:
    """Load the micrometers per pixel value for a given pinceau ID, either from provided scale values 
    or from the raw config."""
    if scale_um is not None or scale_px is not None:
        if scale_um is None or scale_px is None:
            raise AnalysisError("Both scale_um and scale_px must be provided together")
        require_positive(scale_um, "scale_um")
        require_positive(scale_px, "scale_px")
        return scale_um / scale_px

    metadata_path = base_dir / "Outputs" / "scale_metadata.json"
    if not metadata_path.exists():
        raise AnalysisError("No Outputs/scale_metadata.json found. Provide scale_um and scale_px explicitly.")

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    metadata_id = metadata.get("pinceaux_id")
    if metadata_id != pinceaux_id:
        raise AnalysisError(
            f"scale_metadata.json is for pinceaux_{metadata_id}, not pinceaux_{pinceaux_id}. "
            "Provide scale_um and scale_px explicitly or update metadata."
        )

    um_per_px = metadata.get("um_per_px")
    if not isinstance(um_per_px, (int, float)) or um_per_px <= 0:
        raise AnalysisError("Invalid um_per_px in Outputs/scale_metadata.json")

    return float(um_per_px)


def build_z_values(n_slices: int, z_first: float, z_last: float) -> List[float]:
    """Build a list of z-coordinate values for the sections, evenly spaced between z_first and z_last.
    
    If the number of sections is not equal to the z range, the z values will be interpolated.
    """
    if n_slices <= 0:
        return []
    if n_slices == 1:
        return [float(z_first)]
    return np.linspace(z_first, z_last, n_slices).tolist()


def build_processed_name(index: int, z_value: float) -> str:
    return f"z_{index:04d}_{z_value:.6f}.png"


def write_slice_mapping_csv(output_dir: Path, rows: List[SliceMapRow]) -> Path:
    """Write a CSV file mapping section indices to original and processed filenames, along with z-coordinates.
    """
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
    """Write a JSON file containing metadata about the analysis run, including scale, section thickness, 
    z-coordinates, and capture order."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "run_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "pinceaux_id": pinceaux_id,
                "scale_um": scale_um,
                "scale_px": scale_px,
                "slice_thickness_nm": slice_thickness_nm,
                "z_first": z_first,
                "z_last": z_last,
                "capture_order": capture_order,
            },
            handle,
            indent=2,
        )
    return metadata_path


def load_allowed_colors(base_dir: Path, pinceaux_id: int) -> Set[RGB]:
    """Load the colors assigned to MLIs from the pinceau metadata JSON file.
    
    These colors will be used to determine which pixels need to be reassigned during edge correction.
    """
    json_path = base_dir / "Inputs" / f"pinceaux_{pinceaux_id}.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Metadata JSON not found: {json_path}")

    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    layers = data.get("layers")
    if not isinstance(layers, list):
        raise AnalysisError("Invalid JSON structure: 'layers' must be a list")

    hex_colors: Set[str] = set()
    for layer in layers:
        if not isinstance(layer, dict) or layer.get("type") != "segmentation":
            continue

        segment_colors = layer.get("segmentColors")
        if not isinstance(segment_colors, dict):
            continue

        for color in segment_colors.values():
            if isinstance(color, str):
                hex_colors.add(color)

    if not hex_colors:
        raise AnalysisError(f"No colors found in segmentColors for pinceaux_{pinceaux_id}")

    allowed_colors = {hex_to_rgb(color) for color in hex_colors}
    allowed_colors.update(FORCED_ALLOWED_COLORS)
    return allowed_colors




# =========================================================================
# EDGE CORRECTION FUNCTIONS
# =========================================================================




def choose_replacement_color(arr: np.ndarray, y: int, x: int, allowed_colors: Set[RGB]) -> RGB:
    """Due to image compression, pixels on the edge of a colored region (representing the cross section of an MLI) 
    may have colors that are not in the allowed set of colors. This function chooses a replacement color for 
    such pixels by determining which colored region (a neighboring MLI or the background) it most likely belongs to.

    The replacement color is chosen based on the number of known MLI or background pixels in the 
    neighborhood of the discolored pixel.
    
    The function looks at the 5x5 neighborhood of the pixel (excluding the pixel itself) and counts how many 
    pixels of each MLI or background color are present.
    
    The color of the pixel is assigned as the color with the highest count in the neighborhood. 
    
    If no MLIs are found, the pixel is assigned to the background.
    
    If there is a tie among the most frequent MLI colors, the pixel is arbitrarily assigned to one of them,
    while preferencing the background if it is among the most frequent colors.

    NOTE: This function does not modify the input array. It only returns the chosen replacement color.
    This is in order to avoid biasing the neighborhood counts for subsequent pixels in the same pass.
    """
    h, w, _ = arr.shape
    y0 = max(0, y - NEIGHBORHOOD_RADIUS)
    y1 = min(h, y + NEIGHBORHOOD_RADIUS + 1)
    x0 = max(0, x - NEIGHBORHOOD_RADIUS)
    x1 = min(w, x + NEIGHBORHOOD_RADIUS + 1)

    patch = arr[y0:y1, x0:x1]
    patch_colors = [tuple(map(int, color)) for color in patch.reshape(-1, 3)]

    center_index = (y - y0) * (x1 - x0) + (x - x0)
    patch_colors.pop(center_index)

    counts = Counter(color for color in patch_colors if color in allowed_colors)
    if not counts:
        return WHITE

    max_count = max(counts.values())
    top_colors = [color for color, count in counts.items() if count == max_count]
    if WHITE in top_colors:
        return WHITE
    return top_colors[0]


def reassign_non_allowed_pixels(arr: np.ndarray, allowed_colors: Set[RGB]) -> Tuple[np.ndarray, int]:
    """Finds edge pixels (those that are discolored due to compression) in the whole image and reassigns 
    them to an MLI or the background. 
    
    First, all pixels that are not in the allowed set of colors are identified. For each of these pixels, 
    a replacement color is chosen based on the colors of its neighbors (using the choose_replacement_color function). 
    
    After determining the replacement color for all pixels, they are reassigned to the new colors.

    Returns a new array (image) with the reassigned pixels and the count of reassigned pixels.
    """
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
    """Validates that the processed image contains only colors from the allowed set."""
    unique_colors = {tuple(map(int, color)) for color in np.unique(arr.reshape(-1, 3), axis=0)}
    disallowed = sorted(unique_colors - allowed_colors)
    if disallowed:
        sample = ", ".join(str(color) for color in disallowed[:15])
        raise AnalysisError(
            "Processed image contains colors not in JSON allowed set. "
            f"Found {len(disallowed)} unexpected colors. Sample: {sample}"
        )


def process_edge_single_slice(input_path: Path, output_path: Path, allowed_colors: Set[RGB]) -> int:
    """Process a single section for edge correction. This involves:
    1. Loading the image as an RGB array.
    2. Reassigning non-allowed pixels to allowed colors.
    3. Validating the resulting image.
    4. Saving the corrected image.
    
    Returns the number of pixels that were reassigned.
    """
    original = to_rgb_array(input_path)
    cleaned, reassigned_pixels = reassign_non_allowed_pixels(original, allowed_colors)
    validate_only_allowed_colors(cleaned, allowed_colors)
    save_rgb_array(cleaned, output_path)
    return reassigned_pixels


def process_edge_pinceaux(base_dir: Path, pinceaux_id: int) -> int:
    """Performs edge correction on all sections for a given pinceaux ID.
    
    Returns the total number of pixels that were reassigned across all sections.
    """
    input_dir = base_dir / "Inputs" / "Raw" / f"pinceaux_{pinceaux_id}"
    output_dir = base_dir / "Inputs" / "Edge_Corrected" / f"pinceaux_{pinceaux_id}"
    allowed_colors = load_allowed_colors(base_dir, pinceaux_id)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    png_files = sorted(input_dir.glob("*.png"))
    if not png_files:
        raise AnalysisError(f"No PNG files found in {input_dir}")

    total_reassigned = 0
    for png_file in png_files:
        total_reassigned += process_edge_single_slice(png_file, output_dir / png_file.name, allowed_colors)
    return total_reassigned




# =========================================================================
# REMOVAL OF THE PC, SCALEBAR, AND EDGE-TOUCHING MLIs
# =========================================================================




def remove_gray_pixels(arr: np.ndarray) -> Tuple[np.ndarray, int]:
    """Remove the PC and scale bar pixels (gray) from the image by replacing them with white pixels.
    
    Returns a new array (image) with the gray pixels removed and the count of removed pixels.
    """
    working = arr.copy()
    is_pc_gray = np.all(working == PC_GRAY, axis=2)
    is_scalebar_gray = np.all(working == SCALEBAR_GRAY, axis=2)
    gray_mask = is_pc_gray | is_scalebar_gray
    count = int(np.count_nonzero(gray_mask))
    working[gray_mask] = WHITE
    return working, count


def get_neighbors_8(y: int, x: int, h: int, w: int):
    """Yield the 8-connected neighbors of a pixel within the image bounds.

    This helper is used for connected-component analysis, where pixels are considered
    adjacent not only along the four cardinal directions but also diagonally. The
    generator returns each neighboring coordinate `(ny, nx)` that lies inside the
    image and is not the original `(y, x)` pixel itself.
    """
    for ny in range(max(0, y - 1), min(h, y + 2)):
        for nx in range(max(0, x - 1), min(w, x + 2)):
            if ny == y and nx == x:
                continue
            yield ny, nx


def remove_edge_touching_nonwhite_blobs(arr: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """Remove any connected non-white region that touches the image border.

    The function scans all pixels on the outer edge of the image and starts a flood-fill
    from each non-white border pixel using 8-neighbor connectivity. Any connected
    component discovered this way is treated as an edge artifact and replaced with white
    pixels, preventing fragments attached to the frame from being counted as valid MLI
    structures in the later analysis.

    Returns a tuple of:
        - the modified image array,
        - the number of removed connected components,
        - the total number of removed pixels.
    """
    working = arr.copy()
    h, w, _ = working.shape

    visited = np.zeros((h, w), dtype=bool)
    removed_components = 0
    removed_pixels = 0

    if h == 0 or w == 0:
        return working, removed_components, removed_pixels

    edge_coords: List[Tuple[int, int]] = []
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

        # Start a flood-fill from each border pixel that is not white.
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
                    # Same region: keep expanding until the whole border-connected region is found.
                    visited[ny, nx] = True
                    stack.append((ny, nx))

        if component:
            removed_components += 1
            removed_pixels += len(component)
            ys, xs = zip(*component)
            working[list(ys), list(xs)] = WHITE

    return working, removed_components, removed_pixels


def process_mli_single_slice(input_path: Path, output_path: Path) -> Tuple[int, int, int]:
    """Process a single section to remove the PC, scale bar, and any edge-touching MLIs."""
    original = to_rgb_array(input_path)
    no_gray, gray_pixels_whitened = remove_gray_pixels(original)
    cleaned, removed_components, removed_pixels = remove_edge_touching_nonwhite_blobs(no_gray)
    save_rgb_array(cleaned, output_path)
    return gray_pixels_whitened, removed_components, removed_pixels


def process_mli_pinceaux(base_dir: Path, pinceaux_id: int, scale_um: float, scale_px: float) -> int:
    """Process all background-corrected sections for a given pinceaux ID to remove the PC, scale bar, 
    and any edge-touching MLIs."""
    input_dir = base_dir / "Inputs" / "Edge_Corrected" / f"pinceaux_{pinceaux_id}"
    output_dir = base_dir / "Inputs" / "MLI_only" / f"pinceaux_{pinceaux_id}"

    if not input_dir.exists():
        raise FileNotFoundError(f"Edge-corrected input folder not found: {input_dir}")

    png_files = sorted(input_dir.glob("*.png"))
    if not png_files:
        raise AnalysisError(f"No PNG files found in {input_dir}")

    write_scale_metadata(base_dir, pinceaux_id, scale_um, scale_px)

    total_gray_whitened = 0
    total_components_removed = 0
    total_component_pixels_removed = 0
    for png_file in png_files:
        gray_pixels_whitened, removed_components, removed_pixels = process_mli_single_slice(
            png_file, output_dir / png_file.name
        )
        total_gray_whitened += gray_pixels_whitened
        total_components_removed += removed_components
        total_component_pixels_removed += removed_pixels

    return total_gray_whitened + total_components_removed + total_component_pixels_removed




# =========================================================================
# DETECTION OF MLIs AND THEIR CONTOURS
# =========================================================================



"""After edge correction and removal of the PC, scale bar, and edge-touching MLIs, the remaining pixels in the image
represent the cross sections of the MLIs.

The following functions are used to identify and isolate these cross sections, trace their countours, and 
calculate their perimeters and areas.
"""


def extract_component(mask: np.ndarray, start_y: int, start_x: int, visited: np.ndarray) -> np.ndarray:
    """Extracts individual cross sections (connected components) of an MLI from a binary mask, 
    based on a given starting point. 
    
    The binary mask is a 2D array where True values represent pixels belonging to the MLI, and False values 
    represent background pixels or other MLIs.

    Each cross section is a connected component of True pixels.
    
    The function performs a depth-first search (DFS) to find all connected pixels that are 
    part of the same component.

    Args:
        mask (np.ndarray): A binary mask representing all pixels belonging to a given MLI in the section.
        start_y (int): The y-coordinate of the starting pixel for the component extraction.
        start_x (int): The x-coordinate of the starting pixel for the component extraction.
        visited (np.ndarray): A boolean array of the same shape as `mask` that keeps track of which pixels have 
            already been visited during the component extraction.

    Returns:
        np.ndarray: A binary mask representing the extracted connected component (cross section) of the MLI.
    """
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
    """Identifies and isolates all cross sections of an MLI in a given section.
    
    For a given binary mask, this function identifies all distinct connected regions 
    (components) of True pixels.

    Args:
        mask (np.ndarray): A binary mask representing all pixels belonging to a given MLI in the section.

    Returns:
        List[np.ndarray]: A list of binary masks, each representing a single cross section of the MLI.
    """
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    components: List[np.ndarray] = []
    ys, xs = np.where(mask)

    for y, x in zip(ys, xs):
        if visited[y, x]:
            continue
        components.append(extract_component(mask, int(y), int(x), visited))

    return components


def trace_boundary(component_mask: np.ndarray) -> np.ndarray:
    """Trace the boundary of a connected component (cross section of an MLI) and return its contour points.
    
    This function uses the `skimage.measure.find_contours` method to find the contour of the binary mask 
    representing a single connected component, which uses the marching squares algorithm. The contour points 
    are returned as an array of (y, x) coordinates.

    If a component has multiple disjoint contours (e.g., due to holes), the function will return all contours 
    connected in a single chain, separated by NaN values. A break may occur if the component has pixels only 
    connected diagonally, which can result in multiple contours being detected.

    Args:
        component_mask (np.ndarray): A binary mask representing a single connected component.

    Returns:
        np.ndarray: An array of contour points.
    """
    contours = measure.find_contours(component_mask.astype(np.uint8), 0.5)
    if not contours:
        raise AnalysisError("No contour found for component")

    valid_contours = [contour for contour in contours if contour.shape[0] >= 2]
    if not valid_contours:
        raise AnalysisError("No valid contour found for component")

    pieces = []
    for contour in valid_contours:
        if pieces:
            pieces.append(np.array([[np.nan, np.nan]], dtype=float))
        pieces.append(contour.astype(float, copy=False))
    return np.vstack(pieces)


def contour_length_px(points: np.ndarray) -> float:
    """Calculate the perimeter of a contour in pixels.
    
    If the contour has multiple disjoint pieces, the function will sum the lengths of all pieces.

    The length of each piece is calculated as the sum of the Euclidean distances between consecutive points,
    plus the distance between the last point and the first point to close the contour.
    """
    if points.shape[0] < 2:
        return 0.0

    length = 0.0
    start_index = 0
    for index in range(points.shape[0] + 1):
        is_break = index == points.shape[0] or np.isnan(points[index]).any()
        if not is_break:
            continue

        contour = points[start_index:index]
        if contour.shape[0] >= 2:
            diffs = np.diff(contour, axis=0)
            length += float(np.sqrt(np.sum(diffs ** 2, axis=1)).sum())
            length += float(math.hypot(*(contour[-1] - contour[0])))
        start_index = index + 1

    return length


def blob_results_for_color(arr: np.ndarray, color: RGB) -> List[BlobResult]:
    """Find all connected components of an MLI represented by a specific color in the image and 
    return their properties.
    
    The function first creates a binary mask for the specified color, then identifies all connected components
    (cross sections) of that color. For each component, it traces the boundary to obtain the contour points and 
    calculates the perimeter in pixels.

    Each component is represented as a BlobResult containing its color, perimeter in pixels, and contour points.
    """
    mask = np.all(arr == color, axis=2)
    if not np.any(mask):
        return []

    results: List[BlobResult] = []
    for component in find_components(mask):
        contour_points = trace_boundary(component)
        results.append(
            BlobResult(
                color=color,
                perimeter_px=contour_length_px(contour_points),
                contour_points=contour_points,
            )
        )
    return results


def draw_contours(image_shape: Tuple[int, int, int], blobs: List[BlobResult], output_path: Path) -> None:
    """Draw the contours of all detected MLI cross sections on a blank canvas and save the image.
    
    The contours are drawn in the color of the corresponding MLI.
    """
    h, w, _ = image_shape
    canvas = Image.new("RGB", (w, h), WHITE)
    draw = ImageDraw.Draw(canvas)

    for blob in blobs:
        contour_points = blob.contour_points
        if contour_points.shape[0] < 2:
            continue

        start_index = 0
        for index in range(contour_points.shape[0] + 1):
            is_break = index == contour_points.shape[0] or np.isnan(contour_points[index]).any()
            if not is_break:
                continue

            contour = contour_points[start_index:index]
            if contour.shape[0] >= 2:
                xy = [(float(x), float(y)) for y, x in contour]
                xy.append((float(contour[0][1]), float(contour[0][0])))
                draw.line(xy, fill=blob.color, width=1)
            start_index = index + 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def process_slice(arr: np.ndarray) -> Tuple[List[BlobResult], Dict[RGB, float], Dict[RGB, int], Dict[RGB, int]]:
    """Process a single section to detect all MLIs and their contours.

    The area of each MLI cross section is calculated as the number of pixels in the section that match its color.
    
    Args:
        arr (np.ndarray): The RGB image array of the section.
    
    Returns:
        Tuple containing:
            - List of BlobResult for all detected MLIs.
            - Dictionary mapping each MLI color to its total perimeter in linear pixels. (1 pixel = 1 unit length)
            - Dictionary mapping each MLI color to its total cross section ("blob") count.
            - Dictionary mapping each MLI color to its total area in pixels. (1 pixel = 1 unit area)
    """
    unique_colors = np.unique(arr.reshape(-1, 3), axis=0)
    colors = [tuple(map(int, color)) for color in unique_colors if tuple(map(int, color)) != WHITE]

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




# =========================================================================
# TOTAL MLI SURFACE AREA AND VOLUME CALCULATION IN A GIVEN PINCEAU
# =========================================================================




def save_total_area_bar_chart(totals_per_color_area: Dict[RGB, float], output_path: Path) -> None:
    """Save a bar chart of total MLI surface area by color."""
    if not totals_per_color_area:
        raise AnalysisError("No color totals available for bar chart")

    ordered = sorted(totals_per_color_area.items(), key=lambda item: item[1], reverse=True)
    labels = [rgb_to_hex(color) for color, _ in ordered]
    values = [area for _, area in ordered]

    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 0.5), 6), dpi=150)
    ax.bar(labels, values, color=labels)
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
    """Save a bar chart of total MLI volume by color."""
    if not totals_per_color_volume:
        raise AnalysisError("No color totals available for volume bar chart")

    ordered = sorted(totals_per_color_volume.items(), key=lambda item: item[1], reverse=True)
    labels = [rgb_to_hex(color) for color, _ in ordered]
    values = [volume for _, volume in ordered]

    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 0.5), 6), dpi=150)
    ax.bar(labels, values, color=labels)
    ax.set_xlabel("MLI Color")
    ax.set_ylabel("Volume (um^3)")
    ax.set_title("Total MLI Volume by Color")
    ax.tick_params(axis="x", rotation=75)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def analyze_pinceaux(
    base_dir: Path,
    pinceaux_id: int,
    slice_thickness_nm: float,
    scale_um: Optional[float],
    scale_px: Optional[float],
) -> None:
    """Summarizes MLI geometry across all sections in a given pinceau.

    This function reads each section after edge correction and removal of 
    PC/scale bar/edge-touching MLIs, detects connected components for each color, 
    traces their contours, and converts the resulting pixel-based metrics to physical units 
    using information about the scale bar and slice thickness. It writes per-slice CSV summaries, 
    aggregate totals, and contour images for each slice.

    Args:
        base_dir: Project root containing the Inputs and Outputs folders.
        pinceaux_id: Identifier for the current pinceaux stack.
        slice_thickness_nm: Section thickness in nanometers. MLI surface area and volume are calculated by 
            multiplying the perimeter and surface area of the MLI in each section, respectively, by this value,
            then summed up across all sections in the pinceau.
        scale_um: The length of the scalebar in micrometers.
        scale_px: The length of the scalebar in pixels.
    """
    require_positive(slice_thickness_nm, "slice_thickness_nm")
    um_per_px = load_um_per_px(base_dir, pinceaux_id, scale_um, scale_px)
    slice_thickness_um = slice_thickness_nm / 1000.0 # Convert nanometers to micrometers

    input_dir = base_dir / "Inputs" / "MLI_only" / f"pinceaux_{pinceaux_id}"
    output_dir = base_dir / "Outputs" / f"pinceaux_{pinceaux_id}"
    contour_dir = output_dir / "Contours"

    if not input_dir.exists():
        raise FileNotFoundError(f"MLI_only input folder not found: {input_dir}")

    png_files = sorted(input_dir.glob("*.png"))
    if not png_files:
        raise AnalysisError(f"No PNG slices found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    contour_dir.mkdir(parents=True, exist_ok=True)

    # Per-slice outputs track each section (z coordinate) independently; totals aggregate across the full pinceau.
    per_slice_csv = output_dir / "slice_color_perimeter_area.csv"
    totals_csv = output_dir / "color_total_area.csv"
    totals_chart = output_dir / "color_total_area_bar_chart.png"
    per_slice_volume_csv = output_dir / "slice_color_area_volume.csv"
    totals_volume_csv = output_dir / "color_total_volume.csv"
    totals_volume_chart = output_dir / "color_total_volume_bar_chart.png"

    # Accumulators keyed by color (MLI): area, perimeter, individual cross section count, and volume across 
    # all sections.
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
                # Each section is processed independently: find blobs, trace contours, and convert 
                # pixel metrics to um/um^2/um^3.
                arr = to_rgb_array(png_file)
                blobs, per_color_perimeter_px, per_color_blob_count, per_color_area_px = process_slice(arr)

                draw_contours(arr.shape, blobs, contour_dir / png_file.name)

                for color, perimeter_px in per_color_perimeter_px.items():
                    # Convert perimeter from pixels to micrometers, then calculate the contribution to the total 
                    # surface area of the MLI in a given section.
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
                    # Convert cross-section area from pixels to um^2, then caluculate the contribution to the total
                    # MLI volume in a given section.
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

    # Aggregate per-section results into MLI totals for the whole set of sections (the whole pinceau).
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
            fieldnames=["color_hex", "sum_cross_section_area_um2", "slice_thickness_um", "total_volume_um3"],
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


def compute_sa_to_v(base_dir: Path, pinceaux_id: int) -> Path:
    """Compute the surface area to volume ratio for each MLI in a given pinceaux.
    
    This is done to check whether the surface area to volume ratio is within a reasonable range for each MLI.
    
    A physically sensible MLI should have a surface area to volume ratio that is on the order of 3-10 um^-1.
    
    A high SA:V ratio may indicate one of the following:
        1. If all MLIs in the pinceau have a high SA:V, an analysis error yielding incorrect surface area is 
            likely. The volume caluclation is straightforward, so its value is assumed correct and can be used
            to benchmark the surface area calculation.
        2. If only one or a few MLIs have a high SA:V, it may indicate that these MLIs are not truly present in the 
            pinceau. A small number of pixels have by chance been transformed into an allowed color during image
            compression, which resulted in faulty detection by the edge correction algorithm. Such MLIs should
            be disregarded and removed from the analysis.
    """
    base_dir = Path(base_dir)
    output_dir = base_dir / "Outputs" / f"pinceaux_{pinceaux_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    vol_path = output_dir / "color_total_volume.csv"
    area_path = output_dir / "color_total_area.csv"
    if not vol_path.exists() or not area_path.exists():
        raise FileNotFoundError(f"Missing input CSVs in {output_dir}")

    def read_csv(path: Path, value_key: str) -> Dict[str, float]:
        out: Dict[str, float] = {}
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                key = row.get("color_hex")
                if not key:
                    continue
                try:
                    out[key] = float(row.get(value_key, 0) or 0)
                except Exception:
                    out[key] = 0.0
        return out

    volumes = read_csv(vol_path, "total_volume_um3")
    areas = read_csv(area_path, "total_area_um2")

    colors = sorted(set(volumes) | set(areas), key=lambda color: volumes.get(color, 0.0), reverse=True)
    rows: List[Dict[str, str]] = []

    for color in colors:
        volume = volumes.get(color, 0.0)
        area = areas.get(color, 0.0)
        sa = float("nan") if volume == 0.0 else area / volume
        rows.append(
            {
                "color_hex": color,
                "total_volume_um3": f"{volume:.6f}",
                "total_area_um2": f"{area:.6f}",
                "surface_area_to_volume_um_inv": f"{sa:.6f}" if not math.isnan(sa) else "",
            }
        )

    out_csv = output_dir / "sa_to_v.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["color_hex", "total_volume_um3", "total_area_um2", "surface_area_to_volume_um_inv"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    def write_plot(sorted_rows: List[Dict[str, str]], png_name: str, title: str) -> Path:
        out_png = output_dir / png_name
        width = max(6, len(sorted_rows) * 0.25)
        fig, ax = plt.subplots(figsize=(width, 4))
        x = list(range(len(sorted_rows)))
        y = [float(row["surface_area_to_volume_um_inv"]) if row["surface_area_to_volume_um_inv"] else 0.0 for row in sorted_rows]
        labels = [row["color_hex"] for row in sorted_rows]
        colors_for_plot = [row["color_hex"] for row in sorted_rows]

        if any(val > 12 for val in y):
            ax.set_ylim(0, 12)
            for idx, val in enumerate(y):
                if val > 12:
                    ax.annotate(
                        "cropped",
                        xy=(idx, 12),
                        xytext=(0, 6),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=6,
                        color="black",
                    )

        ax.bar(x, y, color=colors_for_plot)
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
    write_plot(by_volume, "sa_to_v_sorted_by_volume.png", f"SA:V by color sorted by total volume for pinceaux_{pinceaux_id}")
    write_plot(by_area, "sa_to_v_sorted_by_area.png", f"SA:V by color sorted by total area for pinceaux_{pinceaux_id}")

    return out_csv




# =========================================================================
# MAIN METHOD FOR FULL ANALYSIS OF A GIVEN PINCEAU
# =========================================================================




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
    """
    Run the full analysis pipeline for a given pinceau ID, including edge correction, MLI-only generation,
    surface area and volume analysis, and contour generation.
    
    Args:
        base_dir: Project root containing the Inputs and Outputs folders.
        pinceaux_id: Identifier for the current pinceau stack.
        scale_um: The length of the scalebar in micrometers.
        scale_px: The length of the scalebar in pixels.
        slice_thickness_nm: The thickness of each section in nanometers.
        z_first: The z-coordinate of the first section.
        z_last: The z-coordinate of the last section.
        capture_order: The order in which the sections were captured.
        write_config: Whether to write the configuration to a file.
    """
    base_dir = Path(base_dir)
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
        raise AnalysisError("capture_order must be 'ascending' or 'descending'")

    require_positive(config.scale_um, "scale_um")
    require_positive(config.scale_px, "scale_px")
    require_positive(config.slice_thickness_nm, "slice_thickness_nm")

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")

    raw_files = sorted(raw_dir.glob("*.png"))
    if not raw_files:
        raise FileNotFoundError(f"No PNG files found in: {raw_dir}")

    z_values = build_z_values(len(raw_files), config.z_first, config.z_last)

    # Load the allowed colors for the pinceaux, i.e. those assigned to MLIs in the images.
    allowed_colors = load_allowed_colors(base_dir, pinceaux_id)

    edge_dir.mkdir(parents=True, exist_ok=True)
    mli_dir.mkdir(parents=True, exist_ok=True)

    rows: List[SliceMapRow] = []

    print(f"Stage 1/3: Edge correction for {len(raw_files)} slices...")
    for index, (raw_path, z_value) in enumerate(zip(raw_files, z_values), start=1):
        processed_name = build_processed_name(index, z_value)
        process_edge_single_slice(raw_path, edge_dir / processed_name, allowed_colors)
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
        process_mli_single_slice(edge_dir / row.processed_filename, mli_dir / row.processed_filename)

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

    try:
        compute_sa_to_v(base_dir=base_dir, pinceaux_id=pinceaux_id)
        print(f"SA:V computation completed and saved in Outputs/pinceaux_{pinceaux_id}")
    except Exception as exc:
        print(f"SA:V computation skipped/error: {exc}")

    if write_config:
        cfg_path = write_raw_config(base_dir, pinceaux_id, config)
        print(f"Raw config saved to: {cfg_path}")

    if config.capture_order == "ascending" and config.z_last < config.z_first:
        print("Warning: capture_order is 'ascending' but z_last < z_first.")
    if config.capture_order == "descending" and config.z_last > config.z_first:
        print("Warning: capture_order is 'descending' but z_last > z_first.")

    print(f"Done. Mapping saved to: {mapping_path}")
    print(f"Run metadata saved to: {metadata_path}")


def discover_pinceaux_ids(raw_root: Path) -> List[int]:
    """Helper function to discover all pinceaux IDs in the Inputs/Raw directory that have both PNG files
    and a config file.
    """
    ids: List[int] = []
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


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the full pinceaux analysis pipeline."""
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
        default=Path(__file__).resolve().parent.parent,
        help="Repository root directory",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for the pinceau analysis pipeline."""
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