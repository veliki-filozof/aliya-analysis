# Pinceau Analysis

Image analysis for pinceau electron microscopy data processing.

## Overview

This pipeline processes EM dataset sections through multiple stages:

1. **Edge Cleanup** – Removes image compression artifacts
2. **MLI Filtering** – Isolates MLIs fully contained in image
3. **Analysis** – Detects MLIs, calculates surface area and volume measurements
4. **Visualization** – Generates contour overlay images, output tables, and bar charts

---

## Quick Start

### 1. Clone and Navigate

```bash
git clone https://github.com/aln222/pinceau-analysis.git
cd pinceau-analysis
```

### 2. Set Up Environment

Choose your preferred method:

#### Option A: Using `conda` (Recommended)

```bash
# Create environment from YAML file
conda env create -f environment.yml

# Activate environment
conda activate pinceau-analysis
```

#### Option B: Using `venv` + `pip`

```bash
# Create virtual environment
python3 -m venv venv

# Activate environment (Linux / macOS)
source venv/bin/activate

# Activate environment (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Prepare Your Data

Place raw EM section images in:
```
Inputs/Raw/pinceaux_X/
```
where `X` is your pinceaux ID (e.g., `pinceaux_5`).

---

## Usage

### Local Analysis

Use the **Jupyter notebook** for local analysis:

```bash
jupyter notebook notebooks/run_full_analysis.ipynb
```

This will:
- Prompt you to enter analysis parameters (scale, z-coordinates, thickness)
- Run the complete analysis
- Generate outputs in `Outputs/pinceaux_X/`

### Batch Processing (All Pinceau)

Process all pinceau with saved configurations:

```bash
python scripts/run_all_pinceaux.py
```

This discovers all pinceau with `analysis_config.json` files and processes them sequentially.

### Interactive Visualization

View processed slices side-by-side across all stages:

```bash
jupyter notebook notebooks/flipbook_viewer.ipynb
```

Features:
- Select pinceaux ID and load all stages
- Slider to browse through slices
- View Raw, Edge_Corrected, MLI_only, and Contours simultaneously

---

## Output Files Explained

After running the analysis, you'll find:

### CSV Tables

- **slice_color_perimeter_area.csv**  
  Per-section perimeter measurements (pixels) for each MLI

- **color_total_area.csv**  
  Aggregated surface area (μm²) calculated from perimeters × section thickness

- **slice_color_area_volume.csv**  
  Per-section cross-section area (μm²) and volume contribution (μm³)

- **color_total_volume.csv**  
  Total volume (μm³) by MLI, summed across all sections

### Images

- **Contours/*.png**  
  Original sections overlaid with colored MLI boundary traces

- **\*_bar_chart.png**  
  Matplotlib bar charts visualizing surface area and volume by color

---

## Configuration Files

### analysis_config.json

Create or auto-generate via `--write-config` in `Inputs/Raw/pinceaux_X/`:

```json
{
  "scale_um": 2.0,
  "scale_px": 90,
  "slice_thickness_nm": 45,
  "z_first": 200,
  "z_last": 400,
  "capture_order": "ascending"
}
```

---

## Algorithms

### MLI Detection
8-connectivity flood-fill algorithm identifies connected pixel regions (MLI cross-sections).

### Perimeter Estimation
Marching squares algorithm finds contour pixels; perimeter = Euclidean distance sum.

### Volume Calculation
Volume per section = cross-section area (μm²) × section thickness (μm)

### Units & Conversion
- Pixels → micrometers using scale bar parameters
- Micrometers² (μm²) for areas
- Micrometers³ (μm³) for volumes

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'numpy'"

Ensure environment is activated:
```bash
# Conda
conda activate pinceau-analysis

# or venv
source venv/bin/activate
```

Then install dependencies:
```bash
pip install -r requirements.txt
```

### "No such file or directory: Inputs/Raw/pinceaux_X"

Create the directory and place `.png` files:
```bash
mkdir -p Inputs/Raw/pinceaux_5
cp your_images/*.png Inputs/Raw/pinceaux_5/
```

### Notebook kernel issues

Ensure Jupyter can find your environment:
```bash
python -m ipykernel install --user --name pinceau-analysis --display-name "Python (pinceau-analysis)"
```

Then select the kernel in Jupyter: *Kernel → Change kernel → Python (pinceau-analysis)*
