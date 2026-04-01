# Aliya Analysis Pipeline

A comprehensive image analysis pipeline for microscopy data processing, designed to extract structural measurements from pinceaux (brush-like) histological samples.

## Overview

This pipeline processes microscopy image slices through multiple stages:

1. **Edge Cleanup** – Removes anti-aliasing artifacts
2. **MLI Filtering** – Isolates biologically relevant pixels
3. **Analysis** – Detects features, calculates perimeter and volume measurements
4. **Visualization** – Generates contour overlays, statistics tables, and bar charts

### Key Features

- 📊 **Quantitative Analysis** – Computes surface area and volume per color channel
- 🗺️ **Spatial Tracking** – Z-coordinate mapping through all processing stages
- 🎯 **Reproducible** – Config-based parameters for consistent results across runs
- 🚀 **CI/CD Ready** – Automated batch processing via GitHub Actions
- 📓 **Interactive Notebooks** – Jupyter interfaces for exploration and visualization

---

## Quick Start

### 1. Clone and Navigate

```bash
git clone https://github.com/veliki-filozof/aliya-analysis.git
cd aliya-analysis
```

### 2. Set Up Environment

Choose your preferred method:

#### Option A: Using `conda` (Recommended)

```bash
# Create environment from YAML file
conda env create -f environment.yml

# Activate environment
conda activate aliya-analysis
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

Place raw microscopy image slices in:
```
Inputs/Raw/pinceaux_X/
```
where `X` is your pinceaux ID (e.g., `pinceaux_5`).

---

## Usage

### Local Interactive Analysis

Use the **Jupyter notebook** for interactive analysis with a user-friendly interface:

```bash
jupyter notebook notebooks/run_full_analysis.ipynb
```

This will:
- Prompt you to enter analysis parameters (scale, z-coordinates, thickness)
- Run the complete pipeline
- Optionally save configuration for CI/batch processing
- Generate outputs in `Outputs/pinceaux_X/`

### Command-Line Full Pipeline

For scripted automation:

```bash
python scripts/full_pipeline.py \
  --id 5 \
  --scale-um 2.0 \
  --scale-px 126 \
  --z-first 0 \
  --z-last 7.2 \
  --capture-order ascending \
  --slice-thickness-nm 40
```

**Parameters:**
- `--id` – Pinceaux ID number
- `--scale-um` – Physical distance per unit (micrometers)
- `--scale-px` – Pixel count per scale unit
- `--z-first` – Z-coordinate of first slice
- `--z-last` – Z-coordinate of last slice
- `--capture-order` – `ascending` or `descending`
- `--slice-thickness-nm` – Thickness between slices (nanometers)
- `--write-config` – Save parameters to `Inputs/Raw/pinceaux_X/analysis_config.json`

### Batch Processing (All Pinceaux)

Process all pinceaux with saved configurations:

```bash
python scripts/run_all_pinceaux.py
```

This discovers all pinceaux with `analysis_config.json` files and processes them sequentially.

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

## Project Structure

```
aliya-analysis/
├── README.md                    # This file
├── requirements.txt             # Pip dependencies
├── environment.yml              # Conda environment definition
│
├── scripts/                     # Python analysis scripts
│   ├── edge_cleanup.py          # Stage 1: Remove anti-aliasing
│   ├── mli_only_from_edge_corrected.py  # Stage 2: MLI filtering
│   ├── perimeter_area_step4.py  # Stage 3: Core analysis engine
│   ├── full_pipeline.py         # Orchestrator script
│   └── run_all_pinceaux.py      # Batch processor
│
├── notebooks/                   # Interactive Jupyter notebooks
│   ├── run_full_analysis.ipynb  # Main analysis interface
│   └── flipbook_viewer.ipynb    # Slice visualization tool
│
├── docs/                        # Documentation
│   └── PIPELINE_CI_NOTES.md     # GitHub Actions CI/CD details
│
├── Inputs/
│   └── Raw/
│       └── pinceaux_X/          # Your image slices (.png)
│
└── Outputs/
    └── pinceaux_X/
        ├── Contours/            # Contour overlay images
        ├── slice_color_perimeter_area.csv          # Per-slice measurements
        ├── color_total_area.csv                    # Aggregated surface area
        ├── color_total_area_bar_chart.png          # Surface area plot
        ├── slice_color_area_volume.csv             # Per-slice volumes
        ├── color_total_volume.csv                  # Aggregated volumes
        ├── color_total_volume_bar_chart.png        # Volume plot
        ├── slice_z_mapping.csv                     # Z→filename mapping
        └── run_metadata.json                       # Pipeline parameters used
```

---

## Output Files Explained

After running the pipeline, you'll find:

### CSV Tables

- **slice_color_perimeter_area.csv**  
  Per-slice perimeter measurements (pixels) for each color

- **color_total_area.csv**  
  Aggregated surface area (μm²) calculated from perimeters × slice thickness

- **slice_color_area_volume.csv**  
  Per-slice cross-section area (μm²) and volume contribution (μm³)

- **color_total_volume.csv**  
  Total volume (μm³) by color, summed across all slices

- **slice_z_mapping.csv**  
  Maps original filenames to z-based names (e.g., `z_042_3.14.png`)

### Images

- **Contours/*.png**  
  Original slices overlaid with colored blob boundary traces

- **\*_bar_chart.png**  
  Matplotlib bar charts visualizing surface area and volume by color

### Metadata

- **run_metadata.json**  
  Stores all pipeline parameters (scale, z-coordinates, thickness) for reproducibility

---

## Configuration Files

### analysis_config.json

Create or auto-generate via `--write-config` in `Inputs/Raw/pinceaux_X/`:

```json
{
  "scale_um": 2.0,
  "scale_px": 126,
  "slice_thickness_nm": 40,
  "z_first": 0.0,
  "z_last": 7.2,
  "capture_order": "ascending"
}
```

This file enables:
- ✅ CI/batch processing without user input
- ✅ Reproducible parameter storage
- ✅ Version control of experimental settings

---

## Algorithms

### Blob Detection
8-connectivity flood-fill algorithm identifies connected pixel regions (blobs).

### Perimeter Estimation
Moore-neighbor boundary tracing extracts ordered edge pixels; perimeter = Euclidean distance sum.

### Volume Calculation
Volume per slice = cross-section area (μm²) × slice thickness (μm)

### Units & Conversion
- Pixels → micrometers using scale bar parameters
- Micrometers² (μm²) for areas
- Micrometers³ (μm³) for volumes

For details, see [PIPELINE_CI_NOTES.md](docs/PIPELINE_CI_NOTES.md).

---

## CI/CD Integration

GitHub Actions automates processing:

1. **Trigger** – On push to `Inputs/Raw/**` or config changes
2. **Discover** – `run_all_pinceaux.py` finds pinceaux with configs
3. **Process** – Full pipeline runs for each pinceaux
4. **Artifact** – Outputs captured and available for download

See [PIPELINE_CI_NOTES.md](docs/PIPELINE_CI_NOTES.md) for detailed workflow configuration.

---

## Requirements Met

- ✅ Multi-stage image processing pipeline
- ✅ Perimeter-based surface area quantification
- ✅ Volume estimation from cross-sections
- ✅ Z-coordinate spatial tracking
- ✅ Reproducible config-based parameters
- ✅ Interactive Jupyter interfaces
- ✅ CI/CD automation with GitHub Actions
- ✅ Comprehensive documentation

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'numpy'"

Ensure environment is activated:
```bash
# Conda
conda activate aliya-analysis

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
python -m ipykernel install --user --name aliya-analysis --display-name "Python (aliya-analysis)"
```

Then select the kernel in Jupyter: *Kernel → Change kernel → Python (aliya-analysis)*

---

## Contributing

Contributions welcome! Please:

1. Create a feature branch
2. Test locally before submitting a PR
3. Include documentation for new features
4. Ensure scripts are executable with `chmod +x scripts/*.py`

---

## License

[Add your license here]

---

## Questions & Support

For issues, bugs, or questions:
- Open a GitHub Issue
- Check [PIPELINE_CI_NOTES.md](docs/PIPELINE_CI_NOTES.md) for technical details
- Review notebook examples for usage patterns

Happy analyzing! 🔬📊
