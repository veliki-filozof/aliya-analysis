# Quick Start Guide

## After Cloning the Repository

### 1. One-time Setup

```bash
# Create environment (choose one method)

# Using conda (recommended):
conda env create -f environment.yml
conda activate aliya-analysis

# OR using pip:
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Prepare Data

Place your microscopy images (.png files) in:
```
Inputs/Raw/pinceaux_X/
```

Example:
```bash
mkdir -p Inputs/Raw/pinceaux_5
cp your_images/*.png Inputs/Raw/pinceaux_5/
```

### 3. Run Analysis

Choose one method:

#### Interactive Notebook (Recommended for New Users)

```bash
jupyter notebook notebooks/run_full_analysis.ipynb
```

- Fill in parameters (scale, z-coordinates, etc.)
- Click "Run Pipeline"
- Optionally save config for batch processing

#### Command Line (Scripted Automation)

```bash
python scripts/full_pipeline.py \
  --id 5 \
  --scale-um 2.0 \
  --scale-px 126 \
  --z-first 0 \
  --z-last 7.2 \
  --capture-order ascending \
  --slice-thickness-nm 40 \
  --write-config
```

#### Batch Processing (Multiple Pinceaux with Saved Configs)

```bash
python scripts/run_all_pinceaux.py
```

Requires `analysis_config.json` files in each `Inputs/Raw/pinceaux_X/` directory.

### 4. Visualize Results

View outputs side-by-side:

```bash
jupyter notebook notebooks/flipbook_viewer.ipynb
```

- Enter pinceaux ID
- Click "Load Stages"
- Move slider to browse slices
- See Raw, Edge_Corrected, MLI_only, and Contours simultaneously

## Output Locations

Results appear in:
```
Outputs/pinceaux_X/
├── Contours/*.png                    # Contour overlays
├── slice_color_perimeter_area.csv    # Per-slice measurements
├── color_total_area.csv              # Aggregated surface area
├── color_total_area_bar_chart.png    # Surface area chart
├── color_total_volume.csv            # Aggregated volumes
├── color_total_volume_bar_chart.png  # Volume chart
└── run_metadata.json                 # Parameters used
```

## Pipeline Parameters Explained

| Parameter | Example | Note |
|-----------|---------|------|
| `scale-um` | 2.0 | Physical distance per unit |
| `scale-px` | 126 | Pixels per scale unit |
| `z-first` | 0 | Z coordinate of first slice |
| `z-last` | 7.2 | Z coordinate of last slice |
| `capture-order` | ascending | `ascending` or `descending` |
| `slice-thickness-nm` | 40 | Slice thickness in **nanometers** |

## Troubleshooting

**Error: "No module named 'numpy'"**
- Check environment is activated: `conda activate aliya-analysis` or `source venv/bin/activate`
- Reinstall: `pip install -r requirements.txt`

**Error: "No such file: Inputs/Raw/pinceaux_X"**
- Create directory: `mkdir -p Inputs/Raw/pinceaux_5`
- Add images: `cp *.png Inputs/Raw/pinceaux_5/`

**Jupyter kernel issues**
- Install kernel: `python -m ipykernel install --user --name aliya-analysis`
- Select in notebook: Kernel → Change kernel → Python (aliya-analysis)

**Imports fail when running scripts**
- Ensure running from repo root: `cd /path/to/aliya-analysis`
- Then: `python scripts/full_pipeline.py ...`

## Next Steps

- See [README.md](../README.md) for full documentation
- Check [docs/PIPELINE_CI_NOTES.md](../docs/PIPELINE_CI_NOTES.md) for CI/GitHub Actions setup
- Explore notebooks for interactive examples
