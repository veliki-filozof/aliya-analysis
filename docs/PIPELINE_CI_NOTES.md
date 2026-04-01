# Pipeline + CI Notes

## Per-pinceaux config location
Store analysis inputs in:

- `Inputs/Raw/pinceaux_X/analysis_config.json`

Required keys:

```json
{
  "scale_um": 2.0,
  "scale_px": 126.0,
  "slice_thickness_nm": 40.0,
  "z_first": 100.0,
  "z_last": 92.8,
  "capture_order": "descending"
}
```

## Local run workflow (recommended)
1. Put raw slices in `Inputs/Raw/pinceaux_X`.
2. Run full pipeline locally with explicit inputs and `--write-config`:

```bash
python full_pipeline.py --id X --scale-um <um> --scale-px <px> --slice-thickness-nm 40 --z-first <z0> --z-last <zN> --capture-order ascending --write-config
```

This writes `analysis_config.json` into the same raw folder.

3. Commit and push raw folder + config to GitHub.

## CI run behavior
- CI executes `run_all_pinceaux.py`.
- It scans all `Inputs/Raw/pinceaux_*` folders.
- It processes folders that have both:
  - PNG slices, and
  - `analysis_config.json`
- Outputs are uploaded as workflow artifacts.

## Manual rerun with saved config only
Once config exists, explicit inputs are optional:

```bash
python full_pipeline.py --id X
```
