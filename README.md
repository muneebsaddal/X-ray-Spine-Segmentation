# TotalSpineSeg 2D X-Ray Adaptation - POC

Adapts the [TotalSpineSeg](https://github.com/neuropoly/totalspineseg) nnU-Net framework from 3D MRI to **2D X-ray vertebral segmentation**.

This repository is a proof of concept for the complete project. It demonstrates that we can:
- generate synthetic 2D spine X-rays from 3D CT data,
- convert them into nnU-Net 2D training format,
- run a segmentation pipeline end to end,
- and produce overlays suitable for review by a client.

The repository is intentionally kept code-only. No generated data, predictions, or visualizations are committed.

For a client-ready handoff summary, see `SUBMISSION.md`.

## Architecture Overview

```
TotalSpineSeg (Original)          This POC
─────────────────────             ──────────────────────────────────
3D MRI (.nii.gz)         →        2D X-Ray (.png / .jpg / .nii.gz)
nnU-Net 3D full res      →        nnU-Net 2D
Dataset101 (step1)       →        Dataset201 (single-stage)
Dataset102 (step2)       →        Optional step2 (full project)
Iterative labeling algo  →        Simplified 2D post-processing
```

## Dataset Used

**Verse2020** (VerSe Challenge) — public CT vertebra dataset.
We generate synthetic 2D X-ray projections (DRRs) from CT volumes using
Digitally Reconstructed Radiographs, then project the 3D masks to 2D.

Alternative: use raw 2D X-ray images if available (drop-in compatible).

For a clean local demo, you can also generate synthetic phantom data from scratch with `src/generate_phantom.py`.

## Pipeline Steps

1. `scripts/01_download_data.sh` — download VerSe sample data
2. `src/generate_drr.py` — generate synthetic X-rays from CT + project masks
3. `src/prepare_nnunet_dataset.py` — convert to nnU-Net 2D format
4. `src/train.py` — train nnU-Net 2D model
5. `src/predict.py` — run inference on new X-ray images
6. `src/visualize.py` — overlay segmentation on original X-ray
7. `tests/` — validation tests for each pipeline stage

## What This POC Proves

The POC validates the technical path for the full project:
- input data can be converted into a 2D X-ray style representation,
- vertebra masks can be projected from 3D to 2D,
- the resulting data can be packaged into nnU-Net v2 format,
- inference can be run through either nnU-Net or a lightweight fallback,
- and outputs can be visualized as segmentation overlays.

This is intentionally a simplified first version:
- the default prepared dataset uses binary vertebra-vs-background labels,
- the fallback predictor is only for pipeline demonstration,
- and the full production version would add stronger training, real X-ray support, and per-vertebra labeling.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate demo CT volumes and masks
python src/generate_phantom.py \
  --output data/raw/verse_sample \
  --num_cases 3

# 3. Generate synthetic X-rays from CT
python src/generate_drr.py \
  --input data/raw/verse_sample \
  --output data/processed/xray_synthetic \
  --projection lateral

# 4. Prepare nnU-Net dataset
python src/prepare_nnunet_dataset.py \
  --input data/processed/xray_synthetic \
  --output data/nnunet_raw \
  --dataset_id 201

# 5. Run nnU-Net preprocessing
nnUNetv2_plan_and_preprocess -d 201 -c 2d --verify_dataset_integrity

# 6. Train (use --fast for quick POC run with fewer epochs)
python src/train.py --dataset_id 201 --fold 0 --fast

# 7. Predict on test images
python src/predict.py \
  --input data/processed/xray_synthetic/test \
  --output outputs/predictions \
  --dataset_id 201

# 8. Visualize results
python src/visualize.py \
  --images data/processed/xray_synthetic/test \
  --masks outputs/predictions \
  --output outputs/visualizations

# 9. Run all tests
python -m pytest tests/ -v
```

## Fastest Demo Path

If you want the quickest demo without requiring a trained nnU-Net model, use the fallback segmenter:

```bash
# Optional: generate synthetic demo cases if you want fresh input data
python src/generate_phantom.py --output data/raw/verse_sample --num_cases 3
python src/generate_drr.py --input data/raw/verse_sample --output data/processed/xray_synthetic --projection lateral

# Run fallback inference
python src/predict.py \
  --input data/processed/xray_synthetic/test \
  --output outputs/predictions \
  --fallback

# Create client-friendly overlay images
python src/visualize.py \
  --images data/processed/xray_synthetic/test \
  --masks outputs/predictions \
  --output outputs/visualizations
```

## Windows Notes

If you are running this on Windows PowerShell:
- use `python src\...` paths instead of `python src/...` if you prefer native separators,
- `scripts/01_download_data.sh` is a Bash script, so run it via Git Bash or WSL if needed,
- if you only need the demo flow, the included local sample data and phantom generator are enough.

## Directory Structure

```
xray_spine_seg/
├── src/
│   ├── generate_drr.py         # CT → synthetic X-ray + mask projection
│   ├── generate_phantom.py     # Synthetic CT + vertebra label generation
│   ├── prepare_nnunet_dataset.py
│   ├── train.py
│   ├── predict.py
│   ├── visualize.py
│   └── utils.py
├── scripts/
│   └── 01_download_data.sh
├── tests/
│   ├── test_drr_generation.py
│   ├── test_dataset_preparation.py
│   ├── test_predict.py
│   └── test_visualize.py
├── requirements.txt
├── README.md
└── SUBMISSION.md
```

Generated folders such as `data/` and `outputs/` are created only when you run the pipeline locally.

## Key Design Decisions

| Decision | Reason |
|---|---|
| nnU-Net 2D config | Designed for 2D slice segmentation; auto-configures all hyperparams |
| DRR synthetic data | No large public 2D labeled spine X-ray dataset exists; DRRs from CT bridge the gap |
| Lateral projection | Better vertebral separation vs AP; standard in clinical use |
| Single-stage (Dataset201) | Sufficient for POC; step2 cascade reserved for full $250 project |
| NIfTI wrapper for X-rays | nnU-Net expects .nii.gz; we wrap PNG→NIfTI with correct spacing metadata |

## POC Deliverable

A working end-to-end pipeline that:
- Accepts a 2D spine X-ray image as input
- Outputs a segmentation mask with labeled vertebrae regions
- Produces a visual overlay (before/after) for client demonstration

## Not Yet Included In The POC

These are expected next steps for the full project rather than gaps in the proof of concept:
- training on a larger curated dataset with proper experiment tracking,
- stronger post-processing and better vertebra separation,
- full per-vertebra labeling across cervical, thoracic, and lumbar levels,
- support for real clinical X-rays and DICOM ingestion at production quality,
- model evaluation with metrics such as Dice, IoU, and per-level accuracy,
- packaging as a polished application or service endpoint.
