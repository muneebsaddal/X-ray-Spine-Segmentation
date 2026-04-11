"""
src/prepare_nnunet_dataset.py

Converts processed 2D X-ray images + masks into the nnU-Net v2 raw dataset
format (Dataset201_SpineXray2D) and writes the required dataset.json.

nnU-Net v2 raw folder structure expected:
    nnUNet_raw/
        Dataset201_SpineXray2D/
            imagesTr/   <case>_0000.nii.gz   (training images)
            labelsTr/   <case>.nii.gz        (training labels)
            imagesTs/   <case>_0000.nii.gz   (test images, no labels)
            dataset.json

Usage:
    python src/prepare_nnunet_dataset.py \\
        --input  data/processed/xray_synthetic \\
        --output data/nnunet_raw \\
        --dataset_id 201
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from pathlib import Path
from typing import List, Tuple

import numpy as np

from utils import (
    VERTEBRA_LABELS,
    build_dataset_json,
    ensure_dir,
    find_files,
    nifti_to_image,
    image_to_nifti,
    setup_logging,
    write_dataset_json,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Label set used for the 2D POC
# ---------------------------------------------------------------------------

# Simplified 2-class segmentation for POC:
#   0  = Background
#   1  = Vertebra body (any level)
#
# Full project will use per-level instance labels (1-25).
POC_LABELS = {
    "background": 0,
    "vertebra": 1,
}

# Full multi-class labels for the $250 project
FULL_LABELS = {
    "background": 0,
    **{name: label_id for label_id, name in VERTEBRA_LABELS.items()},
}


def _remap_to_binary(label_array: np.ndarray) -> np.ndarray:
    """Remap multi-class vertebra labels to binary (0=bg, 1=vertebra)."""
    binary = (label_array > 0).astype(np.uint8)
    return binary


def _copy_or_remap(
    src: Path,
    dst: Path,
    binary: bool = True,
) -> None:
    """
    Copy a label NIfTI, optionally remapping to binary classes.
    If binary=False, copies as-is (for full multi-class project).
    """
    import nibabel as nib

    ensure_dir(dst.parent)

    if binary:
        img = nib.load(str(src))
        data = np.asarray(img.dataobj, dtype=np.float32)
        data = _remap_to_binary(data).astype(np.float32)
        new_img = nib.Nifti1Image(data, img.affine, img.header)
        nib.save(new_img, str(dst))
    else:
        shutil.copy2(str(src), str(dst))


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def prepare_dataset(
    input_dir: Path,
    output_dir: Path,
    dataset_id: int = 201,
    binary_labels: bool = True,
) -> None:
    """
    Build the nnU-Net v2 Dataset{dataset_id} from processed DRR data.

    Args:
        input_dir:     Output of generate_drr.py (contains train/val/test dirs)
        output_dir:    nnUNet_raw root directory
        dataset_id:    nnU-Net dataset ID (default: 201)
        binary_labels: If True, collapse vertebra labels to binary (POC mode).
                       If False, keep per-vertebra instance labels (full project).
    """
    dataset_name = f"Dataset{dataset_id:03d}_SpineXray2D"
    ds_root = ensure_dir(output_dir / dataset_name)

    images_tr = ensure_dir(ds_root / "imagesTr")
    labels_tr = ensure_dir(ds_root / "labelsTr")
    images_ts = ensure_dir(ds_root / "imagesTs")

    # Collect training + validation images (both go into imagesTr for nnU-Net)
    train_dir = input_dir / "train"
    val_dir   = input_dir / "val"
    test_dir  = input_dir / "test"

    def collect_split(split_dir: Path) -> List[Tuple[Path, Path]]:
        """Return list of (image_path, mask_path) tuples from a split dir."""
        if not split_dir.exists():
            return []
        image_files = sorted(split_dir.glob("*_0000.nii.gz"))
        pairs = []
        for img_path in image_files:
            case_id = img_path.name.replace("_0000.nii.gz", "")
            mask_path = split_dir / f"{case_id}.nii.gz"
            if mask_path.exists():
                pairs.append((img_path, mask_path))
            else:
                logger.warning("No mask for %s — skipping", case_id)
        return pairs

    train_pairs = collect_split(train_dir) + collect_split(val_dir)
    test_images = sorted(test_dir.glob("*_0000.nii.gz")) if test_dir.exists() else []

    if not train_pairs:
        raise FileNotFoundError(
            f"No training pairs found in {train_dir} or {val_dir}. "
            "Run generate_drr.py first."
        )

    logger.info("Training + validation pairs : %d", len(train_pairs))
    logger.info("Test images (no label)       : %d", len(test_images))

    # --- Copy training data ---
    for img_path, mask_path in train_pairs:
        case_id = img_path.name.replace("_0000.nii.gz", "")

        # Image
        dst_img = images_tr / img_path.name
        shutil.copy2(str(img_path), str(dst_img))

        # Label
        dst_lbl = labels_tr / f"{case_id}.nii.gz"
        _copy_or_remap(mask_path, dst_lbl, binary=binary_labels)

        logger.debug("Copied train case: %s", case_id)

    # --- Copy test images (no labels needed) ---
    for img_path in test_images:
        dst = images_ts / img_path.name
        shutil.copy2(str(img_path), str(dst))
        logger.debug("Copied test image: %s", img_path.name)

    # --- Write dataset.json ---
    labels = POC_LABELS if binary_labels else FULL_LABELS

    dataset_json = build_dataset_json(
        dataset_name=dataset_name,
        num_training=len(train_pairs),
        channel_names={0: "X-Ray"},
        labels=labels,
        file_ending=".nii.gz",
    )

    write_dataset_json(ds_root / "dataset.json", dataset_json)

    # --- Verify integrity ---
    _verify_dataset(ds_root, len(train_pairs))

    logger.info("Dataset prepared: %s", ds_root)
    logger.info("")
    logger.info("Next steps:")
    logger.info("  export nnUNet_raw=%s", output_dir)
    logger.info("  export nnUNet_preprocessed=data/nnunet_preprocessed")
    logger.info("  export nnUNet_results=data/nnunet_results")
    logger.info("  nnUNetv2_plan_and_preprocess -d %d -c 2d --verify_dataset_integrity", dataset_id)


def _verify_dataset(ds_root: Path, expected_train: int) -> None:
    """Basic integrity checks on the prepared dataset."""
    errors = []

    images_tr = ds_root / "imagesTr"
    labels_tr = ds_root / "labelsTr"
    dataset_json = ds_root / "dataset.json"

    if not dataset_json.exists():
        errors.append("Missing dataset.json")

    img_files = list(images_tr.glob("*_0000.nii.gz"))
    lbl_files = list(labels_tr.glob("*.nii.gz"))

    if len(img_files) != expected_train:
        errors.append(
            f"Expected {expected_train} training images, found {len(img_files)}"
        )

    if len(lbl_files) != expected_train:
        errors.append(
            f"Expected {expected_train} training labels, found {len(lbl_files)}"
        )

    # Check each image has a matching label
    for img_f in img_files:
        case_id = img_f.name.replace("_0000.nii.gz", "")
        lbl_f = labels_tr / f"{case_id}.nii.gz"
        if not lbl_f.exists():
            errors.append(f"Missing label for training case: {case_id}")

    if errors:
        for err in errors:
            logger.error("Dataset integrity error: %s", err)
        raise RuntimeError(
            f"Dataset integrity check failed with {len(errors)} error(s). See log."
        )

    logger.info("Dataset integrity check passed.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare nnU-Net v2 dataset from processed X-ray DRR data."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Root directory from generate_drr.py (contains train/val/test).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="nnUNet_raw root directory.",
    )
    parser.add_argument(
        "--dataset_id",
        type=int,
        default=201,
        help="nnU-Net dataset ID (default: 201).",
    )
    parser.add_argument(
        "--multiclass",
        action="store_true",
        help="Use per-vertebra instance labels instead of binary (for full project).",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    prepare_dataset(
        input_dir=args.input,
        output_dir=args.output,
        dataset_id=args.dataset_id,
        binary_labels=not args.multiclass,
    )


if __name__ == "__main__":
    main()
