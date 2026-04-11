"""
src/utils.py
Shared utilities for the TotalSpineSeg 2D X-Ray pipeline.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Vertebra label mapping (matches TotalSpineSeg / VerSe convention)
# ---------------------------------------------------------------------------

VERTEBRA_LABELS: Dict[int, str] = {
    1: "C1",
    2: "C2",
    3: "C3",
    4: "C4",
    5: "C5",
    6: "C6",
    7: "C7",
    8: "T1",
    9: "T2",
    10: "T3",
    11: "T4",
    12: "T5",
    13: "T6",
    14: "T7",
    15: "T8",
    16: "T9",
    17: "T10",
    18: "T11",
    19: "T12",
    20: "L1",
    21: "L2",
    22: "L3",
    23: "L4",
    24: "L5",
    25: "Sacrum",
    # Simplified semantic classes for 2D POC
    100: "Vertebra",
    200: "IVD",
}

# Colour map for visualization (label_id -> BGR)
LABEL_COLORS: Dict[int, Tuple[int, int, int]] = {
    1: (255, 50, 50),
    2: (255, 100, 50),
    3: (255, 150, 50),
    4: (255, 200, 50),
    5: (255, 255, 50),
    6: (200, 255, 50),
    7: (100, 255, 50),
    8: (50, 255, 100),
    9: (50, 255, 150),
    10: (50, 255, 200),
    11: (50, 255, 255),
    12: (50, 200, 255),
    13: (50, 150, 255),
    14: (50, 100, 255),
    15: (100, 50, 255),
    16: (150, 50, 255),
    17: (200, 50, 255),
    18: (255, 50, 255),
    19: (255, 50, 200),
    20: (255, 50, 150),
    21: (255, 50, 100),
    22: (200, 100, 100),
    23: (100, 200, 100),
    24: (100, 100, 200),
    25: (200, 200, 100),
    100: (0, 200, 100),
    200: (100, 100, 220),
}

DEFAULT_COLOR = (180, 180, 180)


def get_label_color(label_id: int) -> Tuple[int, int, int]:
    """Return RGB color for a given label id."""
    return LABEL_COLORS.get(label_id, DEFAULT_COLOR)


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: Union[str, Path]) -> Path:
    """Create directory (and parents) if it does not exist. Return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def find_files(directory: Union[str, Path], pattern: str) -> List[Path]:
    """Recursively find files matching glob pattern."""
    return sorted(Path(directory).rglob(pattern))


def safe_copy(src: Union[str, Path], dst: Union[str, Path]) -> None:
    """Copy file, creating destination directory if needed."""
    dst = Path(dst)
    ensure_dir(dst.parent)
    shutil.copy2(str(src), str(dst))


# ---------------------------------------------------------------------------
# NIfTI / image metadata helpers
# ---------------------------------------------------------------------------

def load_nifti(path: Union[str, Path]):
    """Load a NIfTI file. Returns (data_array, affine, header)."""
    import nibabel as nib  # lazy import — not needed for all utils

    img = nib.load(str(path))
    data = np.asarray(img.dataobj, dtype=np.float32)
    return data, img.affine, img.header


def save_nifti(
    data: np.ndarray,
    affine: np.ndarray,
    path: Union[str, Path],
    dtype=None,
) -> None:
    """Save numpy array as NIfTI file."""
    import nibabel as nib

    ensure_dir(Path(path).parent)
    if dtype is not None:
        data = data.astype(dtype)
    img = nib.Nifti1Image(data, affine)
    nib.save(img, str(path))
    logger.debug("Saved NIfTI: %s  shape=%s", path, data.shape)


def image_to_nifti(
    image: np.ndarray,
    output_path: Union[str, Path],
    spacing: Tuple[float, float] = (1.0, 1.0),
) -> None:
    """
    Wrap a 2D grayscale image (H, W) into a NIfTI file for nnU-Net.
    nnU-Net 2D expects shape (1, H, W) with a dummy depth of 1.
    """
    import nibabel as nib

    # nnU-Net reads 3D NIfTI; we fake depth=1 for 2D images
    if image.ndim == 2:
        image = image[np.newaxis, ...]  # (1, H, W)

    # Build affine with physical spacing (sx, sy, 1.0)
    sx, sy = spacing
    affine = np.diag([sx, sy, 1.0, 1.0])

    ensure_dir(Path(output_path).parent)
    img = nib.Nifti1Image(image.astype(np.float32), affine)
    nib.save(img, str(output_path))


def nifti_to_image(path: Union[str, Path]) -> np.ndarray:
    """
    Load a NIfTI file and return a 2D (H, W) numpy array.
    Squeezes the dummy depth dimension added by image_to_nifti.
    """
    import nibabel as nib

    img = nib.load(str(path))
    data = np.asarray(img.dataobj)
    if data.ndim == 3 and data.shape[0] == 1:
        data = data[0]  # squeeze (1, H, W) -> (H, W)
    return data.astype(np.float32)


# ---------------------------------------------------------------------------
# Intensity normalisation
# ---------------------------------------------------------------------------

def normalize_xray(image: np.ndarray) -> np.ndarray:
    """
    Normalize X-ray / DRR image to [0, 1] float32.
    Clips extreme outliers (0.5th / 99.5th percentile) before scaling.
    """
    p_low = np.percentile(image, 0.5)
    p_high = np.percentile(image, 99.5)
    if p_high - p_low < 1e-8:
        return np.zeros_like(image, dtype=np.float32)
    clipped = np.clip(image, p_low, p_high)
    return ((clipped - p_low) / (p_high - p_low)).astype(np.float32)


def window_ct(
    volume: np.ndarray,
    window_center: float = 400.0,
    window_width: float = 1800.0,
) -> np.ndarray:
    """Apply HU windowing suitable for bone visualization."""
    low = window_center - window_width / 2
    high = window_center + window_width / 2
    return np.clip(volume, low, high).astype(np.float32)


# ---------------------------------------------------------------------------
# nnU-Net dataset.json helpers
# ---------------------------------------------------------------------------

def build_dataset_json(
    dataset_name: str,
    num_training: int,
    channel_names: Dict[int, str],
    labels: Dict[str, int],
    file_ending: str = ".nii.gz",
) -> Dict:
    """Build a minimal nnU-Net v2 dataset.json dict."""
    return {
        "channel_names": {str(k): v for k, v in channel_names.items()},
        "labels": {v: k for k, v in labels.items()},  # name -> int
        "numTraining": num_training,
        "file_ending": file_ending,
        "name": dataset_name,
        "overwrite_image_reader_writer": "SimpleITKIO",
    }


def write_dataset_json(path: Union[str, Path], content: Dict) -> None:
    """Write dataset.json to disk."""
    ensure_dir(Path(path).parent)
    with open(str(path), "w") as f:
        json.dump(content, f, indent=4)
    logger.info("Wrote dataset.json: %s", path)


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with a sensible format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)-7s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
