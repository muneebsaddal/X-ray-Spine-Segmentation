"""
src/visualize.py

Generates before/after visualization of vertebra segmentation on X-ray images.

Outputs:
  - Side-by-side PNG: original X-ray | segmentation overlay
  - Individual overlay PNGs per case
  - Optional: segmentation mask as colour PNG

Usage:
    python src/visualize.py \\
        --images data/processed/xray_synthetic/test \\
        --masks  outputs/predictions \\
        --output outputs/visualizations

    # With ground truth comparison:
    python src/visualize.py \\
        --images data/processed/xray_synthetic/test \\
        --masks  outputs/predictions \\
        --gt     data/processed/xray_synthetic/test \\
        --output outputs/visualizations
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from utils import (
    LABEL_COLORS,
    VERTEBRA_LABELS,
    ensure_dir,
    get_label_color,
    nifti_to_image,
    normalize_xray,
    setup_logging,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core rendering helpers
# ---------------------------------------------------------------------------

def _to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert float [0,1] image to uint8 [0,255]."""
    clipped = np.clip(image, 0.0, 1.0)
    return (clipped * 255).astype(np.uint8)


def _load_image(path: Path) -> np.ndarray:
    """Load a 2D image from NIfTI, PNG, or JPEG. Returns float32 [0,1]."""
    suffix = path.suffix.lower()
    if suffix in (".nii", ".gz") or path.name.endswith(".nii.gz"):
        img = nifti_to_image(path)
        return normalize_xray(img)
    else:
        from PIL import Image
        pil = Image.open(str(path)).convert("L")
        arr = np.array(pil, dtype=np.float32)
        return normalize_xray(arr)


def _load_mask(path: Path) -> np.ndarray:
    """Load a label mask. Returns int array."""
    if path.name.endswith(".nii.gz") or path.suffix in (".nii", ".gz"):
        mask = nifti_to_image(path)
        return mask.astype(np.int32)
    else:
        from PIL import Image
        return np.array(Image.open(str(path)), dtype=np.int32)


def build_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Blend a colour segmentation mask over a grayscale X-ray image.

    Args:
        image: float32 (H, W) normalised to [0,1]
        mask:  int (H, W) with vertebra label IDs
        alpha: opacity of the mask overlay (0=transparent, 1=opaque)

    Returns:
        overlay: uint8 (H, W, 3) RGB image
    """
    h, w = image.shape[:2]
    # Convert grayscale X-ray to RGB
    xray_rgb = np.stack([_to_uint8(image)] * 3, axis=-1)  # (H, W, 3)

    # Build colour mask
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    unique_labels = np.unique(mask)
    for lbl in unique_labels:
        if lbl == 0:
            continue
        r, g, b = get_label_color(int(lbl))
        color_mask[mask == lbl] = [r, g, b]

    # Blend
    foreground = mask > 0
    overlay = xray_rgb.copy()
    overlay[foreground] = (
        (1 - alpha) * xray_rgb[foreground].astype(np.float32)
        + alpha * color_mask[foreground].astype(np.float32)
    ).astype(np.uint8)

    # Draw contours around each label region
    overlay = _draw_contours(overlay, mask)
    return overlay


def _draw_contours(
    rgb: np.ndarray,
    mask: np.ndarray,
    contour_color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 1,
) -> np.ndarray:
    """Draw white contours around each labeled region."""
    try:
        import cv2
        result = rgb.copy()
        for lbl in np.unique(mask):
            if lbl == 0:
                continue
            binary = (mask == lbl).astype(np.uint8)
            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(result, contours, -1, contour_color, thickness)
        return result
    except ImportError:
        return rgb  # cv2 not available; return without contours


def build_legend(
    labels_present: List[int],
    height: int,
    width: int = 160,
) -> np.ndarray:
    """Build a small colour legend panel."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        img = Image.new("RGB", (width, height), color=(30, 30, 30))
        draw = ImageDraw.Draw(img)

        row_h = max(16, height // (len(labels_present) + 2))
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 11)
        except Exception:
            font = ImageFont.load_default()

        draw.text((5, 4), "Vertebrae", fill=(220, 220, 220), font=font)

        for i, lbl in enumerate(sorted(labels_present)):
            y = 20 + i * row_h
            if y + row_h > height:
                break
            r, g, b = get_label_color(lbl)
            draw.rectangle([5, y, 20, y + row_h - 2], fill=(r, g, b))
            name = VERTEBRA_LABELS.get(lbl, f"Label {lbl}")
            draw.text((25, y + 1), name, fill=(200, 200, 200), font=font)

        return np.array(img)
    except ImportError:
        return np.full((height, width, 3), 30, dtype=np.uint8)


def make_side_by_side(
    image: np.ndarray,
    mask: np.ndarray,
    gt_mask: Optional[np.ndarray] = None,
    title: str = "",
) -> np.ndarray:
    """
    Create a side-by-side comparison image:
      [Original X-Ray] | [Segmentation Overlay] | (optional: [GT Overlay])

    Returns uint8 RGB array.
    """
    original_rgb = np.stack([_to_uint8(image)] * 3, axis=-1)
    pred_overlay = build_overlay(image, mask)

    panels = [original_rgb, pred_overlay]

    if gt_mask is not None:
        gt_overlay = build_overlay(image, gt_mask)
        panels.append(gt_overlay)

    # Add legend
    labels_present = [int(l) for l in np.unique(mask) if l > 0]
    legend = build_legend(labels_present, height=image.shape[0])
    panels.append(legend)

    # Pad all panels to same height
    max_h = max(p.shape[0] for p in panels)
    padded = []
    for p in panels:
        if p.shape[0] < max_h:
            pad = np.zeros((max_h - p.shape[0], p.shape[1], 3), dtype=np.uint8)
            p = np.vstack([p, pad])
        padded.append(p)

    # Add separator lines between panels
    sep = np.full((max_h, 3, 3), 80, dtype=np.uint8)
    combined = padded[0]
    for panel in padded[1:]:
        combined = np.hstack([combined, sep, panel])

    # Add title bar
    if title:
        combined = _add_title_bar(combined, title)

    return combined


def _add_title_bar(
    image: np.ndarray,
    title: str,
    bar_height: int = 28,
) -> np.ndarray:
    """Add a dark title bar at the top of the image."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        bar = Image.new("RGB", (image.shape[1], bar_height), color=(20, 20, 20))
        draw = ImageDraw.Draw(bar)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 13)
        except Exception:
            font = ImageFont.load_default()
        draw.text((8, 5), title, fill=(220, 220, 220), font=font)
        bar_arr = np.array(bar)
        return np.vstack([bar_arr, image])
    except ImportError:
        return image


# ---------------------------------------------------------------------------
# Batch visualization
# ---------------------------------------------------------------------------

def visualize_dataset(
    images_dir: Path,
    masks_dir: Path,
    output_dir: Path,
    gt_dir: Optional[Path] = None,
) -> int:
    """
    Generate visualizations for all cases in `masks_dir`.

    Returns number of successfully visualized cases.
    """
    from PIL import Image

    ensure_dir(output_dir)

    mask_files = sorted(masks_dir.glob("*.nii.gz"))
    if not mask_files:
        raise FileNotFoundError(f"No prediction masks found in: {masks_dir}")

    successes = 0
    for mask_path in mask_files:
        case_id = mask_path.name.replace(".nii.gz", "")

        # Find corresponding image
        img_path = None
        for suffix in [f"_0000.nii.gz", ".nii.gz", ".png", ".jpg"]:
            candidate = images_dir / f"{case_id}{suffix}"
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            logger.warning("No image found for mask: %s — skipping", mask_path.name)
            continue

        try:
            image = _load_image(img_path)
            mask  = _load_mask(mask_path)

            # Align shapes (nnU-Net may add/remove a dim)
            if image.ndim != mask.ndim:
                image = image.squeeze()
                mask  = mask.squeeze()

            if image.shape != mask.shape:
                logger.warning(
                    "Shape mismatch: image %s vs mask %s for %s — resizing mask",
                    image.shape, mask.shape, case_id,
                )
                from skimage.transform import resize
                mask = resize(
                    mask.astype(np.float32),
                    image.shape,
                    order=0,  # nearest neighbor for labels
                    anti_aliasing=False,
                ).astype(np.int32)

            gt_mask = None
            if gt_dir is not None:
                gt_path = gt_dir / mask_path.name
                if gt_path.exists():
                    gt_mask = _load_mask(gt_path).squeeze()

            labels_found = [int(l) for l in np.unique(mask) if l > 0]
            logger.info("Visualizing %s | Labels: %s", case_id, labels_found)

            panel = make_side_by_side(
                image=image,
                mask=mask,
                gt_mask=gt_mask,
                title=f"Spine X-Ray Segmentation — {case_id}",
            )

            out_path = output_dir / f"{case_id}_visualization.png"
            Image.fromarray(panel).save(str(out_path))

            # Also save standalone colour mask
            color_mask_path = output_dir / f"{case_id}_mask_color.png"
            overlay = build_overlay(image, mask, alpha=0.8)
            Image.fromarray(overlay).save(str(color_mask_path))

            successes += 1
            logger.info("Saved: %s", out_path.name)

        except Exception as exc:
            logger.error("Visualization failed for %s: %s", case_id, exc, exc_info=True)

    logger.info(
        "Visualization complete. %d/%d cases saved to %s",
        successes, len(mask_files), output_dir,
    )
    return successes


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize X-ray vertebra segmentation results."
    )
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--masks",  type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/visualizations"))
    parser.add_argument(
        "--gt", type=Path, default=None,
        help="Ground-truth mask directory for side-by-side comparison.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    n = visualize_dataset(
        images_dir=args.images,
        masks_dir=args.masks,
        output_dir=args.output,
        gt_dir=args.gt,
    )
    if n == 0:
        raise SystemExit("No visualizations were produced. Check logs for errors.")


if __name__ == "__main__":
    main()
