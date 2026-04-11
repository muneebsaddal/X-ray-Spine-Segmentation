"""
tests/test_predict.py + test_visualize.py

Tests for the inference pipeline (fallback segmenter) and visualization.
These tests do not require a trained nnU-Net model — they use the
SimpleUNetInference fallback and phantom data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from generate_phantom import generate_and_save
from generate_drr import process_dataset
from predict import SimpleUNetInference, convert_images_to_nifti
from visualize import (
    build_overlay,
    make_side_by_side,
    _load_image,
    _load_mask,
    visualize_dataset,
)
from utils import image_to_nifti, load_nifti, nifti_to_image


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def xray_data(tmp_path_factory):
    """Generate DRR X-ray images for testing predict + visualize."""
    tmp = tmp_path_factory.mktemp("xray_data")
    raw_dir  = tmp / "raw"
    xray_dir = tmp / "xray"

    generate_and_save(raw_dir, num_cases=3, seed=13)
    process_dataset(
        input_dir=raw_dir,
        output_dir=xray_dir,
        projection="lateral",
        train_ratio=0.34,
        val_ratio=0.33,
    )
    return xray_dir


# ---------------------------------------------------------------------------
# Tests: SimpleUNetInference (fallback)
# ---------------------------------------------------------------------------

class TestSimpleUNetInference:

    def test_predict_returns_binary(self):
        seg = SimpleUNetInference()
        image = np.random.rand(128, 64).astype(np.float32)
        mask = seg.predict(image)
        unique = np.unique(mask)
        assert set(unique.tolist()).issubset({0, 1}), \
            f"Fallback segmenter should return 0/1 mask, got: {unique}"

    def test_predict_correct_shape(self):
        seg = SimpleUNetInference()
        h, w = 160, 80
        image = np.random.rand(h, w).astype(np.float32)
        mask = seg.predict(image)
        assert mask.shape == (h, w), f"Expected ({h},{w}), got {mask.shape}"

    def test_predict_uint8(self):
        seg = SimpleUNetInference()
        image = np.random.rand(64, 64).astype(np.float32)
        mask = seg.predict(image)
        assert mask.dtype == np.uint8

    def test_predict_batch(self, tmp_path, xray_data):
        """Run predict_batch on test split."""
        test_dir = xray_data / "test"
        if not test_dir.exists() or not list(test_dir.glob("*_0000.nii.gz")):
            pytest.skip("No test images available in this split")

        output_dir = tmp_path / "preds"
        seg = SimpleUNetInference()
        seg.predict_batch(test_dir, output_dir)

        pred_files = list(output_dir.glob("*.nii.gz"))
        assert len(pred_files) > 0, "No predictions were generated"

    def test_predict_batch_mask_values(self, tmp_path, xray_data):
        """Verify predicted masks are binary 0/1."""
        test_dir = xray_data / "test"
        if not test_dir.exists() or not list(test_dir.glob("*_0000.nii.gz")):
            pytest.skip("No test images available")

        output_dir = tmp_path / "preds_val"
        seg = SimpleUNetInference()
        seg.predict_batch(test_dir, output_dir)

        for pred_path in output_dir.glob("*.nii.gz"):
            data, _, _ = load_nifti(pred_path)
            unique = set(np.unique(data).tolist())
            assert unique.issubset({0.0, 1.0}), \
                f"Prediction {pred_path.name} has unexpected values: {unique}"

    def test_predict_nifti_output_readable(self, tmp_path, xray_data):
        test_dir = xray_data / "test"
        if not test_dir.exists():
            pytest.skip("No test split available")
        img_files = list(test_dir.glob("*_0000.nii.gz"))
        if not img_files:
            pytest.skip("No *_0000.nii.gz in test split")

        output_dir = tmp_path / "preds_readable"
        seg = SimpleUNetInference()
        seg.predict_batch(test_dir, output_dir)

        pred = list(output_dir.glob("*.nii.gz"))[0]
        data = nifti_to_image(pred)
        assert data is not None
        assert data.ndim == 2


# ---------------------------------------------------------------------------
# Tests: PNG conversion
# ---------------------------------------------------------------------------

class TestConvertImageToNifti:

    def test_converts_png(self, tmp_path):
        from PIL import Image
        img_dir = tmp_path / "pngs"
        img_dir.mkdir()
        # Create a small synthetic PNG
        arr = (np.random.rand(128, 64) * 255).astype(np.uint8)
        Image.fromarray(arr, mode="L").save(str(img_dir / "test_xray.png"))

        out_dir = tmp_path / "nifti"
        created = convert_images_to_nifti(img_dir, out_dir)

        assert len(created) == 1
        assert created[0].exists()

    def test_converted_nifti_float32(self, tmp_path):
        from PIL import Image
        img_dir = tmp_path / "pngs2"
        img_dir.mkdir()
        arr = (np.random.rand(64, 64) * 255).astype(np.uint8)
        Image.fromarray(arr, mode="L").save(str(img_dir / "case001.png"))

        out_dir = tmp_path / "nifti2"
        convert_images_to_nifti(img_dir, out_dir)

        nii = out_dir / "case001_0000.nii.gz"
        data, _, _ = load_nifti(nii)
        assert data.dtype == np.float32

    def test_converted_nifti_range(self, tmp_path):
        from PIL import Image
        img_dir = tmp_path / "pngs3"
        img_dir.mkdir()
        arr = (np.random.rand(64, 64) * 255).astype(np.uint8)
        Image.fromarray(arr, mode="L").save(str(img_dir / "case002.png"))

        out_dir = tmp_path / "nifti3"
        convert_images_to_nifti(img_dir, out_dir)

        nii = out_dir / "case002_0000.nii.gz"
        data, _, _ = load_nifti(nii)
        assert data.min() >= 0.0
        assert data.max() <= 1.0

    def test_no_images_raises(self, tmp_path):
        from predict import convert_images_to_nifti
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            convert_images_to_nifti(empty_dir, tmp_path / "out")


# ---------------------------------------------------------------------------
# Tests: Visualization
# ---------------------------------------------------------------------------

class TestVisualization:

    @pytest.fixture
    def sample_image_and_mask(self):
        """Create a synthetic X-ray image and binary mask."""
        h, w = 200, 80
        image = np.random.rand(h, w).astype(np.float32)
        mask  = np.zeros((h, w), dtype=np.int32)
        # Simulate 3 vertebrae
        for i, row in enumerate([40, 90, 140]):
            mask[row:row+20, 20:60] = i + 1
        return image, mask

    def test_overlay_shape(self, sample_image_and_mask):
        image, mask = sample_image_and_mask
        overlay = build_overlay(image, mask)
        assert overlay.shape == (*image.shape, 3), \
            f"Expected RGB shape {(*image.shape,3)}, got {overlay.shape}"

    def test_overlay_dtype(self, sample_image_and_mask):
        image, mask = sample_image_and_mask
        overlay = build_overlay(image, mask)
        assert overlay.dtype == np.uint8

    def test_overlay_background_unchanged(self, sample_image_and_mask):
        """Background pixels (mask==0) should remain close to original grayscale."""
        image, mask = sample_image_and_mask
        image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        overlay = build_overlay(image, mask, alpha=0.45)
        bg = mask == 0
        # Background pixels: overlay channel should roughly match original
        orig_vals = image_uint8[bg].astype(float)
        ov_r = overlay[bg, 0].astype(float)
        ov_g = overlay[bg, 1].astype(float)
        ov_b = overlay[bg, 2].astype(float)
        # All channels should be similar to grayscale (within ±5)
        assert np.abs(ov_r - orig_vals).mean() < 5.0

    def test_side_by_side_shape(self, sample_image_and_mask):
        image, mask = sample_image_and_mask
        panel = make_side_by_side(image, mask)
        assert panel.ndim == 3
        assert panel.shape[2] == 3  # RGB
        assert panel.shape[0] >= image.shape[0]
        assert panel.shape[1] > image.shape[1]  # wider than single image

    def test_side_by_side_with_gt(self, sample_image_and_mask):
        image, mask = sample_image_and_mask
        gt = mask.copy()
        gt[50:70, 25:55] = 1  # slightly different gt
        panel = make_side_by_side(image, mask, gt_mask=gt)
        assert panel.ndim == 3

    def test_visualize_dataset(self, tmp_path, xray_data):
        """End-to-end: run fallback predictions then visualize."""
        test_dir = xray_data / "test"
        if not test_dir.exists():
            # Try train as fallback
            test_dir = xray_data / "train"
        if not list(test_dir.glob("*_0000.nii.gz")):
            pytest.skip("No images available for visualization test")

        pred_dir = tmp_path / "viz_preds"
        seg = SimpleUNetInference()
        seg.predict_batch(test_dir, pred_dir)

        viz_dir = tmp_path / "viz_out"
        n = visualize_dataset(
            images_dir=test_dir,
            masks_dir=pred_dir,
            output_dir=viz_dir,
        )
        assert n > 0, "No visualizations were produced"

        viz_files = list(viz_dir.glob("*_visualization.png"))
        assert len(viz_files) > 0

    def test_visualization_png_readable(self, tmp_path, xray_data):
        """Generated PNGs should be valid image files."""
        from PIL import Image

        test_dir = xray_data / "test"
        if not test_dir.exists():
            test_dir = xray_data / "train"
        if not list(test_dir.glob("*_0000.nii.gz")):
            pytest.skip("No images available")

        pred_dir = tmp_path / "viz_preds2"
        seg = SimpleUNetInference()
        seg.predict_batch(test_dir, pred_dir)

        viz_dir = tmp_path / "viz_out2"
        visualize_dataset(images_dir=test_dir, masks_dir=pred_dir, output_dir=viz_dir)

        for png in viz_dir.glob("*.png"):
            img = Image.open(str(png))
            assert img.size[0] > 0 and img.size[1] > 0
