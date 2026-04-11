"""
tests/test_utils.py

Unit tests for src/utils.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import (
    VERTEBRA_LABELS,
    build_dataset_json,
    ensure_dir,
    find_files,
    get_label_color,
    image_to_nifti,
    load_nifti,
    nifti_to_image,
    normalize_xray,
    save_nifti,
    window_ct,
    write_dataset_json,
)


class TestEnsureDir:
    def test_creates_directory(self, tmp_path):
        new_dir = tmp_path / "a" / "b" / "c"
        assert not new_dir.exists()
        result = ensure_dir(new_dir)
        assert new_dir.exists()
        assert result == new_dir

    def test_idempotent(self, tmp_path):
        d = tmp_path / "existing"
        d.mkdir()
        ensure_dir(d)  # Should not raise


class TestFindFiles:
    def test_finds_nifti(self, tmp_path):
        (tmp_path / "a.nii.gz").touch()
        (tmp_path / "b.nii.gz").touch()
        (tmp_path / "c.txt").touch()
        found = find_files(tmp_path, "*.nii.gz")
        assert len(found) == 2

    def test_recursive(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "deep.nii.gz").touch()
        found = find_files(tmp_path, "*.nii.gz")
        assert len(found) == 1


class TestNormalizeXray:
    def test_range_zero_one(self):
        arr = np.random.rand(100, 100).astype(np.float32) * 1000
        norm = normalize_xray(arr)
        assert norm.min() >= 0.0
        assert norm.max() <= 1.0

    def test_dtype_float32(self):
        arr = np.ones((50, 50))
        norm = normalize_xray(arr)
        assert norm.dtype == np.float32

    def test_uniform_returns_zeros(self):
        arr = np.full((10, 10), 500.0)
        norm = normalize_xray(arr)
        assert np.all(norm == 0.0)

    def test_preserves_shape(self):
        arr = np.random.rand(64, 128)
        norm = normalize_xray(arr)
        assert norm.shape == arr.shape


class TestWindowCT:
    def test_clips_low(self):
        arr = np.full((10, 10), -2000.0)
        windowed = window_ct(arr, window_center=400, window_width=1800)
        expected_low = 400 - 1800 / 2
        assert np.all(windowed == expected_low)

    def test_clips_high(self):
        arr = np.full((10, 10), 5000.0)
        windowed = window_ct(arr, window_center=400, window_width=1800)
        expected_high = 400 + 1800 / 2
        assert np.all(windowed == expected_high)

    def test_preserves_within_window(self):
        arr = np.array([[400.0]], dtype=np.float32)
        windowed = window_ct(arr)
        assert windowed[0, 0] == pytest.approx(400.0)


class TestNiftiRoundtrip:
    def test_save_load_roundtrip(self, tmp_path):
        arr = np.random.rand(64, 64).astype(np.float32)
        affine = np.eye(4)
        path = tmp_path / "test.nii.gz"
        save_nifti(arr, affine, path)
        loaded, loaded_affine, _ = load_nifti(path)
        np.testing.assert_array_almost_equal(arr, loaded.squeeze())

    def test_image_to_nifti_and_back(self, tmp_path):
        arr = np.random.rand(80, 200).astype(np.float32)
        path = tmp_path / "xray.nii.gz"
        image_to_nifti(arr, path, spacing=(1.0, 1.0))
        loaded = nifti_to_image(path)
        assert loaded.shape == arr.shape
        np.testing.assert_array_almost_equal(arr, loaded)

    def test_nifti_2d_wrapping(self, tmp_path):
        """image_to_nifti wraps 2D as (1, H, W); nifti_to_image squeezes back."""
        arr = np.random.rand(100, 50).astype(np.float32)
        path = tmp_path / "wrap.nii.gz"
        image_to_nifti(arr, path)
        result = nifti_to_image(path)
        assert result.shape == (100, 50)

    def test_label_mask_roundtrip(self, tmp_path):
        mask = np.array([[0, 1, 2], [3, 0, 1]], dtype=np.int16)
        affine = np.eye(4)
        path = tmp_path / "mask.nii.gz"
        save_nifti(mask, affine, path, dtype=np.int16)
        loaded, _, _ = load_nifti(path)
        np.testing.assert_array_equal(mask, loaded.squeeze().astype(np.int16))


class TestDatasetJson:
    def test_build_structure(self):
        result = build_dataset_json(
            dataset_name="Dataset201_SpineXray2D",
            num_training=10,
            channel_names={0: "X-Ray"},
            labels={"background": 0, "vertebra": 1},
        )
        assert "channel_names" in result
        assert "labels" in result
        assert result["numTraining"] == 10
        assert result["file_ending"] == ".nii.gz"

    def test_write_and_read(self, tmp_path):
        import json
        content = build_dataset_json(
            dataset_name="Test",
            num_training=5,
            channel_names={0: "X-Ray"},
            labels={"background": 0, "vertebra": 1},
        )
        path = tmp_path / "dataset.json"
        write_dataset_json(path, content)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["numTraining"] == 5


class TestLabelColors:
    def test_known_label_returns_tuple(self):
        color = get_label_color(1)
        assert isinstance(color, tuple)
        assert len(color) == 3

    def test_unknown_label_returns_default(self):
        color = get_label_color(9999)
        assert isinstance(color, tuple)
        assert len(color) == 3

    def test_all_vertebra_labels_have_colors(self):
        for label_id in VERTEBRA_LABELS:
            color = get_label_color(label_id)
            assert len(color) == 3
            for c in color:
                assert 0 <= c <= 255
