"""
tests/test_drr_generation.py

Unit and integration tests for generate_phantom.py and generate_drr.py.
All tests run without downloading real data.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from generate_phantom import generate_phantom_volume, generate_and_save
from generate_drr import generate_drr, project_label_mask, process_case
from utils import load_nifti, save_nifti, normalize_xray


# ---------------------------------------------------------------------------
# Phantom generation tests
# ---------------------------------------------------------------------------

class TestPhantomGeneration:

    def test_volume_shape(self):
        ct, labels = generate_phantom_volume()
        assert ct.shape == (80, 80, 200), f"Unexpected shape: {ct.shape}"
        assert labels.shape == (80, 80, 200)

    def test_label_count(self):
        _, labels = generate_phantom_volume()
        unique = np.unique(labels)
        # Should have background + at least 5 vertebra labels
        assert 0 in unique, "Background (0) should be present"
        n_vertebrae = len(unique) - 1
        assert n_vertebrae >= 5, f"Expected >= 5 vertebra labels, got {n_vertebrae}"

    def test_ct_value_range(self):
        ct, _ = generate_phantom_volume()
        assert ct.min() >= -1025, f"CT min too low: {ct.min()}"
        assert ct.max() <= 3072, f"CT max too high: {ct.max()}"

    def test_reproducibility(self):
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        ct1, _ = generate_phantom_volume(rng=rng1)
        ct2, _ = generate_phantom_volume(rng=rng2)
        np.testing.assert_array_almost_equal(ct1, ct2)

    def test_save_and_reload(self, tmp_path):
        generate_and_save(tmp_path, num_cases=2)
        ct_files = list(tmp_path.glob("*_ct.nii.gz"))
        lbl_files = list(tmp_path.glob("*_seg-vert_msk.nii.gz"))
        assert len(ct_files) == 2, f"Expected 2 CT files, got {len(ct_files)}"
        assert len(lbl_files) == 2, f"Expected 2 label files, got {len(lbl_files)}"

    def test_saved_nifti_readable(self, tmp_path):
        generate_and_save(tmp_path, num_cases=1)
        ct_file = list(tmp_path.glob("*_ct.nii.gz"))[0]
        data, affine, _ = load_nifti(ct_file)
        assert data.ndim == 3
        assert not np.all(data == 0), "CT should not be all zeros"


# ---------------------------------------------------------------------------
# DRR generation tests
# ---------------------------------------------------------------------------

class TestDRRGeneration:

    @pytest.fixture
    def sample_ct_and_labels(self):
        ct, labels = generate_phantom_volume()
        return ct, labels

    def test_drr_lateral_shape(self, sample_ct_and_labels):
        ct, _ = sample_ct_and_labels
        drr = generate_drr(ct, axis=1)
        assert drr.ndim == 2
        # Lateral: project along Y (axis=1), so output is (X, Z) = (80, 200)
        assert drr.shape == (ct.shape[0], ct.shape[2])

    def test_drr_ap_shape(self, sample_ct_and_labels):
        ct, _ = sample_ct_and_labels
        drr = generate_drr(ct, axis=0)
        # AP: project along X (axis=0), so output is (Y, Z) = (80, 200)
        assert drr.shape == (ct.shape[1], ct.shape[2])

    def test_drr_range(self, sample_ct_and_labels):
        ct, _ = sample_ct_and_labels
        drr = generate_drr(ct, axis=1)
        assert drr.dtype == np.float32
        assert drr.min() >= 0.0, f"DRR min should be >= 0, got {drr.min()}"
        assert drr.max() <= 1.0, f"DRR max should be <= 1, got {drr.max()}"

    def test_drr_not_uniform(self, sample_ct_and_labels):
        ct, _ = sample_ct_and_labels
        drr = generate_drr(ct, axis=1)
        # The DRR should have structure (std > 0.05 for a spine phantom)
        assert drr.std() > 0.05, f"DRR appears uniform: std={drr.std()}"

    def test_drr_bone_visible(self, sample_ct_and_labels):
        ct, labels = sample_ct_and_labels
        drr = generate_drr(ct, axis=1)
        # The DRR should be darker where vertebrae are (bone = high attenuation)
        # Mean intensity in vertebra region should differ from background
        label_2d = project_label_mask(labels, axis=1)
        bone_pixels = drr[label_2d > 0]
        bg_pixels   = drr[label_2d == 0]
        if len(bone_pixels) > 0 and len(bg_pixels) > 0:
            # In an X-ray, bone absorbs more → appears brighter in our inverted convention
            assert bone_pixels.mean() != bg_pixels.mean(), \
                "Bone and background should have different intensities in DRR"


# ---------------------------------------------------------------------------
# Mask projection tests
# ---------------------------------------------------------------------------

class TestMaskProjection:

    @pytest.fixture
    def label_volume(self):
        _, labels = generate_phantom_volume()
        return labels

    def test_projection_shape(self, label_volume):
        proj = project_label_mask(label_volume, axis=1)
        assert proj.ndim == 2
        assert proj.shape == (label_volume.shape[0], label_volume.shape[2])

    def test_projection_labels_preserved(self, label_volume):
        proj = project_label_mask(label_volume, axis=1)
        orig_labels = set(np.unique(label_volume).tolist()) - {0}
        proj_labels = set(np.unique(proj).tolist()) - {0}
        # All vertebra labels should be projected (some might merge)
        assert len(proj_labels) > 0, "No vertebra labels in projected mask"
        # Projected labels should be subset of original labels
        assert proj_labels.issubset(orig_labels | {0}), \
            f"Unexpected labels in projection: {proj_labels - orig_labels}"

    def test_projection_no_negatives(self, label_volume):
        proj = project_label_mask(label_volume, axis=1)
        assert proj.min() >= 0, "Label mask should not have negative values"

    def test_projection_dtype(self, label_volume):
        proj = project_label_mask(label_volume, axis=1)
        assert proj.dtype in (np.int16, np.int32, np.int64)


# ---------------------------------------------------------------------------
# End-to-end case processing test
# ---------------------------------------------------------------------------

class TestProcessCase:

    def test_process_phantom_case(self, tmp_path):
        """Full test: generate phantom → save NIfTI → process_case → verify output."""
        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "out"
        ensure_path(raw_dir)
        ensure_path(out_dir)

        # Generate phantom
        from generate_phantom import generate_and_save
        generate_and_save(raw_dir, num_cases=1, seed=7)

        ct_files = list(raw_dir.glob("*_ct.nii.gz"))
        assert len(ct_files) == 1

        ct_path   = ct_files[0]
        case_id   = ct_path.name.replace("_dir-sag_ct.nii.gz", "").replace("_ct.nii.gz", "")
        lbl_path  = raw_dir / ct_path.name.replace("_ct.nii.gz", "_dir-sag_seg-vert_msk.nii.gz")

        ok = process_case(
            ct_path=ct_path,
            label_path=lbl_path if lbl_path.exists() else None,
            output_dir=out_dir,
            axis=1,
            case_id=case_id,
        )

        assert ok, "process_case returned False — check logs"

        # Verify output files exist
        drr_file  = out_dir / f"{case_id}_0000.nii.gz"
        assert drr_file.exists(), f"DRR file not created: {drr_file}"

        # Verify DRR is readable and in range
        drr_data, _, _ = load_nifti(drr_file)
        drr_data = drr_data.squeeze()
        assert drr_data.min() >= 0.0
        assert drr_data.max() <= 1.0


def ensure_path(p):
    p.mkdir(parents=True, exist_ok=True)
