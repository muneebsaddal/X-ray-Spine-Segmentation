"""
tests/test_dataset_preparation.py

Tests for prepare_nnunet_dataset.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from generate_phantom import generate_and_save
from generate_drr import process_dataset
from prepare_nnunet_dataset import prepare_dataset, _verify_dataset
from utils import load_nifti, nifti_to_image


@pytest.fixture(scope="module")
def prepared_dataset(tmp_path_factory):
    """
    Fixture: runs the full phantom → DRR → nnU-Net dataset pipeline.
    Reused across all tests in this module.
    """
    tmp = tmp_path_factory.mktemp("dataset")

    # Step 1: generate phantoms
    raw_dir = tmp / "raw"
    generate_and_save(raw_dir, num_cases=4, seed=99)

    # Step 2: DRR generation
    xray_dir = tmp / "xray"
    process_dataset(
        input_dir=raw_dir,
        output_dir=xray_dir,
        projection="lateral",
        train_ratio=0.5,
        val_ratio=0.25,
    )

    # Step 3: nnU-Net dataset prep
    nnunet_raw = tmp / "nnunet_raw"
    prepare_dataset(
        input_dir=xray_dir,
        output_dir=nnunet_raw,
        dataset_id=201,
        binary_labels=True,
    )

    return {
        "root": tmp,
        "raw_dir": raw_dir,
        "xray_dir": xray_dir,
        "nnunet_raw": nnunet_raw,
        "ds_root": nnunet_raw / "Dataset201_SpineXray2D",
    }


class TestNnUNetDatasetStructure:

    def test_dataset_root_exists(self, prepared_dataset):
        assert prepared_dataset["ds_root"].exists()

    def test_required_directories_exist(self, prepared_dataset):
        ds = prepared_dataset["ds_root"]
        assert (ds / "imagesTr").exists()
        assert (ds / "labelsTr").exists()

    def test_dataset_json_exists(self, prepared_dataset):
        ds_json = prepared_dataset["ds_root"] / "dataset.json"
        assert ds_json.exists()

    def test_dataset_json_valid(self, prepared_dataset):
        ds_json = prepared_dataset["ds_root"] / "dataset.json"
        with open(ds_json) as f:
            content = json.load(f)
        assert "channel_names" in content
        assert "labels" in content
        assert "numTraining" in content
        assert "file_ending" in content
        assert content["file_ending"] == ".nii.gz"

    def test_dataset_json_label_count(self, prepared_dataset):
        ds_json = prepared_dataset["ds_root"] / "dataset.json"
        with open(ds_json) as f:
            content = json.load(f)
        # Binary mode: background + vertebra
        assert len(content["labels"]) >= 2

    def test_training_images_exist(self, prepared_dataset):
        images_tr = prepared_dataset["ds_root"] / "imagesTr"
        img_files = list(images_tr.glob("*_0000.nii.gz"))
        assert len(img_files) >= 1, "No training images found"

    def test_training_labels_exist(self, prepared_dataset):
        labels_tr = prepared_dataset["ds_root"] / "labelsTr"
        lbl_files = list(labels_tr.glob("*.nii.gz"))
        assert len(lbl_files) >= 1, "No training labels found"

    def test_image_label_pairing(self, prepared_dataset):
        """Every image in imagesTr must have a matching label in labelsTr."""
        ds = prepared_dataset["ds_root"]
        images_tr = ds / "imagesTr"
        labels_tr = ds / "labelsTr"

        img_cases = {f.name.replace("_0000.nii.gz", "") for f in images_tr.glob("*_0000.nii.gz")}
        lbl_cases = {f.name.replace(".nii.gz", "") for f in labels_tr.glob("*.nii.gz")}

        missing_labels = img_cases - lbl_cases
        assert len(missing_labels) == 0, f"Missing labels for: {missing_labels}"

    def test_num_training_matches_json(self, prepared_dataset):
        ds = prepared_dataset["ds_root"]
        images_tr = ds / "imagesTr"
        actual_count = len(list(images_tr.glob("*_0000.nii.gz")))

        with open(ds / "dataset.json") as f:
            json_count = json.load(f)["numTraining"]

        assert actual_count == json_count, \
            f"dataset.json numTraining={json_count} but found {actual_count} files"


class TestNnUNetImageFormat:

    def test_image_nifti_readable(self, prepared_dataset):
        images_tr = prepared_dataset["ds_root"] / "imagesTr"
        img_file = next(images_tr.glob("*_0000.nii.gz"))
        data, affine, header = load_nifti(img_file)
        assert data is not None
        assert data.ndim in (2, 3)

    def test_image_float32(self, prepared_dataset):
        images_tr = prepared_dataset["ds_root"] / "imagesTr"
        img_file = next(images_tr.glob("*_0000.nii.gz"))
        data, _, _ = load_nifti(img_file)
        assert data.dtype == np.float32

    def test_label_binary_values(self, prepared_dataset):
        """In binary mode, label mask should only contain 0 and 1."""
        labels_tr = prepared_dataset["ds_root"] / "labelsTr"
        lbl_file = next(labels_tr.glob("*.nii.gz"))
        data, _, _ = load_nifti(lbl_file)
        unique_vals = set(np.unique(data).tolist())
        assert unique_vals.issubset({0.0, 1.0}), \
            f"Expected only {{0,1}} in binary mask, got: {unique_vals}"

    def test_image_label_same_shape(self, prepared_dataset):
        ds = prepared_dataset["ds_root"]
        images_tr = ds / "imagesTr"
        labels_tr = ds / "labelsTr"

        img_file = next(images_tr.glob("*_0000.nii.gz"))
        case_id = img_file.name.replace("_0000.nii.gz", "")
        lbl_file = labels_tr / f"{case_id}.nii.gz"

        if not lbl_file.exists():
            pytest.skip(f"Label not found for {case_id}")

        img_data, _, _ = load_nifti(img_file)
        lbl_data, _, _ = load_nifti(lbl_file)

        assert img_data.shape == lbl_data.shape, \
            f"Shape mismatch: image {img_data.shape} vs label {lbl_data.shape}"

    def test_affine_has_positive_spacing(self, prepared_dataset):
        images_tr = prepared_dataset["ds_root"] / "imagesTr"
        img_file = next(images_tr.glob("*_0000.nii.gz"))
        _, affine, _ = load_nifti(img_file)
        # Diagonal elements represent voxel spacing
        sx, sy = abs(affine[0, 0]), abs(affine[1, 1])
        assert sx > 0, f"X spacing must be positive, got {sx}"
        assert sy > 0, f"Y spacing must be positive, got {sy}"


class TestDatasetIntegrity:

    def test_verify_passes(self, prepared_dataset):
        """_verify_dataset should not raise for a correctly built dataset."""
        ds = prepared_dataset["ds_root"]
        images_tr = ds / "imagesTr"
        n_training = len(list(images_tr.glob("*_0000.nii.gz")))
        # Should not raise
        _verify_dataset(ds, n_training)
