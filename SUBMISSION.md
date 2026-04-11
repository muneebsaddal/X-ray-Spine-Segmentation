# Spine X-Ray Segmentation POC Submission

## Summary

This repository is a proof of concept for a larger spine X-ray segmentation project.

It demonstrates an end-to-end workflow that:
- starts from CT volumes and vertebra masks,
- generates synthetic 2D X-ray-like images using DRR projection,
- converts the data into nnU-Net v2 2D dataset structure,
- runs segmentation inference,
- and produces visual overlays for quick qualitative review.

The goal of this POC is to validate the pipeline design and reduce technical risk before building the complete production version.

The repository is delivered in a clean code-only state. Runtime folders such as `data/` and `outputs/` are generated locally when the pipeline is executed.

## What The Code Does

The repository is organized as a simple linear pipeline:

1. `src/generate_phantom.py`
Creates synthetic CT-like spine volumes and vertebra masks so the system can be tested without depending entirely on external datasets.

2. `src/generate_drr.py`
Converts 3D CT data into 2D synthetic X-ray images using DRR-style projection and projects 3D vertebra labels into 2D masks.

3. `src/prepare_nnunet_dataset.py`
Repackages the generated 2D images and labels into nnU-Net v2 raw dataset format.

4. `src/train.py`
Runs nnU-Net v2 training for the 2D segmentation task.

5. `src/predict.py`
Runs inference on new X-ray inputs. It supports:
- nnU-Net prediction when a trained model is available,
- a lightweight fallback segmenter for demonstration when full training is not available.

6. `src/visualize.py`
Creates side-by-side visual outputs showing the original X-ray and the predicted segmentation overlay.

## What To Tell The Client

You can describe the POC like this:

"This proof of concept shows the full workflow for spine X-ray segmentation. We start with labeled 3D spine data, generate synthetic 2D X-rays, train or simulate a 2D segmentation model, and produce visual overlays that show where the vertebrae were detected. The POC confirms that the technical pipeline works end to end and can be extended into a full production system using real X-ray data, stronger model training, and more detailed vertebra labeling."

Shorter version:

"The POC proves that we can build a full spine X-ray segmentation pipeline, from data preparation to model-ready training format, inference, and visualization."

## Recommended Demo Commands

If you want the simplest runnable demo flow:

```powershell
pip install -r requirements.txt
python src\generate_phantom.py --output data\raw\verse_sample --num_cases 3
python src\generate_drr.py --input data\raw\verse_sample --output data\processed\xray_synthetic --projection lateral
python src\predict.py --input data\processed\xray_synthetic\test --output outputs\predictions --fallback
python src\visualize.py --images data\processed\xray_synthetic\test --masks outputs\predictions --output outputs\visualizations
```

This route is the best choice for a quick demonstration because it does not require a fully trained nnU-Net checkpoint.

## Full Pipeline Commands

If nnU-Net is installed and you want to demonstrate the intended training path:

```powershell
pip install -r requirements.txt
python src\generate_drr.py --input data\raw\verse_sample --output data\processed\xray_synthetic --projection lateral
python src\prepare_nnunet_dataset.py --input data\processed\xray_synthetic --output data\nnunet_raw --dataset_id 201
python src\train.py --dataset_id 201 --fold 0 --fast
python src\predict.py --input data\processed\xray_synthetic\test --output outputs\predictions --dataset_id 201
python src\visualize.py --images data\processed\xray_synthetic\test --masks outputs\predictions --output outputs\visualizations
```

## Scope Of This POC

Included:
- DRR-based synthetic X-ray generation from CT
- 2D mask projection
- nnU-Net dataset preparation
- inference pipeline
- visualization outputs
- automated tests for major pipeline stages

Reserved for the full project:
- larger-scale model training and tuning
- evaluation on real clinical X-ray data
- full vertebra-by-vertebra labeling workflow
- stronger post-processing and deployment packaging

## Practical Notes

- The fallback inference mode exists to prove pipeline connectivity and generate demo outputs quickly.
- No generated artifacts are committed; outputs are created locally when you run the steps.
- Some tests depend on Python packages from `requirements.txt`, especially `nibabel`.

## Submission Positioning

This should be presented as:

"A working proof of concept that validates the architecture and core workflow for 2D spine X-ray segmentation, with a clear path to a production-ready full project."
