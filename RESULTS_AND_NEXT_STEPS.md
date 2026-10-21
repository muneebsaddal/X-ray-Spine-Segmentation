# Spine X-Ray Segmentation POC Results

## What Is Working Now

This project now runs an end-to-end proof-of-concept workflow for spine segmentation visualization:

1. A 3D CT spine volume is converted into a synthetic 2D X-ray-like image.
2. A corresponding 3D vertebra mask is projected into 2D.
3. The 2D image and mask are visualized as a segmentation overlay.
4. A lightweight fallback segmenter can also generate demo masks when a trained model is not available.

The best client-facing outputs are:

- Synthetic phantom prediction:
  `outputs/visualizations_working/phantom/sub-phantom001_visualization.png`
- Real CT-derived projected vertebra labels:
  `outputs/visualizations_working/verse225_projected_labels_clean/verse225_visualization.png`

Important distinction: the second sample uses real/public CT-derived data and its projected labels. It is not a validated model prediction on a real clinical 2D X-ray.

## How 3D CT Is Converted To 2D X-Ray Data

The conversion is implemented in `src/generate_drr.py`.

The input is a 3D CT volume stored as NIfTI, for example:

```text
data/verse225/verse225.nii.gz
```

The matching 3D vertebra segmentation mask is:

```text
data/verse225/verse225_seg.nii.gz
```

The CT volume contains voxel intensities in Hounsfield-unit-like values. The DRR generation method is:

1. Apply a bone CT window to emphasize vertebrae.
2. Convert CT intensity values to simplified X-ray attenuation coefficients.
3. Sum attenuation values along one projection axis.
4. Apply an exponential Beer-Lambert-style transform.
5. Normalize the result to a 2D image in the range `[0, 1]`.

For a lateral projection, the code projects through the Y axis:

```text
Input CT:    X by Y by Z
Projection:  sum through Y
Output DRR:  X by Z
```

This produces a synthetic lateral X-ray-style image from the 3D CT scan.

## How 3D Masks Become 2D Segmentation Masks

The 3D vertebra mask is projected along the same axis as the CT image.

For each pixel in the 2D image, the code looks through the corresponding ray in the 3D mask and counts which vertebra label occupies the most voxels. The winning label becomes the 2D label for that pixel. Background remains `0`.

This means the resulting 2D mask preserves vertebra IDs where available:

```text
0 = background
1 = C1
2 = C2
3 = C3
...
```

For the `verse225` sample, the projected labels include vertebra IDs `1` through `9`.

## How Segmentation Is Done

There are three segmentation paths in this POC:

1. Projected label segmentation
   - Uses the existing 3D CT segmentation mask.
   - Projects that real 3D mask into 2D.
   - This is the cleanest current visualization for CT-derived data.

2. Fallback demo segmentation
   - Implemented in `src/predict.py` as `SimpleUNetInference`.
   - Uses intensity thresholding, small-object removal, binary closing, hole filling, and connected-component filtering.
   - This is not a learned model. It is only for demonstrating the prediction-to-visualization pipeline.

3. Custom U-Net / nnU-Net path
   - The repo contains a custom U-Net architecture and nnU-Net integration.
   - The existing custom checkpoint is not reliable enough for client claims.
   - A real production model needs proper training data and evaluation.

## Current Honest Status

This POC demonstrates that the data pipeline works:

- 3D CT can be converted to synthetic 2D X-ray-like images.
- 3D vertebra labels can be projected into 2D.
- 2D segmentation overlays can be generated.
- The project can run tests and produce visual outputs.

It does not yet prove a production-grade model on real 2D clinical X-rays.

## Commands Used For Current Outputs

Generate corrected CT-derived DRR and projected labels:

```powershell
python src\generate_drr.py --input data\verse225 --output data\processed\verse225_xray_fixed --projection lateral
```

Generate fallback segmentation for the phantom DRR:

```powershell
python src\predict.py --input data\processed\xray_synthetic\test --output outputs\predictions_working\phantom --fallback --device cpu
```

Visualize the phantom prediction:

```powershell
python src\visualize.py --images data\processed\xray_synthetic\test --masks outputs\predictions_working\phantom --output outputs\visualizations_working\phantom
```

Visualize the real CT-derived projected labels:

```powershell
python src\visualize.py --images data\processed\verse225_xray_fixed\train --masks outputs\predictions_working\verse225_projected_labels --output outputs\visualizations_working\verse225_projected_labels_clean
```

## Next Steps To Convert This Into A Real Working Model

1. Collect real 2D spine X-rays.
   - Prefer DICOM or high-quality PNG/JPEG exports.
   - Include lateral and AP views separately.
   - Keep metadata such as view type, pixel spacing, and body region.

2. Create real 2D ground-truth labels.
   - Have annotators mark vertebra bodies on real X-rays.
   - Decide whether the first production target is binary spine/vertebra segmentation or per-vertebra labels.

3. Train a real 2D segmentation model.
   - Use nnU-Net 2D or a U-Net/DeepLab-style model.
   - Start with binary vertebra segmentation, then move to per-level labels.
   - Use CT-derived DRRs only as pretraining or augmentation, not as the final validation set.

4. Validate on held-out real X-rays.
   - Report Dice score, IoU, precision, recall, and failure cases.
   - Evaluate separately for cervical, thoracic, lumbar, AP, and lateral images.

5. Improve domain adaptation.
   - Real X-rays differ from DRRs in contrast, noise, anatomy overlap, scanner artifacts, and patient positioning.
   - Use augmentation, histogram matching, and potentially synthetic-to-real fine-tuning.

6. Add production input support.
   - Add DICOM ingestion.
   - Normalize orientation and spacing.
   - Detect AP versus lateral view automatically or require it as metadata.

7. Package the model for client use.
   - Provide a CLI or small web/API app.
   - Output masks, overlays, and confidence/error warnings.
   - Store model version, input metadata, and processing logs.

## Suggested Client Wording

This proof of concept demonstrates the core technical workflow for spine X-ray segmentation. We start from labeled 3D CT spine data, generate synthetic 2D X-ray projections using DRR-style projection, project the 3D vertebra labels into 2D masks, and produce visual overlays showing the segmented vertebral regions. The current output validates the pipeline and visualization approach. The next phase is to train and validate a dedicated model on real annotated 2D clinical X-rays so the system can perform true real-world X-ray segmentation.
