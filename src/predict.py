import argparse
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional
import numpy as np
import torch
from utils import ensure_dir, image_to_nifti, normalize_xray, nifti_to_image, setup_logging
from model import UNet

logger = logging.getLogger(__name__)

TRAINER_2D = "nnUNetTrainer"
PLAN_NAME  = "nnUNetPlans"

def convert_images_to_nifti(input_dir: Path, output_dir: Path) -> List[Path]:
    from PIL import Image
    ensure_dir(output_dir)
    created = []
    image_files = sorted(list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.jpeg")))
    if not image_files:
        raise FileNotFoundError(f"No PNG/JPEG images found in: {input_dir}")
    for img_path in image_files:
        try:
            img = Image.open(str(img_path))
            if img.mode in ("RGB", "RGBA"):
                img = img.convert("L")
            elif img.mode == "I;16":
                img = img.point(lambda x: x / 256).convert("L")
            arr = np.array(img, dtype=np.float32)
            arr = normalize_xray(arr)
            case_id = img_path.stem
            nii_path = output_dir / f"{case_id}_0000.nii.gz"
            image_to_nifti(arr, nii_path)
            created.append(nii_path)
        except Exception as exc:
            logger.error("Failed to convert %s: %s", img_path, exc)
    return created

def run_nnunet_predict(input_dir: Path, output_dir: Path, dataset_id: int, fold: int, results_dir: Path, raw_dir: Path, preprocessed_dir: Path) -> None:
    os.environ["nnUNet_raw"] = str(raw_dir.resolve())
    os.environ["nnUNet_preprocessed"] = str(preprocessed_dir.resolve())
    os.environ["nnUNet_results"] = str(results_dir.resolve())
    ensure_dir(output_dir)
    cmd = ["nnUNetv2_predict", "-i", str(input_dir.resolve()), "-o", str(output_dir.resolve()), "-d", str(dataset_id), "-c", "2d", "-tr", TRAINER_2D, "-p", PLAN_NAME, "-f", str(fold)]
    logger.info("Running nnU-Net prediction...")
    subprocess.run(cmd, check=True)

def run_custom_predict(input_dir: Path, output_dir: Path, model_path: Path, device: str) -> None:
    ensure_dir(output_dir)
    model = UNet(n_channels=1, n_classes=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    input_files = sorted(list(input_dir.glob("*_0000.nii.gz")))
    if not input_files:
        input_files = sorted(list(input_dir.glob("*.nii.gz")))
    with torch.no_grad():
        for img_path in input_files:
            if "_seg" in img_path.name: continue
            image = nifti_to_image(img_path)
            image = normalize_xray(image)
            image_tensor = torch.from_numpy(image[np.newaxis, np.newaxis, ...]).to(device)
            output = model(image_tensor)
            mask = (torch.sigmoid(output) > 0.5).cpu().numpy().squeeze()
            case_id = img_path.name.replace("_0000.nii.gz", "").replace(".nii.gz", "")
            out_path = output_dir / f"{case_id}.nii.gz"
            image_to_nifti(mask.astype(np.float32), out_path)
            logger.info("Custom Pred: %s -> %s", img_path.name, out_path.name)

def main() -> None:
    args = parse_args()
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    input_dir = args.input
    if args.model_path and args.model_path.exists():
        run_custom_predict(input_dir, args.output, args.model_path, args.device)
    else:
        run_nnunet_predict(input_dir, args.output, args.dataset_id, args.fold, args.results_dir, args.raw_dir, args.preprocessed_dir)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on 2D X-ray images.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/predictions"))
    parser.add_argument("--model_path", type=Path)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])
    parser.add_argument("--dataset_id", type=int, default=201)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--png_input", action="store_true")
    parser.add_argument("--raw_dir", type=Path, default=Path("data/nnunet_raw"))
    parser.add_argument("--preprocessed_dir", type=Path, default=Path("data/nnunet_preprocessed"))
    parser.add_argument("--results_dir", type=Path, default=Path("data/nnunet_results"))
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    main()
