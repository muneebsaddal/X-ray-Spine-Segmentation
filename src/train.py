import argparse
import logging
import os
import subprocess
from pathlib import Path
from utils import ensure_dir, setup_logging
from trainer import train_model

logger = logging.getLogger(__name__)

# Trainer variant for nnU-Net 2D POC
TRAINER_2D = "nnUNetTrainer"
PLAN_NAME  = "nnUNetPlans"
FAST_EPOCHS = 50

def _set_nnunet_env(raw_dir: Path, preprocessed_dir: Path, results_dir: Path) -> None:
    defaults = {
        "nnUNet_raw": str(raw_dir.resolve()),
        "nnUNet_preprocessed": str(preprocessed_dir.resolve()),
        "nnUNet_results": str(results_dir.resolve()),
    }
    for key, val in defaults.items():
        if key not in os.environ:
            os.environ[key] = val

def _run_cmd(cmd: list[str], description: str) -> None:
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=False, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{description} failed with exit code {result.returncode}.")

def plan_and_preprocess(dataset_id: int) -> None:
    cmd = ["nnUNetv2_plan_and_preprocess", "-d", str(dataset_id), "-c", "2d", "--verify_dataset_integrity"]
    _run_cmd(cmd, "nnUNetv2_plan_and_preprocess")

def train_fold(dataset_id: int, fold: int, fast: bool = False, device: str = "cuda") -> None:
    cmd = ["nnUNetv2_train", str(dataset_id), "2d", str(fold), "-tr", TRAINER_2D, "-p", PLAN_NAME, "--device", device]
    if fast:
        os.environ["nnUNet_n_proc_DA"] = "2"
    _run_cmd(cmd, f"nnUNetv2_train fold {fold}")

def main() -> None:
    args = parse_args()
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)

    if args.custom:
        logger.info("Starting Custom U-Net Training...")
        output_path = Path("outputs/models/custom_unet.pth")
        ensure_dir(output_path.parent)
        train_model(
            train_dir=Path("data/processed/verse225_xray/train"),
            val_dir=Path("data/processed/verse225_xray/train"),
            output_model_path=output_path,
            epochs=args.epochs,
            device=args.device
        )
        return

    ensure_dir(args.raw_dir)
    ensure_dir(args.preprocessed_dir)
    ensure_dir(args.results_dir)
    _set_nnunet_env(args.raw_dir, args.preprocessed_dir, args.results_dir)

    if not args.skip_preprocess:
        plan_and_preprocess(args.dataset_id)
    train_fold(args.dataset_id, fold=args.fold, fast=args.fast, device=args.device)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train 2D spine X-ray segmentation model.")
    parser.add_argument("--custom", action="store_true", help="Use custom U-Net instead of nnU-Net")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for custom training")
    parser.add_argument("--dataset_id", type=int, default=201, help="nnU-Net dataset ID")
    parser.add_argument("--fold", type=int, default=0, help="Cross-validation fold")
    parser.add_argument("--fast", action="store_true", help="Fast mode for nnU-Net")
    parser.add_argument("--skip_preprocess", action="store_true", help="Skip preprocessing")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"], help="Device")
    parser.add_argument("--raw_dir", type=Path, default=Path("data/nnunet_raw"))
    parser.add_argument("--preprocessed_dir", type=Path, default=Path("data/nnunet_preprocessed"))
    parser.add_argument("--results_dir", type=Path, default=Path("data/nnunet_results"))
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    main()
