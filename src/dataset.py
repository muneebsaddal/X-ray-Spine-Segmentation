import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from utils import nifti_to_image, normalize_xray

class SpineDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.image_paths = sorted(list(self.data_dir.glob("*_0000.nii.gz")))
        self.mask_paths = [Path(str(p).replace("_0000.nii.gz", ".nii.gz")) for p in self.image_paths]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = nifti_to_image(self.image_paths[idx])
        mask = nifti_to_image(self.mask_paths[idx])
        
        # Ensure image is normalized [0, 1]
        image = normalize_xray(image)
        # Binarize mask for this POC
        mask = (mask > 0).astype(np.float32)

        # Add channel dimension (C, H, W)
        image = image[np.newaxis, ...]
        mask = mask[np.newaxis, ...]

        image_tensor = torch.from_numpy(image)
        mask_tensor = torch.from_numpy(mask)

        if self.transform:
            sample = self.transform({"image": image_tensor, "mask": mask_tensor})
            image_tensor, mask_tensor = sample["image"], sample["mask"]

        return image_tensor, mask_tensor
