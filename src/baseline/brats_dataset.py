import os
import torch.nn.functional as F
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset

# NOTE: hardcoded because this file gets imported by Flower's ClientApp/ServerApp,
# which run from isolated copies (~/.flwr/apps/...), breaking __file__-relative
# path math. Fragile if the project moves machines - acceptable tradeoff given
# the deadline.
DATA_ROOT = r"C:\FedMed\data\brats2020\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData"

MODALITIES = ["t1", "t1ce", "t2", "flair"]

class BraTSDataset(Dataset):
    def __init__(self, patient_ids):
        """
        patient_ids: list of strings like ["BraTS20_Training_001", "BraTS20_Training_002", ...]
        """
        self.patient_ids = patient_ids

    def __len__(self):
        return len(self.patient_ids)

    def _load_and_normalize(self, filepath):
        data = nib.load(filepath).get_fdata().astype(np.float32)
        mean, std = data.mean(), data.std()
        if std > 0:
            data = (data - mean) / std
        return data

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        patient_dir = os.path.join(DATA_ROOT, pid)

        # Load all 4 modalities, stack as channels: shape becomes [4, 240, 240, 155]
        modality_volumes = []
        for mod in MODALITIES:
            filepath = os.path.join(patient_dir, f"{pid}_{mod}.nii")
            modality_volumes.append(self._load_and_normalize(filepath))
        image = np.stack(modality_volumes, axis=0)

        # Load mask, remap label 4 -> 3
        seg_path = os.path.join(patient_dir, f"{pid}_seg.nii")
        mask = nib.load(seg_path).get_fdata().astype(np.int64)
        mask[mask == 4] = 3

                # Pad depth (155) up to 160 -- the nearest multiple of 8 -- so the U-Net's
        # 3 downsampling steps (2^3 = 8) divide evenly, letting skip connections
        # match sizes exactly. Padding with 0 is safe: 0 = background/no-signal
        # for both image intensities (post-normalization, 0 is roughly "average")
        # and mask labels (0 = background class).
        pad_amount = 160 - image.shape[-1]  # 5
        image = np.pad(image, ((0,0), (0,0), (0,0), (0,pad_amount)), mode="constant")
        mask = np.pad(mask, ((0,0), (0,0), (0,pad_amount)), mode="constant")

        # Downsample from 240x240x160 to 96x96x96 -- roughly 16x smaller in
        # memory. Necessary for running 3 concurrent SuperNode processes on
        # one machine without exhausting system RAM. Real deployments with
        # dedicated hardware per node wouldn't need this; documented tradeoff
        # given local multi-process simulation on a single laptop.
        image_t = torch.from_numpy(image).unsqueeze(0)  # add batch dim for interpolate
        image_t = F.interpolate(image_t, size=(96, 96, 96), mode="trilinear", align_corners=False)
        image = image_t.squeeze(0)

        mask_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
        mask_t = F.interpolate(mask_t, size=(96, 96, 96), mode="nearest")
        mask = mask_t.squeeze(0).squeeze(0).long()

        return image, mask


if __name__ == "__main__":
    patient_ids = [f"BraTS20_Training_{i:03d}" for i in range(1, 4)]  # first 3 patients
    dataset = BraTSDataset(patient_ids)

    print(f"Dataset size: {len(dataset)}")

    image, mask = dataset[0]
    print(f"\nImage shape: {image.shape}, dtype: {image.dtype}")
    print(f"Image value range: [{image.min():.2f}, {image.max():.2f}]")
    print(f"Mask shape: {mask.shape}, dtype: {mask.dtype}")
    print(f"Mask unique values: {torch.unique(mask)}")