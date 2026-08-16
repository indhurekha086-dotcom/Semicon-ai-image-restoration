from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from .degradations import randomized_degradation


def load_npy(path):
    arr = np.load(path).astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    if arr.ndim == 2:
        arr = arr[None, ...]
    elif arr.ndim == 3 and arr.shape[-1] in (1, 3):
        arr = np.transpose(arr, (2, 0, 1))
    elif arr.ndim != 3:
        raise ValueError(f"Unsupported shape {arr.shape} in {path}")
    return torch.from_numpy(arr)


class PairedNPYDataset(Dataset):
    def __init__(self, noisy_dir, gt_dir, augment=False):
        self.noisy_dir = Path(noisy_dir)
        self.gt_dir = Path(gt_dir)
        self.augment = augment
        noisy = {p.stem: p for p in self.noisy_dir.glob("*.npy")}
        gt = {p.stem: p for p in self.gt_dir.glob("*.npy")}
        keys = sorted(set(noisy) & set(gt))
        if not keys:
            raise RuntimeError("No matching .npy filenames found between noisy_dir and gt_dir.")
        self.items = [(noisy[k], gt[k]) for k in keys]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        noisy = load_npy(self.items[idx][0])
        gt = load_npy(self.items[idx][1])
        if noisy.shape[0] != gt.shape[0]:
            if noisy.shape[0] == 1:
                noisy = noisy.repeat(gt.shape[0], 1, 1)
            elif gt.shape[0] == 1:
                gt = gt.repeat(noisy.shape[0], 1, 1)
            else:
                raise ValueError("Channel mismatch")
        gt = gt.clamp(0, 1)
        if self.augment:
            if torch.rand(()) < 0.5:
                noisy = randomized_degradation(gt.clone())
            if torch.rand(()) < 0.5:
                noisy = torch.flip(noisy, dims=[-1]); gt = torch.flip(gt, dims=[-1])
            if torch.rand(()) < 0.5:
                noisy = torch.flip(noisy, dims=[-2]); gt = torch.flip(gt, dims=[-2])
        return noisy, gt, self.items[idx][0].stem
