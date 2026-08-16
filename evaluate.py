import argparse
from pathlib import Path

import numpy as np

from utils.metrics import psnr, ssim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--gt_dir", required=True)
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)

    values_psnr = []
    values_ssim = []

    pred_files = sorted(pred_dir.glob("*.npy"))

    if not pred_files:
        raise RuntimeError("No prediction .npy files found.")

    for pred_path in pred_files:
        gt_path = gt_dir / pred_path.name

        if not gt_path.exists():
            continue

        pred = np.load(pred_path).astype(np.float32)
        gt = np.load(gt_path).astype(np.float32)

        pred = np.clip(pred, 0.0, 1.0)
        gt = np.clip(gt, 0.0, 1.0)

        values_psnr.append(psnr(pred, gt))
        values_ssim.append(ssim(pred, gt))

    if not values_psnr:
        raise RuntimeError("No matching prediction/GT filenames found.")

    print(f"Mean PSNR : {np.mean(values_psnr):.4f}")
    print(f"Mean SSIM : {np.mean(values_ssim):.4f}")


if __name__ == "__main__":
    main()
