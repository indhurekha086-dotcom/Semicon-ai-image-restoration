import argparse
from pathlib import Path

import numpy as np
import torch

from models.msr_unet import MSRUNet


def load_npy(path):
    arr = np.load(path).astype(np.float32)

    if arr.ndim == 2:
        arr = arr[None, ...]

    elif arr.ndim == 3 and arr.shape[-1] in (1, 3):
        arr = np.transpose(arr, (2, 0, 1))

    return torch.from_numpy(arr)


def save_npy(path, tensor):
    arr = tensor.detach().cpu().numpy()

    if arr.shape[0] == 1:
        arr = arr[0]

    elif arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))

    np.save(path, arr.astype(np.float32))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scale", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MSRUNet(
        in_channels=1,
        out_channels=1,
        base=32,
        scale=args.scale
    )

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device
    )

    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.npy"))

    if not files:
        raise RuntimeError("No .npy files found in input directory.")

    with torch.no_grad():
        for path in files:
            x = load_npy(path).unsqueeze(0).to(device)

            y = model(x)

            y = torch.clamp(y, 0.0, 1.0)

            save_npy(
                output_dir / path.name,
                y[0]
            )

            print(f"Restored: {path.name}")


if __name__ == "__main__":
    main()
