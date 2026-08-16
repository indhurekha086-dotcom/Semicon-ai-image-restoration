import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from data.dataset import PairedNPYDataset
from models.msr_unet import MSRUNet


def charbonnier_loss(pred, target, eps=1e-3):
    return torch.mean(torch.sqrt((pred - target) ** 2 + eps ** 2))


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--noisy_dir", required=True)
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    args = parser.parse_args()

    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    dataset = PairedNPYDataset(
        args.noisy_dir,
        args.gt_dir,
        augment=True
    )

    val_size = max(1, int(len(dataset) * args.val_ratio))
    train_size = len(dataset) - val_size

    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    model = MSRUNet(
        in_channels=1,
        out_channels=1,
        base=32,
        scale=args.scale
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr
    )

    best_loss = float("inf")

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for noisy, gt, _ in train_loader:
            noisy = noisy.to(device)
            gt = gt.to(device)

            optimizer.zero_grad()

            pred = model(noisy, output_size=gt.shape[-2:])

            loss = charbonnier_loss(pred, gt)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / max(1, len(train_loader))

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for noisy, gt, _ in val_loader:
                noisy = noisy.to(device)
                gt = gt.to(device)

                pred = model(noisy, output_size=gt.shape[-2:])

                loss = charbonnier_loss(pred, gt)
                val_loss += loss.item()

        val_loss /= max(1, len(val_loader))

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f}"
        )

        if val_loss < best_loss:
            best_loss = val_loss

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                checkpoint_dir / "best_model.pth"
            )

            print("Saved best model.")


if __name__ == "__main__":
    main()
