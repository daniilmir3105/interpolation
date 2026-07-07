"""Train SegResNet"""

import argparse
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from monai.networks.nets import SegResNet
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.datasets import HDF5Dataset


def parse_args() -> argparse.Namespace:
    data_dir = Path(__file__).resolve().parent / "data"
    checkpoints_dir = Path(__file__).resolve().parent / "checkpoints"
    parser = argparse.ArgumentParser(
        description="Train SegResNet on train/val HDF5 datasets."
    )
    parser.add_argument(
        "--train-h5",
        type=Path,
        default=data_dir / "train-4-128.hdf5",
        help="Path to the train HDF5 dataset.",
    )
    parser.add_argument(
        "--val-h5",
        type=Path,
        default=data_dir / "val-4-128.hdf5",
        help="Path to the validation HDF5 dataset.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=checkpoints_dir / "SegResNet_best.pth",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--sch-patience", type=int, default=10)
    parser.add_argument("--sch-factor", type=float, default=0.5)
    parser.add_argument("--sch-min-lr", type=float, default=1e-6)
    parser.add_argument("--init-filters", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--plot-losses", action="store_true")
    parser.add_argument("--plot-samples", type=int, default=5)
    return parser.parse_args()


def build_model(init_filters: int, device: torch.device) -> SegResNet:
    return SegResNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        init_filters=init_filters,
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1),
    ).to(device)


def plot_losses(
    train_losses: List[float],
    val_losses: List[float],
    sch_lrs: List[float],
    output_path: Path,
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=100)
    ax1.plot(train_losses[5:], label="Train Loss")
    ax1.plot(val_losses[5:], label="Val Loss")
    ax1.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("MSE")
    ax1.legend()
    ax1.grid(True, alpha=0.5)

    ax2.plot(sch_lrs, linewidth=2, color="red")
    ax2.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("LR")
    ax2.set_title("Learning rate scheduling")
    ax2.grid(True, alpha=0.5)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    if not args.train_h5.is_file():
        raise FileNotFoundError(f"Train HDF5 not found: {args.train_h5}")
    if not args.val_h5.is_file():
        raise FileNotFoundError(f"Val HDF5 not found: {args.val_h5}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    trainset = HDF5Dataset(str(args.train_h5))
    valset = HDF5Dataset(str(args.val_h5))

    trainloader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        num_workers=args.num_workers,
    )
    valloader = DataLoader(
        valset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=args.num_workers,
    )

    model = build_model(args.init_filters, device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=args.sch_patience,
        factor=args.sch_factor,
        min_lr=args.sch_min_lr,
    )

    best_val_loss = float("inf")
    train_losses: List[float] = []
    val_losses: List[float] = []
    sch_lrs: List[float] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        pbar = tqdm(trainloader, desc=f"Epoch {epoch}/{args.epochs} [Train]")
        for batch in pbar:
            inp, tgt = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            pred = model(inp)
            loss = criterion(pred, tgt)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        train_loss /= len(trainloader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in valloader:
                inp, tgt = batch[0].to(device), batch[1].to(device)
                pred = model(inp)
                val_loss += criterion(pred, tgt).item()
        val_loss /= len(valloader)
        val_losses.append(val_loss)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        sch_lrs.append(current_lr)
        print(
            f"Epoch {epoch} | Train MSE: {train_loss:.2e} | "
            f"Val MSE: {val_loss:.2e} | LR: {current_lr:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.checkpoint)
            print(f"Saved best checkpoint to {args.checkpoint}")

    if args.plot_losses:
        plot_losses(
            train_losses,
            val_losses,
            sch_lrs,
            Path(__file__).resolve().parent / "training_curves.png",
        )

    if args.plot_samples > 0:
        import h5py

        from utils.plotting import plot_sample

        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        model.eval()
        with h5py.File(args.val_h5, "r") as h5f:
            val_len = len(h5f["label"])
            for idx in np.random.randint(0, val_len, args.plot_samples):
                inp = torch.tensor(
                    h5f["image"][idx][np.newaxis, :, :, :],
                    dtype=torch.float32,
                    device=device,
                )
                with torch.no_grad():
                    predict = model(inp).detach().cpu().squeeze().numpy()
                plot_sample(
                    h5f["image"][idx][0],
                    predict,
                    h5f["label"][idx][0],
                    show=True,
                )


if __name__ == "__main__":
    main()
