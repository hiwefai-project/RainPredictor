
"""
train.py

Minimal training script wired to the refactored RainPredModel.

This version:
  * Fixes the deprecated GradScaler API:
        torch.cuda.amp.GradScaler()  -> torch.amp.GradScaler("cuda")
  * Avoids using the deprecated lr_scheduler "verbose" argument.
  * Keeps the public CLI interface similar to the original script:
        --epochs, --batch-size, --n (pred_length), etc.

You can adapt the dataset / dataloader code to your project setup.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from rainpred.model import RainPredModel

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dummy dataset placeholder
# ---------------------------------------------------------------------------

class DummyRadarDataset(Dataset):
    """
    Placeholder dataset; replace with your actual dataset implementation.

    Produces random tensors with shapes:
        inputs : (in_length, 1, H, W)
        targets: (pred_length, 1, H, W)
    """

    def __init__(self, length: int, in_length: int, pred_length: int, height: int, width: int):
        super().__init__()
        self.length = length
        self.in_length = in_length
        self.pred_length = pred_length
        self.height = height
        self.width = width

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.randn(self.in_length, 1, self.height, self.width, dtype=torch.float32)
        y = torch.randn(self.pred_length, 1, self.height, self.width, dtype=torch.float32)
        return x, y


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "train.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, mode="a"),
        ],
    )
    log.info("Logging initialized. Log file: %s", log_path)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
    criterion: nn.Module,
    pred_length: int,
) -> float:
    model.train()
    running_loss = 0.0
    num_samples = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)   # (B, in_length, 1, H, W)
        targets = targets.to(device) # (B, pred_length, 1, H, W)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            # AMP path
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                outputs, _ = model(inputs, pred_length=pred_length)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # FP32 path
            outputs, _ = model(inputs, pred_length=pred_length)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        num_samples += batch_size

    return running_loss / max(num_samples, 1)


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    pred_length: int,
) -> float:
    model.eval()
    running_loss = 0.0
    num_samples = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs, _ = model(inputs, pred_length=pred_length)
        loss = criterion(outputs, targets)

        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        num_samples += batch_size

    return running_loss / max(num_samples, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RainPredModel")

    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--in-length", type=int, default=6, help="Number of input frames")
    parser.add_argument("--n", type=int, default=6, help="Number of frames to predict (pred_length)")
    parser.add_argument("--height", type=int, default=704, help="Frame height")
    parser.add_argument("--width", type=int, default=608, help="Frame width")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--data-length", type=int, default=100, help="Number of samples in dummy dataset")
    parser.add_argument("--amp", action="store_true", help="Enable mixed-precision training (AMP)")
    parser.add_argument("--outdir", type=str, default="runs/default", help="Output directory")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    outdir = Path(args.outdir)
    setup_logging(outdir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Using device: %s", device)

    # Build model
    model = RainPredModel(
        in_channels=1,
        out_channels=1,
        hidden_channels=64,
        transformer_d_model=128,
        transformer_nhead=8,
        transformer_num_layers=2,
        pred_length=args.n,
    )

    if torch.cuda.device_count() > 1:
        log.info("Using DataParallel over %d GPUs", torch.cuda.device_count())
        model = nn.DataParallel(model)

    model = model.to(device)

    # Optimizer and LR scheduler (no deprecated verbose parameter)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Loss function
    criterion = nn.SmoothL1Loss()

    # AMP GradScaler: use the new API torch.amp.GradScaler("cuda")
    scaler = None
    if args.amp and device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda")
        log.info("AMP enabled with torch.amp.GradScaler('cuda')")
    else:
        log.info("AMP disabled")

    # Data
    train_dataset = DummyRadarDataset(
        length=args.data_length,
        in_length=args.in_length,
        pred_length=args.n,
        height=args.height,
        width=args.width,
    )
    val_dataset = DummyRadarDataset(
        length=max(args.data_length // 5, 1),
        in_length=args.in_length,
        pred_length=args.n,
        height=args.height,
        width=args.width,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        log.info("Epoch %d/%d", epoch, args.epochs)

        train_loss = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            criterion=criterion,
            pred_length=args.n,
        )
        val_loss = validate_epoch(
            model=model,
            loader=val_loader,
            device=device,
            criterion=criterion,
            pred_length=args.n,
        )

        scheduler.step()

        log.info("Epoch %d - Train loss: %.6f - Val loss: %.6f", epoch, train_loss, val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = outdir / "best_model.pt"
            log.info("New best val loss, saving checkpoint to %s", ckpt_path)
            outdir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "config": vars(args),
                },
                ckpt_path,
            )


if __name__ == "__main__":
    main()
