import os
from typing import Tuple

import numpy as np
from PIL import Image

import torch

from .metrics import (
    criterion_sl1,
    criterion_fl,
    criterion_mae_lambda,
)


def train_epoch(
    model: torch.nn.Module,
    loader,
    optimizer,
    device,
    scaler=None,
    pred_length: int = 6,
) -> float:
    """Train the model for one epoch and return average loss."""
    model.train()
    total_loss = 0.0

    for inputs, targets, mask in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs, logits = model(inputs, pred_length)
                loss = criterion_sl1(outputs, targets) * criterion_mae_lambda
                loss += criterion_fl(logits, mask) * criterion_mae_lambda
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs, logits = model(inputs, pred_length)
            loss = criterion_sl1(outputs, targets) * criterion_mae_lambda
            loss += criterion_fl(logits, mask) * criterion_mae_lambda
            loss.backward()
            optimizer.step()

        total_loss += float(loss.detach().cpu())

    return total_loss / max(1, len(loader))


def save_all_val_predictions(
    model: torch.nn.Module,
    val_loader,
    device,
    out_root: str,
    epoch: int,
    overwrite: bool = False,
    pred_length: int = 6,
) -> None:
    """Run inference on the entire validation set and save predictions as TIFF files."""
    model.eval()
    ep_dir = os.path.join(out_root, f"epoch_{epoch:03d}")
    out_pred = os.path.join(ep_dir, "predictions")
    os.makedirs(out_pred, exist_ok=True)

    saved = set()

    with torch.no_grad():
        for batch in val_loader:
            if len(batch) != 4:
                raise RuntimeError("Validation loader must return (inputs, targets, mask, paths).")

            inputs, targets, mask, paths = batch

            if not paths or len(paths) == 0:
                raise RuntimeError("Missing target_paths in validation.")

            inputs = inputs.to(device, non_blocking=True)
            outputs, _ = model(inputs, pred_length)

            preds = outputs.detach().cpu().numpy()
            preds = preds * 0.5 + 0.5
            if preds.ndim == 5:
                preds = preds[0, :, 0]

            T = preds.shape[0]
            if len(paths) < T:
                raise RuntimeError(
                    f"target_paths ({len(paths)}) fewer than predicted frames ({T})."
                )
            if len(paths) > T:
                paths = list(paths)[:T]

            for t in range(T):
                stem = os.path.splitext(os.path.basename(paths[t]))[0]
                key = stem.lower()
                if key in saved and not overwrite:
                    continue
                saved.add(key)

                out_path = os.path.join(out_pred, f"{stem}_pred.tiff")
                if (not overwrite) and os.path.exists(out_path):
                    continue

                frame = (preds[t] * 255.0).clip(0, 255).astype(np.uint8)
                Image.fromarray(frame).save(out_path)


def save_all_val_targets(
    val_loader,
    out_root: str,
    epoch: int,
    overwrite: bool = False,
) -> None:
    """Save all target frames from validation set as TIFF files."""
    ep_dir = os.path.join(out_root, f"epoch_{epoch:03d}")
    out_targ = os.path.join(ep_dir, "targets")
    os.makedirs(out_targ, exist_ok=True)

    saved = set()

    for batch in val_loader:
        if len(batch) != 4:
            raise RuntimeError("Validation loader must return (inputs, targets, mask, paths).")

        _, targets, _, paths = batch

        if not paths or len(paths) == 0:
            raise RuntimeError("Missing target_paths in validation.")

        targs = targets.detach().cpu().numpy()
        targs = targs * 0.5 + 0.5
        if targs.ndim == 5:
            targs = targs[0, :, 0]

        T = targs.shape[0]
        if len(paths) < T:
            raise RuntimeError(
                f"target_paths ({len(paths)}) fewer than target frames ({T})."
            )
        if len(paths) > T:
            paths = list(paths)[:T]

        for t in range(T):
            stem = os.path.splitext(os.path.basename(paths[t]))[0]
            key = stem.lower()
            if key in saved and not overwrite:
                continue
            saved.add(key)

            out_path = os.path.join(out_targ, f"{stem}_target.tiff")
            if (not overwrite) and os.path.exists(out_path):
                continue

            frame = (targs[t] * 255.0).clip(0, 255).astype(np.uint8)
            Image.fromarray(frame).save(out_path)
