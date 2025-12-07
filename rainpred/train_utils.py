"""
Training utilities: epoch loop, validation previews.
Fully commented version.
"""

import os
import numpy as np
import torch
from PIL import Image

from .metrics import rain_weighted_sl1, criterion_mae_lambda


# ===============================================================
#  TRAIN ONE EPOCH
# ===============================================================
def train_epoch(model, loader, optimizer, device, scaler=None, pred_length=6):
    """
    Train model for one epoch.

    Returns average loss for reporting.
    """
    model.train()
    epoch_loss = 0.0

    for inputs, targets, _ in loader:
        # Move batch to GPU
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)

        # AMP optional
        if scaler is not None:
            with torch.amp.autocast("cuda"):
                outputs, _ = model(inputs, pred_length)
                # rain-aware loss to prevent background collapse
                loss = rain_weighted_sl1(outputs, targets) * criterion_mae_lambda

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            outputs, _ = model(inputs, pred_length)
            loss = rain_weighted_sl1(outputs, targets) * criterion_mae_lambda
            loss.backward()
            optimizer.step()

        epoch_loss += float(loss.detach().cpu())

    return epoch_loss / len(loader)


# ===============================================================
#  SAVE VALIDATION EXAMPLES
# ===============================================================
def save_val_previews(
    model,
    val_loader,
    device,
    out_root,
    epoch,
    pred_length,
    overwrite=False
):
    """
    Save PNG previews of predictions and targets for qualitative inspection.
    """

    model.eval()

    # Create output folders
    ep_dir = os.path.join(out_root, f"epoch_{epoch:03d}")
    dir_pred = os.path.join(ep_dir, "predictions")
    dir_targ = os.path.join(ep_dir, "targets")
    os.makedirs(dir_pred, exist_ok=True)
    os.makedirs(dir_targ, exist_ok=True)

    # Save only a few samples
    saved = 0
    max_save = 8

    with torch.no_grad():
        for batch in val_loader:
            # Some loaders include file paths as 4th element
            if len(batch) == 4:
                inputs, targets, mask, paths = batch
            else:
                inputs, targets, mask = batch
                paths = [None] * targets.shape[1]

            inputs = inputs.to(device)

            outputs, _ = model(inputs, pred_length)

            preds = outputs.cpu().numpy()
            targs = targets.cpu().numpy()

            # Convert back to [0,1] for visualization
            preds = preds * 0.5 + 0.5
            targs = targs * 0.5 + 0.5

            # Remove batch and channel dims
            preds = preds[0, :, 0]   # (T,H,W)
            targs = targs[0, :, 0]

            T = preds.shape[0]

            for t in range(T):
                # Build filename stem from path or index
                if paths and paths[t] is not None:
                    stem = os.path.splitext(os.path.basename(paths[t]))[0]
                else:
                    stem = f"frame_{t:03d}"

                # Output paths
                pred_path = os.path.join(dir_pred, f"{stem}_pred.png")
                targ_path = os.path.join(dir_targ, f"{stem}_target.png")

                # Avoid overwriting unless requested
                if (not overwrite) and os.path.exists(pred_path):
                    continue

                # Convert to 0–255 grayscale
                p = (preds[t] * 255).clip(0, 255).astype(np.uint8)
                g = (targs[t] * 255).clip(0, 255).astype(np.uint8)

                Image.fromarray(p).save(pred_path)
                Image.fromarray(g).save(targ_path)

                saved += 1
                if saved >= max_save:
                    return
