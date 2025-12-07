"""
Fully commented metrics module used for evaluation and training feedback.
Fixes SSIM computation, adds rain-weighted loss, and robust CSI.
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from skimage.metrics import structural_similarity as ssim

# Base SmoothL1 layer
criterion_sl1 = torch.nn.SmoothL1Loss()

# Global multiplier used in TOTAL metric (kept for compatibility)
criterion_mae_lambda = 10.0


# ============================================================
#  RAIN-WEIGHTED SmoothL1 LOSS
# ============================================================
def rain_weighted_sl1(preds, targets, threshold_dbz=15.0, rain_boost=4.0):
    """
    Apply SmoothL1 loss but give more weight to rainy pixels.

    preds, targets: normalized [-1,1] tensors, (B,T,1,H,W)
    """
    # Compute element-wise SmoothL1 loss
    base = F.smooth_l1_loss(preds, targets, reduction="none")

    # Convert targets back to dBZ to identify rain pixels
    targets_01 = targets * 0.5 + 0.5
    targets_dbz = targets_01 * 70.0

    # Rain mask
    rain_mask = (targets_dbz >= threshold_dbz).float()

    # Pixel-level weights: boost rainy areas
    weight = 1.0 + rain_boost * rain_mask

    # Weighted average loss
    weighted = base * weight
    loss = weighted.sum() / weight.sum().clamp_min(1.0)
    return loss


# ============================================================
#  MAIN METRICS: SmoothL1, MAE, TOTAL, SSIM, CSI
# ============================================================
def calculate_metrics(preds, targets, threshold_dbz=15.0):
    """
    Compute all validation metrics.
    preds, targets: (B,T,1,H,W) normalized [-1,1]
    """

    # Base losses in normalized space
    sl1 = F.smooth_l1_loss(preds, targets)
    mae = F.l1_loss(preds, targets)
    total = sl1 * criterion_mae_lambda

    # Convert to dBZ for physical metrics
    p = preds.cpu().numpy()
    t = targets.cpu().numpy()

    p = p * 0.5 + 0.5
    t = t * 0.5 + 0.5
    p_dbz = np.clip(p * 70.0, 0, 70)
    t_dbz = np.clip(t * 70.0, 0, 70)

    B, T, C, H, W = p_dbz.shape

    # ----------------------------
    # SSIM on full images
    # ----------------------------
    ssim_vals = []
    for b in range(B):
        for tt in range(T):
            truth = t_dbz[b, tt, 0]
            pred  = p_dbz[b, tt, 0]

            # avoid division by zero
            dr = max(truth.max() - truth.min(), 1e-6)

            if np.std(truth) < 1e-6 or np.std(pred) < 1e-6:
                ssim_vals.append(1.0)
            else:
                ssim_vals.append(
                    ssim(truth, pred, data_range=dr, win_size=5, channel_axis=None)
                )

    ssim_val = float(np.mean(ssim_vals)) if ssim_vals else 0.0

    # ----------------------------
    # CSI (threshold detection)
    # ----------------------------
    p_bin = (p_dbz > threshold_dbz).astype(np.uint8)
    t_bin = (t_dbz > threshold_dbz).astype(np.uint8)

    cm = confusion_matrix(t_bin.flatten(), p_bin.flatten(), labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0

    csi = tp / (tp + fp + fn + 1e-10)

    return {
        "TOTAL": float(total),
        "SmoothL1": float(sl1),
        "MAE": float(mae),
        "SSIM": float(ssim_val),
        "CSI": float(csi),
    }


# ============================================================
#  EVALUATION LOOP
# ============================================================
def evaluate(model, loader, device, pred_length, threshold_dbz=15.0):
    """
    Evaluate model on validation loader.
    Returns:
        avg_metrics: dict
        cm: confusion matrix (2x2)
    """
    model.eval()

    # accumulators for averaged metrics
    agg = {"TOTAL": 0, "SmoothL1": 0, "MAE": 0, "SSIM": 0, "CSI": 0}
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 4:
                inputs, targets, mask, _ = batch
            else:
                inputs, targets, mask = batch

            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs, _ = model(inputs, pred_length)

            m = calculate_metrics(outputs, targets, threshold_dbz)
            for k in agg:
                agg[k] += m[k]

            # store binary predictions for global CSI
            p = outputs.cpu().numpy()
            t = targets.cpu().numpy()
            p = np.clip((p*0.5+0.5)*70.0, 0, 70)
            t = np.clip((t*0.5+0.5)*70.0, 0, 70)
            all_preds.extend((p > threshold_dbz).astype(np.uint8).flatten())
            all_targets.extend((t > threshold_dbz).astype(np.uint8).flatten())

    # average metrics
    for k in agg:
        agg[k] /= len(loader)

    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1])
    return agg, cm
