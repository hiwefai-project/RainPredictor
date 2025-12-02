from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import confusion_matrix
from skimage.metrics import structural_similarity as ssim

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as vops
from pytorch_msssim import SSIM


criterion_sl1 = nn.SmoothL1Loss()
criterion_mae = nn.L1Loss()
criterion_ssim = SSIM(data_range=1.0, size_average=True, channel=1, win_size=5)
criterion_fl = lambda logits, targets: vops.sigmoid_focal_loss(
    logits,
    targets,
    reduction="mean",
)
criterion_mae_lambda = 10.0


def calculate_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    logits: torch.Tensor = None,
    mask: torch.Tensor = None,
    threshold_dbz: float = 15.0,
) -> Dict[str, float]:
    """Compute multiple evaluation metrics for predicted vs target sequences."""
    preds = preds.detach().cpu()
    targets = targets.detach().cpu()

    sl1 = F.smooth_l1_loss(preds, targets)
    mae = F.l1_loss(preds, targets)

    fl = 0.0
    if (logits is not None) and (mask is not None):
        fl = criterion_fl(logits, mask)

    total = sl1 + fl

    p = preds.numpy().squeeze()
    t = targets.numpy().squeeze()
    p = p * 0.5 + 0.5
    t = t * 0.5 + 0.5

    p_dbz = np.clip(p * 70.0, 0, 70)
    t_dbz = np.clip(t * 70.0, 0, 70)

    ssim_values = []
    for b in range(p_dbz.shape[0]):
        for tt in range(p_dbz.shape[1]):
            dr = max(t_dbz[b, tt].max() - t_dbz[b, tt].min(), 1e-6)
            if np.std(t_dbz[b, tt]) < 1e-6 or np.std(p_dbz[b, tt]) < 1e-6:
                ssim_values.append(1.0)
            else:
                ssim_values.append(
                    ssim(
                        p_dbz[b, tt],
                        t_dbz[b, tt],
                        data_range=dr,
                        win_size=5,
                        multichannel=False,
                    )
                )
    ssim_val = float(np.mean(ssim_values)) if ssim_values else 0.0

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
        "FL": float(fl),
        "SSIM": float(ssim_val),
        "CSI": float(csi),
    }


def generate_evaluation_report(
    metrics_dict: Dict[str, float],
    conf_matrix,
    output_dir: str,
) -> None:
    """Generate a detailed evaluation report containing metrics and confusion matrix."""
    import os

    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "evaluation_report.txt")

    with open(report_path, "w") as f:
        f.write("=== RainPredRNN Evaluation Report ===\n\n")
        f.write("Metrics:\n")
        f.write("-" * 40 + "\n")
        for metric, value in metrics_dict.items():
            f.write(f"{metric:15s}: {value:.4f}\n")
        f.write("\n")

        f.write("Confusion Matrix:\n")
        f.write("-" * 40 + "\n")
        f.write("Format: [[TN, FP],\n        [FN, TP]]\n\n")
        f.write(str(conf_matrix))
        f.write("\n\n")

        tn, fp, fn, tp = conf_matrix.ravel()
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-10)

        f.write("Additional Metrics:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Precision:  {precision:.4f}\n")
        f.write(f"Recall:     {recall:.4f}\n")
        f.write(f"F1 Score:   {f1:.4f}\n")
        f.write(f"Accuracy:   {accuracy:.4f}\n")


def evaluate(
    model: torch.nn.Module,
    loader,
    device,
    threshold_dbz: float = 15.0,
    pred_length: int = 6,
) -> Tuple[Dict[str, float], np.ndarray]:
    """Evaluate the model on a given DataLoader and compute global confusion matrix."""
    model.eval()
    agg = {"MAE": 0.0, "SSIM": 0.0, "CSI": 0.0, "SmoothL1": 0.0, "FL": 0.0, "TOTAL": 0.0}
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 4:
                inputs, targets, mask, _ = batch
            else:
                inputs, targets, mask = batch
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            outputs, logits = model(inputs, pred_length)
            m = calculate_metrics(outputs, targets, logits, mask, threshold_dbz)
            for k in agg:
                agg[k] += float(m[k])

            preds = outputs.detach().cpu().numpy()
            targets_np = targets.detach().cpu().numpy()

            preds = preds * 0.5 + 0.5
            targets_np = targets_np * 0.5 + 0.5

            preds_dbz = np.clip(preds * 70.0, 0, 70)
            targets_dbz = np.clip(targets_np * 70.0, 0, 70)

            preds_bin = (preds_dbz > threshold_dbz).astype(np.uint8)
            targets_bin = (targets_dbz > threshold_dbz).astype(np.uint8)
            all_preds.extend(preds_bin.flatten())
            all_targets.extend(targets_bin.flatten())

    for k in agg:
        agg[k] /= float(max(1, len(loader)))

    conf_matrix = confusion_matrix(all_targets, all_preds, labels=[0, 1])

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return agg, conf_matrix
