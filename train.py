import os
import argparse
import datetime
import json  # Serialize AI-friendly diagnostics to JSON.
import logging  # Use the logging module for structured status output.
import math  # Check metric values for finite diagnostics.

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from rainpred.config import (
    DEVICE,
    NUM_WORKERS,
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    PRED_LENGTH,
    USE_AMP,
    DATA_PATH as DEFAULT_DATA_PATH,
    VAL_PREVIEW_ROOT as DEFAULT_VAL_PREVIEW_ROOT,
    RUNS_DIR as DEFAULT_RUNS_DIR,
    CHECKPOINT_DIR as DEFAULT_CHECKPOINT_DIR,
)
from rainpred.utils import hms, benchmark_train, benchmark_val, set_seed
from rainpred.data import create_dataloaders
from rainpred.model import RainPredModel
from rainpred.metrics import evaluate
from rainpred.train_utils import train_epoch, save_val_previews


# Create a module-level logger to replace print statements.
logger = logging.getLogger(__name__)


def _to_serializable_confusion_matrix(conf_matrix: object) -> list:
    """Convert a confusion matrix into a JSON-serializable nested list."""
    # Use NumPy-style tolist when available to preserve structure.
    if hasattr(conf_matrix, "tolist"):
        # Convert the array-like confusion matrix into nested Python lists.
        return conf_matrix.tolist()
    # Fall back to a list cast for already-iterable structures.
    return list(conf_matrix)


def _build_training_ai_metrics(
    *,
    epoch: int,
    num_epochs: int,
    train_loss: float,
    val_metrics: dict,
    conf_matrix: object,
    csi_threshold: float,
    best_val_total: float,
    t_train: float,
    t_val: float,
) -> dict:
    """Build an AI-friendly diagnostics payload with metrics and suggestions."""
    # Start a suggestions list that downstream tools can consume.
    suggestions: list[str] = []
    # Fetch the composite validation metric for heuristics.
    val_total = float(val_metrics.get("TOTAL", float("nan")))
    # Fetch the CSI metric for rain/no-rain detection guidance.
    val_csi = float(val_metrics.get("CSI", float("nan")))
    # Flag non-finite values to help diagnose numerical instability.
    if not math.isfinite(val_total) or not math.isfinite(train_loss):
        # Suggest checking data normalization or learning rate when losses explode.
        suggestions.append("Non-finite loss detected; verify data normalization and learning rate.")
    # Flag low CSI values with an actionable data/threshold suggestion.
    if math.isfinite(val_csi) and val_csi < 0.2:
        # Suggest focusing on rain detection if CSI remains low.
        suggestions.append("Low CSI: consider more rain events, threshold tuning, or class-balanced sampling.")
    # Detect a large train/val gap as a possible overfitting signal.
    if math.isfinite(val_total) and train_loss * 1.2 < val_total:
        # Suggest regularization or augmentation when validation degrades.
        suggestions.append("Validation TOTAL higher than train loss; consider regularization or more data.")

    # Assemble the structured diagnostics payload for JSON output.
    payload = {
        # Track schema versioning to support future upgrades.
        "schema_version": "v1",
        # Identify the lifecycle stage for downstream filtering.
        "stage": "train",
        # Record the current epoch number (1-based for readability).
        "epoch": epoch + 1,
        # Record the configured number of epochs for context.
        "num_epochs": num_epochs,
        # Capture the training loss for this epoch.
        "train_loss": float(train_loss),
        # Capture validation metrics as a nested dictionary.
        "val_metrics": {key: float(value) for key, value in val_metrics.items()},
        # Capture the CSI threshold used during evaluation.
        "csi_threshold_dbz": float(csi_threshold),
        # Capture the best validation metric observed so far.
        "best_val_total": float(best_val_total),
        # Capture training/validation durations for performance diagnostics.
        "timings_sec": {"train": float(t_train), "val": float(t_val)},
        # Serialize the confusion matrix for downstream analytics.
        "confusion_matrix": _to_serializable_confusion_matrix(conf_matrix),
        # Add human-readable suggestions for next-step actions.
        "suggestions": suggestions,
        # Add an ISO timestamp to align logs with external systems.
        "timestamp": datetime.datetime.now().isoformat(),
    }
    # Return the payload to the caller for writing.
    return payload


def _write_ai_metrics_json(path: str, records: list[dict]) -> None:
    """Write AI-friendly diagnostics to JSON or log them when path is '-'."""
    # Serialize the records to a formatted JSON string for readability.
    json_payload = json.dumps(records, indent=2)
    # Emit to logs when the sentinel "-" path is used.
    if path == "-":
        # Log the JSON payload to stdout via logging.
        logger.info("[train] AI metrics JSON:\n%s", json_payload)
        # Return early since we are not writing to disk.
        return
    # Open the output file in write mode to keep the latest diagnostics.
    with open(path, "w", encoding="utf-8") as handle:
        # Persist the JSON payload to disk for downstream tooling.
        handle.write(json_payload)


def main() -> None:
    """Train RainPredRNN on GeoTIFF radar data."""
    # Configure logging for CLI runs to surface status messages.
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # ---------------- CLI ----------------
    parser = argparse.ArgumentParser(description="Train RainPredRNN nowcasting model.")
    parser.add_argument(
        "--data-path",
        type=str,
        default=DEFAULT_DATA_PATH,
        help="Root directory of prepared dataset (with train/ and val/).",
    )
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Number of epochs.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Initial learning rate.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=NUM_WORKERS,
        help="Number of DataLoader workers.",
    )
    parser.add_argument(
        "--pred-length",
        type=int,
        default=PRED_LENGTH,
        help="Prediction horizon (number of future frames).",
    )
    parser.add_argument(
        "--small-debug",
        action="store_true",
        help="Use a very small subset of the dataset for quick tests.",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to checkpoint (.pth) to resume from.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Save the 'last' checkpoint every N epochs (default: 1).",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional name for TensorBoard run / checkpoints subdir.",
    )
    parser.add_argument(
        "--val-preview-root",
        type=str,
        default=DEFAULT_VAL_PREVIEW_ROOT,
        help="Directory where PNG previews of val predictions are written.",
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default=DEFAULT_RUNS_DIR,
        help="Root directory for TensorBoard runs.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=DEFAULT_CHECKPOINT_DIR,
        help="Directory where checkpoints are stored.",
    )
    parser.add_argument(
        "--csi-threshold",
        type=float,
        default=15.0,
        help="Reflectivity threshold (dBZ) used for CSI during validation.",
    )
    parser.add_argument(
        "--metrics-json",
        type=str,
        default=None,
        help=(
            "Optional path to write AI-friendly JSON diagnostics after each epoch "
            "(use '-' to emit to stdout)."
        ),
    )
    args = parser.parse_args()

    # ---------------- setup ----------------
    set_seed(15)

    data_path = os.path.abspath(args.data_path)
    num_epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    num_workers = args.num_workers
    pred_length = args.pred_length
    small_debug = args.small_debug
    resume_from = args.resume_from
    save_every = max(1, args.save_every)
    csi_threshold = args.csi_threshold
    # Capture the optional JSON diagnostics output path.
    metrics_json_path = args.metrics_json
    # Initialize a list to accumulate epoch-by-epoch diagnostics.
    metrics_json_records: list[dict] = []

    # Run name & directories
    if args.run_name is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"rainpred_{timestamp}"
    else:
        run_name = args.run_name

    runs_dir = os.path.join(args.runs_dir, run_name)
    checkpoint_dir = os.path.join(args.checkpoint_dir, run_name)
    val_preview_root = os.path.join(args.val_preview_root, run_name)

    os.makedirs(runs_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(val_preview_root, exist_ok=True)

    # Log the device choice to document runtime hardware.
    logger.info("[train] Device: %s", DEVICE)
    # Log the data directory for traceability.
    logger.info("[train] Data path: %s", data_path)
    # Log checkpoint directory to show where models are saved.
    logger.info("[train] Checkpoints: %s", checkpoint_dir)
    # Log TensorBoard run directory for experiment tracking.
    logger.info("[train] TensorBoard runs: %s", runs_dir)
    # Log validation preview directory for output inspection.
    logger.info("[train] Val previews: %s", val_preview_root)

    # ---------------- data ----------------
    train_loader, val_loader = create_dataloaders(
        data_path=data_path,
        batch_size=batch_size,
        num_workers=num_workers,
        pred_length=pred_length,
        small_debug=small_debug,
    )

    # Log training steps per epoch to communicate dataset size.
    logger.info("[train] Train steps/epoch: %s", len(train_loader))
    # Log validation steps per epoch for context.
    logger.info("[train]  Val  steps/epoch: %s", len(val_loader))

    # ---------------- model ----------------
    model = RainPredModel(
        in_channels=1,
        out_channels=1,
        hidden_channels=64,
        transformer_d_model=128,
        transformer_nhead=8,
        transformer_num_layers=2,
        pred_length=pred_length,
    )

    # Multi-GPU (DataParallel) if available
    if DEVICE == "cuda":
        n_gpus = torch.cuda.device_count()
        if n_gpus > 1:
            # Log multi-GPU usage to highlight parallel training.
            logger.info("[train] Using DataParallel on %s GPUs", n_gpus)
            model = torch.nn.DataParallel(model)
    model = model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    scaler = torch.amp.GradScaler('cuda') if (USE_AMP and DEVICE == "cuda") else None

    best_val_total = float("inf")
    start_epoch = 0

    # ---------------- optional resume ----------------
    if resume_from is not None and os.path.isfile(resume_from):
        # Log checkpoint resume path for reproducibility.
        logger.info("[train] Resuming from checkpoint: %s", resume_from)
        ckpt = torch.load(resume_from, map_location=DEVICE)

        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
            optimizer.load_state_dict(ckpt.get("optimizer_state_dict", optimizer.state_dict()))
            best_val_total = ckpt.get("best_val", best_val_total)
            start_epoch = ckpt.get("epoch", 0)
        else:
            state_dict = ckpt

        # strip "module." if saved from DataParallel
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k[len("module.") :]: v for k, v in state_dict.items()}

        model_to_load = model.module if isinstance(model, torch.nn.DataParallel) else model
        model_to_load.load_state_dict(state_dict, strict=False)

        # Log the resumed epoch and best validation score.
        logger.info(
            "[train] Resumed at epoch=%s, best_val_total=%.4f",
            start_epoch,
            best_val_total,
        )

    # ---------------- TensorBoard ----------------
    train_writer = SummaryWriter(os.path.join(runs_dir, "train"))
    val_writer = SummaryWriter(os.path.join(runs_dir, "val"))

    # ---------------- main loop ----------------
    for epoch in range(start_epoch, num_epochs):
        # Log the current epoch for progress tracking.
        logger.info("Epoch %s/%s", epoch + 1, num_epochs)

        # --- train ---
        t0 = datetime.datetime.now()
        train_loss = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=DEVICE,
            scaler=scaler,
            pred_length=pred_length,
        )
        t_train = (datetime.datetime.now() - t0).total_seconds()
        # Log training loss and elapsed time for quick monitoring.
        logger.info("\tTrain Loss: %.4f  (%s)", train_loss, hms(t_train))

        # --- validate ---
        v0 = datetime.datetime.now()
        val_metrics, conf_matrix = evaluate(
            model=model,
            loader=val_loader,
            device=DEVICE,
            pred_length=pred_length,
            threshold_dbz=csi_threshold,
        )
        t_val = (datetime.datetime.now() - v0).total_seconds()

        scheduler.step(val_metrics["TOTAL"])

        metrics_str = ", ".join(f"{k}: {float(v):.4f}" for k, v in val_metrics.items())
        # Log validation metrics and elapsed time for evaluation tracking.
        logger.info("\tVal %s  (%s)", metrics_str, hms(t_val))

        # Build AI-friendly diagnostics when JSON output is requested.
        if metrics_json_path:
            # Create a structured payload with metrics and suggestions.
            ai_metrics = _build_training_ai_metrics(
                epoch=epoch,
                num_epochs=num_epochs,
                train_loss=train_loss,
                val_metrics=val_metrics,
                conf_matrix=conf_matrix,
                csi_threshold=csi_threshold,
                best_val_total=best_val_total,
                t_train=t_train,
                t_val=t_val,
            )
            # Append the payload for cumulative JSON output.
            metrics_json_records.append(ai_metrics)
            # Write the updated diagnostics to disk or stdout.
            _write_ai_metrics_json(metrics_json_path, metrics_json_records)

        # log scalars
        train_writer.add_scalar("Loss", train_loss, epoch)
        for k, v in val_metrics.items():
            tag = "Loss" if k.lower() == "total" else k
            val_writer.add_scalar(tag, float(v), epoch)

        # preview PNGs
        save_val_previews(
            model=model,
            val_loader=val_loader,
            device=DEVICE,
            out_root=val_preview_root,
            epoch=epoch,
            pred_length=pred_length,
            overwrite=False,
        )

        # --- best model ---
        if val_metrics["TOTAL"] < best_val_total:
            best_val_total = val_metrics["TOTAL"]
            # Log when a new best model is found to explain checkpoint updates.
            logger.info(
                "\t[train] New best TOTAL=%.4f -> saving best_model.pth",
                best_val_total,
            )
            model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model_to_save.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": val_metrics,
                    "confusion_matrix": conf_matrix,
                    "best_val": best_val_total,
                },
                os.path.join(checkpoint_dir, "best_model.pth"),
            )

        # --- periodic "last" checkpoint ---
        if ((epoch + 1) % save_every == 0) or (epoch + 1 == num_epochs):
            last_path = os.path.join(checkpoint_dir, f"last_epoch_{epoch+1:03d}.pth")
            # Log periodic checkpoint saves for visibility.
            logger.info("\t[train] Saving last checkpoint to %s", last_path)
            model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model_to_save.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": val_metrics,
                    "confusion_matrix": conf_matrix,
                    "best_val": best_val_total,
                },
                last_path,
            )

    train_writer.close()
    val_writer.close()
    # Log completion to signal end of training.
    logger.info("[train] Training completed.")


if __name__ == "__main__":
    main()
