import os
import argparse
import datetime

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


def main() -> None:
    """Train RainPredRNN on GeoTIFF radar data."""

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

    print(f"[train] Device: {DEVICE}")
    print(f"[train] Data path: {data_path}")
    print(f"[train] Checkpoints: {checkpoint_dir}")
    print(f"[train] TensorBoard runs: {runs_dir}")
    print(f"[train] Val previews: {val_preview_root}")

    # ---------------- data ----------------
    train_loader, val_loader = create_dataloaders(
        data_path=data_path,
        batch_size=batch_size,
        num_workers=num_workers,
        pred_length=pred_length,
        small_debug=small_debug,
    )

    print(f"[train] Train steps/epoch: {len(train_loader)}")
    print(f"[train]  Val  steps/epoch: {len(val_loader)}")

    # ---------------- model ----------------
    model = RainPredRNN(
        input_dim=1,
        num_hidden=256,
        max_hidden_channels=128,
        pred_length=pred_length,
    )

    # Multi-GPU (DataParallel) if available
    if DEVICE == "cuda":
        n_gpus = torch.cuda.device_count()
        if n_gpus > 1:
            print(f"[train] Using DataParallel on {n_gpus} GPUs")
            model = torch.nn.DataParallel(model)
    model = model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, verbose=True)
    scaler = torch.cuda.amp.GradScaler() if (USE_AMP and DEVICE == "cuda") else None

    best_val_total = float("inf")
    start_epoch = 0

    # ---------------- optional resume ----------------
    if resume_from is not None and os.path.isfile(resume_from):
        print(f"[train] Resuming from checkpoint: {resume_from}")
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

        print(f"[train] Resumed at epoch={start_epoch}, best_val_total={best_val_total:.4f}")

    # ---------------- TensorBoard ----------------
    train_writer = SummaryWriter(os.path.join(runs_dir, "train"))
    val_writer = SummaryWriter(os.path.join(runs_dir, "val"))

    # ---------------- main loop ----------------
    for epoch in range(start_epoch, num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

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
        print(f"\tTrain Loss: {train_loss:.4f}  ({hms(t_train)})")

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
        print(f"\tVal {metrics_str}  ({hms(t_val)})")

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
            print(f"\t[train] New best TOTAL={best_val_total:.4f} -> saving best_model.pth")
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
            print(f"\t[train] Saving last checkpoint to {last_path}")
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
    print("[train] Training completed.")


if __name__ == "__main__":
    main()
