import os
import datetime
import argparse

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from training.config import (
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
from training.utils import hms, benchmark_train, benchmark_val
from training.data import create_dataloaders
from training.model import RainPredRNN
from training.metrics import (
    criterion_sl1,
    criterion_fl,
    criterion_mae_lambda,
    evaluate,
    generate_evaluation_report,
)
from training.train_utils import (
    train_epoch,
    save_all_val_predictions,
    save_all_val_targets,
)


def main():
    """Main training script entry point with CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Train RainPredRNN radar nowcasting model."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=DEFAULT_DATA_PATH,
        help=f"Path to dataset root (default: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        "--val-preview-root",
        type=str,
        default=DEFAULT_VAL_PREVIEW_ROOT,
        help=f"Path where validation previews will be saved (default: {DEFAULT_VAL_PREVIEW_ROOT})",
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default=DEFAULT_RUNS_DIR,
        help=f"Root directory for TensorBoard runs (default: {DEFAULT_RUNS_DIR})",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=DEFAULT_CHECKPOINT_DIR,
        help=f"Directory where checkpoints will be stored (default: {DEFAULT_CHECKPOINT_DIR})",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=NUM_EPOCHS,
        help=f"Number of training epochs (default: {NUM_EPOCHS})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Training batch size (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=LEARNING_RATE,
        help=f"Learning rate for optimizer (default: {LEARNING_RATE})",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=NUM_WORKERS,
        help=f"Number of DataLoader workers (default: {NUM_WORKERS})",
    )
    parser.add_argument(
        "--pred-length",
        type=int,
        default=PRED_LENGTH,
        help=f"Number of frames to predict (default: {PRED_LENGTH})",
    )
    parser.add_argument(
        "--small-debug",
        action="store_true",
        help="Enable small-debug mode: use a small subset of train/val for quick testing.",
    )

    args = parser.parse_args()

    data_path = os.path.abspath(args.data_path)
    val_preview_root = os.path.abspath(args.val_preview_root)
    runs_dir = os.path.abspath(args.runs_dir)
    checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    num_epochs = int(args.epochs)
    batch_size = int(args.batch_size)
    learning_rate = float(args.lr)
    num_workers = int(args.num_workers)
    pred_length = int(args.pred_length)
    small_debug = bool(args.small_debug)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(runs_dir, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    train_writer = SummaryWriter(log_dir=os.path.join(log_dir, "Train"))
    val_writer = SummaryWriter(log_dir=os.path.join(log_dir, "Validation"))

    model = RainPredRNN(
        input_dim=1,
        num_hidden=256,
        max_hidden_channels=128,
        patch_height=16,
        patch_width=16,
        pred_length=pred_length,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5)
    scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and DEVICE == "cuda"))

    train_loader, val_loader = create_dataloaders(
        data_path=data_path,
        batch_size=batch_size,
        num_workers=num_workers,
        pred_length=pred_length,
        small_debug=small_debug,
    )

    steps_train = len(train_loader)
    steps_val = len(val_loader)

    bps_train = benchmark_train(
        train_loader,
        model,
        optimizer,
        DEVICE,
        criterion_sl1,
        criterion_fl,
        criterion_mae_lambda,
        scaler=scaler,
        warmup=2,
        measure=10,
        pred_length=pred_length,
    )
    bps_val = benchmark_val(
        val_loader,
        model,
        DEVICE,
        warmup=2,
        measure=50,
        pred_length=pred_length,
    )

    eta_train = steps_train / bps_train if bps_train > 0 else float("inf")
    eta_val = steps_val / bps_val if bps_val > 0 else float("inf")

    print(
        f"[Benchmark] Train: ~{bps_train:.2f} batch/s | steps/epoch={steps_train} | "
        f"ETA epoch ≈ {hms(eta_train)}"
    )
    print(
        f"[Benchmark] Val  : ~{bps_val:.2f} batch/s | steps/val  ={steps_val}   | "
        f"ETA val   ≈ {hms(eta_val)}"
    )

    os.makedirs(val_preview_root, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_val = float("inf")

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            DEVICE,
            scaler=scaler,
            pred_length=pred_length,
        )

        val_metrics, conf_matrix = evaluate(
            model,
            val_loader,
            DEVICE,
            threshold_dbz=15.0,
            pred_length=pred_length,
        )

        scheduler.step(val_metrics["TOTAL"])

        print(f"\tTrain Loss: {train_loss:.4f}")
        print(
            "\tVal "
            + ", ".join([f"{k}: {float(v):.4f}" for k, v in val_metrics.items()])
        )

        train_writer.add_scalar("Loss", train_loss, epoch)
        for k, v in val_metrics.items():
            tag = "Loss" if k.lower() == "total" else k
            val_writer.add_scalar(tag, float(v), epoch)

        save_all_val_predictions(
            model,
            val_loader,
            DEVICE,
            val_preview_root,
            epoch,
            overwrite=False,
            pred_length=pred_length,
        )
        save_all_val_targets(
            val_loader,
            val_preview_root,
            epoch,
            overwrite=False,
        )

        if val_metrics["TOTAL"] < best_val:
            best_val = val_metrics["TOTAL"]
            generate_evaluation_report(
                val_metrics,
                conf_matrix,
                os.path.join(checkpoint_dir, "evaluation_reports"),
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": val_metrics,
                    "confusion_matrix": conf_matrix,
                },
                os.path.join(checkpoint_dir, "best_model.pth"),
            )

    train_writer.close()
    val_writer.close()
    print("Training completed.")


if __name__ == "__main__":
    main()
