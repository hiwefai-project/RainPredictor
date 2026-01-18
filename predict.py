import os
import re
import argparse
import datetime as dt
import logging  # Use the logging module for structured status output.
from typing import List

import torch

from rainpred.model import RainPredModel
from rainpred.geo_io import load_sequence_from_dir, save_predictions_as_geotiff


# Create a module-level logger to replace print-based status messages.
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# File-name handling for time-consistent output names
# ----------------------------------------------------------------------
# Expected pattern:
#   <prefix>YYYYMMDDZhhmm<suffix>
# e.g.:
#   rdr0_d01_20241023Z0710_VMI.tiff
FILENAME_RE = re.compile(
    r"^(?P<prefix>.+_)"      # everything up to and including the last "_"
    r"(?P<date>\d{8})"       # YYYYMMDD
    r"Z"
    r"(?P<hour>\d{2})"       # hh
    r"(?P<minute>\d{2})"     # mm
    r"(?P<suffix>.*)$"       # tail, such as "_VMI.tiff"
)


# ----------------------------------------------------------------------
# Device helper
# ----------------------------------------------------------------------
def get_device(prefer_cpu: bool = False) -> torch.device:
    """Return CUDA device if available (and not forced to CPU), else CPU."""
    if (not prefer_cpu) and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ----------------------------------------------------------------------
# Checkpoint loading
# ----------------------------------------------------------------------
def load_model(checkpoint_path: str, device: torch.device, pred_length: int) -> torch.nn.Module:
    """
    Load a RainPredRNN model from checkpoint.
    Compatible with checkpoints produced by train.py.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Log which checkpoint is being loaded for transparency.
    logger.info("[predict] Loading checkpoint from: %s", checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only = False)

    # Extract a state_dict from various checkpoint formats
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        # Raw state_dict or full nn.Module
        if isinstance(ckpt, torch.nn.Module):
            model = ckpt.to(device)
            model.eval()
            return model
        state_dict = ckpt

    # Strip "module." prefix if saved from DataParallel
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k[len("module."):] : v for k, v in state_dict.items()}

    # Instantiate model with same hyperparameters as in train.py
    model = RainPredModel(
        in_channels=1,
        out_channels=1,
        hidden_channels=64,
        transformer_d_model=128,
        transformer_nhead=8,
        transformer_num_layers=2,
        pred_length=pred_length,
    )
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


# ----------------------------------------------------------------------
# Inference helper
# ----------------------------------------------------------------------
def run_inference(
    model: torch.nn.Module,
    sequence_tensor: torch.Tensor,
    device: torch.device,
    n_future: int,
) -> torch.Tensor:
    """
    Run forward prediction on a single input sequence.

    sequence_tensor: output of load_sequence_from_dir, shape (1, T, 1, H, W)
    """
    x = sequence_tensor.to(device)

    # Ensure 5D input (B,T,C,H,W)
    if x.dim() == 3:          # (H,W,C) – very unlikely
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.dim() == 4:        # (T,C,H,W)
        x = x.unsqueeze(0)
    elif x.dim() == 5:
        pass                   # already OK
    else:
        raise ValueError(f"Unexpected input shape: {tuple(x.shape)}")

    with torch.no_grad():
        preds, _ = model(x, pred_length=n_future)

    return preds  # (1, n_future, 1, H, W)


# ----------------------------------------------------------------------
# Time-step / naming utilities
# ----------------------------------------------------------------------
def infer_time_step_minutes(file_list: List[str]) -> int:
    """
    Infer the time step (in minutes) from the last two input filenames.
    Falls back to 10 minutes if parsing fails.
    """
    if len(file_list) < 2:
        return 10

    m2 = FILENAME_RE.match(os.path.basename(file_list[-1]))
    m1 = FILENAME_RE.match(os.path.basename(file_list[-2]))
    if not (m1 and m2):
        return 10

    date2 = dt.datetime.strptime(
        m2.group("date") + m2.group("hour") + m2.group("minute"), "%Y%m%d%H%M"
    )
    date1 = dt.datetime.strptime(
        m1.group("date") + m1.group("hour") + m1.group("minute"), "%Y%m%d%H%M"
    )
    delta = date2 - date1
    minutes = int(delta.total_seconds() // 60) or 10
    return abs(minutes)


def generate_output_basenames(
    file_list: List[str],
    n_future: int,
    step_minutes: int,
) -> List[str]:
    """
    Generate output filenames with consistent timestamps based on the last
    input file and the inferred time step.
    """
    if not file_list:
        return [f"pred_{i:02d}.tif" for i in range(1, n_future + 1)]

    last_name = os.path.basename(file_list[-1])
    match = FILENAME_RE.match(last_name)
    if not match:
        return [f"pred_{i:02d}.tif" for i in range(1, n_future + 1)]

    prefix = match.group("prefix")
    suffix = match.group("suffix")
    base_dt = dt.datetime.strptime(
        match.group("date") + match.group("hour") + match.group("minute"),
        "%Y%m%d%H%M",
    )

    out_names: List[str] = []
    for i in range(1, n_future + 1):
        new_dt = base_dt + dt.timedelta(minutes=step_minutes * i)
        stamp = new_dt.strftime("%Y%m%dZ%H%M")
        out_names.append(f"{prefix}{stamp}{suffix}")
    return out_names


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RainPredRNN inference on a radar sequence."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing a time-ordered sequence of GeoTIFF radar images.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a trained checkpoint (.pth) from train.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where predicted GeoTIFFs will be written.",
    )
    parser.add_argument(
        "-m",
        "--m",
        type=int,
        default=18,
        help="Number of past frames.",
    )
    parser.add_argument(
        "-n",
        "--n",
        type=int,
        default=6,
        help="Number of future frames to predict.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force inference on CPU even if CUDA is available.",
    )
    return parser.parse_args()


def main() -> None:
    # Configure root logging once for the CLI entry point.
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    device = get_device(prefer_cpu=args.cpu)
    # Log the device selection to clarify runtime hardware.
    logger.info("[predict] Using device: %s", device)

    # Load input sequence and metadata
    seq, paths, shape_info, meta = load_sequence_from_dir(args.input_dir, args.m)
    # Log how many frames were loaded from the input directory.
    logger.info(
        "[predict] Loaded sequence with %s frames from %s",
        seq.shape[1],
        args.input_dir,
    )

    # Instantiate and load model
    model = load_model(
        checkpoint_path=args.checkpoint,
        device=device,
        pred_length=args.n,
    )

    # Run inference
    preds = run_inference(
        model=model,
        sequence_tensor=seq,
        device=device,
        n_future=args.n,
    )

    # Build output filenames consistent with input naming
    step = infer_time_step_minutes(paths)
    out_basenames = generate_output_basenames(paths, n_future=args.n, step_minutes=step)

    # Save predictions as GeoTIFF in dBZ
    saved_paths = save_predictions_as_geotiff(
        preds=preds,
        output_dir=args.output_dir,
        shape_info=shape_info,
        meta=meta,
        prefix="pred",
        as_dbz=True,
        out_names=out_basenames,
    )

    # Log the checkpoint used to help with reproducibility.
    logger.info("[predict] Checkpoint used: %s", args.checkpoint)
    # Log the count of predicted frames to confirm output size.
    logger.info("[predict] Predicted %s frames (n=%s).", len(saved_paths), args.n)
    # Log the output directory for user visibility.
    logger.info("[predict] Saved outputs in: %s", args.output_dir)
    for p in saved_paths:
        # Log each saved filename for quick inspection.
        logger.info("  -> %s", os.path.basename(p))


if __name__ == "__main__":
    main()
