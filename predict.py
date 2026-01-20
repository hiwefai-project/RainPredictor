import os
import re
import argparse
import datetime as dt
import json  # Serialize AI-friendly diagnostics to JSON.
import logging  # Use the logging module for structured status output.
import importlib.util  # Check optional dependencies without try/except around imports.
from typing import List

import torch

from rainpred.model import RainPredModel
from rainpred.geo_io import load_sequence_from_dir, save_predictions_as_geotiff


# Create a module-level logger to replace print-based status messages.
logger = logging.getLogger(__name__)


def _summarize_tensor(tensor: torch.Tensor) -> dict:
    """Return basic statistics for a tensor in a JSON-serializable dict."""
    # Move the tensor to CPU to avoid GPU-only serialization issues.
    tensor_cpu = tensor.detach().to("cpu")
    # Compute the minimum value for range diagnostics.
    min_val = float(tensor_cpu.min().item())
    # Compute the maximum value for range diagnostics.
    max_val = float(tensor_cpu.max().item())
    # Compute the mean value for distribution diagnostics.
    mean_val = float(tensor_cpu.mean().item())
    # Return the summarized statistics as a dictionary.
    return {"min": min_val, "max": max_val, "mean": mean_val}


def _build_predict_ai_metrics(
    *,
    args: argparse.Namespace,
    device: torch.device,
    input_frame_count: int,
    output_frame_count: int,
    step_minutes: int,
    preds: torch.Tensor,
    inference_seconds: float,
    total_seconds: float,
) -> dict:
    """Build an AI-friendly diagnostics payload with inference metadata and suggestions."""
    # Initialize a suggestions list for downstream automation.
    suggestions: list[str] = []
    # Suggest using GPU when available and not explicitly forced to CPU.
    if args.cpu and torch.cuda.is_available():
        # Encourage using CUDA for faster inference.
        suggestions.append("CUDA is available; omit --cpu to speed up inference.")
    # Highlight resampling so downstream consumers can track data changes.
    if args.resample_factor != 1.0:
        # Note that resampled inputs differ from original data resolution.
        suggestions.append("Inputs were resampled; ensure this matches training resolution.")
    # Flag unexpected output counts for troubleshooting.
    if output_frame_count != args.n:
        # Suggest verifying input/output settings when counts mismatch.
        suggestions.append("Output frame count mismatch; verify --n and input availability.")
    # Warn if the model is asked to predict too many frames relative to history.
    if args.m <= args.n:
        # Suggest using a longer input history to stabilize predictions.
        suggestions.append("m <= n; consider using more input frames for stable forecasts.")

    # Build the JSON-friendly diagnostics payload.
    payload = {
        # Track schema versioning to support future upgrades.
        "schema_version": "v1",
        # Identify the lifecycle stage for downstream filtering.
        "stage": "predict",
        # Record runtime configuration values for traceability.
        "config": {
            "checkpoint": args.checkpoint,
            "input_dir": args.input_dir,
            "output_dir": args.output_dir,
            "m": int(args.m),
            "n": int(args.n),
            "cpu": bool(args.cpu),
            "resample_factor": float(args.resample_factor),
        },
        # Record the selected device for performance context.
        "device": str(device),
        # Record how many input frames were used.
        "input_frames": int(input_frame_count),
        # Record how many output frames were produced.
        "output_frames": int(output_frame_count),
        # Record the inferred time step for naming consistency.
        "step_minutes": int(step_minutes),
        # Include lightweight statistics about the predictions.
        "prediction_stats": _summarize_tensor(preds),
        # Record timing diagnostics for performance analysis.
        "timings_sec": {"inference": float(inference_seconds), "total": float(total_seconds)},
        # Provide actionable suggestions for next steps.
        "suggestions": suggestions,
        # Add an ISO timestamp to align logs with external systems.
        "timestamp": dt.datetime.now().isoformat(),
    }
    # Return the payload to the caller.
    return payload


def _write_ai_metrics_json(path: str, payload: dict) -> None:
    """Write AI-friendly diagnostics to JSON or log them when path is '-'."""
    # Serialize the payload to a formatted JSON string for readability.
    json_payload = json.dumps(payload, indent=2)
    # Emit to logs when the sentinel "-" path is used.
    if path == "-":
        # Log the JSON payload to stdout via logging.
        logger.info("[predict] AI metrics JSON:\n%s", json_payload)
        # Exit early since we are not writing to disk.
        return
    # Open the output file in write mode to keep the latest diagnostics.
    with open(path, "w", encoding="utf-8") as handle:
        # Persist the JSON payload to disk for downstream tooling.
        handle.write(json_payload)


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
# Optional resampling helper
# ----------------------------------------------------------------------
def resample_image(path: str, factor: float, logger: logging.Logger | None = None) -> bool:
    """Resample an image in-place by the given factor, preserving aspect ratio.

    If the file is a GeoTIFF, the georeferencing information (CRS and affine
    transform) is preserved and properly updated after resampling.

    The image is opened, resized, and saved back to the same path. A factor of
    1.0 leaves the image unchanged. Factors must be greater than 0.
    """
    # Fall back to a module logger if one is not supplied.
    if logger is None:
        # Retrieve the module-level logger for consistent log formatting.
        logger = logging.getLogger(__name__)

    # Short-circuit when no resampling is required.
    if factor == 1.0:
        # Nothing to do when the factor is exactly 1.0.
        return True

    # Guard against invalid scaling factors.
    if factor <= 0:
        # Log invalid factors so users can fix input quickly.
        logger.error("Invalid resample factor %s. It must be > 0.", factor)
        # Signal failure to the caller.
        return False

    # Check if rasterio and affine are available without try/except around imports.
    rasterio_spec = importlib.util.find_spec("rasterio")
    # Check for affine availability to compute updated transforms.
    affine_spec = importlib.util.find_spec("affine")
    # Proceed with GeoTIFF-aware resampling only when both dependencies exist.
    if rasterio_spec is not None and affine_spec is not None:
        # Import rasterio lazily once we know it is available.
        import rasterio
        # Import the resampling enum needed for bilinear resizing.
        from rasterio.enums import Resampling
        # Import Affine to scale the geotransform accurately.
        from affine import Affine

        try:
            # Open the GeoTIFF and gather metadata needed for resizing.
            with rasterio.open(path) as src:
                # Capture the original width and height for logging.
                width, height = src.width, src.height
                # Compute the resampled dimensions while keeping at least 1 pixel.
                new_width = max(1, int(round(width * factor)))
                # Compute the resampled dimensions while keeping at least 1 pixel.
                new_height = max(1, int(round(height * factor)))

                # Read and resample the data into the new shape.
                data = src.read(
                    out_shape=(src.count, new_height, new_width),
                    resampling=Resampling.bilinear,
                )

                # Compute the scale factors used to update the transform.
                scale_x = width / new_width
                # Compute the scale factors used to update the transform.
                scale_y = height / new_height
                # Build a new affine transform that preserves georeferencing.
                new_transform = src.transform * Affine.scale(scale_x, scale_y)

                # Start from the existing profile to preserve GeoTIFF metadata.
                profile = src.profile
                # Update the profile with the new size and transform.
                profile.update(
                    height=new_height,
                    width=new_width,
                    transform=new_transform,
                )

            # Write the resampled data back to the same path.
            with rasterio.open(path, "w", **profile) as dst:
                # Persist the resized data while preserving CRS.
                dst.write(data)

            # Log the resize result for transparency.
            logger.info(
                "GeoTIFF resampled %s from %dx%d to %dx%d (CRS preserved)",
                path, width, height, new_width, new_height,
            )
            # Indicate success to the caller.
            return True
        except Exception as exc:
            # Log the failure and fall back to PIL-based resizing.
            logger.warning(
                "GeoTIFF-aware resampling failed for %s (%s); falling back to PIL.",
                path, exc,
            )
    else:
        # Warn that rasterio/affine are missing and metadata may be lost.
        logger.warning(
            "rasterio/affine not available; falling back to PIL for %s. "
            "GeoTIFF metadata (if any) may not be preserved.",
            path,
        )

    # Check if Pillow is available for a generic resampling fallback.
    pillow_spec = importlib.util.find_spec("PIL")
    # Abort if Pillow is missing since we cannot resize without it.
    if pillow_spec is None:
        # Log the missing dependency to guide installation.
        logger.error("Pillow (PIL) is not available; cannot resample %s.", path)
        # Signal failure to the caller.
        return False

    # Import Pillow only after confirming availability.
    from PIL import Image

    try:
        # Open the image using Pillow for generic resizing.
        with Image.open(path) as img:
            # Capture the original dimensions for logging.
            width, height = img.size
            # Compute the new width while preserving aspect ratio.
            new_width = max(1, int(round(width * factor)))
            # Compute the new height while preserving aspect ratio.
            new_height = max(1, int(round(height * factor)))

            # Perform bilinear resizing for reasonable quality.
            resized = img.resize((new_width, new_height), Image.BILINEAR)
            # Save the resized image back to disk, preserving format.
            resized.save(path, format=img.format)

        # Log the successful PIL-based resize.
        logger.info(
            "Resampled (PIL) %s from %dx%d to %dx%d",
            path, width, height, new_width, new_height,
        )
        # Indicate success to the caller.
        return True
    except Exception as exc:
        # Log any unexpected errors encountered during PIL resize.
        logger.error("Failed to resample image %s: %s", path, exc)
        # Signal failure to the caller.
        return False


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
    # Add a resampling factor flag to optionally scale input frames in-place.
    parser.add_argument(
        "--resample-factor",
        # Parse the factor as a float so fractional scaling is supported.
        type=float,
        # Default to no-op resampling to preserve current behavior.
        default=1.0,
        # Explain how the factor affects input resolution.
        help=(
            "Optional scale factor to resample input frames in-place before "
            "inference (e.g., 0.5 halves resolution, 2.0 doubles it)."
        ),
    )
    parser.add_argument(
        "--metrics-json",
        # Accept a path where JSON diagnostics will be written.
        type=str,
        # Default to disabled so existing behavior is preserved.
        default=None,
        # Explain how to enable JSON diagnostics.
        help="Optional path to write AI-friendly JSON diagnostics (use '-' for stdout).",
    )
    return parser.parse_args()


def list_input_files(input_dir: str, pattern: str = ".tif") -> List[str]:
    """Return sorted input files matching the expected GeoTIFF extensions."""
    # Resolve the directory to an absolute path for consistent listing.
    input_dir = os.path.abspath(input_dir)
    # Collect files that match the .tif/.tiff patterns used in geo_io.
    files = [
        os.path.join(input_dir, filename)
        for filename in os.listdir(input_dir)
        if filename.endswith(pattern) or filename.endswith(pattern + "f")
    ]
    # Sort for deterministic ordering to match load_sequence_from_dir.
    files = sorted(files)
    # Return the ordered list to the caller.
    return files


def main() -> None:
    # Configure root logging once for the CLI entry point.
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # Capture the start time for end-to-end timing diagnostics.
    total_start = dt.datetime.now()
    args = parse_args()

    device = get_device(prefer_cpu=args.cpu)
    # Log the device selection to clarify runtime hardware.
    logger.info("[predict] Using device: %s", device)

    # Load input sequence and metadata
    # Optionally resample the input frames before loading for inference.
    if args.resample_factor != 1.0:
        # Determine the full sorted list so we resample the same frames used in inference.
        input_files = list_input_files(args.input_dir)
        # Trim to the first m frames to match the inference sequence selection.
        input_files = input_files[:args.m]
        # Log how many files will be resampled for clarity.
        logger.info(
            "[predict] Resampling %s input frames by factor %.3f",
            len(input_files),
            args.resample_factor,
        )
        # Resample each input file in-place before loading data.
        for path in input_files:
            # Attempt to resample and abort on failure to avoid mixed resolutions.
            if not resample_image(path, args.resample_factor, logger=logger):
                # Log the failure and stop inference.
                logger.error("[predict] Resampling failed for %s; aborting.", path)
                # Exit early to avoid inconsistent inputs.
                return

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
    # Record inference start time for performance metrics.
    inference_start = dt.datetime.now()
    preds = run_inference(
        model=model,
        sequence_tensor=seq,
        device=device,
        n_future=args.n,
    )
    # Compute inference duration in seconds.
    inference_seconds = (dt.datetime.now() - inference_start).total_seconds()

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
    # Compute total runtime duration in seconds.
    total_seconds = (dt.datetime.now() - total_start).total_seconds()

    # Log the checkpoint used to help with reproducibility.
    logger.info("[predict] Checkpoint used: %s", args.checkpoint)
    # Log the count of predicted frames to confirm output size.
    logger.info("[predict] Predicted %s frames (n=%s).", len(saved_paths), args.n)
    # Log the output directory for user visibility.
    logger.info("[predict] Saved outputs in: %s", args.output_dir)
    for p in saved_paths:
        # Log each saved filename for quick inspection.
        logger.info("  -> %s", os.path.basename(p))

    # Write AI-friendly diagnostics when requested.
    if args.metrics_json:
        # Build the diagnostics payload for inference.
        ai_metrics = _build_predict_ai_metrics(
            args=args,
            device=device,
            input_frame_count=seq.shape[1],
            output_frame_count=len(saved_paths),
            step_minutes=step,
            preds=preds,
            inference_seconds=inference_seconds,
            total_seconds=total_seconds,
        )
        # Persist the diagnostics payload to disk or stdout.
        _write_ai_metrics_json(args.metrics_json, ai_metrics)


if __name__ == "__main__":
    main()
