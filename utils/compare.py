#!/usr/bin/env python
"""
compare.py

Compare truth vs predicted radar TIFF/GeoTIFF images as side-by-side panels
for a sequence of timestamps, and compute quantitative verification metrics.

Features
--------
- Sequence mode between --start and --end timestamps (YYYYMMDDZhhmm)
- Truth vs Pred visualization with two orientations:
    * landscape: Truth row on top, Pred row below, frames arranged horizontally
    * portrait: per-timestamp rows with Truth/Pred columns
- Optional custom radar palette JSON (--palette)
- Robust handling of NaN/inf and constant fields (no vmin > vmax crashes)
- Logging instead of print
- Colorbar placed outside the plot area to avoid overlapping images
- Frame-by-frame and overall metrics (MSE, RMSE, MAE, Bias, Correlation)
- Metrics panel plotted at the bottom of the figure
- Optional JSON dump with all per-frame and overall metrics (--metrics-json)
"""

import argparse
import json
import logging
import os
import re
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

# Optional GeoTIFF-aware reader (rasterio) with fallback to Pillow
try:
    import rasterio

    HAS_RASTERIO = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_RASTERIO = False

try:
    from PIL import Image

    HAS_PIL = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_PIL = False

# Timestamp pattern like 20251202Z1810
TIMESTAMP_RE = re.compile(r"(\d{8}Z\d{4})")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(level_str: str) -> None:
    """Configure the root logger with a simple format."""
    level = getattr(logging, level_str.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Palette handling
# ---------------------------------------------------------------------------
def load_palette(path: Path):
    """
    Load a radar palette JSON file of the form:

    {
      "levels": [-10, -5, 0, 5, ...],
      "colors": ["#646464", "#04e9e7", ...],
      "label": "Reflectivity (dBZ)"
    }

    Returns
    -------
    cmap : matplotlib.colors.ListedColormap
    norm : matplotlib.colors.BoundaryNorm
    label: str
        Label for the colorbar (may be empty).
    """
    logger.info("Loading palette from %s", path)
    with path.open("r") as f:
        pal = json.load(f)

    levels = pal["levels"]
    colors = pal["colors"]
    label = pal.get("label", "")

    if len(colors) != len(levels):
        raise ValueError(
            f"Palette {path}: {len(colors)} colors but {len(levels)} levels."
        )

    # BoundaryNorm expects len(boundaries) = len(colors) + 1
    boundaries = np.array(levels + [levels[-1] + 1], dtype=float)

    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(boundaries, cmap.N)

    logger.info(
        "Palette loaded: %d colors, min=%.3f, max=%.3f",
        len(colors),
        levels[0],
        levels[-1],
    )
    # Attach label to norm for later retrieval
    if label:
        setattr(norm, "label", label)

    return cmap, norm, label


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def extract_timestamp(path: Path):
    """
    Extract timestamp of the form YYYYMMDDZhhmm from file name.

    Parameters
    ----------
    path : Path

    Returns
    -------
    str or None
    """
    m = TIMESTAMP_RE.search(path.name)
    if m:
        return m.group(1)
    return None


def scan_directory_for_tiffs(dir_path: Path):
    """
    Recursively scan directory for .tif/.tiff files and build a mapping:

        timestamp -> Path

    where timestamp is of the form YYYYMMDDZhhmm extracted from the filename.
    """
    mapping: dict[str, Path] = {}
    logger.info("Scanning directory for TIFF files: %s", dir_path)
    for root, _, files in os.walk(dir_path):
        for fname in files:
            if not fname.lower().endswith((".tif", ".tiff")):
                continue
            fpath = Path(root) / fname
            ts = extract_timestamp(fpath)
            if ts is None:
                logger.debug("No timestamp found in file name: %s", fpath)
                continue
            mapping[ts] = fpath
    logger.info(
        "Found %d TIFF files with valid timestamps in %s", len(mapping), dir_path
    )
    return mapping


def read_tiff(path: Path, nodata_value: float | None) -> np.ndarray:
    """
    Read the first band of a TIFF/GeoTIFF as a float32 numpy array.
    Nodata values (if present) are converted to NaN.

    Parameters
    ----------
    path : Path
    nodata_value : float | None
        Optional override for the nodata value to mask.

    Returns
    -------
    np.ndarray
        2D array with dtype float32
    """
    logger.debug("Reading TIFF: %s", path)

    if HAS_RASTERIO:
        # Use rasterio to read the first band for GeoTIFF-aware IO.
        with rasterio.open(path) as src:
            # Cast to float32 to allow NaN masking downstream.
            data = src.read(1).astype(np.float32)
            # Prefer the CLI-provided nodata override when supplied.
            nodata = nodata_value if nodata_value is not None else src.nodata
            # Mask nodata entries to NaN so they do not affect stats.
            if nodata is not None:
                data[data == nodata] = np.nan
        # Return the masked array for downstream plotting/metrics.
        return data

    if HAS_PIL:
        # Read the TIFF via Pillow when rasterio is unavailable.
        im = Image.open(path)
        # Convert to float32 so we can insert NaNs if needed.
        arr = np.array(im, dtype=np.float32)
        # Collapse multiband data to the first channel for consistency.
        if arr.ndim > 2:
            # Take first channel if multi-band
            arr = arr[..., 0]
        # Apply the nodata override since PIL has no nodata metadata.
        if nodata_value is not None:
            # Replace nodata with NaN so color range ignores it.
            arr[arr == nodata_value] = np.nan
        # Return the masked array.
        return arr

    raise RuntimeError(
        "Neither rasterio nor Pillow (PIL) is available. "
        "Install one of them to read TIFF files."
    )


# ---------------------------------------------------------------------------
# Pair building and global stats
# ---------------------------------------------------------------------------
def build_pairs(
    truth_dir: Path, pred_dir: Path, start: str | None, end: str | None
):
    """
    Build a list of (timestamp, truth_path, pred_path) sorted by timestamp.
    Restrict to timestamps between start and end (inclusive) if provided.
    """
    truth_map = scan_directory_for_tiffs(truth_dir)
    pred_map = scan_directory_for_tiffs(pred_dir)

    common_ts = sorted(set(truth_map.keys()) & set(pred_map.keys()))
    logger.info("Found %d common timestamps between truth and pred.", len(common_ts))

    if start:
        common_ts = [ts for ts in common_ts if ts >= start]
    if end:
        common_ts = [ts for ts in common_ts if ts <= end]

    if not common_ts:
        logger.error("No common frames found in the specified range.")
        return []

    logger.info(
        "Using %d frame pairs between %s and %s",
        len(common_ts),
        common_ts[0],
        common_ts[-1],
    )

    pairs = [(ts, truth_map[ts], pred_map[ts]) for ts in common_ts]
    return pairs


def compute_global_range(arrays: list[np.ndarray]):
    """
    Compute a NaN/inf-safe global min and max over a list of arrays.

    Returns
    -------
    vmin, vmax : float
    """
    global_vmin = float("inf")
    global_vmax = float("-inf")

    for arr in arrays:
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            continue
        local_min = float(finite.min())
        local_max = float(finite.max())
        global_vmin = min(global_vmin, local_min)
        global_vmax = max(global_vmax, local_max)

    if not np.isfinite(global_vmin) or not np.isfinite(global_vmax):
        logger.warning("No finite values found across all frames. Using [0, 1].")
        return 0.0, 1.0

    if global_vmin >= global_vmax:
        logger.warning(
            "Global vmin >= vmax (vmin=%.3f, vmax=%.3f). "
            "Expanding range artificially.",
            global_vmin,
            global_vmax,
        )
        global_vmax = global_vmin + 1e-6

    logger.info("Global data range: vmin=%.3f, vmax=%.3f", global_vmin, global_vmax)
    return global_vmin, global_vmax


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_frame_metrics(truth: np.ndarray, pred: np.ndarray):
    """
    Compute basic verification metrics between truth and prediction.

    Returns
    -------
    metrics : dict
        Keys: mse, mae, bias, corr
    """
    mask = np.isfinite(truth) & np.isfinite(pred)
    if not np.any(mask):
        return {"mse": np.nan, "mae": np.nan, "bias": np.nan, "corr": np.nan}

    t = truth[mask].ravel()
    p = pred[mask].ravel()
    diff = p - t

    mse = float(np.mean(diff**2))
    mae = float(np.mean(np.abs(diff)))
    bias = float(np.mean(diff))

    if t.size > 1 and np.std(t) > 0.0 and np.std(p) > 0.0:
        corr = float(np.corrcoef(t, p)[0, 1])
    else:
        corr = np.nan

    return {"mse": mse, "mae": mae, "bias": bias, "corr": corr}


def compute_overall_metrics(truth_list: list[np.ndarray], pred_list: list[np.ndarray]):
    """
    Compute overall metrics across all frames.

    Returns
    -------
    metrics : dict
        Keys: mse, mae, bias, corr
    """
    all_t = []
    all_p = []
    for truth, pred in zip(truth_list, pred_list):
        mask = np.isfinite(truth) & np.isfinite(pred)
        if not np.any(mask):
            continue
        all_t.append(truth[mask].ravel())
        all_p.append(pred[mask].ravel())

    if not all_t:
        return {"mse": np.nan, "mae": np.nan, "bias": np.nan, "corr": np.nan}

    t = np.concatenate(all_t)
    p = np.concatenate(all_p)
    diff = p - t

    mse = float(np.mean(diff**2))
    mae = float(np.mean(np.abs(diff)))
    bias = float(np.mean(diff))

    if t.size > 1 and np.std(t) > 0.0 and np.std(p) > 0.0:
        corr = float(np.corrcoef(t, p)[0, 1])
    else:
        corr = np.nan

    return {"mse": mse, "mae": mae, "bias": bias, "corr": corr}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_sequence(
    pairs: list[tuple[str, Path, Path]],
    save_path: Path | None,
    title: str | None,
    cmap,
    norm,
    nodata_value: float | None,
    orientation: str = "landscape",
    metrics_json: Path | None = None,
    show_metrics: bool = False,  # Control whether metrics are computed and displayed.
):
    """
    Plot a sequence of frame pairs (truth vs pred).

    Orientation = 'landscape':
        - Frames are arranged horizontally by timestamp (columns).
        - First row: all Truth frames.
        - Second row: all Pred frames.
        - Third row: metrics graph.

    Orientation = 'portrait':
        - One row per timestamp.
        - Two columns: Truth (left), Pred (right).
        - Last row: metrics graph.

    If norm is provided (e.g. BoundaryNorm from a palette), it is used.
    Otherwise global vmin/vmax are computed and used.

    Metrics (RMSE, MAE, Bias, R) are visualized in a graph below the frames
    when enabled via the command line.
    Metrics can optionally be dumped to a JSON file when enabled.
    """
    logger.info("Sequence mode: plotting %d frame pairs.", len(pairs))

    # First pass: read data and cache them
    cached: list[tuple[str, np.ndarray, np.ndarray]] = []
    all_truth: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []

    for ts, truth_path, pred_path in pairs:
        # Read the truth TIFF while masking nodata values.
        truth_data = read_tiff(truth_path, nodata_value)
        # Read the prediction TIFF while masking nodata values.
        pred_data = read_tiff(pred_path, nodata_value)
        # Cache arrays for plotting and metrics.
        cached.append((ts, truth_data, pred_data))
        # Accumulate for global min/max computation.
        all_truth.append(truth_data)
        all_pred.append(pred_data)

    # If no palette norm is provided, compute a global vmin/vmax
    if norm is None:
        global_vmin, global_vmax = compute_global_range(all_truth + all_pred)
    else:
        global_vmin = global_vmax = None  # not used

    nframes = len(cached)

    # Prepare metrics only when explicitly requested by the user.
    per_frame_metrics: list[dict] = []  # Initialize the per-frame metrics container.
    # Track overall metrics so we can annotate the metrics panel when enabled.
    overall_with_rmse: dict[str, float] | None = None  # Placeholder for overall stats.
    # Guard all metric computations behind the flag.
    if show_metrics:  # Only compute metrics when the flag is enabled.
        # Precompute metrics (including RMSE) per frame.
        for ts, truth_data, pred_data in cached:  # Loop over each cached frame pair.
            # Compute base metrics for this frame pair.
            m = compute_frame_metrics(truth_data, pred_data)  # Compute base metrics.
            # Derive RMSE from MSE for display.
            rmse = (  # Compute RMSE from MSE when finite.
                float(np.sqrt(m["mse"])) if np.isfinite(m["mse"]) else float("nan")
            )
            # Build a timestamped metrics payload for this frame.
            m_with_ts = {"timestamp": ts, "rmse": rmse}  # Seed the metrics dict.
            # Merge in the base metrics fields for output and plotting.
            m_with_ts.update(m)  # Add MSE/MAE/Bias/Corr to the dict.
            # Store the per-frame metrics for overlays and plotting.
            per_frame_metrics.append(m_with_ts)  # Append for later plotting.
            # Log per-frame diagnostics for debugging.
            logger.debug(  # Emit per-frame metrics to the debug log.
                "Metrics %s: MSE=%.4f, MAE=%.4f, Bias=%.4f, Corr=%.4f",
                ts,  # Include the timestamp in the log entry.
                m["mse"],  # Report the per-frame MSE.
                m["mae"],  # Report the per-frame MAE.
                m["bias"],  # Report the per-frame Bias.
                m["corr"],  # Report the per-frame correlation.
            )

        # Aggregate metrics across all frames for overall reporting.
        overall = compute_overall_metrics(all_truth, all_pred)  # Compute overall stats.
        # Compute RMSE from the overall MSE.
        overall_rmse = (  # Compute overall RMSE when MSE is finite.
            float(np.sqrt(overall["mse"])) if np.isfinite(overall["mse"]) else float("nan")
        )
        # Store the overall metrics with RMSE for rendering.
        overall_with_rmse = dict(overall)  # Copy the overall metrics dict.
        # Attach RMSE to the overall metrics for display.
        overall_with_rmse["rmse"] = overall_rmse  # Add RMSE to overall metrics.

        # Emit a summary of the overall metrics to the log.
        logger.info(  # Log the overall metrics for user visibility.
            "Overall metrics: MSE=%.4f, MAE=%.4f, Bias=%.4f, Corr=%.4f",
            overall["mse"],  # Include overall MSE in the log.
            overall["mae"],  # Include overall MAE in the log.
            overall["bias"],  # Include overall Bias in the log.
            overall["corr"],  # Include overall correlation in the log.
        )

        # Optional JSON dump of metrics when enabled.
        if metrics_json is not None:  # Export metrics when a path is provided.
            # Announce the JSON export location in the log.
            logger.info("Writing metrics JSON to %s", metrics_json)  # Log output path.
            # Build the JSON payload for downstream analysis.
            metrics_payload = {  # Assemble payload with per-frame and overall stats.
                "per_frame": per_frame_metrics,
                "overall": overall_with_rmse,
            }
            # Persist the metrics to disk as formatted JSON.
            with metrics_json.open("w") as f:  # Open the JSON output file.
                json.dump(metrics_payload, f, indent=2)  # Write pretty JSON.
            # Confirm the JSON write succeeded.
            logger.info("Metrics JSON successfully written.")  # Log completion.
    # Warn if metrics JSON is requested while metrics are disabled.
    elif metrics_json is not None:  # Handle a metrics JSON request without metrics.
        # Notify the user that metrics export is skipped without metrics enabled.
        logger.warning(  # Emit a warning about the skipped export.
            "Metrics JSON requested without --metrics; skipping metrics export."
        )

    # ------------------------------------------------------------------
    # Figure layout using GridSpec
    # ------------------------------------------------------------------
    import matplotlib.gridspec as gridspec

    if orientation == "landscape":  # Use the landscape layout when requested.
        # Frames laid out horizontally by time:
        # row 0: Truth, row 1: Pred, row 2: metrics (optional)
        fig_width = max(10, 3 * nframes)  # Scale width by frame count.
        # Keep extra height only when the metrics panel is enabled.
        fig_height = 6 if show_metrics else 5  # Use taller figures with metrics.
        # Set height ratios based on whether metrics are shown.
        height_ratios = [1.0, 1.0, 0.7] if show_metrics else [1.0, 1.0]

        fig = plt.figure(  # Create the figure with a constrained layout.
            figsize=(fig_width, fig_height),
            constrained_layout=True,
        )
        gs = gridspec.GridSpec(  # Define the grid layout for the figure.
            nrows=3 if show_metrics else 2,  # Add a metrics row when enabled.
            ncols=nframes,  # Use one column per timestamp.
            figure=fig,  # Attach the GridSpec to the figure.
            height_ratios=height_ratios,  # Apply height ratios to rows.
        )

        axes_img = np.empty((2, nframes), dtype=object)
        last_im = None

        # Image panels: iterate by column (timestamp)
        for col_idx, (ts, truth_data, pred_data) in enumerate(cached):
            ax_truth = fig.add_subplot(gs[0, col_idx])
            ax_pred = fig.add_subplot(gs[1, col_idx])
            axes_img[0, col_idx] = ax_truth
            axes_img[1, col_idx] = ax_pred

            if norm is not None:
                im_truth = ax_truth.imshow(truth_data, cmap=cmap, norm=norm)
                im_pred = ax_pred.imshow(pred_data, cmap=cmap, norm=norm)
            else:
                im_truth = ax_truth.imshow(
                    truth_data, cmap=cmap, vmin=global_vmin, vmax=global_vmax
                )
                im_pred = ax_pred.imshow(
                    pred_data, cmap=cmap, vmin=global_vmin, vmax=global_vmax
                )

            # Column-level timestamp as title on the Truth row
            ax_truth.set_title(ts, fontsize=9)

            # Label rows once on the leftmost column for clarity
            if col_idx == 0:
                ax_truth.set_ylabel("Truth", fontsize=9)
                ax_pred.set_ylabel("Pred", fontsize=9)

            ax_truth.axis("off")
            ax_pred.axis("off")

            # Annotate per-frame metrics on the Truth panel when enabled.
            if show_metrics:  # Only add overlays when metrics are enabled.
                # Fetch metrics for this frame.
                m = per_frame_metrics[col_idx]  # Select metrics by column index.
                # Compose a multi-line text label for the overlay.
                metric_text = (  # Build a multi-line overlay string.
                    f"RMSE={m['rmse']:.2f}\n"
                    f"MAE={m['mae']:.2f}\n"
                    f"Bias={m['bias']:.2f}\n"
                    f"R={m['corr']:.2f}"
                )
                # Place the metrics in the bottom-left corner of the panel.
                ax_truth.text(  # Render the metric overlay text.
                    0.01,  # Use a small left margin.
                    0.02,  # Use a small bottom margin.
                    metric_text,  # Draw the metrics text.
                    transform=ax_truth.transAxes,  # Place in axes coordinates.
                    fontsize=7,  # Use a small font for overlay readability.
                    va="bottom",  # Anchor the text to the bottom.
                    ha="left",  # Align the text to the left.
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),  # Box.
                )

            last_im = im_pred or im_truth

        # Metrics panel: bottom row, spanning all columns, when enabled.
        ax_metrics = fig.add_subplot(gs[2, :]) if show_metrics else None  # Optional.

    else:
        # Portrait: one row per timestamp, 2 columns (Truth, Pred), last row metrics
        fig_width = 10  # Fix the width for portrait layouts.
        # Scale height to include metrics panel only when enabled.
        fig_height = max(4, 2 * nframes + (2 if show_metrics else 0))  # Scale height.
        # Build height ratios with optional metrics row.
        height_ratios = [1.0] * nframes + ([0.6] if show_metrics else [])  # Ratios.

        fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
        gs = gridspec.GridSpec(
            nrows=nframes + (1 if show_metrics else 0),  # Add a row for metrics.
            ncols=2,  # Two columns for truth and prediction.
            figure=fig,  # Attach to the figure.
            height_ratios=height_ratios,  # Apply height ratios.
        )

        axes_img = np.empty((nframes, 2), dtype=object)
        last_im = None

        # Image panels: iterate by row (timestamp)
        for row_idx, (ts, truth_data, pred_data) in enumerate(cached):
            ax_truth = fig.add_subplot(gs[row_idx, 0])
            ax_pred = fig.add_subplot(gs[row_idx, 1])
            axes_img[row_idx, 0] = ax_truth
            axes_img[row_idx, 1] = ax_pred

            if norm is not None:
                im_truth = ax_truth.imshow(truth_data, cmap=cmap, norm=norm)
                im_pred = ax_pred.imshow(pred_data, cmap=cmap, norm=norm)
            else:
                im_truth = ax_truth.imshow(
                    truth_data, cmap=cmap, vmin=global_vmin, vmax=global_vmax
                )
                im_pred = ax_pred.imshow(
                    pred_data, cmap=cmap, vmin=global_vmin, vmax=global_vmax
                )

            ax_truth.set_title(f"Truth {ts}")
            ax_pred.set_title(f"Pred {ts}")

            ax_truth.axis("off")
            ax_pred.axis("off")

            # Annotate per-frame metrics on the Truth panel when enabled.
            if show_metrics:  # Only draw metrics overlays when enabled.
                # Pull the metrics for this timestamp.
                m = per_frame_metrics[row_idx]  # Select metrics by row index.
                # Build the overlay string for this frame.
                metric_text = (  # Compose the per-frame metrics text.
                    f"RMSE={m['rmse']:.2f}\n"
                    f"MAE={m['mae']:.2f}\n"
                    f"Bias={m['bias']:.2f}\n"
                    f"R={m['corr']:.2f}"
                )
                # Paint the metrics box onto the truth panel.
                ax_truth.text(  # Render the text overlay on the truth axis.
                    0.01,  # Use a left margin.
                    0.02,  # Use a bottom margin.
                    metric_text,  # Add the text content.
                    transform=ax_truth.transAxes,  # Position in axes coords.
                    fontsize=8,  # Slightly larger font for portrait mode.
                    va="bottom",  # Anchor the text to the bottom.
                    ha="left",  # Align left for readability.
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),  # Box.
                )

            last_im = im_pred or im_truth

        # Metrics panel: bottom row, spanning both columns, when enabled.
        ax_metrics = fig.add_subplot(gs[-1, :]) if show_metrics else None  # Optional.

    # Metrics graph (common to both orientations) when enabled.
    if show_metrics and ax_metrics is not None and overall_with_rmse is not None:
        # Use hhmm as short labels for the x-axis.
        frame_labels = [ts[-4:] for ts, _, _ in cached]  # Extract hhmm labels.
        # Gather metric series for plotting.
        rmse_vals = [m["rmse"] for m in per_frame_metrics]  # Collect RMSE values.
        mae_vals = [m["mae"] for m in per_frame_metrics]  # Collect MAE values.
        bias_vals = [m["bias"] for m in per_frame_metrics]  # Collect Bias values.

        # Build the x-axis positions for each frame.
        x = np.arange(nframes)  # Use consecutive indices for frames.

        # Plot the RMSE series.
        ax_metrics.plot(x, rmse_vals, marker="o", label="RMSE")  # Plot RMSE.
        # Plot the MAE series.
        ax_metrics.plot(x, mae_vals, marker="s", label="MAE")  # Plot MAE.
        # Plot the Bias series.
        ax_metrics.plot(x, bias_vals, marker="^", label="Bias")  # Plot Bias.

        # Configure the x-axis tick placement.
        ax_metrics.set_xticks(x)  # Use frame indices as tick positions.
        # Label each tick with the hhmm timestamp suffix.
        ax_metrics.set_xticklabels(frame_labels, rotation=45)  # Add tick labels.
        # Label the x-axis for clarity.
        ax_metrics.set_xlabel("Time (hhmm)")  # Describe the x-axis.
        # Label the y-axis with the units.
        ax_metrics.set_ylabel("Error (dBZ)")  # Describe the y-axis.
        # Add a light grid for readability.
        ax_metrics.grid(True, linestyle="--", alpha=0.3)  # Add grid lines.
        # Add a legend for the plotted series.
        ax_metrics.legend(loc="upper right", fontsize=8)  # Display legend.

        # Add overall metrics as a text box.
        overall_text = (  # Compose the overall metrics string.
            f"Overall: RMSE={overall_with_rmse['rmse']:.2f}, "
            f"MAE={overall_with_rmse['mae']:.2f}, "
            f"Bias={overall_with_rmse['bias']:.2f}, "
            f"R={overall_with_rmse['corr']:.2f}"
        )
        # Place the overall metrics in the top-left of the metrics panel.
        ax_metrics.text(  # Draw the overall metrics text box.
            0.01,  # Use a left margin.
            0.95,  # Use a top margin.
            overall_text,  # Provide the text content.
            transform=ax_metrics.transAxes,  # Position in axes coordinates.
            fontsize=9,  # Use a readable font size.
            va="top",  # Anchor the text to the top.
            ha="left",  # Align text to the left.
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),  # Box styling.
        )

    # Colorbar on the right, outside the image panels
    if last_im is not None:
        cbar = fig.colorbar(
            last_im,
            ax=list(axes_img.ravel()),
            orientation="vertical",
            fraction=0.04,
            pad=0.02,
        )
        label = getattr(norm, "label", None) if norm is not None else None
        if label:
            cbar.set_label(label)

    # Global title (placed slightly high to avoid overlap)
    if title:
        fig.suptitle(title, fontsize=14, y=0.995)

    # Save or show
    if save_path is not None:
        logger.info("Saving figure to %s", save_path)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        logger.info("Displaying figure interactively.")
        plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare truth vs predicted radar TIFF/GeoTIFF sequences."
    )
    parser.add_argument(
        "--truth-dir",
        "--real-dir",  # backward-compatible alias
        dest="truth_dir",
        required=True,
        help="Directory containing truth (ground-truth) TIFF/GeoTIFF frames.",
    )
    parser.add_argument(
        "--pred-dir",
        required=True,
        help="Directory containing predicted TIFF/GeoTIFF frames.",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start timestamp (YYYYMMDDZhhmm), inclusive.",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End timestamp (YYYYMMDDZhhmm), inclusive.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Figure title.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help=(
            "Path to save the output figure (e.g., out.png). "
            "If omitted, the figure is shown interactively."
        ),
    )
    parser.add_argument(
        "--palette",
        type=str,
        default=None,
        help=(
            "Path to palette JSON file (levels + colors). "
            "If omitted, the default 'viridis' colormap is used."
        ),
    )
    parser.add_argument(
        "--orientation",
        type=str,
        choices=["landscape", "portrait"],
        default="landscape",
        help=(
            "Figure orientation: 'landscape' (Truth row, Pred row, metrics below) "
            "or 'portrait' (one row per timestamp, metrics at bottom). "
            "Default: landscape."
        ),
    )
    parser.add_argument(
        "--metrics-json",
        type=str,
        default=None,
        help=(
            "Optional path to a JSON file where per-frame and overall metrics "
            "will be written for post-processing."
        ),
    )
    parser.add_argument(
        "--metrics",  # CLI flag name for enabling metrics output.
        action="store_true",  # Store True when the flag is present.
        help=(  # Describe what enabling metrics does.
            "Enable RMSE/MAE/Bias/R metrics overlays, plots, and optional JSON export."
        ),
    )
    parser.add_argument(
        "--nodata",
        type=float,
        default=None,
        help=(
            "Optional nodata value to mask in TIFFs so it is excluded from "
            "color scaling and metrics."
        ),
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR). Default: INFO.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(args.log_level)

    truth_dir = Path(args.truth_dir).resolve()
    pred_dir = Path(args.pred_dir).resolve()

    if not truth_dir.is_dir():
        logger.error("truth-dir does not exist or is not a directory: %s", truth_dir)
        raise SystemExit(1)
    if not pred_dir.is_dir():
        logger.error("pred-dir does not exist or is not a directory: %s", pred_dir)
        raise SystemExit(1)

    # Palette or default colormap
    if args.palette is not None:
        cmap, norm, _ = load_palette(Path(args.palette))
    else:
        cmap = plt.get_cmap("viridis")
        norm = None

    pairs = build_pairs(truth_dir, pred_dir, args.start, args.end)
    if not pairs:
        raise SystemExit(1)

    save_path = Path(args.save).resolve() if args.save else None
    metrics_json_path = Path(args.metrics_json).resolve() if args.metrics_json else None

    # Ensure metrics JSON export is only attempted when metrics are enabled.
    if metrics_json_path is not None and not args.metrics:
        # Inform the user that metrics need to be enabled for JSON output.
        logger.error("--metrics-json requires --metrics to be enabled.")
        # Exit with a non-zero status to signal misconfiguration.
        raise SystemExit(2)

    plot_sequence(
        pairs=pairs,
        save_path=save_path,
        title=args.title,
        cmap=cmap,
        norm=norm,
        nodata_value=args.nodata,
        orientation=args.orientation,
        metrics_json=metrics_json_path,
        show_metrics=args.metrics,  # Enable metrics visualization when requested.
    )


if __name__ == "__main__":
    main()
