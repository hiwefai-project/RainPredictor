#!/usr/bin/env python
import argparse
import json
import logging
import os
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Optional GeoTIFF-aware reader (rasterio) with fallback to PIL
try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


TIMESTAMP_RE = re.compile(r"(\d{8}Z\d{4})")  # e.g. 20251202Z1810


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(level_str: str) -> None:
    """Configure the root logger."""
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

    Returns:
        cmap  : matplotlib ListedColormap
        norm  : matplotlib BoundaryNorm
        label : string for colorbar label (may be empty)
    """
    logger.info(f"Loading palette from {path}")
    with path.open("r") as f:
        pal = json.load(f)

    levels = pal["levels"]
    colors = pal["colors"]
    label = pal.get("label", "")

    if len(colors) != len(levels):
        raise ValueError(
            f"Palette {path}: {len(colors)} colors but {len(levels)} levels."
        )

    # Matplotlib BoundaryNorm expects (#boundaries == #colors + 1)
    boundaries = np.array(levels + [levels[-1] + 1], dtype=float)

    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(boundaries, cmap.N)

    logger.info(
        "Palette loaded: %d colors, min=%.3f, max=%.3f",
        len(colors),
        levels[0],
        levels[-1],
    )
    return cmap, norm, label


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def extract_timestamp(path: Path):
    """
    Extract timestamp of the form YYYYMMDDZhhmm from file name.
    Returns the string or None if not found.
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
    mapping = {}
    logger.info(f"Scanning directory for TIFF files: {dir_path}")
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
    logger.info("Found %d TIFF files with valid timestamps in %s", len(mapping), dir_path)
    return mapping


def read_tiff(path: Path) -> np.ndarray:
    """
    Read the first band of a TIFF/GeoTIFF as a float32 numpy array.
    Handles nodata as NaN when using rasterio.
    """
    logger.debug("Reading TIFF: %s", path)

    if HAS_RASTERIO:
        with rasterio.open(path) as src:
            data = src.read(1).astype(np.float32)
            nodata = src.nodata
            if nodata is not None:
                data[data == nodata] = np.nan
        return data

    if HAS_PIL:
        im = Image.open(path)
        arr = np.array(im, dtype=np.float32)
        if arr.ndim > 2:
            # Take first channel if multi-band
            arr = arr[..., 0]
        return arr

    raise RuntimeError(
        "Neither rasterio nor Pillow (PIL) is available. "
        "Install one of them to read TIFF files."
    )


# ---------------------------------------------------------------------------
# Pair building and global stats
# ---------------------------------------------------------------------------
def build_pairs(real_dir: Path, pred_dir: Path, start: str | None, end: str | None):
    """
    Build a list of (timestamp, real_path, pred_path) sorted by timestamp.
    Restrict to timestamps between start and end (inclusive) if provided.
    """
    real_map = scan_directory_for_tiffs(real_dir)
    pred_map = scan_directory_for_tiffs(pred_dir)

    common_ts = sorted(set(real_map.keys()) & set(pred_map.keys()))
    logger.info("Found %d common timestamps between real and pred.", len(common_ts))

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

    pairs = [(ts, real_map[ts], pred_map[ts]) for ts in common_ts]
    return pairs


def compute_global_range(arrays: list[np.ndarray]):
    """
    Compute a NaN/inf-safe global min and max over a list of arrays.
    Returns (vmin, vmax).
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
            "Global vmin >= vmax (vmin=%.3f, vmax=%.3f). Expanding range artificially.",
            global_vmin,
            global_vmax,
        )
        global_vmax = global_vmin + 1e-6

    logger.info("Global data range: vmin=%.3f, vmax=%.3f", global_vmin, global_vmax)
    return global_vmin, global_vmax


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_sequence(
    pairs: list[tuple[str, Path, Path]],
    save_path: Path | None,
    title: str | None,
    cmap,
    norm,
):
    """
    Plot a sequence of frame pairs (real vs pred) as a 2-column figure.
    Each row corresponds to a timestamp.

    If norm is provided (e.g. from a palette), it is used.
    Otherwise global vmin/vmax are computed and used.
    """
    logger.info("Sequence mode: plotting %d frame pairs.", len(pairs))

    # First pass: read data and cache them
    cached = []  # list of (ts, real_data, pred_data)
    all_arrays = []

    for ts, real_path, pred_path in pairs:
        real_data = read_tiff(real_path)
        pred_data = read_tiff(pred_path)
        cached.append((ts, real_data, pred_data))
        all_arrays.append(real_data)
        all_arrays.append(pred_data)

    # If no palette norm is provided, compute a global vmin/vmax
    if norm is None:
        global_vmin, global_vmax = compute_global_range(all_arrays)
    else:
        global_vmin = global_vmax = None  # not used

    nrows = len(cached)
    ncols = 2

    fig_width = 8
    fig_height = max(3, 2 * nrows)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fig_width, fig_height),
        squeeze=False,
    )

    # Plot each pair
    last_im = None
    for row_idx, (ts, real_data, pred_data) in enumerate(cached):
        ax_real = axes[row_idx, 0]
        ax_pred = axes[row_idx, 1]

        if norm is not None:
            im_real = ax_real.imshow(real_data, cmap=cmap, norm=norm)
            im_pred = ax_pred.imshow(pred_data, cmap=cmap, norm=norm)
        else:
            im_real = ax_real.imshow(
                real_data, cmap=cmap, vmin=global_vmin, vmax=global_vmax
            )
            im_pred = ax_pred.imshow(
                pred_data, cmap=cmap, vmin=global_vmin, vmax=global_vmax
            )

        ax_real.set_title(f"Real {ts}")
        ax_pred.set_title(f"Pred {ts}")

        ax_real.axis("off")
        ax_pred.axis("off")

        last_im = im_pred  # any one image is fine for the colorbar handle

    # Global title
    if title:
        fig.suptitle(title, fontsize=14)

    # Colorbar
    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), orientation="vertical")
        if hasattr(norm, "boundaries"):
            # Try to deduce label from palette; otherwise generic
            label = getattr(norm, "label", None)
        else:
            label = None
        if label:
            cbar.set_label(label)

    # Layout: avoid tight_layout incompatibility warnings by using subplots_adjust
    fig.subplots_adjust(top=0.92, hspace=0.3, wspace=0.1)

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
        description="Compare real vs predicted radar TIFF/GeoTIFF sequences."
    )
    parser.add_argument(
        "--real-dir",
        required=True,
        help="Directory containing ground-truth TIFF/GeoTIFF frames.",
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
        help="Start timestamp (YYYYMMDDZhhmm). Inclusive.",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End timestamp (YYYYMMDDZhhmm). Inclusive.",
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
        help="Path to save the output figure (e.g., out.png). If omitted, shows interactively.",
    )
    parser.add_argument(
        "--palette",
        type=str,
        default=None,
        help="Path to palette JSON file (levels + colors). If omitted, uses 'viridis'.",
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

    real_dir = Path(args.real_dir).resolve()
    pred_dir = Path(args.pred_dir).resolve()

    if not real_dir.is_dir():
        logger.error("real-dir does not exist or is not a directory: %s", real_dir)
        raise SystemExit(1)
    if not pred_dir.is_dir():
        logger.error("pred-dir does not exist or is not a directory: %s", pred_dir)
        raise SystemExit(1)

    # Palette or default colormap
    if args.palette is not None:
        cmap, norm, label = load_palette(Path(args.palette))
        # Attach label to norm so we can retrieve it later for the colorbar
        if label:
            setattr(norm, "label", label)
    else:
        cmap = plt.get_cmap("viridis")
        norm = None

    pairs = build_pairs(real_dir, pred_dir, args.start, args.end)
    if not pairs:
        raise SystemExit(1)

    save_path = Path(args.save).resolve() if args.save else None
    plot_sequence(
        pairs=pairs,
        save_path=save_path,
        title=args.title,
        cmap=cmap,
        norm=norm,
    )


if __name__ == "__main__":
    main()
