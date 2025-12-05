#!/usr/bin/env python
# -*- coding: utf-8 -*-
# The above shebang and encoding line make the script executable from the shell
# and ensure correct handling of UTF-8 characters in titles and labels.

import argparse              # Import argparse to parse command-line arguments.
import os                    # Import os for filesystem path manipulations.
from typing import List, Tuple  # Import typing helpers for better code clarity.

import numpy as np           # Import NumPy for numerical array operations.
import matplotlib.pyplot as plt  # Import Matplotlib for plotting.
import rasterio              # Import rasterio for reading GeoTIFF files.
from rasterio.transform import xy  # Import helper to convert indices to coordinates.


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def read_geotiff(path: str) -> Tuple[np.ndarray, dict]:
    """
    Read a GeoTIFF file and return the image array and raster metadata.

    Parameters
    ----------
    path : str
        Path to the GeoTIFF file.

    Returns
    -------
    data : np.ndarray
        2D array with the raster values (first band).
    meta : dict
        Raster metadata dictionary containing CRS, transform, etc.
    """
    # Open the GeoTIFF using rasterio's context manager.
    with rasterio.open(path) as ds:
        # Read the first band (index 1) as a 2D NumPy array.
        data = ds.read(1)
        # Copy the dataset metadata to a Python dictionary.
        meta = ds.meta.copy()
    # Return both the data array and the metadata dictionary.
    return data, meta


def compute_extent(meta: dict) -> Tuple[float, float, float, float]:
    """
    Compute the plotting extent (min_x, max_x, min_y, max_y) in coordinate
    space from a raster metadata dictionary.

    Parameters
    ----------
    meta : dict
        Raster metadata containing at least 'transform', 'height', and 'width'.

    Returns
    -------
    extent : tuple
        (min_x, max_x, min_y, max_y) suitable for Matplotlib's imshow.
    """
    # Extract the affine transform from the metadata.
    transform = meta["transform"]
    # Extract the raster height (rows) and width (columns).
    height = meta["height"]
    width = meta["width"]

    # Compute the coordinates of the upper-left pixel (row=0, col=0).
    x_min, y_max = xy(transform, 0, 0)
    # Compute the coordinates of the lower-right pixel (row=height, col=width).
    # Note: using height and width instead of height-1, width-1 is fine here
    # to cover the full raster in plotting extent.
    x_max, y_min = xy(transform, height, width)

    # Return the extent as (min_x, max_x, min_y, max_y).
    return (x_min, x_max, y_min, y_max)


def build_sequence_pairs(
    real_dir: str,
    pred_dir: str,
    start: str = None,
    end: str = None,
) -> List[Tuple[str, str, str]]:
    """
    Build a sorted list of (timestamp, real_path, pred_path) pairs from two
    directories, optionally restricted by a time interval.

    The function assumes that both real and predicted frames use filenames
    that contain a timestamp substring of the form 'YYYYMMDDZhhmm', as in:
      rdr0_d01_20251202Z1510_VMI.tiff

    Parameters
    ----------
    real_dir : str
        Directory containing the reference (observed) frames.
    pred_dir : str
        Directory containing the predicted frames.
    start : str, optional
        Start timestamp filter in the form 'YYYYMMDDZhhmm' (inclusive).
    end : str, optional
        End timestamp filter in the form 'YYYYMMDDZhhmm' (inclusive).

    Returns
    -------
    pairs : list of (timestamp, real_path, pred_path)
        Sorted list of matching timestamped file pairs that satisfy the
        optional time interval.
    """
    # Normalize input directories by expanding '~' and making them absolute.
    real_dir = os.path.abspath(os.path.expanduser(real_dir))
    pred_dir = os.path.abspath(os.path.expanduser(pred_dir))

    # Helper function to extract the timestamp token "YYYYMMDDZhhmm"
    # from a filename. Returns None if it cannot be found.
    def extract_timestamp(name: str) -> str:
        # Split the filename by underscores and search for the token
        # that contains 'Z' and has the expected length (13).
        parts = name.split("_")
        for token in parts:
            if "Z" in token and len(token) >= 13:
                # Return exactly the first 13 characters (YYYYMMDDZhhmm).
                return token[:13]
        # If no valid token is found, return None.
        return None

    # Dictionary mapping timestamp -> real file path.
    real_map = {}
    # Iterate over all entries in the real directory.
    for fname in sorted(os.listdir(real_dir)):
        # Build the absolute path.
        full = os.path.join(real_dir, fname)
        # Skip directories and non-files.
        if not os.path.isfile(full):
            continue
        # Try to extract a timestamp from the file name.
        ts = extract_timestamp(fname)
        # If successful, store the mapping.
        if ts is not None:
            real_map[ts] = full

    # Dictionary mapping timestamp -> predicted file path.
    pred_map = {}
    # Iterate over all entries in the predicted directory.
    for fname in sorted(os.listdir(pred_dir)):
        # Build the absolute path.
        full = os.path.join(pred_dir, fname)
        # Skip directories and non-files.
        if not os.path.isfile(full):
            continue
        # Try to extract a timestamp from the file name.
        ts = extract_timestamp(fname)
        # If successful, store the mapping.
        if ts is not None:
            pred_map[ts] = full

    # Compute the intersection of timestamps present in both maps.
    common_ts = sorted(set(real_map.keys()) & set(pred_map.keys()))

    # If a start timestamp is provided, filter out earlier timestamps.
    if start is not None:
        common_ts = [ts for ts in common_ts if ts >= start]
    # If an end timestamp is provided, filter out later timestamps.
    if end is not None:
        common_ts = [ts for ts in common_ts if ts <= end]

    # Build the final list of (timestamp, real_path, pred_path) pairs.
    pairs: List[Tuple[str, str, str]] = []
    for ts in common_ts:
        pairs.append((ts, real_map[ts], pred_map[ts]))

    # Return the sorted list of matching pairs.
    return pairs


# -----------------------------------------------------------------------------
# Plotting functions
# -----------------------------------------------------------------------------

def plot_single_pair(
    real_path: str,
    pred_path: str,
    title: str = None,
    save_path: str = None,
) -> None:
    """
    Plot a single pair of frames (real vs predicted) side-by-side.

    Parameters
    ----------
    real_path : str
        Path to the reference (observed) GeoTIFF file.
    pred_path : str
        Path to the predicted GeoTIFF file.
    title : str, optional
        Overall figure title.
    save_path : str, optional
        If provided, save the figure as a PNG to this path instead of
        opening an interactive window.
    """
    # Read the real frame and its metadata from disk.
    real_data, real_meta = read_geotiff(real_path)
    # Read the predicted frame and its metadata from disk.
    pred_data, pred_meta = read_geotiff(pred_path)

    # Compute the plotting extent from the real frame metadata.
    # We assume real and predicted frames share the same grid.
    extent = compute_extent(real_meta)

    # Compute global color limits using both arrays to have a consistent scale.
    vmin = min(real_data.min(), pred_data.min())
    vmax = max(real_data.max(), pred_data.max())

    # Create a Matplotlib figure with two subplots side-by-side.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot the real frame in the left subplot.
    im0 = axes[0].imshow(
        real_data,
        origin="upper",
        extent=extent,
        vmin=vmin,
        vmax=vmax,
    )
    # Set axis labels for the left subplot.
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    axes[0].set_title("Observed")

    # Plot the predicted frame in the right subplot.
    im1 = axes[1].imshow(
        pred_data,
        origin="upper",
        extent=extent,
        vmin=vmin,
        vmax=vmax,
    )
    # Set axis labels for the right subplot.
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Latitude")
    axes[1].set_title("Predicted")

    # Add a colorbar that is shared between the two subplots.
    fig.colorbar(im1, ax=axes.ravel().tolist(), label="Reflectivity (dBZ)")

    # If a global title is provided, set it as the figure suptitle.
    if title:
        fig.suptitle(title, fontsize=14)

    # Adjust layout to avoid overlapping labels and titles.
    plt.tight_layout()

    # If a save path is given, save the figure instead of showing it.
    if save_path is not None:
        # Ensure the target directory exists.
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        # Save the figure as a PNG, with reasonable DPI and tight bounding box.
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        # Close the figure to free memory when running in batch mode.
        plt.close(fig)
    else:
        # Otherwise, display the figure interactively.
        plt.show()


def plot_sequence(
    pairs: List[Tuple[str, str, str]],
    title: str = None,
    save_path: str = None,
) -> None:
    """
    Plot a time sequence of real vs predicted frames.

    The resulting figure has one row per timestamp and two columns:
    - left column: observed frame
    - right column: predicted frame

    Parameters
    ----------
    pairs : list of (timestamp, real_path, pred_path)
        List of sorted frame pairs produced by build_sequence_pairs().
    title : str, optional
        Overall figure title.
    save_path : str, optional
        If provided, save the figure as a PNG to this path instead of
        opening an interactive window.
    """
    # If there are no pairs to plot, simply return without doing anything.
    if not pairs:
        print("[compare] No matching frames found to plot.")
        return

    # Read the first real frame to get metadata and extent.
    first_real_data, first_meta = read_geotiff(pairs[0][1])
    extent = compute_extent(first_meta)

    # Prepare lists to track global min and max values across all frames.
    global_vmin = float("inf")
    global_vmax = float("-inf")

    # First pass: read all frames and cache them to compute global color limits.
    cached_data = []  # Each entry: (timestamp, real_data, pred_data)
    for ts, real_path, pred_path in pairs:
        # Read the real and predicted frames for this timestamp.
        real_data, _ = read_geotiff(real_path)
        pred_data, _ = read_geotiff(pred_path)
        # Update global min and max using both frames.
        local_min = min(real_data.min(), pred_data.min())
        local_max = max(real_data.max(), pred_data.max())
        global_vmin = min(global_vmin, local_min)
        global_vmax = max(global_vmax, local_max)
        # Store data in the cache list for a second plotting pass.
        cached_data.append((ts, real_data, pred_data))

    # Determine the number of rows in the figure from the number of pairs.
    n_rows = len(cached_data)
    # Create a Matplotlib figure with n_rows rows and 2 columns.
    fig, axes = plt.subplots(
        n_rows,
        2,
        figsize=(12, 4 * n_rows),
        squeeze=False,  # Force axes to always be 2D array.
    )

    # Second pass: plot each pair using the global color limits.
    for row_idx, (ts, real_data, pred_data) in enumerate(cached_data):
        # Access the subplot for the observed frame in this row.
        ax_real = axes[row_idx, 0]
        # Access the subplot for the predicted frame in this row.
        ax_pred = axes[row_idx, 1]

        # Plot the observed frame.
        im_real = ax_real.imshow(
            real_data,
            origin="upper",
            extent=extent,
            vmin=global_vmin,
            vmax=global_vmax,
        )
        # Label the axes for the observed frame.
        ax_real.set_xlabel("Longitude")
        ax_real.set_ylabel("Latitude")
        # Set the subplot title including the timestamp.
        ax_real.set_title(f"Observed @ {ts}")

        # Plot the predicted frame.
        im_pred = ax_pred.imshow(
            pred_data,
            origin="upper",
            extent=extent,
            vmin=global_vmin,
            vmax=global_vmax,
        )
        # Label the axes for the predicted frame.
        ax_pred.set_xlabel("Longitude")
        ax_pred.set_ylabel("Latitude")
        # Set the subplot title including the timestamp.
        ax_pred.set_title(f"Predicted @ {ts}")

    # Add a shared colorbar for all subplots, using the last image handle.
    fig.colorbar(im_pred, ax=axes.ravel().tolist(), label="Reflectivity (dBZ)")

    # If a global figure title is provided, set it as the suptitle.
    if title:
        fig.suptitle(title, fontsize=16)

    # Adjust layout so that titles, labels, and colorbar are not overlapping.
    plt.tight_layout(rect=(0, 0, 1, 0.97))

    # If a save path is given, save the figure as a PNG file.
    if save_path is not None:
        # Ensure the output directory exists.
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        # Save the figure with tight bounding box and reasonable DPI.
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        # Close the figure after saving to release resources.
        plt.close(fig)
    else:
        # Otherwise, show the interactive figure on screen.
        plt.show()


# -----------------------------------------------------------------------------
# Command-line interface
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the compare.py script.

    Returns
    -------
    args : argparse.Namespace
        The parsed arguments.
    """
    # Create the argument parser with a brief description.
    parser = argparse.ArgumentParser(
        description=(
            "Compare radar GeoTIFF frames (real vs predicted) in lon/lat. "
            "Supports single-frame comparison or a sequence over a time interval."
        )
    )

    # Argument for a single real (input) frame path.
    parser.add_argument(
        "--input",
        type=str,
        help="Path to a single observed GeoTIFF frame.",
    )

    # Argument for a single predicted frame path.
    parser.add_argument(
        "--pred",
        type=str,
        help="Path to a single predicted GeoTIFF frame.",
    )

    # Optional title for the figure.
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional title for the figure.",
    )

    # Directory containing real frames for sequence comparison.
    parser.add_argument(
        "--real-dir",
        type=str,
        default=None,
        help=(
            "Directory with observed frames for sequence comparison. "
            "If provided together with --pred-dir, sequence mode is used."
        ),
    )

    # Directory containing predicted frames for sequence comparison.
    parser.add_argument(
        "--pred-dir",
        type=str,
        default=None,
        help=(
            "Directory with predicted frames for sequence comparison. "
            "Must be used together with --real-dir."
        ),
    )

    # Optional start timestamp for filtering the sequence.
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help=(
            "Start timestamp for sequence filtering, in the form 'YYYYMMDDZhhmm'. "
            "Only frames with timestamps >= start are included."
        ),
    )

    # Optional end timestamp for filtering the sequence.
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help=(
            "End timestamp for sequence filtering, in the form 'YYYYMMDDZhhmm'. "
            "Only frames with timestamps <= end are included."
        ),
    )

    # Optional path to save the figure instead of opening a window.
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help=(
            "If provided, save the resulting figure as a PNG to this path "
            "instead of displaying it on the screen."
        ),
    )

    # Parse and return all arguments.
    return parser.parse_args()


def main() -> None:
    """
    Main entry point for the compare.py script.
    """
    # Parse the command-line arguments.
    args = parse_args()

    # Decide whether to run in single-pair mode or sequence mode.
    sequence_mode = args.real_dir is not None or args.pred_dir is not None

    # If sequence_mode is requested, ensure both directories are provided.
    if sequence_mode:
        if args.real_dir is None or args.pred_dir is None:
            # If only one of the directories is defined, print an error message.
            raise ValueError(
                "Both --real-dir and --pred-dir must be provided for sequence mode."
            )

        # Build the list of matching (timestamp, real_path, pred_path) pairs.
        pairs = build_sequence_pairs(
            real_dir=args.real_dir,
            pred_dir=args.pred_dir,
            start=args.start,
            end=args.end,
        )

        # If there are no pairs, inform the user and exit early.
        if not pairs:
            print("[compare] No matching real/predicted frames found "
                  "in the given directories and time interval.")
            return

        # Print a short log of how many frames will be plotted.
        print(f"[compare] Sequence mode: plotting {len(pairs)} frame pairs.")

        # Plot the full sequence using the dedicated plotting function.
        plot_sequence(
            pairs=pairs,
            title=args.title,
            save_path=args.save,
        )
    else:
        # In single-pair mode, both --input and --pred must be provided.
        if args.input is None or args.pred is None:
            raise ValueError(
                "For single-frame comparison, both --input and --pred must be provided. "
                "To compare a sequence, use --real-dir and --pred-dir."
            )

        # Print a short log indicating that single-pair mode is being used.
        print(f"[compare] Single-frame mode: {args.input} vs {args.pred}")

        # Plot the single pair or save it depending on --save.
        plot_single_pair(
            real_path=args.input,
            pred_path=args.pred,
            title=args.title,
            save_path=args.save,
        )


# Standard Python idiom to allow running the script directly.
if __name__ == "__main__":
    main()
