#!/usr/bin/env python3
"""
radar_info.py

Reads a GeoTIFF VMI radar reflectivity image and logs:
- Maximum VMI (dBZ)
- Average VMI (dBZ)
- Count of pixels with VMI greater than a configurable threshold

Usage:
    python radar_info.py radar_image.tiff --threshold 15
"""

import sys
import argparse
import numpy as np
import rasterio
import logging

# ------------------------------------------------------
# Configure logging output format and default verbosity
# ------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,                                # default log level
    format="%(asctime)s [%(levelname)s] %(message)s",   # timestamp + level + message
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)                   # create logger instance


def read_vmi_dbz(path):
    """
    Reads radar VMI from a GeoTIFF and converts values to dBZ.
    Handles scale/offset correction if present in metadata.
    """

    logger.info(f"Opening file: {path}")  # Log which file is being processed

    # Open GeoTIFF with rasterio
    with rasterio.open(path) as ds:
        # Read band 1 (VMI). Convert to float for mathematical operations.
        band = ds.read(1).astype("float32")

        # Handle NoData values if the file contains them
        nodata = ds.nodata
        if nodata is not None:
            logger.info(f"Detected NoData value: {nodata}")
            band[band == nodata] = np.nan

        # Read possible scale/offset metadata
        tags = ds.tags()
        scale = tags.get("Scale") or tags.get("scale")
        offset = tags.get("Offset") or tags.get("offset")

        # Convert to float if present
        if scale is not None:
            scale = float(scale)
            logger.info(f"Detected scale factor: {scale}")

        if offset is not None:
            offset = float(offset)
            logger.info(f"Detected offset: {offset}")

        # Apply dBZ conversion if metadata exists
        if scale is not None or offset is not None:
            band = band * (scale if scale else 1.0) + (offset if offset else 0.0)
            logger.info("Applied scale/offset to convert raw values to dBZ")
        else:
            logger.info("No scale/offset found — assuming values are already in dBZ")

        # Return the VMI matrix (in dBZ)
        return band


def parse_arguments():
    """
    Parses command-line arguments.
    """

    parser = argparse.ArgumentParser(description="Compute VMI statistics for a radar GeoTIFF.")

    # Positional argument: path to GeoTIFF
    parser.add_argument("file", help="Path to the VMI radar GeoTIFF")

    # Optional argument: threshold for counting pixels
    parser.add_argument(
        "--threshold",
        type=float,
        default=10.0,
        help="Threshold in dBZ for counting pixels (default: 10 dBZ)",
    )

    return parser.parse_args()


def main():
    # Parse command-line arguments
    args = parse_arguments()

    # Load radar image and convert to dBZ
    vmi = read_vmi_dbz(args.file)

    # Compute maximum (ignoring NaNs)
    max_vmi = np.nanmax(vmi)

    # Compute mean (ignoring NaNs)
    avg_vmi = np.nanmean(vmi)

    # Count pixels above the user-defined threshold
    count_above_threshold = np.sum(vmi > args.threshold)

    # Log results
    logger.info("=== Radar VMI Statistics ===")
    logger.info(f"File: {args.file}")
    logger.info(f"Maximum VMI (dBZ): {max_vmi:.2f}")
    logger.info(f"Average VMI (dBZ): {avg_vmi:.2f}")
    logger.info(f"Pixels with VMI > {args.threshold} dBZ: {int(count_above_threshold)}")


if __name__ == "__main__":
    main()

