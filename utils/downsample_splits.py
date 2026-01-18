#!/usr/bin/env python3
"""Downsample dataset splits by stride while logging progress."""  # Describe script purpose.

import argparse  # Parse command-line arguments.
import logging  # Emit structured logs.
import math  # Provide rounding utilities.
import os  # Interact with the filesystem.
import shutil  # Provide file copy fallback when symlinks fail.


def setup_logging() -> logging.Logger:
    """Configure logging and return the module logger."""  # Document logging setup helper.
    logging.basicConfig(  # Initialize root logging configuration.
        level=logging.INFO,  # Default to INFO verbosity for progress logs.
        format="%(asctime)s [%(levelname)s] %(message)s",  # Use a readable log format.
    )
    return logging.getLogger(__name__)  # Return a module-scoped logger instance.


def gather_sorted(root: str) -> list[str]:
    """Collect TIFF paths from a directory tree, sorted lexicographically."""  # Explain behavior.
    files: list[str] = []  # Accumulate file paths in a list.
    for cur, _, fnames in os.walk(root, followlinks=True):  # Walk the directory tree.
        for fname in fnames:  # Inspect each file name.
            if fname.lower().endswith((".tif", ".tiff")):  # Filter for TIFF files.
                files.append(os.path.join(cur, fname))  # Store the full path.
    files.sort()  # Sort paths for deterministic ordering.
    return files  # Return the sorted file list.


def take_stride(files: list[str], target: int) -> list[str]:
    """Select a stride-based subset aiming for a target count."""  # Explain downsampling policy.
    if target >= len(files):  # Check if we already have fewer than the target.
        return files[:]  # Return a shallow copy of the full list.
    stride = max(1, len(files) // target)  # Compute stride to approximate target size.
    picked = files[::stride]  # Take every Nth element.
    return picked[:target]  # Trim to exact target count.


def link_preserve(src_root: str, dst_root: str, files: list[str]) -> None:
    """Mirror file selection into destination, preserving relative paths."""  # Describe linking behavior.
    for src in files:  # Iterate through chosen source files.
        rel = os.path.relpath(src, src_root)  # Compute relative path to maintain structure.
        dst = os.path.join(dst_root, rel)  # Build destination path.
        os.makedirs(os.path.dirname(dst), exist_ok=True)  # Ensure destination directory exists.
        try:  # Attempt to create a symlink first.
            if not os.path.exists(dst):  # Avoid overwriting existing outputs.
                os.symlink(src, dst)  # Create a symlink to save space.
        except Exception:  # Fall back if symlinks are not permitted.
            shutil.copy2(src, dst)  # Copy file while preserving metadata.


def count_tiffs(dst: str) -> int:
    """Count TIFF files in a destination subtree."""  # Explain counting helper.
    count = 0  # Initialize the file counter.
    for cur, _, fnames in os.walk(dst):  # Traverse the destination directory.
        for fname in fnames:  # Inspect each file name.
            if fname.lower().endswith((".tif", ".tiff")):  # Check for TIFF extension.
                count += 1  # Increment the counter for each TIFF.
    return count  # Return the final count.


def main() -> None:
    """Parse arguments, downsample splits, and log results."""  # Summarize script workflow.
    logger = setup_logging()  # Initialize logging for the script.
    parser = argparse.ArgumentParser()  # Create a CLI parser.
    parser.add_argument(  # Add the source split path argument.
        "--src",  # Define the CLI flag name.
        required=True,  # Require the argument.
        help="cartella split sorgente (con train/val/test)",  # Describe the argument.
    )
    parser.add_argument(  # Add the destination path argument.
        "--dst",  # Define the CLI flag name.
        required=True,  # Require the argument.
        help="cartella split destinazione ridotti",  # Describe the argument.
    )
    parser.add_argument(  # Add target total count argument.
        "--total",  # Define the CLI flag name.
        type=int,  # Parse as integer.
        default=5000,  # Use default total of 5000 images.
        help="totale desiderato (default 5000)",  # Describe the argument.
    )
    args = parser.parse_args()  # Parse CLI arguments into a namespace.

    t_train = int(round(args.total * 0.70))  # Compute train target count (70%).
    t_val = int(round(args.total * 0.15))  # Compute validation target count (15%).
    t_test = args.total - t_train - t_val  # Compute test target count (remaining).

    logger.info(  # Log the computed target counts.
        "Target: total=%s -> train=%s, val=%s, test=%s",  # Log message template.
        args.total,  # Log total target.
        t_train,  # Log train target.
        t_val,  # Log validation target.
        t_test,  # Log test target.
    )

    src_train = os.path.join(args.src, "train")  # Build source train directory.
    src_val = os.path.join(args.src, "val")  # Build source validation directory.
    src_test = os.path.join(args.src, "test")  # Build source test directory.

    dst_train = os.path.join(args.dst, "train")  # Build destination train directory.
    dst_val = os.path.join(args.dst, "val")  # Build destination validation directory.
    dst_test = os.path.join(args.dst, "test")  # Build destination test directory.

    for dst in (dst_train, dst_val, dst_test):  # Iterate destination directories.
        os.makedirs(dst, exist_ok=True)  # Create each destination directory if missing.

    f_train = gather_sorted(src_train)  # Collect sorted train TIFFs.
    f_val = gather_sorted(src_val)  # Collect sorted validation TIFFs.
    f_test = gather_sorted(src_test)  # Collect sorted test TIFFs.

    logger.info(  # Log the counts found in the source split directories.
        "Sorgente: train=%s, val=%s, test=%s",  # Log message template.
        len(f_train),  # Log number of train files found.
        len(f_val),  # Log number of validation files found.
        len(f_test),  # Log number of test files found.
    )

    pick_train = take_stride(f_train, t_train)  # Select train subset by stride.
    pick_val = take_stride(f_val, t_val)  # Select validation subset by stride.
    pick_test = take_stride(f_test, t_test)  # Select test subset by stride.

    logger.info(  # Log the counts selected for each split.
        "Selezionati: train=%s, val=%s, test=%s",  # Log message template.
        len(pick_train),  # Log selected train count.
        len(pick_val),  # Log selected validation count.
        len(pick_test),  # Log selected test count.
    )

    link_preserve(src_train, dst_train, pick_train)  # Mirror train files.
    link_preserve(src_val, dst_val, pick_val)  # Mirror validation files.
    link_preserve(src_test, dst_test, pick_test)  # Mirror test files.

    logger.info(  # Log final counts in the destination directories.
        "Conteggi finali (dest): train=%s, val=%s, test=%s",  # Log message template.
        count_tiffs(dst_train),  # Count train files in destination.
        count_tiffs(dst_val),  # Count validation files in destination.
        count_tiffs(dst_test),  # Count test files in destination.
    )


if __name__ == "__main__":
    main()  # Run main when executed as a script.
