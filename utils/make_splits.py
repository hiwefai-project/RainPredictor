#!/usr/bin/env python3
import os
import argparse
import random
import logging
import shutil
from pathlib import Path
from collections import Counter
from datetime import datetime, timedelta
from tqdm import tqdm


# -------------------------------------------------------------
# Configure logging: console + optional file
# -------------------------------------------------------------
def configure_logging(log_file=None, level=logging.INFO):
    # Create logger
    logger = logging.getLogger("dataset_splitter")
    logger.setLevel(level)

    # Avoid adding multiple handlers if configure_logging is called twice
    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_formatter = logging.Formatter("%(levelname)s - %(message)s")
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # Optional file handler
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s"
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
    else:
        # If we already have handlers and a log_file is requested, add it
        if log_file and not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s"
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

    return logger


# -------------------------------------------------------------
# Collect all files matching prefix/suffix with a scan progress bar
# -------------------------------------------------------------
def collect_files(base_path, prefix, suffix, logger):
    collected = []

    # Progress bar with unknown total; we update by one per scanned file
    with tqdm(desc="Scanning dataset", unit="files") as pbar:
        for root, dirs, files in os.walk(base_path):
            for f in files:
                pbar.update(1)
                if f.startswith(prefix) and f.endswith(suffix):
                    collected.append(Path(root) / f)

    logger.info(
        f"Collected {len(collected)} files matching prefix='{prefix}' suffix='{suffix}'."
    )
    return collected


# -------------------------------------------------------------
# Detect missing frames:
#  - Build expected 10-minute time series between min/max timestamp
#  - Report gaps and daily/hourly missing-slot summaries
# -------------------------------------------------------------
def detect_missing_files(files, prefix, logger):
    timestamps = []

    # Extract timestamps from filenames
    for p in files:
        # Expected name pattern segment after prefix: YYYYMMDDZhhmm
        # Example: rdr0_d01_20250610Z1530_VMI.tiff -> 20250610Z1530
        try:
            ts_str = p.name.replace(prefix, "").split("_")[0]
            ts = datetime.strptime(ts_str, "%Y%m%dZ%H%M")
            timestamps.append(ts)
        except Exception:
            logger.warning(f"Skipping malformed filename: {p.name}")

    if not timestamps:
        logger.warning("No valid timestamps found; skipping missing-data analysis.")
        return {
            "gaps": [],
            "missing_timestamps": [],
            "daily": Counter(),
            "hourly": Counter(),
        }

    # Sort and deduplicate timestamps
    timestamps = sorted(set(timestamps))
    first = timestamps[0]
    last = timestamps[-1]

    # Build full expected 10-minute time series from first to last
    expected = []
    cur = first
    while cur <= last:
        expected.append(cur)
        cur += timedelta(minutes=10)

    existing_set = set(timestamps)
    missing_ts = [t for t in expected if t not in existing_set]

    if missing_ts:
        logger.warning(f"Detected {len(missing_ts)} missing 10-minute slots in dataset.")
    else:
        logger.info("No missing frames detected in the 10-minute time series.")

    # Build gap list from missing timestamps (consecutive groups)
    gaps = []
    if missing_ts:
        start_gap = missing_ts[0]
        prev = missing_ts[0]
        for t in missing_ts[1:]:
            if (t - prev) > timedelta(minutes=10):
                # gap ended at 'prev'
                gaps.append((start_gap, prev))
                start_gap = t
            prev = t
        # last gap
        gaps.append((start_gap, prev))

    # Daily and hourly aggregates
    daily_counter = Counter(t.date() for t in missing_ts)
    hourly_counter = Counter((t.date(), t.hour) for t in missing_ts)

    # Log daily summary
    if daily_counter:
        logger.warning("Daily missing-frame summary:")
        for day, count in sorted(daily_counter.items()):
            logger.warning(f"  Missing frames on {day}: {count} slots of 10 min")

    # Log hourly summary
    if hourly_counter:
        logger.warning("Hourly missing-frame summary:")
        for (day, hour), count in sorted(hourly_counter.items()):
            logger.warning(
                f"  {day} hour {hour:02d}: {count} missing slots (10-min each)"
            )

    # Log human-readable gap ranges
    if gaps:
        logger.warning("Temporal gaps in dataset:")
        for a, b in gaps:
            logger.warning(f"  Gap from {a} to {b}")

    return {
        "gaps": gaps,
        "missing_timestamps": missing_ts,
        "daily": daily_counter,
        "hourly": hourly_counter,
    }


# -------------------------------------------------------------
# Ensure train/val/test directories exist
# -------------------------------------------------------------
def create_split_dirs(out_dir, logger):
    split_dirs = {
        "train": Path(out_dir) / "train",
        "val": Path(out_dir) / "val",
        "test": Path(out_dir) / "test",
    }

    for d in split_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output split directories created under {out_dir}.")
    return split_dirs


# -------------------------------------------------------------
# Populate a split directory:
#   - symlink mode (default) or copy mode (--copy)
# -------------------------------------------------------------
def populate_split(file_list, out_dir, logger, copy_mode=False):
    mode_str = "copy" if copy_mode else "symlink"
    logger.info(f"Populating '{out_dir.name}' using {mode_str} mode.")

    for f in tqdm(file_list, desc=f"Populating {out_dir.name}", unit="files"):
        target = out_dir / f.name

        # Remove old file/link if exists
        if target.exists() or target.is_symlink():
            try:
                target.unlink()
            except Exception as e:
                logger.error(f"Failed to remove existing target {target}: {e}")
                continue

        # Create copy or symlink
        try:
            if copy_mode:
                shutil.copy2(f, target)
            else:
                target.symlink_to(f)
        except Exception as e:
            logger.error(f"Failed to create {mode_str} for {f} → {target}: {e}")


# -------------------------------------------------------------
# Main program
# -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Split radar dataset into train/val/test subsets."
    )

    # Root of dataset
    parser.add_argument(
        "--base-path",
        type=str,
        required=True,
        help="Root folder of the dataset (e.g., data/dataset/rdr0/)",
    )

    # Filename prefix and suffix
    parser.add_argument(
        "--prefix",
        type=str,
        default="rdr0_d01_",
        help="Filename prefix, e.g. 'rdr0_d01_'",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_VMI.tiff",
        help="Filename suffix, e.g. '_VMI.tiff'",
    )

    # Output directory for splits
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for train/val/test splits.",
    )

    # Ratios for train/val/test
    parser.add_argument(
        "--ratios",
        nargs=3,
        type=float,
        default=[0.9, 0.05, 0.05],
        help="Train/Val/Test ratios (must sum to 1).",
    )

    # Random seed for reproducibility
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling.",
    )

    # Copy mode flag
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of creating symlinks.",
    )

    # Optional log file
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Optional log file path.",
    )

    args = parser.parse_args()

    # Validate ratios
    if not abs(sum(args.ratios) - 1.0) < 1e-6:
        raise ValueError("Train/Val/Test ratios must sum to 1.0")

    # Initialize logger
    logger = configure_logging(log_file=args.log_file)

    logger.info("Starting dataset split process...")
    logger.info(f"Base path: {args.base_path}")
    logger.info(f"Output path: {args.output}")
    logger.info(f"Prefix: {args.prefix}  Suffix: {args.suffix}")
    logger.info(f"Ratios (train/val/test): {args.ratios}")
    logger.info(f"Copy mode: {'ON' if args.copy else 'OFF'}")

    base_path = Path(args.base_path)
    out_dir = Path(args.output)

    # Step 1 — Collect files (with scan progress bar)
    files = collect_files(base_path, args.prefix, args.suffix, logger)
    files = sorted(files)

    if not files:
        logger.error("No files found. Exiting.")
        return

    # Step 2 — Missing data detection + daily/hourly summary
    missing_info = detect_missing_files(files, args.prefix, logger)

    # Step 3 — Shuffle and split
    random.seed(args.seed)
    random.shuffle(files)

    n = len(files)
    n_train = int(args.ratios[0] * n)
    n_val = int(args.ratios[1] * n)

    train_files = files[:n_train]
    val_files = files[n_train : n_train + n_val]
    test_files = files[n_train + n_val :]

    logger.info(f"Total files: {n}")
    logger.info(f"Train set: {len(train_files)} files")
    logger.info(f"Validation set: {len(val_files)} files")
    logger.info(f"Test set: {len(test_files)} files")

    # Step 4 — Prepare output dirs
    split_dirs = create_split_dirs(out_dir, logger)

    # Step 5 — Populate splits (symlinks or copies)
    populate_split(train_files, split_dirs["train"], logger, copy_mode=args.copy)
    populate_split(val_files, split_dirs["val"], logger, copy_mode=args.copy)
    populate_split(test_files, split_dirs["test"], logger, copy_mode=args.copy)

    logger.info("Dataset split completed successfully.")


# -------------------------------------------------------------
# Entry point
# -------------------------------------------------------------
if __name__ == "__main__":
    main()

