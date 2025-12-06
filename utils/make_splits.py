#!/usr/bin/env python3  # Shebang to run this script with Python when executed as a program

# ==========================
# Imports
# ==========================
import os               # Standard library module for interacting with the operating system (file system walking, etc.)
import argparse         # Standard library module for parsing command-line arguments
import random           # Standard library module for random operations (here used for shuffling file lists)
import logging          # Standard library module for logging messages with various severity levels
import csv              # Standard library module for handling CSV files
import shutil           # Standard library module for high-level file operations (copy, remove trees, etc.)
from pathlib import Path        # Provides an object-oriented interface to filesystem paths
from collections import Counter  # Counter is a dict subclass for counting hashable objects
from datetime import datetime, timedelta  # Used for timestamp parsing and time interval calculations
from tqdm import tqdm            # Third-party library providing progress bars for loops


# ==========================
# Logging configuration
# ==========================
def configure_logging(log_file=None, level=logging.INFO):
    """
    Configure the logger used throughout the script.

    Parameters:
        log_file (str or None): Optional path to a file where logs will be written.
        level (int): Logging level (e.g., logging.INFO, logging.DEBUG).

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger("dataset_splitter")  # Create/get a logger named 'dataset_splitter'
    logger.setLevel(level)                          # Set the global logging level for this logger

    # Avoid adding multiple handlers if configure_logging is called more than once
    if not logger.handlers:
        # Create a handler that logs to the console (stderr)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)  # Console handler minimum log level

        # Set formatting for console logs: only level and message
        console_formatter = logging.Formatter("%(levelname)s - %(message)s")
        console_handler.setFormatter(console_formatter)  # Attach formatter to console handler

        # Add the console handler to the logger
        logger.addHandler(console_handler)

        # If a log file path is provided, also log to that file
        if log_file is not None:
            file_handler = logging.FileHandler(log_file)  # Create file handler
            file_handler.setLevel(level)                  # Set file handler log level

            # Define format for file logging: time, level, message
            file_formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s"
            )
            file_handler.setFormatter(file_formatter)     # Attach formatter to file handler

            # Add the file handler to the logger
            logger.addHandler(file_handler)

    # Return the configured logger
    return logger


# ==========================
# Collect files by scanning a directory tree
# ==========================
def collect_files(base_path, prefix, suffix, logger):
    """
    Walk the directory tree under base_path and collect all files whose names
    start with 'prefix' and end with 'suffix'.

    Parameters:
        base_path (Path): Root folder of the dataset.
        prefix (str): Filename prefix to match.
        suffix (str): Filename suffix to match.
        logger (logging.Logger): Logger instance for messages.

    Returns:
        list[Path]: List of matching file paths.
    """
    collected = []  # Initialize an empty list that will contain all matching file paths

    # tqdm provides a progress bar; total is unknown so we let it auto-update
    with tqdm(desc="Scanning dataset", unit="files") as pbar:
        # os.walk recursively traverses the directory tree starting from base_path
        for root, dirs, files in os.walk(base_path):
            # Iterate over each file name found in the current directory
            for f in files:
                pbar.update(1)  # Increment the progress bar for each file encountered
                # Check if the filename matches the required prefix and suffix
                if f.startswith(prefix) and f.endswith(suffix):
                    # Build the full path using root and filename, then convert to Path object
                    collected.append(Path(root) / f)

    # Log how many files were collected
    logger.info(
        f"Collected {len(collected)} files matching prefix='{prefix}' suffix='{suffix}'."
    )
    # Return the list of collected file paths
    return collected


# ==========================
# Load files from a sequences CSV file
# ==========================
def load_files_from_sequences(csv_path, prefix, suffix, logger):
    """
    Load file paths from a CSV file. The CSV must contain a 'path' column.
    Optionally filter each row by filename prefix and suffix.

    Parameters:
        csv_path (Path): Path to the CSV file containing file paths.
        prefix (str): Filename prefix to filter on (optional but used for consistency).
        suffix (str): Filename suffix to filter on.
        logger (logging.Logger): Logger instance for messages.

    Returns:
        list[Path]: List of Path objects from the CSV, filtered by prefix/suffix.
    """
    logger.info(f"Loading file list from sequences CSV: {csv_path}")  # Inform about CSV source
    files = []  # Initialize the list that will store paths from the CSV

    try:
        # Open the CSV file in text mode with default encoding
        with open(csv_path, newline="") as f:
            # DictReader interprets the first line of the CSV as fieldnames
            reader = csv.DictReader(f)

            # Check if the required 'path' column is present in the CSV header
            if "path" not in reader.fieldnames:
                # Raise an error if the column is missing, so the caller can handle it
                raise ValueError(
                    "Sequences CSV must contain a 'path' column with file paths."
                )

            # Iterate over each row in the CSV file
            for row in reader:
                raw_path = row["path"].strip()  # Extract the 'path' field and strip whitespace
                if not raw_path:
                    # Skip empty paths if any
                    continue

                p = Path(raw_path)  # Convert the raw string path into a Path object

                # Extract the filename (last component of the path) for prefix/suffix filtering
                name = p.name
                # If a prefix is specified and filename does not start with it, skip
                if prefix and not name.startswith(prefix):
                    continue
                # If a suffix is specified and filename does not end with it, skip
                if suffix and not name.endswith(suffix):
                    continue

                # If all filters pass, append the path to the list
                files.append(p)
    except Exception as e:
        # Log any error that occurs while reading or processing the CSV
        logger.error(f"Failed to read sequences CSV {csv_path}: {e}")
        # Return an empty list if reading fails
        return []

    # Log the number of files collected from the CSV
    logger.info(f"Collected {len(files)} files from sequences CSV.")
    # Return the list of Path objects
    return files


# ==========================
# Missing data detection and reporting
# ==========================
def detect_missing_files(files, prefix, logger):
    """
    Analyze timestamps extracted from filenames to detect missing frames.

    Filenames are expected to contain a timestamp component after 'prefix' of the form:
        YYYYMMDDZhhmm
    For example: rdr0_d01_20250610Z1530_VMI.tiff → timestamp 20250610Z1530

    The function builds an expected 10-minute time series between the earliest and latest
    timestamps and identifies missing timestamps, gaps, and daily/hourly aggregates.

    Parameters:
        files (list[Path]): List of file paths to analyze.
        prefix (str): Filename prefix before the timestamp portion.
        logger (logging.Logger): Logger instance.

    Returns:
        dict: {
            "gaps": list[(datetime, datetime)],
            "missing_timestamps": list[datetime],
            "daily": Counter,
            "hourly": Counter
        }
    """
    timestamps = []  # Will store all successfully parsed timestamps

    # Loop through each file path to extract and parse its timestamp
    for p in files:
        try:
            # Extract filename from path
            fname = p.name
            # Remove the prefix from the filename, then split by '_' and take the first part
            # The result should look like 'YYYYMMDDZhhmm'
            ts_str = fname.replace(prefix, "").split("_")[0]
            # Parse the timestamp string into a datetime object using the format 'YYYYMMDDZhhmm'
            ts = datetime.strptime(ts_str, "%Y%m%dZ%H%M")
            # Append the parsed datetime to the list
            timestamps.append(ts)
        except Exception:
            # If anything goes wrong (e.g., unexpected filename format), log a warning
            logger.warning(f"Skipping malformed filename: {p.name}")

    # If no timestamps could be parsed, missing-data analysis cannot proceed
    if not timestamps:
        logger.warning("No valid timestamps found; skipping missing-data analysis.")
        # Return empty structures indicating no analysis
        return {
            "gaps": [],
            "missing_timestamps": [],
            "daily": Counter(),
            "hourly": Counter(),
        }

    # Sort timestamps in ascending order and remove duplicates by converting to a set first
    timestamps = sorted(set(timestamps))
    # The earliest timestamp in the dataset
    first = timestamps[0]
    # The latest timestamp in the dataset
    last = timestamps[-1]

    # Build the full list of expected timestamps with 10-minute frequency
    expected = []     # Will store all expected timestamps in the range
    cur = first       # Start from the first timestamp
    while cur <= last:
        expected.append(cur)                 # Add the current timestamp to expected list
        cur += timedelta(minutes=10)         # Move ahead by 10 minutes

    # Convert the actual timestamps to a set for fast membership checks
    existing_set = set(timestamps)
    # Build a list of timestamps that are expected but not present in existing_set
    missing_ts = [t for t in expected if t not in existing_set]

    # Identify contiguous gaps in the missing timestamps list
    gaps = []  # Each gap will be a tuple: (gap_start, gap_end)
    if missing_ts:
        # Initialize the first gap start and previous timestamp
        gap_start = missing_ts[0]
        prev = missing_ts[0]

        # Iterate over the missing timestamps starting from the second element
        for t in missing_ts[1:]:
            # If the difference between current and previous missing timestamp is more than 10 minutes,
            # it means the previous gap ended at 'prev'
            if t - prev > timedelta(minutes=10):
                gaps.append((gap_start, prev))  # Save the gap interval
                gap_start = t                   # Start a new gap at the current missing timestamp
            prev = t                            # Update 'prev' to current timestamp
        # After loop ends, close the last gap
        gaps.append((gap_start, prev))

    # Daily and hourly counters for missing timestamps
    daily_counter = Counter()   # Key: 'YYYY-MM-DD', Value: number of missing slots that day
    hourly_counter = Counter()  # Key: 'YYYY-MM-DD HH:00', Value: number of missing slots in that hour

    # Iterate over each missing timestamp and increment the corresponding counters
    for t in missing_ts:
        day_key = t.strftime("%Y-%m-%d")       # Format for day-level grouping
        hour_key = t.strftime("%Y-%m-%d %H:00")  # Format for hour-level grouping
        daily_counter[day_key] += 1            # Increment daily counter
        hourly_counter[hour_key] += 1          # Increment hourly counter

    # Log summary of missing data analysis
    logger.info("Missing-data analysis:")
    logger.info(f"  Time coverage: {first} to {last}")
    logger.info(f"  Total frames (expected, 10-min steps): {len(expected)}")
    logger.info(f"  Existing frames: {len(timestamps)}")
    logger.info(f"  Missing frames: {len(missing_ts)}")
    logger.info(f"  Number of gaps: {len(gaps)}")

    # Log detailed gap intervals, if any
    if gaps:
        logger.info("  Gaps:")
        for (a, b) in gaps:
            logger.info(f"    from {a} to {b}")

    # Log daily missing statistics in sorted date order
    logger.info("Daily missing-frame counts:")
    for day, count in sorted(daily_counter.items()):
        logger.info(f"  {day}: {count} missing slots")

    # Log hourly missing statistics in sorted hour order
    logger.info("Hourly missing-frame counts:")
    for hour, count in sorted(hourly_counter.items()):
        logger.info(f"  {hour}: {count} missing slots")

    # Return a dictionary summarizing the missing data analysis
    return {
        "gaps": gaps,
        "missing_timestamps": missing_ts,
        "daily": daily_counter,
        "hourly": hourly_counter,
    }


# ==========================
# Prepare train/val/test directories
# ==========================
def create_split_dirs(out_dir, logger):
    """
    Create (or clean) the train/val/test directories within out_dir.

    Parameters:
        out_dir (Path): Root output directory.
        logger (logging.Logger): Logger for messages.

    Returns:
        dict[str, Path]: Mapping from split name ('train', 'val', 'test') to its directory.
    """
    # Dictionary mapping split names to their respective subdirectories
    split_dirs = {
        "train": Path(out_dir) / "train",
        "val": Path(out_dir) / "val",
        "test": Path(out_dir) / "test",
    }

    # Iterate over each split and ensure that its directory exists and is cleaned
    for split_name, split_dir in split_dirs.items():
        if split_dir.exists():
            # If the directory already exists, we clear its contents
            logger.info(
                f"Cleaning existing directory for split '{split_name}': {split_dir}"
            )
            # Iterate over the items in this directory
            for item in split_dir.iterdir():
                if item.is_dir():
                    # If it's a subdirectory, remove it and everything under it
                    shutil.rmtree(item)
                else:
                    # If it's a file (including symlinks), remove it
                    item.unlink()
        else:
            # If the directory does not exist, we create it with parents
            logger.info(f"Creating directory for split '{split_name}': {split_dir}")
            split_dir.mkdir(parents=True, exist_ok=True)

    # Return the dictionary of split directories
    return split_dirs


# ==========================
# Populate a split with files (copy or symlink)
# ==========================
def populate_split(file_list, out_dir, logger, copy_mode=False):
    """
    Populate a given split directory with either symlinks or copies of the files.

    Parameters:
        file_list (list[Path]): List of source files for this split.
        out_dir (Path): Target directory for this split.
        logger (logging.Logger): Logger instance.
        copy_mode (bool): If True, copy files instead of linking them.

    Returns:
        None
    """
    # Choose a human-readable mode string based on copy_mode
    mode_str = "copy" if copy_mode else "symlink"
    logger.info(f"Populating '{out_dir.name}' using {mode_str} mode.")

    # Wrap the loop with tqdm to show a progress bar for this split
    for f in tqdm(file_list, desc=f"Populating {out_dir.name}", unit="files"):
        # Build the target path inside the split directory (same filename)
        target = out_dir / f.name

        # If the target already exists (file or symlink), remove it first
        if target.exists() or target.is_symlink():
            try:
                target.unlink()  # Remove existing file/symlink
            except Exception as e:
                logger.error(f"Failed to remove existing target {target}: {e}")
                # Skip to the next file if we cannot clean the target
                continue

        try:
            # If copy_mode is True, physically copy the file with metadata
            if copy_mode:
                shutil.copy2(f, target)
            else:
                # Otherwise, create a symbolic link from target to source file
                target.symlink_to(f)
        except Exception as e:
            # Log any error that occurs when creating copies or links
            logger.error(f"Failed to create {mode_str} for {f} → {target}: {e}")


# ==========================
# Main function: argument parsing and orchestration
# ==========================
def main():
    """
    Main entry point of the script. Parses arguments, collects files (from a base
    directory or a sequences CSV), detects missing data, splits the dataset into
    train/val/test subsets, and populates output directories with copies/symlinks.
    """
    # Create an ArgumentParser to define command-line options and help text
    parser = argparse.ArgumentParser(
        description="Split radar dataset into train/val/test subsets."
    )

    # Create a mutually exclusive group for --base-path and --sequences
    # Exactly one of these arguments must be provided
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--base-path",
        type=str,
        help="Root folder of the dataset (e.g., data/dataset/rdr0/).",
    )
    group.add_argument(
        "--sequences",
        type=str,
        help=(
            "CSV file with a 'path' column listing the files to be used "
            "for splitting (alternative to --base-path)."
        ),
    )

    # Argument for the filename prefix (used for both scanning and timestamp parsing)
    parser.add_argument(
        "--prefix",
        type=str,
        default="rdr0_d01_",
        help="Filename prefix, e.g. 'rdr0_d01_'",
    )

    # Argument for the filename suffix (last part of the name)
    parser.add_argument(
        "--suffix",
        type=str,
        default="_VMI.tiff",
        help="Filename suffix, e.g. '_VMI.tiff'",
    )

    # Required argument specifying where to store the split files
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for train/val/test splits.",
    )

    # Ratios for train/val/test splits; must sum to 1.0
    parser.add_argument(
        "--ratios",
        nargs=3,
        type=float,
        default=[0.9, 0.05, 0.05],
        help="Train/Val/Test ratios (must sum to 1).",
    )

    # Random seed for reproducible shuffling of the file list
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling.",
    )

    # Flag to enable copy mode instead of symlink mode
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of creating symlinks.",
    )

    # Optional argument specifying a file to log messages to
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Optional log file path.",
    )

    # Parse all the command-line arguments provided by the user
    args = parser.parse_args()

    # Validate that the three ratios sum to 1.0 (within a small numerical tolerance)
    if not abs(sum(args.ratios) - 1.0) < 1e-6:
        # Raise a ValueError if they do not sum to 1
        raise ValueError("Train/Val/Test ratios must sum to 1.0")

    # Initialize the logger using the optionally provided log file
    logger = configure_logging(log_file=args.log_file)

    # Log basic information about the current run
    logger.info("Starting dataset split process...")
    if args.sequences:
        logger.info(f"Using sequences CSV: {args.sequences}")
    else:
        logger.info(f"Base path: {args.base_path}")
    logger.info(f"Output path: {args.output}")
    logger.info(f"Prefix: {args.prefix}  Suffix: {args.suffix}")
    logger.info(f"Ratios (train/val/test): {args.ratios}")
    logger.info(f"Copy mode: {'ON' if args.copy else 'OFF'}")

    # Convert the output directory string to a Path object
    out_dir = Path(args.output)

    # Step 1 — Collect files either from a directory or from a sequences CSV
    if args.sequences:
        # If --sequences is used, call load_files_from_sequences
        files = load_files_from_sequences(
            Path(args.sequences), args.prefix, args.suffix, logger
        )
    else:
        # If --base-path is used, scan the directory tree
        base_path = Path(args.base_path)
        files = collect_files(base_path, args.prefix, args.suffix, logger)

    # Sort the collected file paths in ascending order
    files = sorted(files)

    # If no files were found, log an error and exit the function
    if not files:
        logger.error("No files found. Exiting.")
        return

    # Step 2 — Perform missing data detection and log daily/hourly summaries
    missing_info = detect_missing_files(files, args.prefix, logger)
    # Note: 'missing_info' is currently not used later, but can be used for export if needed

    # Step 3 — Shuffle the file list and split it into train/val/test subsets
    random.seed(args.seed)  # Set seed for reproducible randomized splits
    random.shuffle(files)   # Shuffle the list of file paths in-place

    # Total number of files
    n = len(files)
    # Number of files assigned to train split
    n_train = int(args.ratios[0] * n)
    # Number of files assigned to validation split
    n_val = int(args.ratios[1] * n)
    # The rest of the files will be used for the test split

    # Slice the list into train, val, and test subsets
    train_files = files[:n_train]
    val_files = files[n_train : n_train + n_val]
    test_files = files[n_train + n_val :]

    # Log the number of files in each split
    logger.info(f"Total files: {n}")
    logger.info(f"Train set: {len(train_files)} files")
    logger.info(f"Validation set: {len(val_files)} files")
    logger.info(f"Test set: {len(test_files)} files")

    # Step 4 — Ensure train/val/test directories are created (and old contents cleared)
    split_dirs = create_split_dirs(out_dir, logger)

    # Step 5 — Populate each split directory with symlinks or copies
    populate_split(train_files, split_dirs["train"], logger, copy_mode=args.copy)
    populate_split(val_files, split_dirs["val"], logger, copy_mode=args.copy)
    populate_split(test_files, split_dirs["test"], logger, copy_mode=args.copy)

    # Final log message indicating successful completion
    logger.info("Dataset split completed successfully.")


# ==========================
# Script entry point
# ==========================
if __name__ == "__main__":  # This condition is True when the script is run directly
    main()                  # Call the main function
