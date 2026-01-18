# fill_holes.py  # Script name comment for clarity.
import logging  # Provide structured logging output.
import os  # Access filesystem utilities.
import shutil  # Copy files when replacing holes.
import sys  # Read command-line arguments.


def setup_logging() -> logging.Logger:
    """Configure logging and return the module logger."""  # Explain logging helper.
    logging.basicConfig(  # Configure root logger formatting and level.
        level=logging.INFO,  # Emit info-level logs by default.
        format="%(asctime)s [%(levelname)s] %(message)s",  # Use readable log format.
    )
    return logging.getLogger(__name__)  # Return a module-scoped logger.


logger = setup_logging()  # Initialize logging as soon as the module runs.

ROOT = sys.argv[1] if len(sys.argv) > 1 else "data/rdr0_splits/train/"  # Choose input root.
MIN_SIZE = 1024  # byte: < MIN_SIZE lo consideriamo buco (es. 128 B)
LOG_PATH = os.path.join(ROOT, "_fillholes_log.txt")  # Build log file path.

replaced = 0  # Track number of files replaced.
first_bad = 0  # Track number of initial holes with no previous file.
checked = 0  # Track number of files inspected.

with open(LOG_PATH, "w", encoding="utf-8") as log:  # Open log file for writing.
    for cur, _, files in os.walk(ROOT):  # Traverse the root directory tree.
        tif = sorted(  # Sort TIFF files to preserve temporal order.
            [f for f in files if f.lower().endswith((".tif", ".tiff"))]  # Filter TIFFs.
        )
        prev_valid = None  # Keep track of the last valid file path.
        for fname in tif:  # Iterate over each TIFF filename.
            path = os.path.join(cur, fname)  # Build the full file path.
            checked += 1  # Increment the checked file counter.
            try:  # Guard file size access.
                size = os.path.getsize(path)  # Read the file size in bytes.
            except FileNotFoundError:  # Skip files that vanish mid-run.
                continue  # Move to the next file.

            if size < MIN_SIZE:  # Detect holes based on minimal file size.
                if prev_valid:  # Ensure we have a previous valid file to copy.
                    shutil.copy2(prev_valid, path)  # Replace hole with the previous valid file.
                    replaced += 1  # Increment replacement counter.
                    log.write(f"REPL {path} <- {prev_valid}\n")  # Log the replacement action.
                else:  # Handle the first file being a hole.
                    first_bad += 1  # Increment first-bad counter.
                    log.write(f"SKIP(no-prev) {path}\n")  # Log that we skipped the first hole.
            else:  # Handle valid file sizes.
                prev_valid = path  # Update the previous valid file reference.

logger.info(  # Log summary statistics after processing.
    "Controllati: %s  |  Rimpiazzati: %s  |  Primi buchi: %s",  # Log message template.
    checked,  # Include number of files checked.
    replaced,  # Include number of replacements.
    first_bad,  # Include number of first holes.
)
logger.info("Log: %s", LOG_PATH)  # Log the path to the detailed log file.
