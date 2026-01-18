# make_splits_clean.py (estratto minimo)  # Preserve original description comment.
import logging  # Provide structured logging output.
import math  # Support split size calculations.
import os  # Access filesystem utilities.
import shutil  # Copy files when symlinks fail.


def setup_logging() -> logging.Logger:
    """Configure logging and return the module logger."""  # Describe logging helper.
    logging.basicConfig(  # Configure the root logger.
        level=logging.INFO,  # Use INFO level for progress messages.
        format="%(asctime)s [%(levelname)s] %(message)s",  # Provide consistent log format.
    )
    return logging.getLogger(__name__)  # Return the module-scoped logger.


logger = setup_logging()  # Initialize logging when the script runs.

ROOT = "/storage/external_01/hiwefi/data"  # Base storage path.
DATA_DIR = os.path.join(ROOT, "rdr0_3k")  # Source dataset directory.
OUT_DIR = os.path.join(ROOT, "rdr0_3k_splits_clean")  # Destination split directory.
os.makedirs(OUT_DIR, exist_ok=True)  # Ensure the output directory exists.


def gather_images(root: str) -> list[str]:
    """Collect TIFF files in a directory tree."""  # Explain helper intent.
    files: list[str] = []  # Accumulate file paths.
    for cur, _, fnames in os.walk(root, followlinks=True):  # Walk the directory tree.
        for fname in fnames:  # Inspect each file name.
            if fname.lower().endswith((".tif", ".tiff")):  # Filter TIFF files.
                files.append(os.path.join(cur, fname))  # Store full file path.
    return sorted(files)  # Return a sorted list of file paths.


files = gather_images(DATA_DIR)  # Gather all source TIFF files.
n = len(files)  # Count total files.
r_train, r_val, r_test = 0.90, 0.09, 0.01  # Define split ratios.
n_train = int(math.floor(n * r_train))  # Compute train count.
n_val = int(math.floor(n * r_val))  # Compute validation count.
train = files[:n_train]  # Slice train files.
val = files[n_train : n_train + n_val]  # Slice validation files.
test = files[n_train + n_val :]  # Slice test files.


def link_into(subset: list[str], subset_dir: str) -> None:
    """Link or copy files into a split directory."""  # Describe linking behavior.
    for src in subset:  # Iterate each source file in the subset.
        rel = os.path.relpath(src, DATA_DIR)  # Compute relative path.
        dst_dir = os.path.join(subset_dir, os.path.dirname(rel))  # Build destination directory.
        os.makedirs(dst_dir, exist_ok=True)  # Ensure destination directory exists.
        dst = os.path.join(dst_dir, os.path.basename(src))  # Build destination file path.
        try:  # Attempt to create a symlink.
            if not os.path.exists(dst):  # Avoid overwriting an existing file.
                os.symlink(src, dst)  # Create a symlink to the source file.
        except Exception:  # Fall back when symlinks are not allowed.
            shutil.copy2(src, dst)  # Copy file while preserving metadata.


for name, subset in [("train", train), ("val", val), ("test", test)]:  # Iterate splits.
    link_into(subset, os.path.join(OUT_DIR, name))  # Populate each split directory.
logger.info("Split creato in %s", OUT_DIR)  # Log completion and output path.
