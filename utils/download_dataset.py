#!/usr/bin/env python3

# Import standard libraries
import argparse                           # For parsing command-line arguments
import os                                 # For filesystem operations
import sys                                # For process exit codes
from datetime import datetime, timedelta  # For dates and time intervals
from concurrent.futures import ThreadPoolExecutor, as_completed  # For parallel downloads
import logging                            # For logging to console and optional file
import hashlib                            # For checksum (SHA256) verification

# Third-party libraries
import requests                           # For HTTP file downloads
from tqdm import tqdm                     # For progress bar visualization


# Time step between products (in minutes)
TIME_STEP_MINUTES = 10


def parse_datetime_yyyymmddhhmm(s: str) -> datetime:
    """Parse datetime string YYYYMMDDHHMM."""
    try:
        return datetime.strptime(s, "%Y%m%dZ%H%M")
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"Invalid datetime '{s}'. Expected format YYYYMMDDZHHMM."
        ) from e


def build_filename(dt: datetime, prefix: str, postfix: str) -> str:
    """Build the radar filename according to user prefix/postfix."""
    timestamp = dt.strftime("%Y%m%dZ%H%M")
    return f"{prefix}{timestamp}{postfix}"


def build_url(dt: datetime, base_url: str, prefix: str, postfix: str) -> str:
    """Build the remote URL for a radar file."""
    year = dt.strftime("%Y")
    month = dt.strftime("%m")
    day = dt.strftime("%d")
    filename = build_filename(dt, prefix, postfix)
    return f"{base_url}/{year}/{month}/{day}/{filename}"


def build_output_path(output_dir: str, dt: datetime, prefix: str, postfix: str) -> str:
    """Construct local path mirroring server folder structure."""
    year = dt.strftime("%Y")
    month = dt.strftime("%m")
    day = dt.strftime("%d")
    filename = build_filename(dt, prefix, postfix)

    dir_path = os.path.join(output_dir, year, month, day)
    os.makedirs(dir_path, exist_ok=True)

    return os.path.join(dir_path, filename)


def download_file(url: str, dest_path: str, timeout: float = 30.0, logger=None) -> bool:
    """Download remote file."""
    if logger is None:
        logger = logging.getLogger(__name__)

    try:
        response = requests.get(url, stream=True, timeout=timeout)
    except requests.RequestException as e:
        logger.error("Request failed for %s: %s", url, e)
        return False

    if response.status_code == 404:
        logger.warning("Not found (404): %s", url)
        return False

    if not response.ok:
        logger.error("HTTP %s for %s", response.status_code, url)
        return False

    try:
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    except OSError as e:
        logger.error("Write failed for %s: %s", dest_path, e)
        return False

    logger.info("Saved %s -> %s", url, dest_path)
    return True


def iterate_datetimes(start: datetime, end: datetime, step_minutes: int):
    """Yield datetimes from start to end inclusive."""
    current = start
    delta = timedelta(minutes=step_minutes)

    while current <= end:
        yield current
        current += delta


def load_checksums(path: str, logger=None):
    """Load filename → sha256 dictionary."""
    if logger is None:
        logger = logging.getLogger(__name__)

    checksums = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) != 2:
                    logger.warning("Malformed checksum line: %s", line)
                    continue
                checksums[parts[0]] = parts[1].lower()
    except OSError as e:
        logger.error("Cannot read checksum file %s: %s", path, e)
        return {}

    logger.info("Loaded %d checksums", len(checksums))
    return checksums


def compute_sha256(path: str) -> str:
    """Compute SHA256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_checksum(path: str, expected_sha: str, logger=None) -> bool:
    """Verify SHA256 checksum."""
    if logger is None:
        logger = logging.getLogger(__name__)

    try:
        actual = compute_sha256(path)
    except OSError as e:
        logger.error("Checksum read failed for %s: %s", path, e)
        return False

    if actual.lower() == expected_sha.lower():
        logger.info("Checksum OK: %s", path)
        return True

    logger.error(
        "Checksum mismatch for %s: expected %s, got %s",
        path, expected_sha, actual
    )
    return False



def resample_image(path: str, factor: float, logger=None) -> bool:
    """Resample an image in-place by the given factor, preserving aspect ratio.

    If the file is a GeoTIFF, the georeferencing information (CRS and affine
    transform) is preserved and properly updated after resampling.

    The image is opened, resized, and saved back to the same path. A factor of
    1.0 leaves the image unchanged. Factors must be greater than 0.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if factor == 1.0:
        # Nothing to do
        return True

    if factor <= 0:
        logger.error("Invalid resample factor %s. It must be > 0.", factor)
        return False

    # First, try a GeoTIFF-aware path using rasterio. This preserves CRS and
    # geotransform if the image is georeferenced.
    try:
        import rasterio
        from rasterio.enums import Resampling
        from affine import Affine

        with rasterio.open(path) as src:
            width, height = src.width, src.height
            new_width = max(1, int(round(width * factor)))
            new_height = max(1, int(round(height * factor)))

            data = src.read(
                out_shape=(src.count, new_height, new_width),
                resampling=Resampling.bilinear,
            )

            scale_x = width / new_width
            scale_y = height / new_height
            new_transform = src.transform * Affine.scale(scale_x, scale_y)

            profile = src.profile
            profile.update(
                height=new_height,
                width=new_width,
                transform=new_transform,
            )

        # Write back in-place, preserving CRS and most profile fields
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(data)

        logger.info(
            "GeoTIFF resampled %s from %dx%d to %dx%d (CRS preserved)",
            path, width, height, new_width, new_height
        )
        return True

    except ImportError:
        # rasterio / affine not installed: fall back to a generic PIL-based
        # resize without geospatial metadata handling.
        logger.warning(
            "rasterio/affine not available; falling back to PIL for %s. "
            "GeoTIFF metadata (if any) may not be preserved.", path
        )
    except Exception as e:
        # If something went wrong specifically in the rasterio path, log and
        # fall back to PIL below.
        logger.warning(
            "GeoTIFF-aware resampling failed for %s (%s); falling back to PIL.",
            path, e
        )

    # Generic PIL-based resampling (no GeoTIFF awareness).
    try:
        from PIL import Image  # Imported lazily in case Pillow is not installed at runtime

        with Image.open(path) as img:
            width, height = img.size
            new_width = max(1, int(round(width * factor)))
            new_height = max(1, int(round(height * factor)))

            resized = img.resize((new_width, new_height), Image.BILINEAR)
            resized.save(path, format=img.format)

        logger.info(
            "Resampled (PIL) %s from %dx%d to %dx%d",
            path, width, height, new_width, new_height
        )
        return True
    except Exception as e:
        logger.error("Failed to resample image %s: %s", path, e)
        return False



def download_one(
    dt: datetime,
    output_dir: str,
    base_url: str,
    prefix: str,
    postfix: str,
    skip_existing: bool,
    retries: int,
    resample_factor: float,
    checksums: dict | None,
    verify_checksums: bool,
    logger=None,
) -> bool:
    """Download a single file with retry, optional checksum, and optional resampling."""
    if logger is None:
        logger = logging.getLogger(__name__)

    url = build_url(dt, base_url, prefix, postfix)
    dest_path = build_output_path(output_dir, dt, prefix, postfix)
    filename = os.path.basename(dest_path)

    # Optionally skip if the file already exists
    if skip_existing and os.path.exists(dest_path):
        logger.info("Skipping existing: %s", dest_path)
        if verify_checksums and checksums and filename in checksums:
            return verify_checksum(dest_path, checksums[filename], logger=logger)
        return True

    attempts = max(1, retries)

    for attempt in range(1, attempts + 1):
        logger.info("Downloading %s (attempt %d/%d)", url, attempt, attempts)
        ok = download_file(url, dest_path, logger=logger)

        if not ok:
            if attempt < attempts:
                continue
            return False

        # checksum verification
        if verify_checksums and checksums and filename in checksums:
            if not verify_checksum(dest_path, checksums[filename], logger=logger):
                if attempt < attempts:
                    logger.warning("Retry due to checksum mismatch: %s", filename)
                    continue
                return False

        # Optional resampling of the downloaded image (after a successful download
        # and optional checksum verification). This is done in-place and preserves
        # the original aspect ratio. GeoTIFFs are handled in a CRS-aware manner
        # when rasterio/affine are available.
        if resample_factor != 1.0:
            if not resample_image(dest_path, resample_factor, logger=logger):
                if attempt < attempts:
                    logger.warning("Retry due to resampling failure: %s", filename)
                    continue
                return False

        return True

    return False


def setup_logger(log_file: str | None) -> logging.Logger:
    """Setup logging (console + optional file)."""
    logger = logging.getLogger("rdr0_downloader")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        fmt = "%(asctime)s - %(levelname)s - %(message)s"
        formatter = logging.Formatter(fmt, "%Y-%m-%d %H:%M:%S")

        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    return logger


def main():

    parser = argparse.ArgumentParser(description="Download radar TIFF dataset.")

    parser.add_argument("start", type=parse_datetime_yyyymmddhhmm)
    parser.add_argument("end", type=parse_datetime_yyyymmddhhmm)

    parser.add_argument("-o", "--output-dir", default="downloads")

    parser.add_argument("--base-url",
                        default="https://data.meteo.uniparthenope.it/instruments/rdr0")

    # NEW
    parser.add_argument("--prefix", default="rdr0_d02_",
                        help="Filename prefix (default: rdr0_d02_)")
    parser.add_argument("--postfix", default="_VMI.tiff",
                        help="Filename postfix (default: _VMI.tiff)")

    parser.add_argument("--dryrun", action="store_true")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument( "--resample", type=float, default=1.0, help=("Resample factor for images; e.g. 0.5 halves width/height, 2.0 doubles them. 1.0 = no resampling."))

    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--retries", type=int, default=3)

    parser.add_argument("--log-file")
    parser.add_argument("--checksum-file")

    args = parser.parse_args()

    logger = setup_logger(args.log_file)

    # Validate date range
    if args.end < args.start:
        logger.error("End datetime must be >= start datetime.")
        sys.exit(1)

    logger.info("Downloading from %s to %s", args.start, args.end)
    logger.info("Prefix: %s", args.prefix)
    logger.info("Postfix: %s", args.postfix)

    # Load checksum manifest
    checksums = None
    verify_checksums = False
    if args.checksum_file:
        checksums = load_checksums(args.checksum_file, logger)
        verify_checksums = bool(checksums)

    # Build datetime list
    datetimes = list(iterate_datetimes(args.start, args.end, TIME_STEP_MINUTES))

    if args.dryrun:
        for dt in tqdm(datetimes, desc="Dryrun"):  # Iterate timestamps for dry-run output.
            logger.info(  # Log each URL instead of printing to stdout.
                build_url(dt, args.base_url, args.prefix, args.postfix)  # Build the URL to preview.
            )
        sys.exit(0)  # Exit after dry-run listing.

    failed = 0

    # Sequential mode
    if not args.parallel:
        for dt in tqdm(datetimes, desc="Downloading"):
            ok = download_one(
                dt,
                args.output_dir,
                args.base_url,
                args.prefix,
                args.postfix,
                args.skip_existing,
                args.retries,
                args.resample,
                checksums,
                verify_checksums,
                logger,
            )
            if not ok:
                failed += 1

    else:
        # Parallel mode
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    download_one,
                    dt,
                    args.output_dir,
                    args.base_url,
                    args.prefix,
                    args.postfix,
                    args.skip_existing,
                    args.retries,
                    args.resample,
                    checksums,
                    verify_checksums,
                    logger,
                ): dt
                for dt in datetimes
            }

            for future in tqdm(as_completed(futures),
                               total=len(futures),
                               desc="Downloading (parallel)"):
                try:
                    if not future.result():
                        failed += 1
                except Exception as e:
                    logger.error("Task exception: %s", e)
                    failed += 1

    # Exit code logic
    if failed > 0:
        logger.error("Completed with %d failed downloads.", failed)
        sys.exit(1)

    logger.info("All downloads completed successfully.")
    sys.exit(0)


if __name__ == "__main__":
    main()
