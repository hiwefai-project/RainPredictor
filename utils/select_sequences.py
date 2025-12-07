#!/usr/bin/env python3
"""
Weather Radar GeoTIFF dataset scanner.

Features:
1. Detects genuine GeoTIFFs (optional deletion of invalid files).
2. Computes pixel rate above a reflectivity threshold.
3. Detects time-consistent 10-minute sequences of minimum length.
4. Optional CSV listing missing or invalid expected GeoTIFFs.
5. Supports custom filename prefix and postfix.
6. Verbose mode: per-timestamp status (genuine / invalid / missing) + stats.
"""

import argparse
import csv
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any
import re

import numpy as np
import rasterio

FILENAME_REGEX = re.compile(r".*?(?P<date>\d{8})Z(?P<time>\d{4})")


# ---------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="GeoTIFF radar dataset scanner.")

    parser.add_argument(
        "--base-path", "-b", required=True,
        help="Base directory containing YYYY/MM/DD folders."
    )
    parser.add_argument(
        "--threshold", "-t", type=float, required=True,
        help="Reflectivity threshold in dBZ."
    )
    parser.add_argument(
        "--rate", "-r", type=float, required=True,
        help="Minimum fraction of pixels > threshold to select a file."
    )
    parser.add_argument(
        "--csv", "-o", required=True,
        help="Output CSV for selected sequences."
    )
    parser.add_argument(
        "--min-length", "-m", type=int, default=36,
        help="Minimum sequence length (10-min steps)."
    )

    parser.add_argument(
        "--log-level", default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)."
    )

    parser.add_argument(
        "--delete-invalid", action="store_true",
        help="Delete non-GeoTIFF files that have .tiff extension."
    )

    parser.add_argument(
        "--missing-csv",
        help="Write a CSV listing missing or invalid GeoTIFF timestamps."
    )

    parser.add_argument(
        "--prefix", default="rdr0_d01_",
        help="Prefix of radar files (default: rdr0_d01_)."
    )

    parser.add_argument(
        "--postfix", default="_VMI.tiff",
        help="Postfix of radar files (default: _VMI.tiff)."
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show per-timestamp status: genuine/invalid/missing, max dBZ, rate, selection."
    )

    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s"
    )


# ---------------------------------------------------------------
# File Helpers
# ---------------------------------------------------------------
def is_geotiff(path: Path) -> bool:
    """Return True only if file is a valid GeoTIFF."""
    try:
        with rasterio.open(path) as ds:
            return ds.driver == "GTiff" and ds.count > 0
    except Exception:
        return False


def compute_stats(path: Path, threshold: float, verbose: bool = False):
    """
    Compute:
      - max_dbz: maximum reflectivity (dBZ)
      - rate: fraction of pixels > threshold

    Returns (max_dbz, rate).
    """
    with rasterio.open(path) as ds:
        data = ds.read(1).astype(np.float32)
        nodata = ds.nodata

        if verbose:
            try:
                logging.debug(
                    "STATS RAW %s: nodata=%r min=%.2f max=%.2f",
                    path,
                    nodata,
                    float(np.nanmin(data)),
                    float(np.nanmax(data)),
                )
            except Exception:
                pass

        # Handle nodata carefully
        if nodata is not None and not np.isnan(nodata):
            mask = data != nodata
            if mask.any():
                valid = data[mask]
            else:
                # All pixels equal to nodata → fall back to using full array
                if verbose:
                    logging.debug(
                        "All pixels equal to nodata (%r) in %s – using full array",
                        nodata, path
                    )
                valid = data.reshape(-1)
        else:
            valid = data.reshape(-1)

        # Drop NaNs if present
        valid = valid[~np.isnan(valid)]

        if valid.size == 0:
            return float("nan"), 0.0

        max_dbz = float(valid.max())
        rate = float((valid > threshold).sum()) / float(valid.size)
        return max_dbz, rate


def extract_datetime(path: Path) -> datetime:
    """Extract datetime object from filename."""
    m = FILENAME_REGEX.search(path.name)
    if not m:
        raise ValueError(f"File does not match timestamp pattern: {path}")
    return datetime.strptime(m.group("date") + m.group("time"), "%Y%m%d%H%M")


def dt_to_stamp(dt: datetime) -> str:
    """Convert datetime → YYYYMMDDZhhmm."""
    return dt.strftime("%Y%m%dZ%H%M")


def minutes_ok(dt: datetime) -> bool:
    return dt.minute in {0, 10, 20, 30, 40, 50}


# ---------------------------------------------------------------
# Sequence Detection
# ---------------------------------------------------------------
def find_sequences(entries: List[Dict[str, Any]], min_length: int):
    if not entries:
        return []

    entries.sort(key=lambda e: e["dt"])
    sequences: List[List[Dict[str, Any]]] = []
    seq: List[Dict[str, Any]] = [entries[0]]

    for prev, cur in zip(entries, entries[1:]):
        if cur["dt"] - prev["dt"] == timedelta(minutes=10) and minutes_ok(cur["dt"]):
            seq.append(cur)
        else:
            if len(seq) >= min_length:
                logging.info(
                    "Sequence found: %s → %s length=%d",
                    dt_to_stamp(seq[0]["dt"]),
                    dt_to_stamp(seq[-1]["dt"]),
                    len(seq)
                )
                sequences.append(seq)
            seq = [cur]

    # Last sequence
    if len(seq) >= min_length:
        logging.info(
            "Sequence found: %s → %s length=%d",
            dt_to_stamp(seq[0]["dt"]),
            dt_to_stamp(seq[-1]["dt"]),
            len(seq)
        )
        sequences.append(seq)

    return sequences


# ---------------------------------------------------------------
# CSV Writers
# ---------------------------------------------------------------
def write_sequences_csv(sequences, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sequence_id", "index", "timestamp", "path", "rate"])
        for sid, seq in enumerate(sequences, start=1):
            for idx, e in enumerate(seq):
                writer.writerow([
                    sid,
                    idx,
                    dt_to_stamp(e["dt"]),
                    str(e["path"]),
                    f"{e['rate']:.6f}",
                ])


def write_missing_csv(missing, csv_path: Path) -> None:
    if not missing:
        logging.info("No missing timestamps detected.")
        return

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "directory", "expected_path"])
        for m in missing:
            writer.writerow([m["timestamp"], m["directory"], m["expected_path"]])

    logging.info("Missing timestamps written to: %s", csv_path)


# ---------------------------------------------------------------
# Main Program
# ---------------------------------------------------------------
def main():
    args = parse_args()
    setup_logging(args.log_level)

    base = Path(args.base_path)
    if not base.is_dir():
        logging.error("Base path does not exist or is not a directory: %s", base)
        raise SystemExit(1)

    accepted: List[Dict[str, Any]] = []
    missing: List[Dict[str, str]] = []

    total_files = 0
    valid_count = 0
    rate_count = 0

    prefix = args.prefix
    postfix = args.postfix

    logging.info("Using prefix='%s' postfix='%s'", prefix, postfix)

    # ---------------------------------------------------------------
    # Traverse YEAR / MONTH / DAY
    # ---------------------------------------------------------------
    for year in sorted(base.glob("20??")):
        if not year.is_dir():
            continue

        logging.info("Scanning YEAR %s", year.name)

        for month in sorted(year.glob("[01][0-9]")):
            if not month.is_dir():
                continue

            logging.info("Scanning YEAR %s / MONTH %s", year.name, month.name)

            for day in sorted(month.glob("[0-3][0-9]")):
                if not day.is_dir():
                    continue

                day_str = year.name + month.name + day.name

                # Build expected timeline for this day
                dt = datetime.strptime(day_str + "0000", "%Y%m%d%H%M")
                expected: List[datetime] = []
                while dt.strftime("%Y%m%d") == day_str:
                    expected.append(dt)
                    dt += timedelta(minutes=10)

                # Map real files found for this day
                found: Dict[datetime, Path] = {}
                for file in day.glob(f"{prefix}*{postfix}"):
                    total_files += 1
                    try:
                        file_dt = extract_datetime(file)
                        found[file_dt] = file
                    except Exception:
                        # Filename not matching pattern → ignored
                        continue

                # For each expected timestamp, decide: missing / invalid / genuine
                for dt in expected:
                    stamp = dt_to_stamp(dt)
                    expected_path = day / f"{prefix}{stamp}{postfix}"

                    # Missing file completely
                    if dt not in found:
                        missing.append({
                            "timestamp": stamp,
                            "directory": str(day),
                            "expected_path": str(expected_path),
                        })
                        if args.verbose:
                            logging.info(
                                "TIMESTAMP %s STATUS=MISSING expected=%s",
                                stamp, expected_path
                            )
                        continue

                    file = found[dt]

                    # Exists but not genuine GeoTIFF → invalid
                    if not is_geotiff(file):
                        missing.append({
                            "timestamp": stamp,
                            "directory": str(day),
                            "expected_path": str(file),
                        })

                        deleted = False
                        if args.delete_invalid:
                            try:
                                file.unlink()
                                deleted = True
                            except Exception as e:
                                logging.error("Failed to delete '%s': %s", file, e)

                        if args.verbose:
                            logging.info(
                                "TIMESTAMP %s FILE=%s STATUS=INVALID deleted=%s",
                                stamp, file, "yes" if deleted else "no"
                            )
                        continue

                    # Genuine GeoTIFF
                    valid_count += 1

                    try:
                        max_dbz, rate = compute_stats(file, args.threshold, args.verbose)
                    except Exception as e:
                        if args.verbose:
                            logging.info(
                                "TIMESTAMP %s FILE=%s STATUS=ERROR error=%s",
                                stamp, file, e
                            )
                        continue

                    selected = rate > args.rate
                    if selected:
                        rate_count += 1
                        accepted.append({"dt": dt, "path": file, "rate": rate})

                    if args.verbose:
                        logging.info(
                            "TIMESTAMP %s FILE=%s STATUS=GENUINE max_dbZ=%.2f rate=%.6f selected=%s",
                            stamp,
                            file,
                            max_dbz,
                            rate,
                            "yes" if selected else "no",
                        )

    logging.info("Total files (matching prefix/postfix) scanned: %d", total_files)
    logging.info("Valid GeoTIFF files: %d", valid_count)
    logging.info("Files satisfying rate threshold: %d", rate_count)

    # ---------------------------------------------------------------
    # Detect sequences
    # ---------------------------------------------------------------
    sequences = find_sequences(accepted, args.min_length)
    logging.info("Total sequences found: %d", len(sequences))

    # ---------------------------------------------------------------
    # Write outputs
    # ---------------------------------------------------------------
    write_sequences_csv(sequences, Path(args.csv))

    if args.missing_csv:
        write_missing_csv(missing, Path(args.missing_csv))


if __name__ == "__main__":
    main()

