# Radar Dataset Splitter (`make_splits.py`)

This script scans a radar image dataset and splits it into **train**, **validation**, and **test** subsets.
It can create either **symbolic links** (default) or **physical copies** of the original files, and it also
performs consistency checks on the time series, reporting missing frames with **daily** and **hourly** summaries.

## 1. Dataset Layout

The script expects a directory tree like:

```text
BASE_PATH/
  YYYY/
    MM/
      DD/
        rdr0_d01_YYYYMMDDZhhmm_VMI.tiff
```

Where:

- `YYYY` – year (4 digits)
- `MM` – month (2 digits, `01`–`12`)
- `DD` – day of month (2 digits, `01`–`31`)
- `hh` – hour of day (`00`–`23`)
- `mm` – minutes (`00`, `10`, `20`, `30`, `40`, `50`)

The filename pattern is configurable via **prefix** and **suffix** arguments:

- **Prefix** (default: `rdr0_d01_`)
- **Suffix** (default: `_VMI.tiff`)

Only files whose names start with the given prefix and end with the given suffix are considered part of the dataset.

## 2. Features

- Recursive scan of the dataset tree under `--base-path`
- Progress bar for dataset scanning
- Configurable filename **prefix** and **suffix**
- Configurable **train/val/test** ratios
- Optional **copy mode** (`--copy`) instead of symbolic links
- Logging to console and optional log file (`--log-file`)
- Detection of missing radar frames in the time series (10-minute spacing)
- Daily and hourly summary of missing time slots

## 3. Command-line Arguments

```text
--base-path PATH      Root folder of the dataset (e.g., data/dataset/rdr0/)
--output PATH         Output directory where train/val/test folders will be created
--prefix STR          Filename prefix, e.g. 'rdr0_d01_' (default: 'rdr0_d01_')
--suffix STR          Filename suffix, e.g. '_VMI.tiff' (default: '_VMI.tiff')
--ratios R1 R2 R3     Train/Val/Test ratios (default: 0.9 0.05 0.05)
--seed INT            Random seed for shuffling (default: 42)
--copy                If set, copy files instead of creating symlinks
--log-file PATH       Optional log file path; if set, logs are written to this file in addition to the console
```

## 4. Basic Usage Examples

### 4.1. Using symbolic links (default)

This is usually preferred to avoid duplicating the dataset on disk:

```bash
python make_splits.py   --base-path data/dataset/rdr0   --output data/splits   --prefix rdr0_d01_   --suffix _VMI.tiff   --ratios 0.9 0.05 0.05
```

After running, the script will create:

```text
data/splits/
  train/
  val/
  test/
```

Each folder will contain either symlinks (default) or copies (if `--copy` is used) of the original TIFF files.

### 4.2. Using copy mode

If your environment does not support symbolic links well (e.g., some network filesystems or Windows setups),
you can enable copy mode:

```bash
python make_splits.py   --base-path data/dataset/rdr0   --output data/splits_copy   --prefix rdr0_d01_   --suffix _VMI.tiff   --ratios 0.8 0.1 0.1   --copy
```

This will physically copy the files into the split directories.

### 4.3. Logging to a file

To keep a persistent log of the operations and the detected missing frames:

```bash
python make_splits.py   --base-path data/dataset/rdr0   --output data/splits   --log-file split_radar.log
```

The log file will contain entries such as:

```text
2025-12-04 17:02:10 - INFO - Collected 12840 files matching prefix='rdr0_d01_' suffix='_VMI.tiff'.
2025-12-04 17:02:11 - WARNING - Missing frames on 2025-06-10: 2 slots of 10 min
2025-12-04 17:02:11 - WARNING -   2025-06-10 hour 15: 1 missing slots
2025-12-04 17:02:11 - WARNING -   2025-06-10 hour 16: 1 missing slots
```

## 5. Missing-data Reporting

The script assumes that radar images are acquired every **10 minutes**.

1. It parses timestamps from filenames using the pattern `YYYYMMDDZhhmm`.
2. It builds the expected regular time series between the first and last timestamps.
3. It computes the set of **missing timestamps** (slots where a file should exist but does not).
4. It aggregates missing timestamps:
   - by **day** (`YYYY-MM-DD`)
   - by **day + hour** (`YYYY-MM-DD`, `HH`)

The summary is written to the log as warnings, e.g.:

```text
WARNING - Missing frames on 2025-06-10: 3 slots of 10 min
WARNING -   2025-06-10 hour 15: 2 missing slots
WARNING -   2025-06-10 hour 16: 1 missing slots
```

This helps quickly identify problematic days or hours in the dataset.

## 6. Reproducibility

The splitting is randomized but controlled by the `--seed` argument (default: `42`). Using the same seed and
The same input dataset will produce the same train/val/test partition.

## 7. Notes

- Make sure `--base-path` points to the root directory that contains the `YYYY/` folders.
- The script operates **read-only** on the dataset; it never deletes or modifies original TIFF files.
- In symlink mode, deleting the split directories does **not** affect the original dataset.
- In copy mode, each split directory will contain complete copies of the TIFF files, so ensure that you have
  enough disk space.
