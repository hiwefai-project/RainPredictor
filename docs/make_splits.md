# Radar Dataset Splitter (`make_splits.py`)

This script splits a radar-image dataset into **train**, **validation**, and **test** subsets.  
It supports two ways of defining the dataset:

1. **Directory scan mode** — recursively scan a base folder (`--base-path`)
2. **Sequence-file mode** — read an explicit list of file paths from a CSV (`--sequences`)

The script can create **symbolic links** (default) or **physical copies**, and it performs time-series consistency checks with **daily** and **hourly** summaries of missing frames.

---

## 1. Dataset Layout

### 1.1 Directory scan mode (`--base-path`)

The dataset is expected to follow this directory structure:

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
- `hh` – hour (`00`–`23`)  
- `mm` – minutes (`00`, `10`, `20`, `30`, `40`, `50`)  

The script identifies valid files using:

- **Prefix** (`rdr0_d01_` by default)
- **Suffix** (`_VMI.tiff` by default)

Only files matching both are processed.

---

### 1.2 Sequence-file mode (`--sequences`)

Instead of scanning a directory tree, you may provide a CSV file listing the **exact files** to include:

Example `sequences.csv`:

```csv
path
/projects/data/rdr0/2025/12/01/rdr0_d01_20251201Z0010_VMI.tiff
/projects/data/rdr0/2025/12/01/rdr0_d01_20251201Z0020_VMI.tiff
...
```

This mode is useful when:

- You want to use a **filtered** dataset  
- You want to enforce a **specific ordering**  
- You want to split only **validated sequences**  
- The dataset structure is nonstandard  

Prefix and suffix filters still apply unless changed via CLI.

---

## 2. Features

- **Two dataset input modes:** directory-tree scan or sequence-file CSV
- Progress bar for scanning files
- Configurable filename **prefix** and **suffix**
- Configurable **train/val/test ratios**
- Optional **copy mode** (`--copy`) instead of symbolic links
- Optional **log file** (`--log-file`)
- Detection of missing 10-minute radar frames
- Daily and hourly summary of missing slots
- Fully deterministic splits via `--seed`

---

## 3. Command-line Arguments

```text
--base-path PATH       Root folder of the dataset (e.g., data/dataset/rdr0/)
--sequences CSV        CSV file with a "path" column defining explicit files to split
--output PATH          Output directory for train/val/test
--prefix STR           Filename prefix (default: rdr0_d01_)
--suffix STR           Filename suffix (default: _VMI.tiff)
--ratios R1 R2 R3      Train/Val/Test ratios (default: 0.9 0.05 0.05)
--seed INT             Random seed (default: 42)
--copy                 Copy files instead of creating symlinks
--log-file PATH        Optional log file path
```

**Note:**  
`--base-path` **and** `--sequences` are *mutually exclusive*.  
You must provide exactly one.

---

## 4. Usage Examples

### 4.1 Directory scan mode (default method)

```bash
python make_splits.py     --base-path data/dataset/rdr0     --output data/splits     --prefix rdr0_d01_     --suffix _VMI.tiff     --ratios 0.9 0.05 0.05
```

Result:

```text
data/splits/
  train/
  val/
  test/
```

---

### 4.2 Sequence-file mode (CSV input)

Use this when the dataset is pre-filtered or non-hierarchical:

```bash
python make_splits.py     --sequences sequences.csv     --output data/splits     --prefix rdr0_d01_     --suffix _VMI.tiff
```

The script will:

- Read all paths from the CSV  
- Optionally filter them by prefix/suffix  
- Perform missing-data analysis  
- Split into train/val/test  

---

### 4.3 Copy mode

```bash
python make_splits.py     --base-path data/dataset/rdr0     --output data/splits_copy     --copy
```

Use if symlinks are not supported (e.g., some shared filesystems, Windows).

---

### 4.4 Logging to a file

```bash
python make_splits.py     --base-path data/dataset/rdr0     --output data/splits     --log-file split_radar.log
```

---

## 5. Missing-data Reporting

The script assumes radar images arrive every **10 minutes**.

Steps performed:

1. Extract timestamps from filenames (`YYYYMMDDZhhmm`)  
2. Build a complete expected timeline between min/max timestamps  
3. Identify missing timestamps  
4. Report:
   - Total missing frames  
   - **Gaps** (continuous missing intervals)  
   - **Daily missing count**  
   - **Hourly missing count**

Example log snippet:

```text
Missing frames on 2025-06-10: 3 slots
  2025-06-10 hour 15: 2 missing slots
  2025-06-10 hour 16: 1 missing slots
```

---

## 6. Reproducibility

Splits are deterministic when specifying the same:

- dataset or sequences file
- prefix/suffix filters
- ratios
- `--seed` value

---

## 7. Notes

- The original dataset is **never modified**.  
- Symlink mode is recommended for space efficiency.  
- Copy mode requires significantly more disk space.  
- The script cleans the output directory before generating splits.  
- Sequence-file mode is ideal for **curated ML/AI datasets**.

