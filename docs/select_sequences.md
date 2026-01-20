# Sequence selector (`utils/select_sequences.py`)

## Overview

`utils/select_sequences.py` is a Python tool for scanning large structured weather radar datasets stored as GeoTIFF files.  
It validates files, computes reflectivity statistics, detects time-consistent 10-minute radar sequences, and generates multiple CSV reports.

The tool is designed to work with datasets organized as:

```
<BASE_PATH> / YYYY / MM / DD / rdr0_d01_YYYYMMDDZhhmm_VMI.tiff
```

Where:
- `YYYY` — Year (e.g., `2024`)
- `MM` — Month (`01`–`12`)
- `DD` — Day (`01`–`31`)
- `hhmm` — Time (`hh` ∈ `00..23`, `mm` ∈ `{00,10,20,30,40,50}`)
- File format is **GeoTIFF**, containing **VMI reflectivity in dBZ**

## Main Features

### GeoTIFF Validation  
- Detects fake `.tiff` files (e.g., JSON files wrongly named `.tiff`)  
- Ensures files are readable GTiff with at least 1 band  
- Optional automatic deletion of invalid files (`--delete-invalid`)

### Pixel Statistics  
For each genuine GeoTIFF:
- Maximum reflectivity (max_dBZ)
- Rate of pixels exceeding a given dBZ threshold  
- Selection based on `rate > threshold_rate`

### Time-Consistent Sequence Detection  
Sequences must:
- Use 10-minute intervals  
- Contain **no missing frames**  
- Be at least *N* frames long (`--min-length`, default: 36)

### Missing File Report  
Creates a CSV listing missing or invalid timestamps if requested.

### Verbose Mode  
Shows:
- Genuine / invalid / missing
- max_dBZ  
- Threshold rate  
- Whether file satisfies selection criteria  
- Deletion events

### Custom Prefix/Postfix  
Compatible with alternative radar products:
```
--prefix rdr0_d02_
--postfix _VMI.tiff
```

## Installation

### Requirements
- Python ≥ 3.9
- `rasterio`
- `numpy`

### Install dependencies

```bash
pip install rasterio numpy
```

## Usage

### Basic Example

```bash
python utils/select_sequences.py \
  --base-path /projects/data/weather_radar/600x700 \
  --threshold 20 \
  --rate 0.01 \
  --csv sequences.csv
```

## Full Example with All Features

```bash
python utils/select_sequences.py \
  --base-path /projects/data/weather_radar/600x700 \
  --prefix rdr0_d01_ \
  --postfix _VMI.tiff \
  --threshold 25 \
  --rate 0.02 \
  --min-length 36 \
  --csv selected_sequences.csv \
  --missing-csv missing_frames.csv \
  --delete-invalid \
  --verbose \
  --log-level DEBUG
```

## Command-Line Options

| Option | Description |
|--------|-------------|
| `--base-path PATH` | Base directory of the radar dataset |
| `--threshold FLOAT` | dBZ threshold used for pixel counting |
| `--rate FLOAT` | Minimum pixel ratio above threshold |
| `--csv FILE` | Output CSV for valid sequences |
| `--missing-csv FILE` | Optional CSV listing missing/invalid timestamps |
| `--min-length INT` | Minimum sequence length (default 36 frames) |
| `--delete-invalid` | Delete files that are not genuine GeoTIFF |
| `--prefix PREFIX` | Radar filename prefix (default: `rdr0_d01_`) |
| `--postfix POSTFIX` | Radar filename suffix (default: `_VMI.tiff`) |
| `--verbose` | Prints detailed per-file status |
| `--log-level LEVEL` | DEBUG, INFO, WARNING, ERROR |

## Output Files

### 1. Sequence CSV (`--csv`)
Contains all selected time-consistent sequences.

Columns:
- `sequence_id`
- `index`
- `timestamp` (YYYYMMDDZhhmm)
- `path`
- `rate`

### 2. Missing/Invalid CSV (`--missing-csv`)  
Includes all:
- Missing timestamps
- Invalid GeoTIFFs
- Files deleted (if `--delete-invalid`)

Columns:
- `timestamp`
- `directory`
- `expected_path`

## Verbose Output Example

With `--verbose`, the script prints:

```
TIMESTAMP 20240210Z1030 FILE=... STATUS=GENUINE max_dbZ=47.32 rate=0.012345 selected=yes
TIMESTAMP 20240210Z1040 FILE=... STATUS=INVALID deleted=yes
TIMESTAMP 20240210Z1050 STATUS=MISSING expected=.../rdr0_d01_20240210Z1050_VMI.tiff
```

## Troubleshooting

### max_dBZ is always NaN
This may happen when:
- nodata masks are incorrect
- all values equal nodata
- file is not a real GeoTIFF

The improved `compute_stats` function:
- Prints debug ranges
- Falls back to full array when nodata mask removes everything

## License

MIT License (unless otherwise specified)

## Author

Raffaele Montella  
University of Naples "Parthenope"
