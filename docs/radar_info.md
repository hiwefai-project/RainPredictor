# Radar metadata inspector (`utils/radar_info.py`)

`utils/radar_info.py` is a command-line tool for analyzing **VMI radar GeoTIFF images** such as those produced by the Italian Department of Civil Protection.  
It reads the radar reflectivity values (VMI) from the GeoTIFF file, converts them to **dBZ**, and computes:

- **Maximum reflectivity (dBZ)**
- **Average reflectivity (dBZ)**
- **Count of all pixels whose VMI exceeds a configurable threshold**

The script automatically detects and applies **scale/offset** metadata commonly stored inside radar GeoTIFFs.

---

## Features

- Reads GeoTIFF radar images using `rasterio`
- Converts raw values to **dBZ** if scale/offset metadata is present
- Handles NoData values
- Allows setting a custom reflectivity threshold via:
  ```
  --threshold <value>
  ```
- Uses Python's `logging` module for structured output
- Fully commented and clean for integration in workflows (Hi-WeFAI, RainPredictor, HPC pipelines)

---

## Installation

Make sure the following Python packages are installed:

```bash
pip install numpy rasterio
```

---

## Usage

### Basic usage (default threshold = 10 dBZ)

```bash
python utils/radar_info.py path/to/radar_image.tiff
```

### Custom threshold

```bash
python utils/radar_info.py path/to/radar_image.tiff --threshold 20
```

---

## Output Example

```
2025-12-05 22:31:01 [INFO] Opening file: rdr1_d01_20251202Z1900_VMI.tiff
2025-12-05 22:31:01 [INFO] Detected scale factor: 0.5
2025-12-05 22:31:01 [INFO] Detected offset: -32.0
2025-12-05 22:31:01 [INFO] Applied scale/offset to convert raw values to dBZ
2025-12-05 22:31:01 [INFO] === Radar VMI Statistics ===
2025-12-05 22:31:01 [INFO] File: rdr1_d01_20251202Z1900_VMI.tiff
2025-12-05 22:31:01 [INFO] Maximum VMI (dBZ): 46.30
2025-12-05 22:31:01 [INFO] Average VMI (dBZ): 3.81
2025-12-05 22:31:01 [INFO] Pixels with VMI > 10 dBZ: 12453
```

---

## Command Line Arguments

| Argument | Description | Default |
|---------|-------------|---------|
| `file` | Path to the GeoTIFF radar image | — |
| `--threshold <dBZ>` | Minimum reflectivity for pixel counting | `10` |

---

## Example: Using the script in a pipeline

This script can be used in:

- RainPredictor model evaluation  
- Hi-WeFAI ingestion pipelines  
- Radar quality checks  
- Automated workflows (Snakemake, Nextflow, SLURM jobs)

Example:

```bash
for f in data/*.tiff; do
    python utils/radar_info.py "$f" --threshold 15
done
```

---

## License

MIT License.  
You may use the script freely in research and operational workflows.

---

## Author

Generated for the Hi-WeFAI project radar processing tasks.
