
# compare.py — Truth vs Prediction Radar Image Comparison

`compare.py` is a visualization utility for comparing **Radar Ground-Truth (Truth)** images with **Predicted** images produced by weather nowcasting models.  
It loads TIFF/GeoTIFF files, extracts timestamps from filenames, aligns truth/pred pairs, and generates a vertical sequence plot with two columns:

- **Left column:** Truth (ground truth radar reflectivity)
- **Right column:** Prediction

Each row represents a timestamp, and the script automatically determines correct ordering.

---

## ✨ Features

- Extracts timestamps automatically (`YYYYMMDDZhhmm`)
- Matches Truth and Pred pairs automatically
- Sequence comparison from `--start` to `--end`
- Supports **GeoTIFF** metadata (via `rasterio`)
- Handles NaNs, NODATA values, constant fields safely
- Optional **custom palette JSON** for color-mapped reflectivity (`--palette`)
- Logging instead of print
- Colorbar placed outside the image panel (no overlapping)
- Works in both **interactive mode** and **save-to-file mode**

---

## 📦 Requirements

One of:

- `rasterio` (recommended)
- `Pillow` (fallback)

Plus:

```
numpy
matplotlib
```

Install:

```bash
pip install rasterio pillow numpy matplotlib
```

---

## 🗂 Directory Structure

Both truth and prediction directories must contain TIFF/GeoTIFF files named with timestamps, e.g.:

```
truth/
   rdr0_d01_20251202Z1810_VMI.tiff
   rdr0_d01_20251202Z1820_VMI.tiff
   ...

output/
   rdr0_d01_20251202Z1810_pred.tiff
   rdr0_d01_20251202Z1820_pred.tiff
   ...
```

Files do **not** need the same prefix or suffix — only the timestamp matters.

---

## 🧠 Timestamp Extraction

A timestamp is automatically detected using the regular expression:

```
(\d{8}Z\d{4})
```

Example: `20251202Z1810`

---

## 🚀 Basic Usage

### Compare a full sequence

```bash
python compare.py \
  --truth-dir truth \
  --pred-dir output \
  --title "First Train" \
  --start 20251202Z1810 \
  --end 20251202Z1900
```

### Save to a PNG file

```bash
python compare.py \
  --truth-dir truth \
  --pred-dir output \
  --save comparison.png
```

---

## 🎨 Using a Custom Palette (Recommended for Radar dbZ)

You may pass a JSON palette file, for example:

```json
{
  "levels": [-10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
  "colors": [
    "#646464","#04e9e7","#019ff4","#0300f4","#02fd02",
    "#01c501","#008e00","#fdf802","#e5bc00","#fd9500",
    "#fd0000","#d40000","#bc0000","#f800fd","#9854c6"
  ],
  "label": "Reflectivity (dBZ)"
}
```

Use it:

```bash
python compare.py \
  --truth-dir truth \
  --pred-dir output \
  --palette palette.json \
  --save compare.png
```

`compare.py` automatically builds a `ListedColormap` + `BoundaryNorm` from the palette.

---

## 🛠 Command-Line Arguments

| Flag | Description |
|------|-------------|
| `--truth-dir` | Directory containing ground-truth radar TIFFs |
| `--pred-dir` | Directory containing predicted radar TIFFs |
| `--start` | First timestamp to include (inclusive) |
| `--end` | Last timestamp to include (inclusive) |
| `--title` | Title displayed above the plot |
| `--save` | File path to save the figure (omit to view interactively) |
| `--palette` | Path to a palette JSON file |
| `--log-level` | DEBUG, INFO, WARNING, ERROR (default: INFO) |

---

## 📜 Example Full Command

```bash
python compare.py \
  --truth-dir truth \
  --pred-dir output \
  --title "Nowcasting Comparison" \
  --start 20251202Z1810 \
  --end 20251202Z1900 \
  --palette palette.json \
  --save 20251202Z1810-1900.png \
  --log-level DEBUG
```

---

## 🖼 Output Figure

The generated figure contains:

- A sequence of Truth vs Pred rows
- A colorbar placed outside the right edge
- Clean layout (no overlapping)
- Optional radar reflectivity labels (from palette)

---

## 🧩 Notes

- Only timestamps present in **both** directories are included.
- TIFFs with multiple bands use the **first band**.
- NODATA values are converted to NaN when rasterio provides metadata.

---

## 📄 License

MIT License — free to use, modify, and integrate in research workflows.

---

## 📬 Support

If you need:
- multi-panel layouts  
- animated outputs  
- automatic GIF/MP4 generation  
- overlay of truth/pred heatmaps  
- integration with training pipelines  

I can extend the script for you.

