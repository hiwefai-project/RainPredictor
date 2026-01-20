
# compare.py — Truth vs Prediction Radar Image Comparison Tool

`compare.py` is a comprehensive visualization and evaluation tool designed to compare **ground‑truth (Truth)** radar reflectivity fields against **Predicted** fields (typically produced by nowcasting models).  
It generates side‑by‑side comparison panels, computes verification metrics, and optionally exports metrics to JSON for advanced post‑processing.

---

## ✨ Features

- **Automatic timestamp extraction** (`YYYYMMDDZhhmm`)
- **Truth vs Pred** comparison with two-column layout
- **Landscape or Portrait orientation** (`--orientation`)
- **Custom radar palette support** via JSON (`--palette`)
- **Robust TIFF/GeoTIFF reader** (supports rasterio or PIL)
- **NaN‑safe and inf‑safe rendering**
- **Colorbar placed outside** image area (no overlap)
- **Optional per-frame metrics** (enable with `--metrics`), including  
  - RMSE  
  - MSE  
  - MAE  
  - Bias  
  - Correlation coefficient
- **Optional overall metrics** across the full sequence (enable with `--metrics`)
- **Optional metrics panel** plotted below the images (enable with `--metrics`)
- **Optional JSON export** (`--metrics-json`, requires `--metrics`) for post-processing
- Logging support (`--log-level`)

---

## 📦 Installation Requirements

Install dependencies:

```bash
pip install numpy matplotlib rasterio pillow
```

Rasterio is preferred for correct GeoTIFF handling.

---

## 🗂 Directory Structure

Your dataset must contain TIFF/GeoTIFF radar images with timestamps in the filename:

```
truth/
   rdr0_d01_20251202Z1810_VMI.tiff
   rdr0_d01_20251202Z1820_VMI.tiff
   ...

pred/
   rdr0_d01_20251202Z1810_pred.tiff
   rdr0_d01_20251202Z1820_pred.tiff
   ...
```

Only the timestamp matters — filenames do not need to match otherwise.

Timestamp extraction uses:

```
(\d{8}Z\d{4})
```

Example: `20251202Z1810`

---

## 🎨 Using a Custom Radar Palette

Example `palette.json`:

```json
{
  "levels": [-10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
  "colors": ["#646464","#04e9e7","#019ff4","#0300f4","#02fd02","#01c501",
             "#008e00","#fdf802","#e5bc00","#fd9500","#fd0000","#d40000",
             "#bc0000","#f800fd","#9854c6"],
  "label": "Reflectivity (dBZ)"
}
```

Use it with:

```bash
--palette palette.json
```

---

## 🚀 Command Line Usage

### Basic comparison

```bash
python compare.py \
  --truth-dir truth \
  --pred-dir pred
```

### With timestamps

```bash
python compare.py \
  --truth-dir truth \
  --pred-dir pred \
  --start 20251202Z1810 \
  --end 20251202Z1900
```

### Save to file

```bash
--save output.png
```

### Orientation control

```bash
--orientation {landscape,portrait}
```

### Enable metrics overlays and plots

```bash
--metrics
```

### Export metrics to JSON

```bash
--metrics --metrics-json metrics.json
```

---

## 📊 JSON Export Format

The `--metrics-json` output looks like:

```json
{
  "per_frame": [
    {
      "timestamp": "20251202Z1810",
      "mse": 1.23,
      "rmse": 1.11,
      "mae": 0.85,
      "bias": -0.10,
      "corr": 0.92
    }
  ],
  "overall": {
    "mse": 1.10,
    "rmse": 1.05,
    "mae": 0.80,
    "bias": -0.05,
    "corr": 0.93
  }
}
```

Perfect for ingestion into pandas:

```python
import json
import pandas as pd

metrics = json.load(open("metrics.json"))
df = pd.DataFrame(metrics["per_frame"])
print(df)
```

---

## 🖼 Example Output

The generated figure contains:

- Left column: **Truth**
- Right column: **Pred**
- Per-frame metrics overlay (when `--metrics` is enabled)
- Bottom metrics graph (RMSE, MAE, Bias) when `--metrics` is enabled
- External colorbar with optional palette label

---

## 🧪 Recommended Workflow

1. Compare sequences visually  
2. Export metrics:  
   ```bash
   --metrics --metrics-json metrics.json
   ```
3. Analyze model performance across lead times  
4. Tune model & architecture  
5. Regenerate comparison to validate improvements

---

## 📬 Support

If you want:

- GIF/MP4 animation support  
- Difference heatmaps  
- Skill score computation (CSI, POD, FAR, ETS)  
- Multi-model comparison mode  

Just ask — I can extend the tool.

---

## 📝 License

MIT License
