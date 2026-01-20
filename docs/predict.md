# Inference guide (`predict.py`)

The `predict.py` script is the **inference entry point** for this repository.
It loads **m** GeoTIFF radar frames, predicts the next **n** frames with the
`RainPredModel`, and writes GeoTIFF outputs that preserve geospatial metadata.

Key consistency points with `predict.py`:

- Uses the `rainpred` package for model definition and GeoTIFF I/O.
- Loads the exact checkpoint format expected by `train.py`.
- Normalizes input frames the same way as training (`normalize_image` + `ToTensor`
  + `Normalize(mean=[0.5], std=[0.5])`).
- Outputs GeoTIFF predictions in **dBZ** units by default.

## 1. Relevant files

```text
RainPredictor/
├─ predict.py
└─ rainpred/
   ├─ model.py
   ├─ geo_io.py
   └─ data.py
```

- `predict.py`:
  - CLI script that loads a checkpoint, runs inference, and saves predictions.
- `rainpred/geo_io.py`:
  - `load_sequence_from_dir` handles GeoTIFF input, padding, and normalization.
  - `save_predictions_as_geotiff` writes GeoTIFF outputs with metadata.
- `rainpred/model.py`:
  - `RainPredModel` architecture used for inference.

## 2. Installation

Create a fresh virtual environment (recommended), then install dependencies:

```bash
pip install -r requirements.txt
```

The script assumes you already have a trained checkpoint from `train.py`.
The checkpoint is expected to be a `.pth` file containing either:

- A dictionary with key `model_state_dict` (as saved by `torch.save({...})`), or
- A raw `state_dict` (as saved by `torch.save(model.state_dict())`).

## 3. Expected data format

Input is a sequence of **m** radar frames stored as **single-band GeoTIFF** files:

```text
INPUT_DIR/
  frame_0000.tif
  frame_0001.tif
  ...
  frame_00XX.tif
```

The script:

1. Lists all files in `INPUT_DIR` ending with `.tif` or `.tiff`.
2. Sorts them lexicographically.
3. Takes the first **m** files (defaults to `m=18`).
4. Applies the same preprocessing pipeline used in training:
   - Read band 1 as float32.
   - Normalize dBZ to `[0, 1]` using the shared `normalize_image` helper.
   - Pad to patch-aligned dimensions (no resize).
   - Convert to 8-bit grayscale (`0–255`) for PIL.
   - Convert to tensor and normalize with `mean=[0.5]`, `std=[0.5]`.
5. Stacks the **m** frames into a tensor of shape `(1, m, 1, H, W)`.

## 4. Running inference

Basic usage (matches `predict.py`):

```bash
python predict.py \
  --checkpoint /path/to/best_model.pth \
  --input-dir /path/to/input_frames \
  --m 18 \
  --n 6 \
  --output-dir /path/to/output_predictions
```

- `--checkpoint`:
  - Path to the trained `RainPredRNN` checkpoint.
- `--input-dir`:
  - Directory containing at least **m** `.tif` frames.
- `--m`:
  - Number of input frames the model conditions on.
  - Must be **greater than** `n` (the number of predicted frames).
- `--n`:
  - Number of future frames to predict.
- `--output-dir`:
  - Directory where predicted frames are written as GeoTIFFs.
- `--cpu`:
  - Optional flag to force inference on CPU even if CUDA is available.
- `--resample-factor`:
  - Optional scale factor to resample input frames in-place before inference.
  - Use `1.0` to leave inputs unchanged; values must be greater than 0.
- `--metrics-json`:
  - Optional path to write AI-friendly JSON diagnostics (use `-` for stdout).

Example:

```bash
python predict.py \
  --checkpoint /models/rainpred/best_model.pth \
  --input-dir /data/radar_sequence_example \
  --m 18 \
  --n 6 \
  --output-dir ./predictions
```

This will:

1. Load the model on GPU if available (or CPU with `--cpu`).
2. Read the first 18 frames from `/data/radar_sequence_example`.
3. Predict the next 6 frames.
4. Save them to `./predictions/`, using time-consistent basenames when the
   input filenames contain timestamps (see below).

To emit structured diagnostics for automation or debugging, include:

```bash
python predict.py \
  --checkpoint /models/rainpred/best_model.pth \
  --input-dir /data/radar_sequence_example \
  --m 18 \
  --n 6 \
  --output-dir ./predictions \
  --metrics-json ./predict_metrics.json
```

## 5. Output naming and time consistency

`predict.py` maintains **time-consistent output filenames** when the input
filenames follow the pattern:

```
<prefix>YYYYMMDDZhhmm<suffix>
```

For example:

```
rdr0_d01_20241023Z0710_VMI.tiff
```

The script:

1. Infers the time step (in minutes) from the **last two** input filenames.
2. Uses the timestamp of the last input file as the base time.
3. Generates `n` future names by stepping forward in minutes.

If the filenames do not match the pattern, it falls back to `pred_01.tif`,
`pred_02.tif`, etc.

## 6. Reproducibility and consistency with training

The architecture and preprocessing in this project are designed to be
**compatible** with the training project you already have:

- Same UNet encoder/decoder layout.
- Same temporal Transformer (patch size, number of heads, etc.).
- Same spatial resolution (padding only; no resize).
- Same intensity normalization:
  - Input dBZ values assumed in [0, 70].
  - Normalized to [0, 1], then to [-1, 1] via `Normalize(mean=0.5, std=0.5)`.

Predictions are returned by the model in the normalized range `[-1, 1]`. The
saving utility:

1. Maps back to `[0, 1]` using `x = x * 0.5 + 0.5`.
2. Converts to dBZ via `x * 70.0` and clips to `[0, 70]`.
3. Writes each frame as a **float32 GeoTIFF** with original metadata.

Note: if you prefer normalized `[0, 1]` outputs instead of dBZ, set
`as_dbz=False` in `save_predictions_as_geotiff`.

## 7. Requirements

A minimal `requirements.txt` is provided and includes:

- torch
- torchvision
- numpy
- pillow
- rasterio
- einops

Install them via:

```bash
pip install -r requirements.txt
```

## 8. Summary

- **Input**: sequence of **m** GeoTIFF radar frames (m > n).
- **Output**: sequence of the next **n** frames predicted by the model, saved
  as GeoTIFFs in dBZ with preserved metadata.
- **Usage**: `python predict.py --checkpoint ... --input-dir ... --m M --n N --output-dir ...`.

This repository keeps inference and training code together, and `predict.py`
mirrors the data handling and model configuration used during training for
consistent results.
