# RainPredictor-Inference

Minimal **inference-only** project for the RainPredRNN radar nowcasting model.

This project takes as input a sequence of **m** radar frames and predicts the
next **n** frames, under the constraint **m > n**, using the architecture
compatible with the training project you already have.

The implementation here **does not include any training code**: it only
provides the model definition, preprocessing, and a CLI entry point for
inference.

## 1. Project structure

```text
RainPredictor-Inference/
├─ run_inference.py
├─ README.md
├─ requirements.txt
└─ rainpredictor_inference/
   ├─ __init__.py
   ├─ model.py
   ├─ preprocess.py
   └─ io_utils.py
```

- `rainpredictor_inference/model.py`:
  - `RainPredRNN` model: U-Net encoder/decoder + temporal Transformer.
- `rainpredictor_inference/preprocess.py`:
  - Functions to load radar TIFFs, normalize, and build input tensors.
- `rainpredictor_inference/io_utils.py`:
  - Functions to save predicted frames as TIFF images.
- `run_inference.py`:
  - CLI script that wires everything together.

## 2. Installation

Create a fresh virtual environment (recommended), then install dependencies:

```bash
pip install -r requirements.txt
```

The project assumes you already have a trained checkpoint for the `RainPredRNN`
model (for example, from the training project we previously created). The
checkpoint is expected to be a `best_model.pth` file containing either:

- A dictionary with key `model_state_dict` (as saved by `torch.save({...})`), or
- A raw `state_dict` (as saved by `torch.save(model.state_dict())`).

## 3. Expected data format

Input is a sequence of **m** radar frames stored as single-band TIFF files:

```text
INPUT_DIR/
  frame_0000.tif
  frame_0001.tif
  ...
  frame_00XX.tif
```

The script:

1. Lists all files in `INPUT_DIR` matching the extension `.tif` (or a custom
   extension via `--pattern`).
2. Sorts them lexicographically.
3. Takes the first **m** files.
4. Applies the same preprocessing pipeline used in training:
   - Read band 1 as float32.
   - Map dBZ to [0, 1] by clipping to [0, 70] and scaling.
   - Convert to 8-bit grayscale (`0–255`) for PIL.
   - Resize to 224×224.
   - Convert to tensor and normalize with `mean=[0.5]`, `std=[0.5]`.
5. Stacks the **m** frames into a tensor of shape `(1, m, 1, 224, 224)`.

## 4. Running inference

Basic usage:

```bash
python run_inference.py \
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
  - Directory where predicted frames are written as `pred_000.tiff`, `pred_001.tiff`, etc.
- `--cpu`:
  - Optional flag to force inference on CPU even if CUDA is available.
- `--pattern`:
  - File extension/pattern to match input frames (default: `.tif`; `.tiff` also accepted).

Example:

```bash
python run_inference.py \
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
4. Save them to `./predictions/pred_000.tiff`, ..., `pred_005.tiff`.

## 5. Reproducibility and consistency with training

The architecture and preprocessing in this project are designed to be
**compatible** with the training project you already have:

- Same UNet encoder/decoder layout.
- Same temporal Transformer (patch size, number of heads, etc.).
- Same spatial resolution (224×224).
- Same intensity normalization:
  - Input dBZ values assumed in [0, 70].
  - Normalized to [0, 1], then to [-1, 1] via `Normalize(mean=0.5, std=0.5)`.

Predictions are returned by the model in the normalized range `[-1, 1]`. The
saving utility:

1. Maps back to `[0, 1]` using `x = x * 0.5 + 0.5`.
2. Converts to 8-bit `0–255`.
3. Writes each frame as a grayscale TIFF.

Note: if in your downstream pipeline you need physical dBZ values instead of
8-bit grayscale, you can adapt the saving function in
`rainpredictor_inference/io_utils.py` to do:

```python
dbz = np.clip(x * 70.0, 0.0, 70.0)
```

and then write to a floating-point TIFF.

## 6. Requirements

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

## 7. Summary

- **Input**: sequence of **m** radar frames (m > n).
- **Output**: sequence of the next **n** frames predicted by the model.
- **Usage**: `python run_inference.py --checkpoint ... --input-dir ... --m M --n N --output-dir ...`.

This project is intentionally focused on **inference only**, making it
suitable for deployment, testing on new sequences, or integration into a
larger nowcasting pipeline without bringing in the full training
infrastructure.
