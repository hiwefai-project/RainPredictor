# Installation guide

This guide walks through setting up RainPredictor for both training and
prediction workflows using either pip or conda.

## 1. Prerequisites

- **Python 3.10+** (required by the training and prediction scripts).
- **CUDA-capable GPU (optional)** for faster training/inference; CPU-only
  installs are supported.
- **System libraries** required by GeoTIFF tooling (notably `rasterio`).
  On Linux, these typically arrive via GDAL/PROJ packages from your OS
  package manager.

## 2. Install with pip (virtual environment)

```bash
python -m venv .venv               # Create an isolated virtual environment.
source .venv/bin/activate          # Activate the virtual environment (Linux/macOS).
python -m pip install --upgrade pip # Ensure pip is up to date.
pip install -r requirements.txt    # Install core dependencies (training + inference).
```

### 2.1. Optional: install a specific PyTorch build

If you need a CUDA-enabled or CPU-only build of PyTorch, follow the official
PyTorch selector and then re-run the remaining dependencies:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 # Example CUDA 12.1 build.
pip install -r requirements.txt                                                  # Install the rest of the deps.
```

## 3. Install with conda (recommended for GDAL/rasterio)

```bash
conda create -n rainpredictor python=3.10 -y # Create a dedicated conda env.
conda activate rainpredictor                 # Activate the environment.
conda install -c conda-forge gdal proj -y    # Install GeoTIFF prerequisites.
python -m pip install --upgrade pip          # Keep pip up to date inside conda.
pip install -r requirements.txt              # Install Python dependencies.
```

### 3.1. Optional: GPU-accelerated PyTorch in conda

```bash
conda install -c pytorch -c nvidia pytorch torchvision pytorch-cuda=12.1 -y # GPU-ready PyTorch.
pip install -r requirements.txt                                             # Ensure remaining deps are installed.
```

## 4. Validate the installation

### 4.1. Check training CLI availability

```bash
python train.py --help # Confirm the training entry point is available.
```

### 4.2. Check prediction CLI availability

```bash
python predict.py --help # Confirm the prediction entry point is available.
```

## 5. Next steps: training and prediction

Once dependencies are installed, follow the workflow guides:

- **Training**: use the [training guide](train.md) to prepare GeoTIFF splits and
  launch `train.py`.
- **Prediction**: use the [inference guide](predict.md) to run `predict.py` on
  GeoTIFF sequences and write predicted frames.

For a quick example, once your dataset and checkpoints are ready:

```bash
python train.py --data-path /data/rdr0_splits --epochs 10 # Launch training with your dataset.
python predict.py --checkpoint checkpoints/best_model.pth \\ # Point to a trained checkpoint.
  --input-dir /data/radar_seq \\                           # Select the input GeoTIFF sequence.
  --output-dir /data/preds \\                              # Choose where predictions are written.
  --m 18 \\                                                # Provide the number of input frames.
  --n 6                                                    # Provide the number of future frames.
```
