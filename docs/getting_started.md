# Getting started guide

This guide provides a step-by-step walkthrough to install RainPredictor, set
up your environment, run training, and generate predictions. For deeper dives,
see the dedicated [installation](install.md), [training](train.md), and
[prediction](predict.md) guides.

## 1. Clone the repository

```bash
git clone <YOUR_REPO_URL>
cd RainPredictor
```

## 2. Set up a Python environment

Choose one of the supported environment setups below.

### 2.1. Pip virtual environment

```bash
python -m venv .venv               # Create an isolated virtual environment.
source .venv/bin/activate          # Activate the virtual environment (Linux/macOS).
python -m pip install --upgrade pip # Ensure pip is up to date.
pip install -r requirements.txt    # Install core dependencies (training + inference).
```

### 2.2. Conda environment (recommended for GDAL/rasterio)

```bash
conda create -n rainpredictor python=3.10 -y # Create a dedicated conda env.
conda activate rainpredictor                 # Activate the environment.
conda install -c conda-forge gdal proj -y    # Install GeoTIFF prerequisites.
python -m pip install --upgrade pip          # Keep pip up to date inside conda.
pip install -r requirements.txt              # Install Python dependencies.
```

### 2.3. (Optional) Use a GPU-ready PyTorch build

If you want GPU acceleration, install a CUDA-compatible PyTorch build before
installing the rest of the requirements:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 # Example CUDA 12.1 build.
pip install -r requirements.txt                                                  # Install remaining deps.
```

## 3. Prepare the dataset

1. Download or collect your GeoTIFF radar sequence data.
2. Split the dataset into training/validation/test folders using
   `utils/make_splits.py`:

```bash
python utils/make_splits.py \
  --data-dir "<YOUR_DATASET_DIR>" \
  --out-dir "<YOUR_OUTPUT_DIR>/splits" \
  --pattern "**/*.tiff" \
  --split-ratios-train 0.90 \
  --split-ratios-val 0.09 \
  --split-ratios-test 0.01
```

When the script finishes, your directory layout should look like:

```text
<YOUR_OUTPUT_DIR>/splits/
├─ train/
├─ val/
└─ test/
```

## 4. Train a model

1. Point `train.py` at the dataset splits folder created above.
2. Choose the training hyperparameters (epochs, batch size, learning rate).
3. Run the training command:

```bash
python train.py \
  --data-path "<YOUR_OUTPUT_DIR>/splits" \
  --epochs 20 \
  --batch-size 4 \
  --lr 1e-3 \
  --num-workers 8 \
  --pred-length 6
```

Key outputs:

- `checkpoints/last_checkpoint.pth` and `checkpoints/best_model.pth` (model
  checkpoints).
- `runs/` (TensorBoard logs, if enabled).

## 5. Run prediction (inference)

1. Collect an input sequence directory with the `m` GeoTIFF frames you want to
   use as input.
2. Run `predict.py` with the trained checkpoint:

```bash
python predict.py \
  --checkpoint checkpoints/best_model.pth \
  --input-dir "<YOUR_SEQUENCE_DIR>" \
  --output-dir "<YOUR_OUTPUT_DIR>/preds" \
  --m 18 \
  --n 6
```

After the command completes, the future `n` frames are written as GeoTIFFs
inside `<YOUR_OUTPUT_DIR>/preds`, preserving CRS and georeferencing metadata.

## 6. Optional: Inspect results

Use the comparison utility to visualize a prediction alongside a ground-truth
frame:

```bash
python utils/compare.py \
  --input "<YOUR_SEQUENCE_DIR>/rdr0_d01_20251202Z1800_VMI.tiff" \
  --pred "<YOUR_OUTPUT_DIR>/preds/rdr0_d01_20251202Z1810_VMI.tiff" \
  --title "t=1800 vs t+1 prediction"
```
