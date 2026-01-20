# Training guide (`train.py`)

This repository contains a modular PyTorch implementation of a radar rainfall
nowcasting model (`RainPredRNN`) composed of:

- A U-Net style encoder/decoder for per-frame spatial features.
- A temporal Transformer block over encoded feature maps.
- A training/evaluation pipeline with:
  - Mixed precision support (AMP).
  - Benchmarking of train/val throughput.
  - Automatic saving of predictions and targets as TIFF.
  - Evaluation metrics and confusion matrix report.
## 1. Data layout

The code assumes that you have the entire dataset (including its directory substructure) in a single folder (for example, `dataset/`).

Let's run the script to reorganize the dataset into three sub-datasets:
1) Training set - 90% of the entire dataset
2) Validation set - 9% of the entire dataset
3) Test set - 1% of the entire dataset

```bash
python utils/make_splits.py \
  --data-dir "<YOUR-dataset-DIR>" \
  --out-dir "<DECIDE-YOUR-OUT-DIR>/splits" \
  --pattern "**/*.tiff" \
  --split-ratios-train 0.90 \
  --split-ratios-val 0.09 \
  --split-ratios-test 0.01
```

| Parameter               | Type          | Default       | Description                                                                                       |
|-------------------------|---------------|---------------|---------------------------------------------------------------------------------------------------|
| `--data-dir`            | string (path) | —             | Path to the directory containing the dataset.                                                     |
| `--out-dir`             | string (path) | —             | Output directory where the dataset splits (`train`, `val`, `test`) will be created.               |
| `--pattern`             | string        | `"**/*.tiff"` | Glob pattern used to match the input files within the dataset directory and its subdirectories.   |
| `--split-ratios-train`  | float         | `0.90`        | Fraction of the dataset to assign to the training set (value between `0.0` and `1.0`).            |
| `--split-ratios-val`    | float         | `0.09`        | Fraction of the dataset to assign to the validation set (value between `0.0` and `1.0`).          |
| `--split-ratios-test`   | float         | `0.01`        | Fraction of the dataset to assign to the test set (value between `0.0` and `1.0`).                |


The result:
```text
<YOUR-OUT-DIR>/splits/
├── train/
│   ├── <radar_YYYYMMDDHHMM_0000>.tif(f)
│   ├── <radar_YYYYMMDDHHMM_0001>.tif(f)
│   └── ...
├── val/
│   ├── <radar_YYYYMMDDHHMM_0000>.tif(f)
│   ├── <radar_YYYYMMDDHHMM_0001>.tif(f)
│   └── ...
└── test/
    ├── <radar_YYYYMMDDHHMM_0000>.tif(f)
    ├── <radar_YYYYMMDDHHMM_0001>.tif(f)
    └── ...
```

Where each file is a single-band GeoTIFF (or TIFF) containing radar
reflectivity in dBZ. File naming does **not** have to match this pattern
exactly: the dataset uses the sorted list of `.tif`/`.tiff` files
in each split and builds sliding temporal windows over them.

Default paths in `rainpred/config.py` are:

- `DATA_PATH = "/data/rdr0_splits/"`
- `VAL_PREVIEW_ROOT = "/previews"`
- `RUNS_DIR = "runs"`
- `CHECKPOINT_DIR = "checkpoints"`

You will typically override these via CLI arguments (see below).

## 2. Running training

Basic usage:

```bash
python train.py \
  --data-path /path/to/rdr0_splits \
  --val-preview-root /path/to/previews \
  --runs-dir /path/to/runs \
  --checkpoint-dir /path/to/checkpoints
```

This will:

1. Create train/val `DataLoader`s from `/path/to/rdr0_splits/train` and
   `/path/to/rdr0_splits/val`.
2. Instantiate `RainPredRNN` with default hyperparameters.
3. Benchmark train/val throughput.
4. Train for the default number of epochs (`NUM_EPOCHS` from `config.py`,
   overridable via CLI).
5. At each epoch:
   - Log train/val metrics to TensorBoard.
   - Save validation predictions and targets as TIFF images organized by epoch.
   - Save the best model checkpoint and an evaluation report.

### CLI hyperparameters

All key paths and hyperparameters are configurable via CLI:

```bash
python train.py \
  --data-path /path/to/rdr0_splits \
  --val-preview-root /path/to/previews \
  --runs-dir /path/to/runs \
  --checkpoint-dir /path/to/checkpoints \
  --epochs 20 \
  --batch-size 8 \
  --lr 5e-4 \
  --num-workers 8 \
  --pred-length 6
```

- `--epochs`: number of training epochs.
- `--batch-size`: training batch size.
- `--lr`: learning rate for Adam.
- `--num-workers`: number of `DataLoader` workers.
- `--pred-length`: number of future frames to predict.

### Small-debug mode

For quick debugging, you can enable `--small-debug`:

```bash
python train.py \
  --data-path /path/to/rdr0_splits \
  --epochs 1 \
  --batch-size 2 \
  --num-workers 2 \
  --small-debug
```

In this mode, the dataset is trimmed to small subsets:

- Up to 64 training windows.
- Up to 16 validation windows.

This is useful for quickly verifying that:

- The dataset paths and TIFFs are readable.
- The model forward/backward passes work.
- Logging, checkpointing, and preview saving work end-to-end.

## 3. Reproducibility notes

The pipeline includes several mechanisms to improve reproducibility:

- A `set_seed(15)` call in `rainpred/data.py::create_dataloaders` sets:
  - NumPy RNG seed.
  - PyTorch CPU and CUDA RNG seeds.
- `torch.backends.cudnn.benchmark = True` and
  `torch.backends.cudnn.deterministic = False` are used to favor
  performance on convolution-heavy workloads (strict bit-level
  determinism is not enforced).
- Dataset windows are built deterministically from the sorted file list.
- Training and validation splits are separated at the filesystem level
  (`train/` and `val/` directories), Only valid TIFFs are used.

For **bitwise reproducibility**, you may want to:

- Switch `cudnn.deterministic = True` and `cudnn.benchmark = False` in
  `rainpred/utils.py::set_seed`.
- Fix `num_workers=0` in `create_dataloaders` to avoid multi-process
  nondeterminism.
- Ensure your environment (CUDA, drivers, PyTorch version) is frozen,
  e.g., using a container.

## 4. Outputs

During training, the following artifacts are produced:

- TensorBoard logs:
  - Under `<runs-dir>/<timestamp>/Train` and `/Validation`.
  - Contains train loss and validation metrics per epoch.
- Validation previews:
  - Under `<val-preview-root>/epoch_XXX/predictions` and `/targets`.
  - Predicted and ground-truth TIFF frames with names derived from the
    original target file stems.
- Checkpoints and evaluation reports:
  - Under `<checkpoint-dir>`.
  - `best_model.pth` with:
    - `model_state_dict`
    - `optimizer_state_dict`
    - `metrics`
    - `confusion_matrix`
  - `evaluation_reports/evaluation_report.txt` describing:
    - Average metrics over the validation loader.
    - Confusion matrix, precision, recall, F1, accuracy.

## 5. Project structure

```text
RainPredictor/
├─ train.py
├─ predict.py
├─ README.md
├─ requirements.txt
├─ utils/
│  ├─ make_splits.py
│  └─ compare.py
└─ rainpred/
   ├─ config.py
   ├─ data.py
   ├─ geo_io.py
   ├─ model.py
   ├─ utils.py
   ├─ metrics.py
   └─ train_utils.py
```

- `rainpred/config.py`: default hyperparameters and paths.
- `rainpred/utils.py`: seeding and benchmarking helpers.
- `rainpred/data.py`: `RadarDataset`, augmentations, and dataloaders.
- `rainpred/model.py`: `RainPredRNN` definition (encoder, transformer, decoder).
- `rainpred/metrics.py`: scalar metrics, confusion matrix, evaluation report.
- `rainpred/train_utils.py`: training loop per epoch and prediction/target saving.
- `train.py`: CLI-based training entry point tying everything together.

## 6. How to reproduce your experiments

1. Clone or unpack this project.
2. Create and activate a Python 3.10+ virtual environment.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Ensure your radar dataset is organized under `DATA_ROOT` with `train/` and
   `val/` subdirectories containing `.tif`/`.tiff` files.
5. Run training with explicit paths and hyperparameters:
   ```bash
   python train.py \
     --data-path /absolute/path/to/DATA_ROOT \
     --val-preview-root /absolute/path/to/previews \
     --runs-dir /absolute/path/to/runs \
     --checkpoint-dir /absolute/path/to/checkpoints \
     --epochs 10 \
     --batch-size 4 \
     --lr 1e-3 \
     --num-workers 8 \
     --pred-length 6
   ```
6. After training:
   - Inspect TensorBoard logs with:
     ```bash
     tensorboard --logdir /absolute/path/to/runs
     ```
   - Inspect saved TIFF previews for qualitative evaluation.
   - Inspect `evaluation_report.txt` and `best_model.pth` under
     `/absolute/path/to/checkpoints`.

This should provide everything needed to reproduce and extend your
RainPredRNN radar nowcasting experiments.
