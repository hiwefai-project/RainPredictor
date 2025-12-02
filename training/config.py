# Import operating system utilities
import os
# Import torch to detect device availability
import torch

# Detect computation device: use CUDA if available, otherwise CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Number of workers used by DataLoader
NUM_WORKERS = 8
# Default training batch size
BATCH_SIZE = 4
# Whether to pin host memory for faster GPU transfers
PIN_MEMORY = True

# Initial learning rate for optimizer
LEARNING_RATE = 1e-3
# Number of training epochs
NUM_EPOCHS = 1
# Number of frames to predict into the future
PRED_LENGTH = 6
# Enable mixed precision (AMP) when running on CUDA
USE_AMP = True

# Default path to dataset containing train/val splits
DATA_PATH = os.path.abspath("/home/v.bucciero/data/instruments/rdr0_splits/")
# Default path where validation previews (predictions/targets) are saved
VAL_PREVIEW_ROOT = os.path.abspath("/home/v.bucciero/data/instruments/rdr0_previews_h100gpu")
# Default root for TensorBoard run logs
RUNS_DIR = "runs"
# Default directory where model checkpoints are stored
CHECKPOINT_DIR = "checkpoints"
