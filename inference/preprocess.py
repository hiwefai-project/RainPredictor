# Import operating system utilities
import os
# Import typing helpers
from typing import List

# Import numpy for numerical operations
import numpy as np
# Import PIL Image for image conversion
from PIL import Image

# Import torch for tensor handling
import torch
# Import torchvision transforms for image preprocessing
import torchvision.transforms as transforms

# Import rasterio for reading TIFF images
import rasterio


# Define function to normalize radar images from dBZ to [0, 1]
def normalize_image(img: np.ndarray) -> np.ndarray:
    """Normalize radar reflectivity (dBZ) image to [0, 1] range."""
    # Replace NaNs and infinities with zeros
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    # Clip negative values to zero
    img = np.clip(img, 0, None)
    # Define minimum and maximum dBZ values
    min_dbz, max_dbz = 0, 70
    # Clip image to [min_dbz, max_dbz]
    img = np.clip(img, min_dbz, max_dbz)
    # Scale to [0, 1] range
    return (img - min_dbz) / (max_dbz - min_dbz)


# Define factory for the same transform used during training
def get_inference_transform() -> transforms.Compose:
    """Return torchvision transform matching the training preprocessing pipeline."""
    # Resize to 224x224, convert to tensor, and normalize to mean=0.5, std=0.5
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )


# Define function to load a single TIFF frame into a tensor
def load_single_frame(path: str, transform: transforms.Compose) -> torch.Tensor:
    """Load a single-band radar TIFF file and return a preprocessed tensor (C, H, W)."""
    # Open raster file using rasterio
    with rasterio.open(path) as src:
        # Read first band as float32 array
        img = src.read(1).astype(np.float32)
    # Normalize dBZ image to [0, 1]
    img = normalize_image(img)
    # Convert to 8-bit image for PIL pipeline
    img = Image.fromarray((img * 255.0).astype(np.uint8))
    # Apply transform pipeline (resize, to tensor, normalize)
    tensor = transform(img)
    # Return tensor of shape (C, H, W)
    return tensor


# Define function to load a sequence of frames from a directory
def load_sequence_from_dir(
    input_dir: str,
    m: int,
    pattern: str = ".tif",
) -> torch.Tensor:
    """Load a sequence of m frames from input_dir and stack into shape (1, m, C, H, W)."""
    # Resolve absolute path for input directory
    input_dir = os.path.abspath(input_dir)
    # Build list of all files in directory
    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(pattern) or f.endswith(pattern + "f")]
    # Sort files lexicographically to preserve temporal order
    files = sorted(files)
    # Check that we have at least m frames
    if len(files) < m:
        raise RuntimeError(f"Not enough frames in {input_dir}: found {len(files)}, need at least {m}.")
    # Take the first m frames (you may change to last m if desired)
    files = files[:m]

    # Create transform matching training preprocessing
    transform = get_inference_transform()
    # Initialize list for per-frame tensors
    frames: List[torch.Tensor] = []

    # Loop over selected file paths
    for path in files:
        # Load and preprocess frame
        tensor = load_single_frame(path, transform)
        # Append tensor to list
        frames.append(tensor)

    # Stack into shape (m, C, H, W)
    seq = torch.stack(frames, dim=0)
    # Add batch dimension -> (1, m, C, H, W)
    seq = seq.unsqueeze(0)
    # Return sequence tensor
    return seq, files
