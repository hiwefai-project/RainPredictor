# Import operating system utilities
import os
# Import typing helpers
from typing import List

# Import numpy for array operations
import numpy as np
# Import PIL for saving images
from PIL import Image

# Import torch for tensor operations
import torch


# Define function to save predicted frames to output directory
def save_predictions_as_tiff(
    preds: torch.Tensor,
    output_dir: str,
    prefix: str = "pred",
) -> List[str]:
    """Save predicted frames (1, n, 1, H, W) as 8-bit TIFF images and return their paths."""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Move predictions to CPU and convert to numpy
    arr = preds.detach().cpu().numpy()
    # Undo normalization from [-1, 1] to [0, 1]
    arr = arr * 0.5 + 0.5

    # If tensor has shape (1, n, 1, H, W) keep only (n, H, W)
    if arr.ndim == 5:
        arr = arr[0, :, 0]

    # Initialize list of output paths
    out_paths: List[str] = []

    # Loop over time dimension
    for t in range(arr.shape[0]):
        # Get frame t of shape (H, W)
        frame = arr[t]
        # Convert to 0-255 uint8
        frame_uint8 = (frame * 255.0).clip(0, 255).astype(np.uint8)
        # Create PIL Image
        img = Image.fromarray(frame_uint8)
        # Build output file path
        out_path = os.path.join(output_dir, f"{prefix}_{t:03d}.tiff")
        # Save image as TIFF
        img.save(out_path)
        # Append path to list
        out_paths.append(out_path)

    # Return list of saved file paths
    return out_paths
