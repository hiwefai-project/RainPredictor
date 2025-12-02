# Import operating system utilities
import os
# Import argparse for CLI argument parsing
import argparse

# Import torch core library
import torch

# Import model architecture
from inference.model import RainPredRNN
# Import preprocessing utilities
from inference.preprocess import load_sequence_from_dir
# Import output saving utilities
from inference.io_utils import save_predictions_as_tiff


# Define helper to automatically select device
def get_device(force_cpu: bool = False) -> torch.device:
    """Return CUDA device if available and not forced to CPU, otherwise CPU."""
    if (not force_cpu) and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# Define function to load model and checkpoint
def load_model(checkpoint_path: str, device: torch.device, pred_length: int) -> RainPredRNN:
    """Instantiate RainPredRNN and load weights from a checkpoint file."""
    # Create model instance with same architecture as training
    model = RainPredRNN(
        input_dim=1,
        num_hidden=256,
        max_hidden_channels=128,
        patch_height=16,
        patch_width=16,
        pred_length=pred_length,
    ).to(device)

    # Load checkpoint file
    ckpt = torch.load(checkpoint_path, map_location=device)
    # If checkpoint is a dict with 'model_state_dict', use that
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        # Otherwise, assume the checkpoint is already a state_dict
        state_dict = ckpt
    # Load state dictionary into model
    model.load_state_dict(state_dict)
    # Set model to evaluation mode
    model.eval()
    # Return loaded model
    return model


# Define main CLI entry point
def main():
    """Run inference: given m input frames, predict next n frames and save them."""
    # Create argument parser
    parser = argparse.ArgumentParser(
        description="RainPredictor-Inference: predict future radar frames from a sequence."
    )
    # Add argument for checkpoint path
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained RainPredRNN checkpoint (best_model.pth).",
    )
    # Add argument for input directory containing frames
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing input radar frames (.tif or .tiff).",
    )
    # Add argument for number of input frames m
    parser.add_argument(
        "--m",
        type=int,
        default=18,
        help="Number of input frames to condition on (m > n).",
    )
    # Add argument for number of predicted frames n
    parser.add_argument(
        "--n",
        type=int,
        default=6,
        help="Number of future frames to predict (n < m).",
    )
    # Add argument for output directory to store predictions
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where predicted frames will be saved.",
    )
    # Add optional flag to force CPU usage
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force inference on CPU even if CUDA is available.",
    )
    # Add optional pattern for input file extension
    parser.add_argument(
        "--pattern",
        type=str,
        default=".tif",
        help="File extension pattern to select input frames (default: .tif).",
    )

    # Parse command-line arguments
    args = parser.parse_args()

    # Check that m is strictly greater than n as requested
    if not (args.m > args.n):
        raise ValueError(f"Constraint m > n not satisfied: got m={args.m}, n={args.n}.")

    # Resolve absolute paths
    checkpoint_path = os.path.abspath(args.checkpoint)
    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)

    # Select computation device
    device = get_device(force_cpu=args.cpu)

    # Load model with checkpoint
    model = load_model(checkpoint_path, device, pred_length=args.n)

    # Load input sequence of m frames from directory
    seq, file_list = load_sequence_from_dir(input_dir, m=args.m, pattern=args.pattern)

    # Move sequence to selected device
    seq = seq.to(device)

    # Disable gradient computation for inference
    with torch.no_grad():
        # Run model forward pass, providing pred_length=n
        outputs, _ = model(seq, pred_length=args.n)

    # Save predictions as TIFF images in the output directory
    saved_paths = save_predictions_as_tiff(outputs, output_dir, prefix="pred")

    # Print summary to user
    print(f"Loaded {len(file_list)} input frames from: {input_dir}")
    print(f"Checkpoint used: {checkpoint_path}")
    print(f"Predicted {len(saved_paths)} future frames (n={args.n}).")
    print(f"Saved predictions under: {output_dir}")


# Invoke main when the script is executed directly
if __name__ == "__main__":
    main()
