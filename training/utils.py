# Import time utilities for benchmarking
import time
# Import numpy for numerical operations
import numpy as np
# Import torch for seeding and device synchronization
import torch


# Define a function to set random seeds and CuDNN settings
def set_seed(seed: int = 15) -> None:
    """Set random seeds and backend flags for reproducibility and stable performance."""
    # Set NumPy random seed
    np.random.seed(seed)
    # Set PyTorch CPU random seed
    torch.manual_seed(seed)
    # Set PyTorch CUDA random seed on all devices
    torch.cuda.manual_seed_all(seed)
    # Allow non-deterministic CuDNN operations for better performance
    torch.backends.cudnn.deterministic = False
    # Enable CuDNN auto-tuner for best convolution algorithms
    torch.backends.cudnn.benchmark = True


# Define a helper function to format seconds as HH:MM:SS
def hms(sec: float) -> str:
    """Convert seconds to HH:MM:SS formatted string."""
    # Cast seconds to integer
    sec = int(sec)
    # Compute hours from total seconds
    h = sec // 3600
    # Compute minutes from remaining seconds
    m = (sec % 3600) // 60
    # Compute leftover seconds
    s = sec % 60
    # Format as zero-padded HH:MM:SS string
    return f"{h:02d}:{m:02d}:{s:02d}"


# Define a function to benchmark training throughput
def benchmark_train(
    loader,
    model,
    optimizer,
    device,
    criterion_sl1,
    criterion_fl,
    mae_lambda: float,
    scaler=None,
    warmup: int = 2,
    measure: int = 10,
    pred_length: int = 6,
) -> float:
    """Benchmark training throughput (batches per second)."""
    # Put model in training mode
    model.train()
    # Create iterator from data loader
    it = iter(loader)

    # Perform warmup iterations (not timed)
    for _ in range(warmup):
        # Get next batch from iterator
        batch = next(it)
        # Unpack inputs, targets and mask
        inputs, targets, mask = batch[:3]
        # Move inputs to target device
        inputs = inputs.to(device, non_blocking=True)
        # Move targets to target device
        targets = targets.to(device, non_blocking=True)
        # Move mask to target device
        mask = mask.to(device, non_blocking=True)
        # Clear gradients before backward pass
        optimizer.zero_grad(set_to_none=True)
        # If mixed precision scaler is provided
        if scaler is not None:
            # Use autocast context for mixed precision
            with torch.cuda.amp.autocast():
                # Forward pass through model
                outputs, logits = model(inputs, pred_length)
                print("===============================================")
                print(f"output.shape: {outputs.shape}")
                print(f"targets.shape: {targets.shape}")
                print(f"logits.shape: {logits.shape}")
                print(f"mask.shape: {mask.shape}")
                print("================================================")
                # Compute SmoothL1 loss term
                loss = criterion_sl1(outputs, targets) * mae_lambda + criterion_fl(logits, mask) * mae_lambda
                # Add focal loss term
                # loss += criterion_fl(logits, mask) * mae_lambda
            # Backward pass with scaled loss
            scaler.scale(loss).backward()
            # Perform optimizer step
            scaler.step(optimizer)
            # Update scaling factor
            scaler.update()
        # If not using mixed precision
        else:
            # Forward pass in full precision
            outputs, logits = model(inputs, pred_length)
            # Compute SmoothL1 loss term
            loss = criterion_sl1(outputs, targets) * mae_lambda
            # Add focal loss term
            loss += criterion_fl(logits, mask) * mae_lambda
            # Backward pass
            loss.backward()
            # Optimizer step
            optimizer.step()

    # Synchronize CUDA device before starting timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    # Record start time for benchmarking
    t0 = time.perf_counter()
    # Initialize counter of measured batches
    counted = 0

    # Measure throughput over a fixed number of iterations
    for _ in range(measure):
        try:
            # Try to get next batch from iterator
            batch = next(it)
        except StopIteration:
            # Stop if no more batches are available
            break
        # Unpack inputs, targets and mask
        inputs, targets, mask = batch[:3]
        # Move inputs to device
        inputs = inputs.to(device, non_blocking=True)
        # Move targets to device
        targets = targets.to(device, non_blocking=True)
        # Move mask to device
        mask = mask.to(device, non_blocking=True)
        # Reset gradients
        optimizer.zero_grad(set_to_none=True)
        # If mixed precision is enabled
        if scaler is not None:
            # Use autocast for mixed precision
            with torch.cuda.amp.autocast():
                # Forward pass
                outputs, logits = model(inputs, pred_length)
                # Compute SmoothL1 loss term
                loss = criterion_sl1(outputs, targets) * mae_lambda
                # Add focal loss term
                loss += criterion_fl(logits, mask) * mae_lambda
            # Backpropagate scaled loss
            scaler.scale(loss).backward()
            # Optimizer step
            scaler.step(optimizer)
            # Update scale factor
            scaler.update()
        # If not using mixed precision
        else:
            # Forward pass
            outputs, logits = model(inputs, pred_length)
            # Compute SmoothL1 loss term
            loss = criterion_sl1(outputs, targets) * mae_lambda
            # Add focal loss term
            loss += criterion_fl(logits, mask) * mae_lambda
            # Backpropagation
            loss.backward()
            # Apply optimizer step
            optimizer.step()
        # Increment number of measured batches
        counted += 1

    # Synchronize CUDA device after timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    # Compute elapsed time in seconds
    dt = time.perf_counter() - t0
    # Return measured batches per second
    return counted / dt if dt > 0 else 0.0


# Define a function to benchmark validation throughput
def benchmark_val(
    loader,
    model,
    device,
    warmup: int = 2,
    measure: int = 20,
    pred_length: int = 6,
) -> float:
    """Benchmark validation throughput (batches per second)."""
    # Put model into evaluation mode
    model.eval()
    # Create iterator from loader
    it = iter(loader)
    # Disable gradient computations
    with torch.no_grad():
        # Perform warmup iterations
        for _ in range(warmup):
            # Get next batch
            batch = next(it)
            # Extract inputs from batch
            inputs = batch[0].to(device, non_blocking=True)
            # Forward pass
            _ = model(inputs, pred_length)

        # Synchronize CUDA before timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        # Record start time
        t0 = time.perf_counter()
        # Initialize counter of measured batches
        counted = 0

        # Measure a fixed number of iterations
        for _ in range(measure):
            try:
                # Get next batch
                batch = next(it)
            except StopIteration:
                # Stop when loader is exhausted
                break
            # Move inputs to device
            inputs = batch[0].to(device, non_blocking=True)
            # Forward pass
            _ = model(inputs, pred_length)
            # Increment measured batch count
            counted += 1

        # Synchronize CUDA after timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    # Compute elapsed time in seconds
    dt = time.perf_counter() - t0
    # Return batches per second
    return counted / dt if dt > 0 else 0.0
