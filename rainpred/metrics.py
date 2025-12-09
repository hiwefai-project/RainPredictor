"""
metrics.py
==========
Metrics and evaluation utilities for RainPredictor.

This module implements:
  - a rain-weighted SmoothL1 loss (rain_weighted_sl1),
  - a PyTorch-only calculation of SSIM and a soft CSI,
  - a composite TOTAL metric that depends on SmoothL1, MAE, SSIM and CSI,
  - an evaluation loop (evaluate) that aggregates metrics over a DataLoader.

The public interfaces of the following functions are preserved:
  * rain_weighted_sl1(preds, targets, threshold_dbz=15.0, rain_boost=4.0)
  * calculate_metrics(preds, targets, threshold_dbz=15.0)
  * evaluate(model, loader, device, pred_length, threshold_dbz=15.0)
"""

# Import the core PyTorch package
import torch  # type: ignore
# Import functional API (for loss functions and convolutions)
import torch.nn.functional as F  # type: ignore
# Import confusion_matrix to build a global CSI confusion matrix in evaluate
from sklearn.metrics import confusion_matrix  # type: ignore

# ------------------------------------------------------------
# Global configuration / hyperparameters
# ------------------------------------------------------------

# Maximum reflectivity (in dBZ) corresponding to value 1.0 in [0,1] space
MAX_DBZ = 70.0  # 70 dBZ is a typical upper bound for weather radar reflectivity

# Global multiplier historically used in TOTAL in the original code
# We keep it for scale compatibility when combining losses
criterion_mae_lambda = 10.0  # This scales the SmoothL1 contribution

# Weights for the different components in the TOTAL metric
W_SL1 = 1.0  # Weight for SmoothL1 loss in normalized space
W_MAE = 1.0  # Weight for MAE loss in normalized space
W_SSIM = 0.5  # Weight for structural similarity (converted to a loss)
W_CSI = 0.5  # Weight for soft CSI (converted to a loss)

# Sharpness of the soft threshold used for CSI (sigmoid slope)
CSI_ALPHA = 5.0  # Larger values approximate a hard threshold more closely

# Base SmoothL1 loss layer used by rain_weighted_sl1 (no reduction)
criterion_sl1 = torch.nn.SmoothL1Loss(reduction="none")  # compute element-wise SmoothL1


# ------------------------------------------------------------
# Utility functions for SSIM (all in PyTorch)
# ------------------------------------------------------------

def _gaussian_window(window_size: int, sigma: float, device, dtype):
    """
    Create a 2D Gaussian window to be used as a convolution kernel.

    Args:
        window_size (int): The size of the square kernel (e.g., 5x5).
        sigma (float): Standard deviation of the Gaussian.
        device: PyTorch device on which to allocate the tensor.
        dtype: PyTorch dtype for the resulting tensor.

    Returns:
        torch.Tensor: 2D Gaussian kernel of shape (1, 1, window_size, window_size).
    """
    # Create a 1D coordinate grid centered around zero
    coords = torch.arange(window_size, device=device, dtype=dtype)  # values 0..window_size-1
    coords = coords - (window_size - 1) / 2.0  # shift so that the center is at zero

    # Apply the 1D Gaussian formula along the coordinate axis
    g_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))  # unnormalized 1D Gaussian

    # Normalize the 1D kernel so that it sums to 1
    g_1d = g_1d / g_1d.sum()  # ensure area under the curve is 1

    # Build a 2D separable kernel via the outer product of the 1D kernel
    g_2d = torch.outer(g_1d, g_1d)  # shape (window_size, window_size)

    # Add batch and channel dimensions so it can be used with conv2d
    window = g_2d.unsqueeze(0).unsqueeze(0)  # shape becomes (1, 1, K, K)
    return window  # return the Gaussian kernel


def _ssim_torch(x, y, window_size: int = 5, sigma: float = 1.5, data_range: float = MAX_DBZ):
    """
    Compute SSIM (Structural Similarity Index) between two image batches using PyTorch.

    Args:
        x (torch.Tensor): Reference images of shape (N, 1, H, W).
        y (torch.Tensor): Predicted images of shape (N, 1, H, W).
        window_size (int): Size of the Gaussian window used for local statistics.
        sigma (float): Standard deviation of the Gaussian window.
        data_range (float): Max-min range of the pixel values (used in SSIM constants).

    Returns:
        torch.Tensor: Scalar tensor containing the mean SSIM over batch and spatial dimensions.
    """
    # Make sure x and y live on the same device and use the same dtype
    device = x.device  # device where tensors reside (CPU or GPU)
    dtype = x.dtype  # data type used by the tensors

    # Build the Gaussian window once for this call
    window = _gaussian_window(window_size, sigma, device, dtype)  # (1,1,K,K)

    # Compute padding so output of conv2d has same H, W as input
    padding = window_size // 2  # symmetric padding on all sides

    # Compute local means via convolution
    mu_x = torch.nn.functional.conv2d(x, window, padding=padding, groups=1)  # local mean of x
    mu_y = torch.nn.functional.conv2d(y, window, padding=padding, groups=1)  # local mean of y

    # Compute squares of the local means
    mu_x2 = mu_x * mu_x  # mu_x squared
    mu_y2 = mu_y * mu_y  # mu_y squared

    # Compute product of local means
    mu_xy = mu_x * mu_y  # mu_x times mu_y

    # Compute local variances and covariance
    sigma_x2 = torch.nn.functional.conv2d(x * x, window, padding=padding, groups=1) - mu_x2  # variance of x
    sigma_y2 = torch.nn.functional.conv2d(y * y, window, padding=padding, groups=1) - mu_y2  # variance of y
    sigma_xy = torch.nn.functional.conv2d(x * y, window, padding=padding, groups=1) - mu_xy  # covariance of x,y

    # Standard SSIM constants scaled by data_range
    C1 = (0.01 * data_range) ** 2  # contrast constant C1
    C2 = (0.03 * data_range) ** 2  # contrast constant C2

    # Compute SSIM numerator: (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)  # SSIM numerator

    # Compute SSIM denominator: (mu_x^2 + mu_y^2 + C1) * (sigma_x^2 + sigma_y^2 + C2)
    den = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)  # SSIM denominator

    # Avoid division by zero by adding a tiny epsilon
    ssim_map = num / (den + 1e-12)  # point-wise SSIM values

    # Return the mean SSIM over all pixels and all images
    return ssim_map.mean()  # scalar tensor with SSIM


# ------------------------------------------------------------
# Rain-weighted SmoothL1 loss
# ------------------------------------------------------------

def rain_weighted_sl1(preds, targets, threshold_dbz: float = 15.0, rain_boost: float = 4.0):
    """
    Compute a rain-weighted SmoothL1 loss.

    This function gives more importance to pixels where it is actually
    raining according to the ground truth, i.e. where reflectivity
    exceeds the specified threshold in dBZ.

    Args:
        preds (torch.Tensor): Model predictions in normalized space,
                              shape (B, T, 1, H, W), values in [-1, 1].
        targets (torch.Tensor): Ground truth in normalized space,
                                same shape and range as preds.
        threshold_dbz (float): Reflectivity threshold (in dBZ) that defines "rain".
        rain_boost (float): Additional multiplicative weight for rainy pixels.

    Returns:
        torch.Tensor: Scalar tensor containing the rain-weighted SmoothL1 loss.
    """
    # Compute element-wise SmoothL1 loss without any reduction
    base_loss = criterion_sl1(preds, targets)  # shape (B, T, 1, H, W)

    # Convert targets from [-1, 1] to [0, 1]
    targets_01 = targets * 0.5 + 0.5  # normalize target values to [0, 1]

    # Map [0, 1] normalized values to [0, MAX_DBZ] dBZ
    targets_dbz = targets_01 * MAX_DBZ  # reflectivity in dBZ scale

    # Build a binary mask of rainy pixels according to ground truth
    rain_mask = (targets_dbz >= threshold_dbz).float()  # 1 where rain, 0 elsewhere

    # Compute pixel-wise weights: base weight 1, plus a boost on rainy pixels
    weights = 1.0 + rain_boost * rain_mask  # shape (B, T, 1, H, W)

    # Apply weights to the element-wise loss
    weighted_loss = base_loss * weights  # shape (B, T, 1, H, W)

    # Compute the normalized average: sum(loss * w) / sum(w)
    loss = weighted_loss.sum() / weights.sum().clamp_min(1.0)  # scalar tensor

    # Return the final scalar loss tensor
    return loss  # rain-weighted SmoothL1 loss


# ------------------------------------------------------------
# Main metrics: SmoothL1, MAE, TOTAL, SSIM, CSI
# ------------------------------------------------------------

def calculate_metrics(preds, targets, threshold_dbz: float = 15.0):
    """
    Compute all validation metrics in a PyTorch-friendly way.

    This function keeps the original public interface used in the
    training and evaluation code. It returns a dictionary with scalar
    floats so that the caller can easily log them.

    Args:
        preds (torch.Tensor): Model predictions in normalized space,
                              shape (B, T, 1, H, W), values in [-1, 1].
        targets (torch.Tensor): Ground truth in normalized space,
                                same shape and range as preds.
        threshold_dbz (float): Reflectivity threshold used for CSI (in dBZ).

    Returns:
        dict: Dictionary with keys:
              'TOTAL'    : composite metric combining all components,
              'SmoothL1' : unweighted SmoothL1 loss in normalized space,
              'MAE'      : L1 loss in normalized space,
              'SSIM'     : structural similarity in dBZ space,
              'CSI'      : soft critical success index in dBZ space.
    """
    # Ensure preds and targets are floating-point tensors
    preds = preds.float()  # cast predictions to float if needed
    targets = targets.float()  # cast targets to float if needed

    # --------------------------------------------------------
    # Base losses in normalized space (range [-1, 1])
    # --------------------------------------------------------

    # Standard SmoothL1 loss averaged over all elements
    sl1 = torch.nn.functional.smooth_l1_loss(preds, targets)  # scalar tensor with SmoothL1

    # Standard L1 loss (MAE) averaged over all elements
    mae = torch.nn.functional.l1_loss(preds, targets)  # scalar tensor with MAE

    # --------------------------------------------------------
    # Conversion to dBZ for physically meaningful metrics
    # --------------------------------------------------------

    # Map from [-1, 1] to [0, 1] for both predictions and targets
    preds_01 = preds * 0.5 + 0.5  # normalized predictions in [0, 1]
    targets_01 = targets * 0.5 + 0.5  # normalized targets in [0, 1]

    # Map from [0, 1] to [0, MAX_DBZ] and clamp to the valid range
    preds_dbz = torch.clamp(preds_01 * MAX_DBZ, 0.0, MAX_DBZ)  # predictions in dBZ
    targets_dbz = torch.clamp(targets_01 * MAX_DBZ, 0.0, MAX_DBZ)  # targets in dBZ

    # Unpack batch dimensions for clarity
    B, T, C, H, W = preds_dbz.shape  # batch, time, channel, height, width

    # --------------------------------------------------------
    # SSIM computation in dBZ space
    # --------------------------------------------------------

    # Flatten time dimension into the batch dimension for convenience
    preds_flat = preds_dbz.view(B * T, 1, H, W)  # merge B and T into N = B*T
    targets_flat = targets_dbz.view(B * T, 1, H, W)  # same reshape for targets

    # Compute SSIM using the pure PyTorch helper function
    ssim_val = _ssim_torch(
        targets_flat,  # reference images
        preds_flat,  # predicted images
        window_size=5,  # size of the Gaussian kernel
        sigma=1.5,  # standard deviation of Gaussian
        data_range=MAX_DBZ,  # reflectivity range for SSIM constants
    )  # scalar tensor with SSIM

    # --------------------------------------------------------
    # Soft CSI computation in dBZ space
    # --------------------------------------------------------

    # Build soft masks for predictions and targets using a sigmoid
    # The CSI_ALPHA factor controls how sharp the threshold is
    soft_pred = torch.sigmoid(CSI_ALPHA * (preds_dbz - threshold_dbz))  # soft rain mask for preds
    soft_true = torch.sigmoid(CSI_ALPHA * (targets_dbz - threshold_dbz))  # soft rain mask for targets

    # Compute soft intersection between prediction and target masks
    intersection = (soft_pred * soft_true).sum()  # sum over all dimensions

    # Compute soft union between prediction and target masks
    union = (soft_pred + soft_true - soft_pred * soft_true).sum()  # soft union

    # Handle the corner case of no rain at all (union ~ 0)
    csi_val = torch.where(
        union > 0.0,  # condition: rain is present somewhere
        intersection / (union + 1e-12),  # normal CSI when union > 0
        torch.ones_like(intersection),  # perfect score (1.0) when no rain at all
    )  # scalar tensor with CSI

    # --------------------------------------------------------
    # Composite TOTAL metric
    # --------------------------------------------------------

    # Combine the different components into a single TOTAL metric.
    # SSIM and CSI are "higher is better", so we turn them into
    # loss-like quantities via (1 - metric).
    total = (
        criterion_mae_lambda * W_SL1 * sl1  # scaled SmoothL1 contribution
        + W_MAE * mae  # MAE contribution
        + W_SSIM * (1.0 - ssim_val)  # structural dissimilarity contribution
        + W_CSI * (1.0 - csi_val)  # missed/false rain detection contribution
    )  # scalar tensor with TOTAL

    # --------------------------------------------------------
    # Prepare a dictionary of scalar floats for logging
    # --------------------------------------------------------

    metrics = {
        "TOTAL": float(total.detach().cpu()),  # composite TOTAL metric as Python float
        "SmoothL1": float(sl1.detach().cpu()),  # SmoothL1 as Python float
        "MAE": float(mae.detach().cpu()),  # MAE as Python float
        "SSIM": float(ssim_val.detach().cpu()),  # SSIM as Python float
        "CSI": float(csi_val.detach().cpu()),  # CSI as Python float
    }  # dictionary with scalar metrics

    # Return the metrics dictionary
    return metrics  # contains TOTAL, SmoothL1, MAE, SSIM, CSI


# ------------------------------------------------------------
# Evaluation loop
# ------------------------------------------------------------

def evaluate(model, loader, device, pred_length, threshold_dbz: float = 15.0):
    """
    Evaluate the model on a validation loader and compute metrics.

    This function preserves the original interface and behaviour:
    it iterates over the DataLoader, runs the model in evaluation
    mode, accumulates metrics from calculate_metrics, and builds a
    global confusion matrix for rain / no-rain classification.

    Args:
        model (torch.nn.Module): The model to evaluate.
        loader (DataLoader): DataLoader providing validation batches.
        device: PyTorch device where computations are performed.
        pred_length (int): Number of future frames the model should predict.
        threshold_dbz (float): Reflectivity threshold (in dBZ) used for CSI.

    Returns:
        tuple: (avg_metrics, cm) where
               avg_metrics is a dict with averaged metrics over the loader,
               cm is a 2x2 confusion matrix (no-rain vs rain).
    """
    # Put the model in evaluation mode (disables dropout, etc.)
    model.eval()  # set model to eval mode

    # Initialize accumulators for averaged metrics
    agg = {
        "TOTAL": 0.0,  # accumulator for TOTAL metric
        "SmoothL1": 0.0,  # accumulator for SmoothL1
        "MAE": 0.0,  # accumulator for MAE
        "SSIM": 0.0,  # accumulator for SSIM
        "CSI": 0.0,  # accumulator for CSI
    }  # dictionary for aggregated metrics

    # Lists used to build the global confusion matrix
    all_preds = []  # will store flattened predicted rain / no-rain labels
    all_targets = []  # will store flattened target rain / no-rain labels

    # Disable gradient computation during evaluation
    with torch.no_grad():  # no gradients needed
        # Iterate over all batches from the DataLoader
        for batch in loader:  # loop over validation batches
            # Some loaders may provide (inputs, targets, mask, extra)
            if len(batch) == 4:  # when four elements are returned
                inputs, targets, mask, _ = batch  # unpack inputs, targets, mask, and ignore extra
            else:  # otherwise we expect (inputs, targets, mask)
                inputs, targets, mask = batch  # unpack inputs, targets, mask

            # Move inputs and targets to the desired device
            inputs = inputs.to(device)  # send input batch to device
            targets = targets.to(device)  # send target batch to device

            # Forward pass: model returns predictions and optional extra outputs
            outputs, _ = model(inputs, pred_length)  # run the model for pred_length steps

            # Compute metrics for this batch using the provided function
            batch_metrics = calculate_metrics(outputs, targets, threshold_dbz)  # dict of floats

            # Accumulate metrics for later averaging
            for key in agg:  # iterate over all metric names
                agg[key] += batch_metrics[key]  # add current batch metric to accumulator

            # ----------------------------------------------------
            # Build arrays for the global confusion matrix
            # ----------------------------------------------------

            # Convert predictions and targets back to [0, 1] on CPU
            preds_01 = (outputs.detach().cpu() * 0.5) + 0.5  # predictions in [0, 1]
            targets_01 = (targets.detach().cpu() * 0.5) + 0.5  # targets in [0, 1]

            # Convert [0, 1] to [0, MAX_DBZ] in dBZ
            preds_dbz = torch.clamp(preds_01 * MAX_DBZ, 0.0, MAX_DBZ)  # predictions in dBZ
            targets_dbz = torch.clamp(targets_01 * MAX_DBZ, 0.0, MAX_DBZ)  # targets in dBZ

            # Build hard binary masks for rain / no-rain using the threshold
            preds_bin = (preds_dbz > threshold_dbz).to(torch.uint8).view(-1)  # flatten predictions
            targets_bin = (targets_dbz > threshold_dbz).to(torch.uint8).view(-1)  # flatten targets

            # Extend the Python lists with the batch labels
            all_preds.extend(preds_bin.tolist())  # add predicted labels for this batch
            all_targets.extend(targets_bin.tolist())  # add target labels for this batch

    # Compute the number of batches processed (avoid division by zero)
    num_batches = max(len(loader), 1)  # at least 1 to avoid division by zero

    # Average the accumulated metrics over the number of batches
    for key in agg:  # loop over all metric names
        agg[key] /= num_batches  # compute the mean value for each metric

    # Build the 2x2 confusion matrix using sklearn (0 = no-rain, 1 = rain)
    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1])  # confusion matrix

    # Return averaged metrics and the confusion matrix
    return agg, cm  # (avg_metrics, confusion_matrix)

