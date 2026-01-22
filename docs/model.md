# RainPredictor Model Details

## Overview

RainPredictor uses a compact CNN + Transformer architecture designed for radar nowcasting on full-resolution GeoTIFF frames.
The model is optimized to ingest a sequence of past radar images and predict a fixed-length sequence of future frames
without resizing inputs, preserving the original spatial resolution and georeferencing.

## Core architecture (rainpred/model.py)

The model is implemented in `rainpred/model.py` and is organized into three primary stages.
Each step below maps directly to the operations in the forward pass, so you can trace the computation end-to-end.

1. **Per-time-step spatial encoding (Encoder2D)**
   - Each input frame is passed through two 2D convolutional blocks (Conv2d → BatchNorm → ReLU).
   - Spatial resolution is preserved by using stride 1 with padding.
   - This produces a feature map for each time step that still aligns with the input grid.

2. **Temporal modeling (Transformer over time)**
   - The encoded feature maps are projected to the Transformer embedding dimension with a 1×1 convolution.
   - Spatial dimensions are globally averaged, so the Transformer only sees a sequence of time tokens.
   - A Transformer Encoder processes the sequence to capture temporal dynamics across the input frames.
   - The final time token summarizes the observed sequence and is broadcast back over the spatial grid.

3. **Frame synthesis (prediction head)**
   - The temporal summary is fused with the last encoded spatial map via element-wise addition.
   - A 1×1 convolution projects the fused features back to the hidden channel width.
   - A final 3×3 convolution predicts `pred_length × out_channels` maps in one shot.
   - The tensor is reshaped into `(batch, pred_length, out_channels, height, width)` for downstream use.

## Input and output tensor shapes

The model follows strict shape conventions throughout the pipeline:

- **Input**: `(batch, in_length, in_channels, height, width)`
- **Output**: `(batch, pred_length, out_channels, height, width)`

`pred_length` is fixed at initialization time. If a different value is passed at inference time, the model
throws a validation error so that training and inference remain aligned.

## Configuration parameters

The `RainPredModel` constructor exposes the following tunable parameters:

- `in_channels`: number of channels per input frame (default: 1 for VMI radar reflectivity).
- `out_channels`: number of channels per output frame (default: 1).
- `hidden_channels`: width of the CNN encoder/decoder features (default: 64).
- `transformer_d_model`: Transformer embedding dimension (default: 128).
- `transformer_nhead`: number of attention heads (default: 8).
- `transformer_num_layers`: number of Transformer encoder layers (default: 2).
- `pred_length`: number of future frames predicted per forward pass (default: 6).

A helper, `build_model_from_config`, reads these values from a config dictionary so training code can
be driven by configuration files or CLI overrides.

## Training and inference considerations

- The model expects full-resolution GeoTIFF inputs; preprocessing uses zero-padding to satisfy patch constraints.
- Predictions retain the original geospatial metadata when saved, ensuring GIS compatibility.
- Multi-GPU training uses `torch.nn.DataParallel`, and checkpoints remain loadable for single-GPU inference.

## When to adjust the model

- Increase `hidden_channels` or `transformer_d_model` for more capacity on large datasets.
- Increase `pred_length` when you want longer forecast horizons, but keep training data and evaluation
  in sync with that choice.
- Adjust `transformer_nhead` and `transformer_num_layers` together to balance temporal context vs. runtime.
