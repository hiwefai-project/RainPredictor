
"""
rainpred/model.py

Refactored RainPredModel to fix the Transformer embedding-dimension mismatch.

This implementation assumes inputs of shape:
    (batch, in_length, in_channels, height, width)

and produces outputs of shape:
    (batch, pred_length, out_channels, height, width)

where out_channels is typically 1 (single VMI radar channel).
"""

import logging
from typing import Tuple, Optional, Any

import torch
from torch import nn
from torch.nn import functional as F

log = logging.getLogger(__name__)


class ConvBlock2D(nn.Module):
    """
    Simple 2D convolutional block: Conv2d -> BatchNorm2d -> ReLU.
    Keeps spatial resolution (stride=1, padding=1).
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class Encoder2D(nn.Module):
    """
    Lightweight 2D encoder applied independently to each time step.

    Input:  (B * T, in_channels, H, W)
    Output: (B * T, hidden_channels, H, W)  # same spatial resolution
    """

    def __init__(self, in_channels: int, hidden_channels: int):
        super().__init__()
        self.block1 = ConvBlock2D(in_channels, hidden_channels)
        self.block2 = ConvBlock2D(hidden_channels, hidden_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        return x


class RainPredModel(nn.Module):
    """
    CNN + temporal Transformer model for radar nowcasting.

    The Transformer is applied along the time dimension only.
    Spatial dimensions are globally averaged before the Transformer and
    re-injected via broadcasting onto the last encoder feature map.

    Forward signature matches existing training code:

        outputs, extra = model(inputs, pred_length)

    where:
        inputs  : (B, in_length, in_channels, H, W)
        outputs : (B, pred_length, out_channels, H, W)
        extra   : dict with optional auxiliary data (currently empty).
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        hidden_channels: int = 64,
        transformer_d_model: int = 128,
        transformer_nhead: int = 8,
        transformer_num_layers: int = 2,
        pred_length: int = 6,
    ) -> None:
        super().__init__()

        if transformer_d_model % transformer_nhead != 0:
            raise ValueError(
                f"transformer_d_model ({transformer_d_model}) must be divisible "
                f"by transformer_nhead ({transformer_nhead})."
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.transformer_d_model = transformer_d_model
        self.pred_length = pred_length

        # Spatial encoder (per time step)
        self.encoder = Encoder2D(in_channels=in_channels, hidden_channels=hidden_channels)

        # 1x1 conv to match encoder channels to Transformer d_model
        self.enc_to_trans = nn.Conv2d(hidden_channels, transformer_d_model, kernel_size=1)
        self.trans_to_dec = nn.Conv2d(transformer_d_model, hidden_channels, kernel_size=1)

        # Temporal Transformer (sequence over time, batch_first = True => (B, T, E))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_d_model,
            nhead=transformer_nhead,
            dim_feedforward=transformer_d_model * 4,
            batch_first=True,
            dropout=0.1,
            activation="relu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=transformer_num_layers,
        )

        # Final 2D conv that produces one feature map per future time step.
        # Shape: (B, pred_length, H, W) which is then reshaped to
        # (B, pred_length, out_channels, H, W).
        self.frame_head = nn.Conv2d(hidden_channels, pred_length * out_channels, kernel_size=3, padding=1)

        log.info(
            "Initialized RainPredModel: in_channels=%d, out_channels=%d, "
            "hidden_channels=%d, transformer_d_model=%d, pred_length=%d",
            in_channels,
            out_channels,
            hidden_channels,
            transformer_d_model,
            pred_length,
        )

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode the input sequence with the 2D CNN encoder.

        Args:
            x: (B, T, C_in, H, W)

        Returns:
            enc_stack: (B, T, transformer_d_model, H, W)
        """
        b, t, c, h, w = x.shape
        if c != self.in_channels:
            raise ValueError(f"Expected in_channels={self.in_channels}, got {c}")

        # Merge batch and time, encode, then un-merge.
        x_flat = x.view(b * t, c, h, w)                       # (B*T, C_in, H, W)
        feat_flat = self.encoder(x_flat)                      # (B*T, hidden_channels, H, W)
        feat_flat = self.enc_to_trans(feat_flat)              # (B*T, transformer_d_model, H, W)
        feat = feat_flat.view(b, t, self.transformer_d_model, h, w)  # (B, T, E, H, W)
        return feat

    def forward(
        self,
        x: torch.Tensor,
        pred_length: Optional[int] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass.

        Args:
            x: Input tensor, shape (B, in_length, in_channels, H, W).
            pred_length: Number of future steps to predict. If None, uses
                         the configured self.pred_length. If provided and
                         different, an error is raised.

        Returns:
            outputs: (B, pred_length, out_channels, H, W)
            extra:   dict (currently empty, placeholder for future use).
        """
        if pred_length is None:
            pred_length = self.pred_length
        elif pred_length != self.pred_length:
            raise ValueError(
                f"pred_length={pred_length} does not match model.pred_length={self.pred_length}. "
                f"For this version, they must be equal."
            )

        # ---- Encode full observed sequence --------------------------------
        enc_stack = self.encode_sequence(x)                   # (B, T, E, H, W)
        b, t, e, h, w = enc_stack.shape

        # ---- Temporal Transformer over globally pooled features -----------
        # Global average pool spatial dims: (B, T, E, H, W) -> (B, T, E)
        seq_feat = enc_stack.mean(dim=(-1, -2))               # (B, T, E)

        # Transformer expects (B, T, E) because batch_first=True.
        trans_feat = self.transformer(seq_feat)               # (B, T, E)

        # Use the last time step of the Transformer output as global
        # temporal summary and broadcast it over the last encoded frame.
        last_token = trans_feat[:, -1, :]                     # (B, E)
        last_token_map = last_token.view(b, e, 1, 1)          # (B, E, 1, 1)

        # Last encoded spatial map:
        last_enc = enc_stack[:, -1, :, :, :]                  # (B, E, H, W)

        # Fuse temporal and spatial information
        fused = last_enc + last_token_map                     # (B, E, H, W)

        # Map back to decoder channels
        hidden = self.trans_to_dec(fused)                     # (B, hidden_channels, H, W)

        # Produce future frames in a single shot:
        # (B, pred_length * out_channels, H, W)
        frames = self.frame_head(hidden)

        # Reshape to (B, pred_length, out_channels, H, W)
        frames = frames.view(
            b,
            self.pred_length,
            self.out_channels,
            h,
            w,
        )

        extra: dict[str, Any] = {}
        return frames, extra


def build_model_from_config(cfg: dict) -> RainPredModel:
    """
    Convenience builder so that existing code using a config dict can still work.

    Expected keys in cfg (all optional, with defaults matching RainPredModel.__init__):
        - in_channels
        - out_channels
        - hidden_channels
        - transformer_d_model
        - transformer_nhead
        - transformer_num_layers
        - pred_length
    """
    return RainPredModel(
        in_channels=cfg.get("in_channels", 1),
        out_channels=cfg.get("out_channels", 1),
        hidden_channels=cfg.get("hidden_channels", 64),
        transformer_d_model=cfg.get("transformer_d_model", 128),
        transformer_nhead=cfg.get("transformer_nhead", 8),
        transformer_num_layers=cfg.get("transformer_num_layers", 2),
        pred_length=cfg.get("pred_length", 6),
    )
