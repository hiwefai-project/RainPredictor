"""
Fully commented version of the corrected RainPred model.
Removes tanh activation, supports higher dynamic range,
and prepares the architecture for stable learning.
"""

import torch
import torch.nn as nn
from einops import rearrange

from .config import PRED_LENGTH, PATCH_HEIGHT, PATCH_WIDTH


# =============================
#  U-NET ENCODER
# =============================
class UNet_Encoder(nn.Module):
    """U-Net encoder extracting spatial features from each radar image."""

    def __init__(self, input_channels: int):
        super().__init__()

        # First convolution block: keep resolution, increase channels to 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, 1, 1, bias=False),  # conv layer
            nn.BatchNorm2d(32),                                  # normalize
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )

        # Second block: downsample spatial size by factor 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 2, 1, bias=False),  # stride=2 halves H,W
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, 1, 1, bias=False),  # increase channels to 128
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )

    def forward(self, x):
        """
        Args:
            x: (B, 1, H, W) normalized radar input frame

        Returns:
            x_down: low-resolution features (B, 128, H/2, W/2)
            skip1:  high-resolution features for U-Net skip connection (B,64,H,W)
        """
        x = self.conv1(x)
        skip1 = x                   # high-resolution skip before downsampling
        x = self.conv2(x)           # downsampled representation
        return x, skip1


# =============================
#  U-NET DECODER
# =============================
class UNet_Decoder(nn.Module):
    """Reconstruct spatial radar fields from low-resolution + skip connections."""

    def __init__(self, output_channels: int):
        super().__init__()

        # First decoder block: merge encoder + transformer features (128+128=256)
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )

        # Second refinement block
        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )

        # Upsampling: return from H/2,W/2 to H,W
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, 1),  # reduce channels
        )

        # Third U-Net merge block: combine skip connection
        self.conv7 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, 3, 1, 1, bias=False),  # skip1 has 64 channels
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )

        # Final refinement
        self.conv8 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )

        # Final mapping to 1 channel (no activation)
        self.final_conv = nn.Conv2d(64, output_channels, kernel_size=1)

    def forward(self, x, skip0, skip1):
        """
        Args:
            x:     transformer features (B,128,H/2,W/2)
            skip0: encoder low-res features
            skip1: encoder high-res features

        Returns:
            out:     prediction (B,1,H,W)
            out_raw: pre-activation output
        """
        # Concatenate transformer output + encoder features
        x = torch.cat([skip0, x], dim=1)

        # Decode low-res fused feature
        x = self.conv5(x)
        x = self.conv6(x)

        # Upsample
        x = self.up1(x)

        # Merge with high-resolution skip connection
        x = torch.cat([skip1, x], dim=1)
        x = self.conv7(x)
        x = self.conv8(x)

        # Final output (NO tanh, full-range regression)
        out_raw = self.final_conv(x)
        return out_raw, out_raw  # for compatibility


# =============================
#  TEMPORAL TRANSFORMER
# =============================
class TemporalTransformerBlock(nn.Module):
    """
    Temporal + spatial attention over 2D feature sequences.
    """

    def __init__(self, d_model=128, nhead=4, num_layers=2):
        super().__init__()

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x):
        """
        Args:
            x: (B,T,128,H/2,W/2)

        Returns:
            (B,T,128,H/2,W/2)
        """
        B, T, C, H, W = x.shape

        # Flatten spatial dims so transformer attends over time+patches
        x = rearrange(x, "b t c h w -> b t (c h w)")

        # Apply transformer
        x = self.encoder(x)

        # Restore shape
        x = rearrange(x, "b t (c h w) -> b t c h w", c=C, h=H, w=W)
        return x


# =============================
#  MAIN RAINPRED MODEL
# =============================
class RainPredRNN(nn.Module):
    """UNet encoder + Temporal Transformer + UNet decoder."""

    def __init__(
        self,
        input_dim=1,
        num_hidden=256,
        max_hidden_channels=128,
        patch_height=PATCH_HEIGHT,
        patch_width=PATCH_WIDTH,
        pred_length=PRED_LENGTH,
    ):
        super().__init__()

        # Spatial encoder for each frame
        self.encoder = UNet_Encoder(input_dim)

        # Spatial decoder to reconstruct frames
        self.decoder = UNet_Decoder(output_channels=input_dim)

        # Temporal transformer operating on encoded features
        self.transformer = TemporalTransformerBlock(
            d_model=max_hidden_channels,
            nhead=4,
            num_layers=2,
        )

        self.pred_length = pred_length

    def forward(self, x, pred_length=None):
        """
        Args:
            x: (B,T_in,1,H,W) normalized radar sequence
        Returns:
            (B,T_out,1,H,W)
        """
        if pred_length is None:
            pred_length = self.pred_length

        B, T_in, C, H, W = x.shape

        # ----------------------------
        # Encode each input frame
        # ----------------------------
        enc_list = []
        skip1_list = []
        for t in range(T_in):
            enc, skip1 = self.encoder(x[:, t])
            enc_list.append(enc)
            skip1_list.append(skip1)

        # Stack temporally
        enc_stack = torch.stack(enc_list, 1)     # (B,T_in,128,H/2,W/2)
        skip1_stack = torch.stack(skip1_list, 1) # (B,T_in,64,H,W)

        # ----------------------------
        # Temporal transformer
        # ----------------------------
        trans_out = self.transformer(enc_stack)  # same shape

        # Use last encoded frame to produce all outputs
        last_enc = enc_stack[:, -1]
        last_trans = trans_out[:, -1]
        last_skip1 = skip1_stack[:, -1]

        # Repeat over prediction horizon
        preds = []
        preds_raw = []
        for t in range(pred_length):
            out, out_raw = self.decoder(last_trans, last_enc, last_skip1)
            preds.append(out)
            preds_raw.append(out_raw)

        return torch.stack(preds, 1), torch.stack(preds_raw, 1)
