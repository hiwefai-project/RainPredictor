# Import math for positional encoding computations
import math
# Import typing helpers for type hints
from typing import Tuple

# Import torch core library and neural network modules
import torch
import torch.nn as nn
# Import einops rearrange for patch reshaping
from einops import rearrange
# Import einops Rearrange layer for patch embedding
from einops.layers.torch import Rearrange


# Define a U-Net encoder for a single radar frame
class UNetEncoder(nn.Module):
    """U-Net encoder that extracts spatial features from a single-frame input."""

    # Initialize encoder with number of input channels
    def __init__(self, input_channels: int):
        # Call base class constructor
        super().__init__()
        # First convolutional block: conv -> batchnorm -> ReLU
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # Second convolutional block
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # Max pooling to downsample spatial resolution by factor 2
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Third convolutional block, doubling channels
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        # Fourth convolutional block at same channel size
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

    # Forward pass for a single frame
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input x and return encoded features and skip connection."""
        # Apply first conv block
        x = self.conv1(x)
        # Apply second conv block
        x = self.conv2(x)
        # Save skip connection at full resolution
        skip1 = x
        # Downsample spatially with max pooling
        x = self.pool1(x)
        # Apply third conv block at lower resolution
        x = self.conv3(x)
        # Apply fourth conv block
        x = self.conv4(x)
        # Return encoded features and skip connection
        return x, skip1


# Define U-Net decoder for a single radar frame
class UNetDecoder(nn.Module):
    """U-Net decoder that reconstructs a frame from encoded features and skips."""

    # Initialize decoder with desired number of output channels
    def __init__(self, output_channels: int):
        # Call base class constructor
        super().__init__()
        # First decoder conv block, input is concatenation of two 128-channel tensors (256 channels)
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        # Second decoder conv block
        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        # Transposed convolution for upsampling back to original resolution
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # Third decoder conv block after concatenation with skip connection
        self.conv7 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # Fourth decoder conv block
        self.conv8 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # Final 1x1 convolution to map to desired output channels
        self.final_conv = nn.Conv2d(64, output_channels, kernel_size=1)

    # Forward pass for a single decoded frame
    def forward(
        self,
        x: torch.Tensor,
        skip0: torch.Tensor,
        skip1: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode features x using skip connections skip0 and skip1."""
        # Concatenate encoded features at this time step with temporal context
        x = torch.cat([skip0, x], dim=1)
        # Apply conv5
        x = self.conv5(x)
        # Apply conv6
        x = self.conv6(x)
        # Upsample spatially
        x = self.up1(x)
        # Concatenate with high-resolution skip connection
        x = torch.cat([skip1, x], dim=1)
        # Apply conv7
        x = self.conv7(x)
        # Apply conv8
        x = self.conv8(x)
        # Preserve last linear feature map before activation
        x_last = x
        # Apply tanh activation to get outputs in [-1, 1]
        x = torch.tanh(x)
        # Return activated output and raw feature map
        return x, x_last


# Define function to generate sinusoidal positional encodings
def generate_positional_encoding(
    seq_len: int,
    d_model: int,
    device: torch.device,
) -> torch.Tensor:
    """Generate sinusoidal positional encodings of shape (1, seq_len, d_model)."""
    # Initialize encoding tensor with zeros
    pe = torch.zeros(seq_len, d_model, device=device)
    # Create vector of positions [0, 1, ..., seq_len-1]
    position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)
    # Compute scaling terms for even dimensions
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device).float() * (-math.log(10000.0) / d_model)
    )
    # Apply sine to even indices in the embedding
    pe[:, 0::2] = torch.sin(position * div_term)
    # Apply cosine to odd indices in the embedding
    pe[:, 1::2] = torch.cos(position * div_term)
    # Add batch dimension and return
    return pe.unsqueeze(0)


# Define temporal Transformer over encoded spatio-temporal patches
class TemporalTransformerBlock(nn.Module):
    """Temporal Transformer block applied over patch-embedded feature sequences."""

    # Initialize temporal Transformer block
    def __init__(
        self,
        channels: int,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        pred_length: int,
        patch_height: int,
        patch_width: int,
    ):
        # Call base class constructor
        super().__init__()
        # Store prediction length
        self.pred_length = pred_length
        # Store patch height
        self.patch_height = patch_height
        # Store patch width
        self.patch_width = patch_width
        # Compute flattened patch dimension
        patch_dim = channels * patch_height * patch_width

        # Define patch embedding: rearrange -> layer norm -> linear -> layer norm
        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b t c (h p1) (w p2) -> b (t h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, d_model),
            nn.LayerNorm(d_model),
        )

        # Define Transformer encoder stack
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True),
            num_layers=num_encoder_layers,
        )

        # Define projection back to patch space
        self.to_feature_map = nn.Sequential(
            nn.Linear(d_model, patch_dim),
            nn.LayerNorm(patch_dim),
        )

    # Forward pass for temporal Transformer
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass given encoded inputs x of shape (B, Tin, C, H, W)."""
        # Extract height and width from tensor
        H, W = x.shape[-2:]
        # Compute number of patches along height
        ph = H // self.patch_height
        # Compute number of patches along width
        pw = W // self.patch_width

        # Convert to patch embeddings: shape becomes (B, Tin*ph*pw, d_model)
        x = self.to_patch_embedding(x)
        # Extract batch size, total tokens, and model dimension
        B, T, D = x.shape

        # Build positional encodings for T tokens
        pe = generate_positional_encoding(T, D, x.device)
        # Apply Transformer encoder to positionally encoded sequence
        mem = self.encoder(x + pe)

        # Number of tokens per frame
        tokens_per_frame = ph * pw
        # Total number of tokens needed for pred_length frames
        needed = self.pred_length * tokens_per_frame
        # Ensure we have enough tokens for the requested prediction length
        assert needed <= T, (
            f"pred_length({self.pred_length}) * ph*pw({tokens_per_frame}) > Tin*ph*pw({T})"
        )

        # Take the last `needed` tokens as prediction context
        mem = mem[:, -needed:, :]
        # Project tokens back to patch space
        out = self.to_feature_map(mem)

        # Rearrange patches back to feature map of shape (B, Tout, C, H, W)
        out = rearrange(
            out,
            "b (t h w) (p1 p2 c) -> b t c (h p1) (w p2)",
            t=self.pred_length,
            h=ph,
            w=pw,
            p1=self.patch_height,
            p2=self.patch_width,
        )
        # Return reconstructed feature maps for each predicted time step
        return out


# Define the main RainPredRNN model for nowcasting
class RainPredRNN(nn.Module):
    """Rain prediction model that combines UNet encoder/decoder with a temporal Transformer."""

    # Initialize model with architectural hyperparameters
    def __init__(
        self,
        input_dim: int = 1,
        num_hidden: int = 256,
        max_hidden_channels: int = 128,
        patch_height: int = 16,
        patch_width: int = 16,
        pred_length: int = 6,
    ):
        # Call base class constructor
        super().__init__()
        # Create per-frame encoder
        self.encoder = UNetEncoder(input_dim)
        # Create per-frame decoder
        self.decoder = UNetDecoder(input_dim)
        # Store prediction length
        self.pred_length = pred_length
        # Create temporal Transformer over encoded feature maps
        self.transformer_block = TemporalTransformerBlock(
            channels=max_hidden_channels,
            d_model=num_hidden,
            nhead=8,
            num_encoder_layers=3,
            pred_length=pred_length,
            patch_height=patch_height,
            patch_width=patch_width,
        )

    # Forward pass over a sequence of input frames
    def forward(
        self,
        input_sequence: torch.Tensor,
        pred_length: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: input_sequence (B, Tin, C, H, W) -> predictions (B, Tout, C, H, W)."""
        # Unpack shape of input sequence
        B, Tin, C, H, W = input_sequence.size()
        # Initialize list to store encoded features for each time step
        enc_feats = []
        # Initialize list to store skip connections for each time step
        skip1 = []

        # Loop over time dimension
        for t in range(Tin):
            # Encode frame t
            x, sk1 = self.encoder(input_sequence[:, t])
            # Append encoded features
            enc_feats.append(x)
            # Append skip connection
            skip1.append(sk1)

        # Stack encoded features into shape (B, Tin, C_enc, H/2, W/2)
        enc_feats = torch.stack(enc_feats, dim=1)
        # Stack skip connections similarly
        skip1 = torch.stack(skip1, dim=1)

        # Apply temporal Transformer to encoded features
        pred_feats = self.transformer_block(enc_feats)

        # Initialize lists for activated predictions and raw logits
        preds = []
        preds_noact = []

        # Decode each predicted time step
        for t in range(pred_length):
            # Decode time step t using encoder context
            y, y_no = self.decoder(pred_feats[:, t], enc_feats[:, t], skip1[:, t])
            # Append activated output
            preds.append(y)
            # Append raw output (before tanh)
            preds_noact.append(y_no)

        # Stack predictions along time dimension and return both activated and raw versions
        return torch.stack(preds, dim=1), torch.stack(preds_noact, dim=1)
