from src.models.feature_extractor import VGGFeatureExtractor
from src.models.transformer_encoder import TransformerEncoder
from src.models.projection_network import ProjectionNetwork

import torch
import torch.nn as nn


class AudioEncoder(nn.Module):
    """
    Combines VGG-16 and Transformer Encoder for audio modality.

    Methods:
        forward(x): Extracts and encodes audio features.
    """
    def __init__(self, embed_dim=256, num_heads=4, num_layers=2):
        super(AudioEncoder, self).__init__()
        self.feature_extractor = VGGFeatureExtractor()
        self.projection_network = ProjectionNetwork(input_dim=512, output_dim=embed_dim)
        self.transformer_encoder = TransformerEncoder(embed_dim, num_heads, num_layers)
        self.positional_encoding = nn.Parameter(torch.randn(1, 49, embed_dim))

    def forward(self, x):
        """
        Forward pass through the audio feature extractor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, C, H, W).

        Returns:
            torch.Tensor: Encoded audio features of shape (batch_size, embed_dim).
        """
        batch_size, _, _, _ = x.shape
        x = self.feature_extractor(x)  # Shape: (batch_size, 512, H/32, W/32)
        seq_len = x.size(2) * x.size(3)  # Sequence length is H/32 * W/32

        x = x.view(batch_size, 512, seq_len).permute(0, 2, 1)  # Shape: (batch_size, seq_len, 512)
        x = self.projection_network(x)  # Shape: (batch_size, seq_len, embed_dim)
        
        x = x + self.positional_encoding
        x = self.transformer_encoder(x) # Shape: (batch_size, seq_len, embed_dim)
        x = x.mean(dim=1)  # Shape: (batch_size, embed_dim)
        return x
