import torch
import torch.nn as nn


class TransformerFusion(nn.Module):
    """
    Combines features from different modalities using a Transformer Encoder.

    Methods:
        forward(video_features, audio_features, text_features): Combines features and encodes them.
    """
    def __init__(self, embed_dim=256, num_heads=4, num_layers=2):
        super(TransformerFusion, self).__init__()
        self.positional_encoding = nn.Parameter(torch.randn(1, 3, embed_dim))
        transformer_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)

    def forward(self, modalities_features):
        """
        Forward pass through the fusion module.

        Args:
            modalities_features (torch.Tensor): Encoded combined features of shape (batch_size, embed_dim).

        Returns:
            torch.Tensor: Fused features of shape (batch_size, embed_dim).
        """
        features = torch.stack(modalities_features, dim=1)  # Shape: (batch_size, 3, embed_dim)
        features += self.positional_encoding
        x = self.transformer(features)  # Shape: (batch_size, 3, embed_dim)
        x = x.mean(dim=1)  # Shape: (batch_size, embed_dim)
        return x
