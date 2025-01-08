import torch.nn as nn


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder for processing sequence data.

    Methods:
        forward(x): Passes the input through the transformer encoder.
    """
    def __init__(self, embed_dim=256, num_heads=4, num_layers=2):
        super(TransformerEncoder, self).__init__()
        transformer_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)

    def forward(self, x):
        """
        Forward pass through the transformer encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim).

        Returns:
            torch.Tensor: Encoded features of shape (batch_size, seq_len, embed_dim).
        """
        x = self.transformer(x)
        return x
    