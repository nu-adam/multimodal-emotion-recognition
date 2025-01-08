import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, embed_dim, num_classes):
        """
        Decoder to generate answers from fused features.

        Args:
            embed_dim (int): Dimensionality of the fused features.
            num_classes (int): Number of possible answers (classification).
        """
        super(Decoder, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embed_dim // 2, num_classes),
            nn.Sigmoid()
        )

    def forward(self, fused_features):
        """
        Forward pass for the decoder.

        Args:
            fused_features (torch.Tensor): Batch of fused features (shape: [batch_size, fused_dim]).

        Returns:
            torch.Tensor: Predicted probabilities for each class (shape: [batch_size, num_classes]).
        """
        x = self.classifier(fused_features)
        return x
