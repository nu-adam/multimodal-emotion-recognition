import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, fused_dim, num_classes):
        """
        Decoder to generate answers from fused features.

        Args:
            fused_dim (int): Dimensionality of the fused features.
            num_classes (int): Number of possible answers (classification).
        """
        super(Decoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fused_dim // 2, num_classes)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, fused_features):
        """
        Forward pass for the decoder.

        Args:
            fused_features (torch.Tensor): Batch of fused features (shape: [batch_size, fused_dim]).

        Returns:
            torch.Tensor: Predicted probabilities for each class (shape: [batch_size, num_classes]).
        """
        logits = self.fc(fused_features)
        probabilities = self.softmax(logits)
        return probabilities
