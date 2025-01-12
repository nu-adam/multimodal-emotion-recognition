import torch.nn as nn
from torchvision import models


class VGGFeatureExtractor(nn.Module):
    """
    Feature Extractor based on the VGG-19 architecture for the visual modality.
    """
    def __init__(self):
        super(VGGFeatureExtractor, self).__init__()
        self.vgg = models.vgg19(weights=None).features

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, H, W).

        Returns:
            torch.Tensor: Feature maps of shape (batch_size, 512, 7, 7).
        """
        x = self.vgg(x)
        return x  # Output: (batch_size, 512, 7, 7)


class ProjectionNetwork(nn.Module):
    """
    Projection Network for dimensionality reduction.
    """
    def __init__(self, input_dim=512, output_dim=256):
        super(ProjectionNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim).

        Returns:
            torch.Tensor: Reduced-dimensionality tensor of shape (batch_size, seq_len, output_dim).
        """
        x = self.fc(x)
        x = self.relu(x)
        return x  # Output: (batch_size, seq_len, output_dim)
