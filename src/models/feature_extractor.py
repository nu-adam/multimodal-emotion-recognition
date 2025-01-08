import torch.nn as nn
from torchvision import models


class VGGFeatureExtractor(nn.Module):
    """
    Feature Extractor based on the VGG-16 architecture.
    
    Extracts spatial features from input images.
    
    Methods:
        forward(x): Passes the input through the VGG-16 feature extractor.
    """
    def __init__(self):
        super(VGGFeatureExtractor, self).__init__()
        self.vgg = models.vgg16(weights=None).features

    def forward(self, x):
        """
        Forward pass through VGG-16 feature extractor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, C, H, W).

        Returns:
            torch.Tensor: Extracted feature maps of shape (batch_size, 512, H/32, W/32).
        """
        x = self.vgg(x)
        return x
