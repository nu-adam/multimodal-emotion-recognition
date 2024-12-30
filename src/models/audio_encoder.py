# File: src/models/audio_encoder.py
import torch
import torch.nn as nn
from torchvision import models


class AudioEncoder(nn.Module):
    """
    Audio Encoder to process Mel spectrograms and extract feature embeddings.
    """
    def __init__(self, embed_dim=256):
        super(AudioEncoder, self).__init__()
        self.cnn = models.vgg16(weights=None).features  # Use VGG-16 for feature extraction
        self.fc = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, embed_dim)
        )

    def forward(self, x):
        """
        Forward pass for audio encoder.

        Args:
            x (torch.Tensor): Input Mel spectrogram (shape: [batch_size, 1, H, W]).

        Returns:
            torch.Tensor: Audio feature embeddings (shape: [batch_size, embed_dim]).
        """
        x = self.cnn(x)  # Extract CNN features
        x = torch.flatten(x, start_dim=1)  # Flatten features
        x = self.fc(x)  # Fully connected layers
        return x
