import torch
import torch.nn as nn
from torchvision import models


class VGG16Encoder(nn.Module):
    """
    VGG-16-based image encoder for feature extraction.

    The model uses the convolutional layers of the VGG-16 model and
    replaces the classifier to output a custom feature dimension.

    Attributes:
        features (nn.Sequential): Convolutional layers from VGG-16.
        avgpool (nn.AdaptiveAvgPool2d): Average pooling layer to reduce spatial dimensions.
        fc (nn.Sequential): Custom fully connected layers for feature extraction.
    """
    def __init__(self, output_dim=1024):
        """
        Initialize the VGG-16 encoder.

        Args:
            output_dim (int): The dimensionality of the output feature vector.
        """
        super(VGG16Encoder, self).__init__()
        # Load the VGG-16 model
        vgg16 = models.vgg16(weights=None)
        
        # Extract the convolutional and pooling layers (feature extractor)
        self.features = vgg16.features
        
        # Retain the original average pooling layer
        self.avgpool = vgg16.avgpool

        # Replace the original classifier with a custom one
        self.fc = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, output_dim)
        )

    def forward(self, x):
        """
        Forward pass through the VGG-16 encoder.

        Args:
            x (torch.Tensor): Batch of images with shape [batch_size, 3, 224, 224].

        Returns:
            torch.Tensor: Feature vectors with shape [batch_size, output_dim].
        """
        # Pass input through convolutional layers
        x = self.features(x)
        
        # Apply average pooling to reduce spatial dimensions
        x = self.avgpool(x)
        
        # Flatten the tensor for the fully connected layers
        x = torch.flatten(x, 1)
        
        # Pass through the fully connected layers
        x = self.fc(x)
        return x


class VGGFeatureExtractor(nn.Module):
    """
    Feature Extractor based on the VGG-19 architecture.
    """
    def __init__(self):
        super(VGGFeatureExtractor, self).__init__()
        self.vgg = models.vgg19(weights=None).features

    def forward(self, x):
        x = self.vgg(x)
        return x  # Output: (batch_size, 512, 7, 7)


class TransformerEncoder(nn.Module):
    """
    Transformer model for processing image features.
    """
    def __init__(self, embed_dim=256, num_heads=4, num_layers=2):
        super(TransformerEncoder, self).__init__()
        transformer_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
        
    def forward(self, x):
        x = self.transformer(x)
        return x  # Output: (batch_size, 49, embed_dim)
    

class ProjectionNetwork(nn.Module):
    """
    A projection network to reduce the dimensionality of the VGG feature maps.
    """
    def __init__(self, input_dim=512, output_dim=256):
        super(ProjectionNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, output_dim)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        return x  # Output: (batch_size, 49, output_dim)


class FaceEmotionModel(nn.Module):
    """
    Face Emotion Recognition Model combining VGG-19 and Transformer.

    This model extracts features from facial images using VGG19, then processes
    these features through a Transformer to learn temporal and spatial
    representations, and finally classifies the emotions present in the input.
    """
    def __init__(self, embed_dim=256, num_heads=4, num_layers=2, num_classes=7):
        super(FaceEmotionModel, self).__init__()
        self.feature_extractor = VGGFeatureExtractor()
        self.transformer = TransformerEncoder(embed_dim, num_heads, num_layers)
        self.projection = ProjectionNetwork(input_dim=512, output_dim=embed_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, 49, embed_dim))
        self.classifier = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.shape[0], 512, 7 * 7).permute(0, 2, 1) # Shape: (batch_size, 49, 512)
        x = self.projection(x) # Shape: (batch_size, 49, embed_dim)
        x = x + self.positional_encoding
        x = self.transformer(x)
        x = x.mean(dim=1) # Shape: (batch_size, embed_dim)
        x = self.classifier(x) # Shape: (batch_size, num_classes)
        return x # Output: (batch_size, num_classes)
