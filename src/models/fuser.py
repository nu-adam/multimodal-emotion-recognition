import torch
import torch.nn as nn
from src.models.visual.vgg19 import VGGFeatureExtractor, ProjectionNetwork
from src.models.text.text import TextFeatureExtractor, ProjectionNetwork as TextProjectionNetwork


class MultimodalTransformer(nn.Module):
    """
    Multimodal Transformer combining visual, audio, and text modalities.
    """
    def __init__(self, enabled_modalities=None, embed_dim=256, num_heads=4, num_layers=2, num_classes=8):
        super(MultimodalTransformer, self).__init__()
        self.enabled_modalities = enabled_modalities or ['visual']
        self.modalities_dim = {'visual': 512, 'audio': 768, 'text': 768}

        # Feature extractors and projectors
        if 'visual' in self.enabled_modalities:
            self.visual_extractor = VGGFeatureExtractor()
            self.visual_projector = ProjectionNetwork(input_dim=512, output_dim=embed_dim)
        if 'audio' in self.enabled_modalities:
            self.audio_projector = ProjectionNetwork(input_dim=768, output_dim=embed_dim)
        if 'text' in self.enabled_modalities:
            self.text_extractor = TextFeatureExtractor()
            self.text_projector = TextProjectionNetwork(input_dim=768, output_dim=embed_dim)

        # Total input dimensions after concatenation
        # self.total_dim = embed_dim * len(self.enabled_modalities)
        self.total_dim = embed_dim  # Per modality embedding dimension

        # Transformer Encoder
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,  # Use the reduced embedding dimension for each modality
            nhead=num_heads,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)

        # Classifier
        self.classifier = nn.Linear(self.total_dim, num_classes)


    def forward(self, features):
        """
        Args:
            features (dict): Dictionary of modality tensors with keys like 'visual', 'audio', 'text'.

        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes).
        """
        combined_features = []

        # Visual features
        if 'visual' in self.enabled_modalities and 'visual' in features:
            visual_features = self.visual_extractor(features['visual'])
            visual_features = visual_features.view(visual_features.size(0), 49, -1)  # Flatten spatial dimensions
            visual_features = self.visual_projector(visual_features)  # Shape: (batch_size, 49, embed_dim)
            combined_features.append(visual_features)

        # Audio features
        if 'audio' in self.enabled_modalities and 'audio' in features:
            audio_features = features['audio']  # Shape: (batch_size, 768)
            audio_features = self.audio_projector(audio_features.unsqueeze(1))  # Add sequence dim and project
            combined_features.append(audio_features)  # Shape: (batch_size, 1, embed_dim)

        # Text features
        if 'text' in self.enabled_modalities and 'text' in features:
            text_features = self.text_extractor(features['text'])  # Shape: (batch_size, 768)
            text_features = self.text_projector(text_features.unsqueeze(1))  # Add sequence dim and project
            combined_features.append(text_features)  # Shape: (batch_size, 1, embed_dim)

        # Combine all features
        if not combined_features:
            raise ValueError("No valid modalities provided.")

        x = torch.cat(combined_features, dim=1)  # Concatenate along sequence dimension
        x = self.transformer(x)  # Shape: (batch_size, seq_len, embed_dim)
        x = x.mean(dim=1)  # Global average pooling over sequence dimension
        return self.classifier(x)  # Output: (batch_size, num_classes)
