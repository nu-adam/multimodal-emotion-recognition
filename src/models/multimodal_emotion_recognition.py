from src.models.encoders.video_encoder import VideoEncoder
from src.models.encoders.audio_encoder import AudioEncoder
from src.models.encoders.text_encoder import TextEncoder
from src.models.fusion import TransformerFusion
from src.models.decoder import Decoder

import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class MultimodalEmotionRecognition(nn.Module):
    """
    Multimodal Emotion Recognition model.

    This model integrates video, audio, and text encoders, combines their outputs
    using a fusion module, and decodes the fused embeddings into emotion predictions.
    """
    def __init__(self, enabled_modalities=None, embed_dim=256, num_heads=4, num_layers=2, num_classes=7):
        super(MultimodalEmotionRecognition, self).__init__()
        self.enabled_modalities = enabled_modalities or ['video']

        if 'video' in self.enabled_modalities:
            self.video_encoder = VideoEncoder(embed_dim=embed_dim)
        if 'audio' in self.enabled_modalities:
            self.audio_encoder = AudioEncoder(embed_dim=embed_dim)
        if 'text' in self.enabled_modalities:
            self.text_encoder = TextEncoder(embed_dim=embed_dim)

        self.fusion = TransformerFusion(embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers)
        self.decoder = Decoder(embed_dim=embed_dim, num_classes=num_classes)

    def forward(self, video, audio, text_input_ids, text_attention_mask):
        """
        Forward pass through the Multimodal Emotion Recognition model.

        Args:
            video (torch.Tensor): Video input tensor of shape (batch_size, 3, H, W).
            audio (torch.Tensor): Audio input tensor of shape (batch_size, 1, H, W).
            text_input_ids (torch.Tensor): Tokenized text input IDs of shape (batch_size, seq_len).
            text_attention_mask (torch.Tensor): Attention mask of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Predicted class probabilities of shape (batch_size, num_classes).
        """
        modalities_embeddings = []
        if 'video' in self.enabled_modalities:
            # video_features = self.video_encoder(video)
            video_features = checkpoint(self.video_encoder, video)
            modalities_embeddings.append(video_features)
        if 'audio' in self.enabled_modalities:
            # audio_features = self.audio_encoder(audio)
            audio_features = checkpoint(self.audio_encoder, audio)
            modalities_embeddings.append(audio_features)
        if 'text' in self.enabled_modalities:
            # text_features = self.text_encoder(text_input_ids, text_attention_mask)
            text_features = checkpoint(self.text_encoder, text_input_ids, text_attention_mask)
            modalities_embeddings.append(text_features)

        fused_features = self.fusion(modalities_embeddings)
        predictions = self.decoder(fused_features)
        return predictions
