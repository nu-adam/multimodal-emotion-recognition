from src.models.video_encoder import VideoEncoder
from src.models.audio_encoder import AudioEncoder
from src.models.text_encoder import TextEncoder
from src.models.fusion import TransformerFusion
from src.models.decoder import Decoder

import torch.nn as nn


class MultimodalEmotionRecognition(nn.Module):
    """
    Multimodal Emotion Recognition model.

    This model integrates video, audio, and text encoders, combines their outputs
    using a fusion module, and decodes the fused embeddings into emotion predictions.
    """
    def __init__(self, video_dim=256, audio_dim=256, text_dim=256, fused_dim=256, num_classes=7):
        super(MultimodalEmotionRecognition, self).__init__()
        self.video_encoder = VideoEncoder(embed_dim=video_dim)
        self.audio_encoder = AudioEncoder(embed_dim=audio_dim)
        self.text_encoder = TextEncoder(embed_dim=text_dim)
        self.fusion = TransformerFusion(embed_dim=video_dim)
        self.decoder = Decoder(embed_dim=fused_dim, num_classes=num_classes)

    def forward(self, video, audio, text_input_ids, text_attention_mask):
        """
        Forward pass through the Multimodal Emotion Recognition model.

        Args:
            video_input (torch.Tensor): Video input tensor of shape (batch_size, 3, H, W).
            audio_input (torch.Tensor): Audio input tensor of shape (batch_size, 1, H, W).
            text_input_ids (torch.Tensor): Tokenized text input IDs of shape (batch_size, seq_len).
            text_attention_mask (torch.Tensor): Attention mask of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Predicted class probabilities of shape (batch_size, num_classes).
        """
        video_features = self.video_encoder(video)
        audio_features = self.audio_encoder(audio)
        text_features = self.text_encoder(text_input_ids, text_attention_mask)
        fused_features = self.fusion(video_features, audio_features, text_features)
        predictions = self.decoder(fused_features)
        return predictions
