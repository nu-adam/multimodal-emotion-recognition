from src.models.video_encoder import FaceEmotionModel
from src.models.audio_encoder import AudioEncoder
from src.models.text_encoder import TextEncoder
from src.models.fusion import FusionModule
from src.models.decoder import Decoder

import torch.nn as nn

class HERModel(nn.Module):
    """
    Multimodal Human Emotion Recognition (HER) model.

    This model integrates video, audio, and text encoders, combines their outputs
    using a fusion module, and decodes the fused embeddings into emotion predictions.
    """
    def __init__(self, video_dim=256, audio_dim=256, text_dim=256, fused_dim=512, num_classes=7):
        super(HERModel, self).__init__()

        # Encoders for each modality
        self.video_encoder = FaceEmotionModel(embed_dim=video_dim, num_classes=video_dim)
        self.audio_encoder = AudioEncoder(embed_dim=audio_dim)
        self.text_encoder = TextEncoder(output_dim=text_dim)

        # Fusion and decoder modules
        self.fusion = FusionModule(video_dim=video_dim, audio_dim=audio_dim, text_dim=text_dim, fused_dim=fused_dim)
        self.decoder = Decoder(fused_dim=fused_dim, num_classes=num_classes)

    def forward(self, video, audio, text_input_ids, text_attention_mask):
        """
        Forward pass for the HER model.

        Args:
            video (torch.Tensor): Batch of video frames (shape: [batch_size, 3, H, W]).
            audio (torch.Tensor): Batch of audio Mel spectrograms (shape: [batch_size, 1, H, W]).
            text_input_ids (torch.Tensor): Tokenized text input IDs (shape: [batch_size, seq_length]).
            text_attention_mask (torch.Tensor): Attention masks for text (shape: [batch_size, seq_length]).

        Returns:
            torch.Tensor: Predicted probabilities for each emotion class (shape: [batch_size, num_classes]).
        """
        # Encode each modality
        video_features = self.video_encoder(video)
        audio_features = self.audio_encoder(audio)
        text_features = self.text_encoder(text_input_ids, text_attention_mask)

        # Fuse features from all modalities
        fused_features = self.fusion(video_features, audio_features, text_features)

        # Decode fused features to predict emotions
        predictions = self.decoder(fused_features)
        return predictions
