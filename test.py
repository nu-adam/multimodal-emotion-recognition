import torch
from src.models.video_encoder import VideoEncoder
from src.models.audio_encoder import AudioEncoder
from src.models.text_encoder import TextEncoder
from src.models.fusion import TransformerFusion
from src.models.decoder import Decoder
from src.models.multimodal_emotion_recognition import MultimodalEmotionRecognition

# Define dummy input shapes
batch_size = 8
video_input = torch.randn(batch_size, 3, 224, 224)  # Video input (batch_size, channels, height, width)
audio_input = torch.randn(batch_size, 3, 224, 224)  # Audio spectrogram input
text_input_ids = torch.randint(0, 30522, (batch_size, 50))  # Tokenized text input
text_attention_mask = torch.ones(batch_size, 50)  # Attention mask for text input

# Initialize individual components
# video_encoder = VideoEncoder(embed_dim=256)
# audio_encoder = AudioEncoder(embed_dim=256)
# text_encoder = TextEncoder(model_name='roberta-base', embed_dim=256)
# fusion = TransformerFusion(embed_dim=256, num_heads=4, num_layers=2)
# decoder = Decoder(embed_dim=256, num_classes=7)

# Initialize the multimodal pipeline
pipeline = MultimodalEmotionRecognition()

# Forward pass through the pipeline
output = pipeline(
    video=video_input,
    audio=audio_input,
    text_input_ids=text_input_ids,
    text_attention_mask=text_attention_mask
)

# Print the output
print("Output shape:", output.shape)
print("Output values:", output)
