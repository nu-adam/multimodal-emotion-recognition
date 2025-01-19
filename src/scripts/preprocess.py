from src.preprocessing.video_preprocess import preprocess_video
from src.preprocessing.audio_preprocess import preprocess_audio
from src.preprocessing.text_preprocess import preprocess_text


def preprocess_multimodal_input(enabled_modalities, video_path, audio_path, text_path):
    """
    Preprocesses video, audio, and text inputs for multimodal emotion recognition.

    Args:
    - enabled_modalities (list): List of enabled modalities (e.g., ['video', 'audio', 'text']).
    - video_path (str): Path to the video file.
    - audio_path (str): Path to save the extracted audio file.
    - text_path (str): Path to save the transcribed subtitles.

    Returns:
    - torch.Tensor: Preprocessed data including video frames, audio spectrogram, and tokenized text.
    """
    pass
