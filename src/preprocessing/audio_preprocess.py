import os
import subprocess
import librosa
import numpy as np
import torch

def extract_audio(video_path, audio_path):
    """
    Extracts audio from a video file and saves it, unless the audio file already exists.

    Args:
    - video_path (str): Path to the video file.
    - audio_path (str): Path to save the extracted audio file.

    Returns:
    - str: Path to the saved audio file.
    """
    if os.path.exists(audio_path):
        print(f"Audio file already exists: {audio_path}. Skipping conversion.")
    else:
        subprocess.run(["ffmpeg", "-i", video_path, audio_path], check=True)
        print(f"Audio extraction complete: {audio_path}")
    return audio_path


def preprocess_audio(video_path):
    """
    Loads and preprocesses audio for spectrogram extraction.

    Args:
    - video_path (str): Path to the video file.

    Returns:
    - torch.Tensor: Batch of preprocessed audio tensors.
    """
    # Define the output audio path (e.g., with a `.wav` extension)
    audio_path = os.path.splitext(video_path)[0] + ".wav"
    
    # Extract audio from the video
    audio_path = extract_audio(video_path, audio_path)
    
    # Load the audio file
    y, sr = librosa.load(audio_path)
    
    # Generate Mel spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_db_mel = librosa.amplitude_to_db(S, ref=np.max)  # Shape: (128, 143)

    # Normalize the spectrogram to [0, 1]
    S_db_mel_normalized = (S_db_mel - S_db_mel.min()) / (S_db_mel.max() - S_db_mel.min())
    
    # Resize to (3, 224, 224)
    S_resized = torch.nn.functional.interpolate(
        torch.tensor(S_db_mel_normalized).unsqueeze(0).unsqueeze(0),  # Shape: (1, 1, 128, 143)
        size=(224, 224),
        mode="bilinear",
        align_corners=False,
    ).squeeze()  # Shape: (224, 224)
    
    # Duplicate channels to create (3, 224, 224)
    S_tensor = S_resized.repeat(3, 1, 1)
    return S_tensor
