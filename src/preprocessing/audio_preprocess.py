import os
import torch
import librosa
from torchaudio.transforms import MelSpectrogram


# Constants
SAMPLE_RATE = 16000


def preprocess_audio(audio_path):
    """Preprocess audio into Mel spectrogram."""
    waveform, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
    mel_spec_transform = MelSpectrogram(sample_rate=SAMPLE_RATE)
    waveform_tensor = torch.tensor(waveform).unsqueeze(0)  # Add channel dimension
    mel_spectrogram = mel_spec_transform(waveform_tensor)
    return torch.log1p(mel_spectrogram)


def extract_audio_from_video(video_path):
    """Extract audio from video."""
    audio_path = video_path.replace('.mp4', '.wav')
    command = f"ffmpeg -i {video_path} -q:a 0 -map a {audio_path} -y"
    os.system(command)
    return audio_path