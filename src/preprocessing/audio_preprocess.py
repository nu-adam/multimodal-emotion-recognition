import librosa
import numpy as np
import moviepy.editor as mp


def extract_audio_from_video(video_path, audio_save_path):
    """
    Extracts audio from a video file and saves it.

    Args:
        video_path (str): Path to the video file.
        audio_save_path (str): Path to save the extracted audio file.

    Returns:
        str: Path to the saved audio file.
    """
    video = mp.VideoFileClip(video_path)
    video.audio.write_audiofile(audio_save_path)
    return audio_save_path


def preprocess_audio(audio_path, sr=16000):
    """
    Loads and preprocesses audio for spectrogram extraction.

    Args:
        audio_path (str): Path to the audio file.
        sr (int): Target sampling rate.

    Returns:
        np.ndarray: Mel spectrogram of the audio.
    """
    audio, _ = librosa.load(audio_path, sr=sr)
    mel_spectrogram = librosa.feature.melspectrogram(audio, sr=sr, n_mels=128, fmax=8000)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return mel_spectrogram
