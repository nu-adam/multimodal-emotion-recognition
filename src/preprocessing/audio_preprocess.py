import ray
import librosa
import numpy as np
import torch
import subprocess
from io import BytesIO
import os
from glob import glob


def extract_audio_as_waveform(video_path, sr=16000):
    """
    Extracts audio directly from a video file using FFmpeg and validates it.

    Args:
        video_path (str): Path to the video file.
        sr (int): Sampling rate for the audio.

    Returns:
        Tuple[np.ndarray, int]: Audio waveform as a NumPy array and its sampling rate.
        If no audio exists, returns (None, None).
    """
    # FFmpeg command to extract audio
    command = [
        "ffmpeg", "-i", video_path, "-f", "wav", "-ar", str(sr), "-ac", "1", "pipe:1"
    ]
    try:
        process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        audio_data = BytesIO(process.stdout)

        # Load audio waveform using Librosa
        waveform, sample_rate = librosa.load(audio_data, sr=sr)

        # Check if waveform has significant energy (to avoid silent audio)
        if np.sum(np.abs(waveform)) < 1e-6:  # Threshold for silent audio
            print(f"[INFO] Silent audio detected in video: {video_path}")
            return None, None

        return waveform, sample_rate
    except subprocess.CalledProcessError:
        print(f"[WARNING] No audio stream found in video: {video_path}")
        return None, None
    except Exception as e:
        print(f"[ERROR] Failed to extract audio from video {video_path}: {e}")
        return None, None


def generate_vgg_mel_spectrogram(waveform, sr, n_mels=128):
    """
    Generates a Mel spectrogram, normalizes it, resizes it to 224x224, 
    and duplicates channels to create a 3x224x224 tensor for VGG input.

    Args:
        waveform (np.ndarray): Audio waveform.
        sr (int): Sampling rate of the audio.
        n_mels (int): Number of mel bands.

    Returns:
        torch.Tensor: Mel spectrogram tensor of shape (3, 224, 224).
    """
    # Generate Mel spectrogram
    S = librosa.feature.melspectrogram(y=waveform, sr=sr, n_mels=n_mels)
    S_db_mel = librosa.amplitude_to_db(S, ref=np.max)  # Convert to decibel scale

    # Normalize to [0, 1]
    S_db_mel_normalized = (S_db_mel - S_db_mel.min()) / (S_db_mel.max() - S_db_mel.min())

    # Resize to (224, 224)
    S_resized = torch.nn.functional.interpolate(
        torch.tensor(S_db_mel_normalized).unsqueeze(0).unsqueeze(0),  # Shape: (1, 1, 128, time)
        size=(224, 224),
        mode="bilinear",
        align_corners=False,
    ).squeeze()  # Shape: (224, 224)

    # Duplicate channels to create (3, 224, 224)
    S_tensor = S_resized.repeat(3, 1, 1)
    return S_tensor


@ray.remote
class AudioProcessor:
    """
    Ray Actor for audio extraction and mel spectrogram generation.
    """

    def __init__(self, sampling_rate=16000, n_mels=128):
        """
        Initializes the AudioProcessor.

        Args:
            sampling_rate (int): Sampling rate for audio extraction.
            n_mels (int): Number of mel bands for spectrogram generation.
        """
        self.sampling_rate = sampling_rate
        self.n_mels = n_mels

    def process_audio(self, video_path):
        """
        Processes a video to extract audio and generate a 3x224x224 mel spectrogram.

        Args:
            video_path (str): Path to the video file.

        Returns:
            torch.Tensor: Mel spectrogram tensor of shape (3, 224, 224).
        """
        try:
            # Extract audio waveform directly from video
            waveform, sr = extract_audio_as_waveform(video_path, sr=self.sampling_rate)
            # Generate Mel spectrogram for VGG input
            mel_tensor = generate_vgg_mel_spectrogram(waveform, sr, n_mels=self.n_mels)
            return mel_tensor
        except Exception as e:
            print(f"[ERROR] Failed to process audio for video {video_path}: {e}")
            return None


@ray.remote
def preprocess_audio_task(video_path, audio_processor_actor):
    """
    Processes the audio for a video file and saves the VGG-compatible mel spectrogram.

    Args:
        video_path (str): Path to the video file.
        audio_processor_actor (ray.actor.ActorHandle): A Ray Actor for audio processing.

    Returns:
        str: Path to the saved tensor file, or None if processing failed.
    """
    mel_tensor = ray.get(audio_processor_actor.process_audio.remote(video_path))
    if mel_tensor is not None:
        # Save tensor to processed/audio directory
        output_path = video_path.replace("raw", "processed").replace(".mp4", ".pth").replace("/video/", "/audio/")
        torch.save(mel_tensor, output_path)
        return output_path
    else:
        print(f"[INFO] No tensor saved for video: {video_path}")
        return None
    

def get_video_paths(raw_dir):
    """
    Recursively finds all video files in the dataset directory.

    Args:
        raw_dir (str): Path to the raw dataset directory.

    Returns:
        List[str]: A list of paths to video files.
    """
    # return glob(os.path.join(raw_dir, "**", "*.mp4"), recursive=True)
    return glob(os.path.join(raw_dir, "*.mp4"), recursive=True)


def main():
    """
    Main function to preprocess all audio in the dataset using Ray Actors.
    """
    ray.init()  # Initialize Ray

    # Step 1: Start the AudioProcessor Actor
    audio_processor_actor = AudioProcessor.remote()

    # Step 2: Get all video paths
    raw_dir = "data/raw/test/disgust"
    video_paths = get_video_paths(raw_dir)

    # Step 3: Launch preprocessing tasks in parallel
    futures = [preprocess_audio_task.remote(video_path, audio_processor_actor) for video_path in video_paths]

    # Step 4: Wait for all tasks to complete and get results
    results = ray.get(futures)
    print("All audio processed. Saved to:")
    for result in results:
        if result:
            print(result)


if __name__ == "__main__":
    main()
