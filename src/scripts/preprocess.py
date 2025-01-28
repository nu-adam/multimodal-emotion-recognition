import ray
import os
import torch
import cv2
import librosa
import numpy as np
import subprocess
from io import BytesIO
from transformers import RobertaTokenizer
from insightface.app import FaceAnalysis
import whisper
import torchvision.transforms as T


def get_video_paths(raw_dir):
    """
    Recursively find all video files in the dataset directory.

    Args:
        raw_dir (str): Path to the raw dataset directory.

    Returns:
        List[str]: A list of paths to video files.
    """
    return [
        os.path.join(dp, f)
        for dp, dn, filenames in os.walk(raw_dir)
        for f in filenames if f.endswith(".mp4")
    ]


def ensure_directory_exists(file_path):
    """
    Ensures the parent directory of the given file path exists.

    Args:
        file_path (str): The file path whose parent directory needs to be created.
    """
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def save_chunk(output_dir, modality, chunk_id, tensors, labels):
    """
    Saves a chunk of tensors and labels to a .pth file.

    Args:
        output_dir (str): Base directory for processed files.
        modality (str): Modality ('video', 'audio', or 'text').
        chunk_id (int): Chunk ID (used for file naming).
        tensors (list[torch.Tensor]): List of tensors to save.
        labels (list[str]): List of emotion labels corresponding to the tensors.
    """
    output_path = os.path.join(output_dir, modality, f"{modality}_chunk_{chunk_id:04d}.pth")
    ensure_directory_exists(output_path)  # Ensure the directory exists
    torch.save({"tensors": tensors, "labels": labels}, output_path)
    print(f"[INFO] Saved chunk {chunk_id} to {output_path}")


@ray.remote(num_gpus=1, num_cpus=3)
class VideoProcessor:
    def __init__(self):
        self.face_detector = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider"])
        self.face_detector.prepare(ctx_id=0)

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % fps == 0:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

        batched_faces = []
        for frame in frames:
            faces = self.face_detector.get(frame)
            for face in faces:
                # Extract bounding box
                x1, y1, x2, y2 = map(int, face.bbox)

                # Validate bounding box dimensions
                if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0] or x2 <= x1 or y2 <= y1:
                    print(f"[WARNING] Invalid bounding box in video: {video_path}")
                    continue

                # Crop and resize the face
                cropped = frame[y1:y2, x1:x2]
                if cropped.size == 0:  # Ensure the cropped region is not empty
                    print(f"[WARNING] Empty cropped region in video: {video_path}")
                    continue

                resized = cv2.resize(cropped, (224, 224))
                tensor = T.ToTensor()(resized)
                batched_faces.append(tensor)

        if not batched_faces:
            print(f"[INFO] No faces detected in video: {video_path}")
            return None
        return torch.stack(batched_faces)



@ray.remote(num_cpus=3)
class AudioProcessor:
    def __init__(self, sampling_rate=16000, n_mels=128):
        self.sampling_rate = sampling_rate
        self.n_mels = n_mels

    def process_audio(self, video_path):
        """
        Extracts audio and generates a VGG-compatible Mel spectrogram tensor.

        Args:
            video_path (str): Path to the video file.

        Returns:
            torch.Tensor or None: Mel spectrogram tensor, or None if no audio exists.
        """
        command = ["ffmpeg", "-i", video_path, "-f", "wav", "-ar", str(self.sampling_rate), "-ac", "1", "pipe:1"]
        try:
            process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            audio_data = BytesIO(process.stdout)
            waveform, sr = librosa.load(audio_data, sr=self.sampling_rate)

            if np.sum(np.abs(waveform)) < 1e-6:
                print(f"[INFO] Silent audio detected in video: {video_path}")
                return None

            mel_spec = librosa.feature.melspectrogram(y=waveform, sr=sr, n_mels=self.n_mels)
            mel_spec_db = librosa.amplitude_to_db(mel_spec, ref=np.max)
            mel_spec_db_normalized = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
            mel_resized = torch.nn.functional.interpolate(
                torch.tensor(mel_spec_db_normalized).unsqueeze(0).unsqueeze(0),
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
            ).squeeze()
            return mel_resized.repeat(3, 1, 1)
        except subprocess.CalledProcessError:
            print(f"[INFO] No audio stream found in video: {video_path}")
            return None
        except Exception as e:
            print(f"[ERROR] Failed to process audio for video {video_path}: {e}")
            return None


@ray.remote(num_gpus=1, num_cpus=3)
class TextProcessor:
    def __init__(self, whisper_model="base", roberta_model="roberta-base"):
        self.whisper_model = whisper.load_model(whisper_model)
        self.tokenizer = RobertaTokenizer.from_pretrained(roberta_model)

    def process_text(self, video_path):
        """
        Transcribes audio from a video and tokenizes the text.

        Args:
            video_path (str): Path to the video file.

        Returns:
            dict or None: Tokenized data containing "input_ids" and "attention_mask",
                          or None if no audio exists.
        """
        try:
            result = self.whisper_model.transcribe(video_path)
            text = result["text"]
            encoded = self.tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=512,
            )
            return {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}
        except Exception as e:
            print(f"[ERROR] Failed to process text for video {video_path}: {e}")
            return None


@ray.remote
def preprocess_task(video_paths, processor_actor, output_dir, modality, chunk_size=32):
    """
    Processes multiple video files for a specific modality, accumulates tensors/labels,
    and saves them in chunks.

    Args:
        video_paths (list[str]): List of video file paths.
        processor_actor (ray.actor.ActorHandle): Ray Actor for processing.
        output_dir (str): Base directory for processed files.
        modality (str): Modality ('video', 'audio', or 'text').
        chunk_size (int): Number of tensors per chunk.
    """
    tensors = []
    labels = []
    chunk_id = 0

    for video_path in video_paths:
        # Extract emotion label from the directory structure
        label = os.path.basename(os.path.dirname(video_path))

        # Process the video for the given modality
        result = ray.get(
            processor_actor.process_video.remote(video_path)
            if modality == "video"
            else processor_actor.process_audio.remote(video_path)
            if modality == "audio"
            else processor_actor.process_text.remote(video_path)
        )

        if result is not None:
            tensors.append(result)
            labels.append(label)

        # If chunk size is reached, save the chunk
        if len(tensors) == chunk_size:
            save_chunk(output_dir, modality, chunk_id, tensors, labels)
            tensors = []  # Reset tensors and labels for the next chunk
            labels = []
            chunk_id += 1

    # Save any remaining tensors as the final chunk
    if tensors:
        save_chunk(output_dir, modality, chunk_id, tensors, labels)


def main():
    ray.init()

    # Initialize processors
    video_processor = VideoProcessor.remote()
    audio_processor = AudioProcessor.remote()
    text_processor = TextProcessor.remote()

    raw_dir = "data/raw"
    output_dir = "data/processed"
    video_paths = get_video_paths(raw_dir)

    # Split video paths into smaller batches for chunk processing
    chunk_size = 32  # Number of tensors per chunk
    batch_size = 64  # Number of videos per batch
    video_batches = [video_paths[i:i + batch_size] for i in range(0, len(video_paths), batch_size)]

    # Launch tasks for all modalities in parallel
    futures = []
    for batch in video_batches:
        futures.append(preprocess_task.remote(batch, video_processor, output_dir, "video", chunk_size))
        futures.append(preprocess_task.remote(batch, audio_processor, output_dir, "audio", chunk_size))
        futures.append(preprocess_task.remote(batch, text_processor, output_dir, "text", chunk_size))

    # Wait for all tasks to complete
    ray.get(futures)
    print("Processing complete. Chunks saved in:", output_dir)


if __name__ == "__main__":
    main()
