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


def save_chunk(output_dir, modality, chunk_id, video_ids, labels, tensors):
    """
    Saves a chunk of tensors, labels, and video IDs to a .pth file.

    Args:
        output_dir (str): Base directory for processed files.
        modality (str): Modality ('video', 'audio', or 'text').
        chunk_id (int): Chunk ID (used for file naming).
        video_ids (list[str]): List of video filenames corresponding to the tensors.
        labels (list[str]): List of emotion labels corresponding to the tensors.
        tensors (list[torch.Tensor]): List of tensors to save.        
    """
    output_path = os.path.join(output_dir, modality, f"{modality}_chunk_{chunk_id:04d}.pth")
    ensure_directory_exists(output_path)  # Ensure the directory exists
    torch.save({"video_ids": video_ids, "labels": labels, "tensors": tensors}, output_path)
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


@ray.remote(num_cpus=8)
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
def preprocess_task(video_paths, processor_actor, output_dir, modality, chunk_size=32, split="train"):
    """
    Processes multiple video files for a specific modality and split,
    accumulates tensors/labels/video_ids, and saves them in chunks.

    Args:
        video_paths (list[str]): List of video file paths.
        processor_actor (ray.actor.ActorHandle): Ray Actor for processing.
        output_dir (str): Base directory for processed files.
        modality (str): Modality ('video', 'audio', or 'text').
        chunk_size (int): Number of tensors per chunk.
        split (str): Dataset split ('train', 'val', or 'test').

    Returns:
        None
    """
    tensors = []
    labels = []
    video_ids = []
    chunk_id = 0

    for video_path in video_paths:
        # Extract emotion label and video ID
        label = os.path.basename(os.path.dirname(video_path))
        video_id = os.path.basename(video_path)

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
            video_ids.append(video_id)

        # Save chunk when size limit is reached
        if len(tensors) == chunk_size:
            save_chunk(os.path.join(output_dir, split), modality, chunk_id, video_ids, labels, tensors)
            video_ids = []
            labels = []
            tensors = []
            chunk_id += 1

    # Save remaining tensors
    if tensors:
        save_chunk(os.path.join(output_dir, split), modality, chunk_id, tensors, labels, video_ids)


def main():
    ray.init(include_dashboard=True)

    # Initialize processors
    video_processor = VideoProcessor.remote()
    audio_processor = AudioProcessor.remote()
    text_processor = TextProcessor.remote()

    raw_dir = "data/raw"
    output_dir = "data/processed"

    # Get split-specific video paths
    splits = ["train", "val", "test"]
    video_paths_by_split = {split: get_video_paths(os.path.join(raw_dir, split)) for split in splits}

    # Process each split separately
    futures = []
    for split, video_paths in video_paths_by_split.items():
        futures.append(preprocess_task.remote(video_paths, video_processor, output_dir, "video", split=split))
        futures.append(preprocess_task.remote(video_paths, audio_processor, output_dir, "audio", split=split))
        futures.append(preprocess_task.remote(video_paths, text_processor, output_dir, "text", split=split))

    # Wait for all tasks to complete
    ray.get(futures)
    print("Processing complete.")


if __name__ == "__main__":
    main()

    # Load a video chunk
    video_data = torch.load("data/processed/train/video/video_chunk_0000.pth")
    audio_data = torch.load("data/processed/train/audio/audio_chunk_0000.pth")
    text_data = torch.load("data/processed/train/text/text_chunk_0000.pth")

    # Match data using video IDs
    video_ids = set(video_data["video_ids"])  # Get the video IDs from the video chunk
    aligned_video_tensors = []
    aligned_audio_tensors = []
    aligned_text_tensors = []
    aligned_labels = []

    for i, video_id in enumerate(video_data["video_ids"]):
        if video_id in video_ids:
            aligned_video_tensors.append(video_data["tensors"][i])
            aligned_audio_tensors.append(audio_data["tensors"][i])
            aligned_text_tensors.append(text_data["tensors"][i])
            aligned_labels.append(video_data["labels"][i])  # Assuming labels are consistent across modalities

    # Check alignment
    print(f"Aligned {len(aligned_video_tensors)} samples.")
