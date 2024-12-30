import os
import torch
from torch.utils.data import Dataset
from src.preprocessing.audio_preprocess import extract_audio_from_video, preprocess_audio
from src.preprocessing.video_preprocess import extract_frames, preprocess_frames
from src.preprocessing.text_preprocess import extract_text_from_audio, preprocess_text


class MELTDataset(Dataset):
    def __init__(self, data_dir, face_detector):
        """Dataset for MELT videos with modular preprocessing."""
        self.data_dir = data_dir
        self.face_detector = face_detector
        self.video_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.mp4')]

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]

        # Video preprocessing
        frames = extract_frames(video_path)
        video_tensor = preprocess_frames(frames)

        # Audio preprocessing
        audio_path = extract_audio_from_video(video_path)
        audio_tensor = preprocess_audio(audio_path)

        # Text preprocessing
        text = extract_text_from_audio(audio_path)
        text_input_ids, text_attention_mask = preprocess_text(text)

        # Placeholder for label (replace with actual labels)
        label = torch.tensor(0)  # Update with real labels

        return {
            'video': video_tensor,
            'audio': audio_tensor,
            'text_input_ids': text_input_ids.squeeze(0),
            'text_attention_mask': text_attention_mask.squeeze(0),
            'label': label
        }
