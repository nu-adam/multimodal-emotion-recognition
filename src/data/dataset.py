import os
import torch
from torch.utils.data import Dataset
from preprocessing.multimodal_preprocessing import preprocess_multimodal_input

class MultimodalDataset(Dataset):
    """
    Custom Dataset for Multimodal Emotion Recognition.

    Args:
        data_dir (str): Root directory containing data.
        transform (dict): Dictionary of transforms for 'video_frames', 'mel_spectrogram', and 'text'.
        frame_rate (int): Frames per second to extract from videos.
        sr (int): Sampling rate for audio.
    """
    def __init__(self, data_dir, transforms=None, frame_rate=1, sr=16000):
        self.data_dir = data_dir
        self.transforms = transforms
        self.frame_rate = frame_rate
        self.sr = sr
        self.samples = []
        self.labels = []

        # Map labels to folder names
        self.label_map = {label: idx for idx, label in enumerate(sorted(os.listdir(data_dir)))}

        # Collect all samples with labels
        for label in self.label_map:
            label_dir = os.path.join(data_dir, label)
            for file_name in os.listdir(label_dir):
                if file_name.endswith(('.mp4', '.avi', '.mkv')):
                    self.samples.append(os.path.join(label_dir, file_name))
                    self.labels.append(self.label_map[label])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Get video path and corresponding label
        video_path = self.samples[idx]
        label = self.labels[idx]

        # Generate paths for audio and transcript
        audio_save_path = video_path.replace('.mp4', '.wav').replace('.avi', '.wav').replace('.mkv', '.wav')
        transcript_save_path = video_path.replace('.mp4', '.srt').replace('.avi', '.srt').replace('.mkv', '.srt')

        # Preprocess video, audio, and text
        data = preprocess_multimodal_input(
            video_path=video_path,
            audio_save_path=audio_save_path,
            transcript_save_path=transcript_save_path,
            frame_rate=self.frame_rate,
            sr=self.sr
        )

        # Apply transforms
        if self.transform:
            data['train'] = [self.transform['video_frames'](frame) for frame in data['video_frames']]

        # Convert video frames to tensor
        video_tensor = torch.stack(data['video_frames']) if data['video_frames'] else torch.empty(0)

        return {
            'video': video_tensor,
            'audio': torch.tensor(data['mel_spectrogram'], dtype=torch.float32),
            'text': data['tokenized_text'],
            'label': torch.tensor(label, dtype=torch.long)
        }