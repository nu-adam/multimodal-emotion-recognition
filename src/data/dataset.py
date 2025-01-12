import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.io import read_video
from torchvision.transforms import ToPILImage

class EmotionDataset(Dataset):
    def __init__(self, data_dir, transform=None, enabled_modalities=['visual', 'audio', 'text']):
        """
        Dataset for loading emotion classification data.

        Args:
            data_dir (str): Root directory of the dataset (e.g., 'data/train').
            transform (callable, optional): Transformations for visual data.
            enabled_modalities (list): List of enabled modalities (e.g., ['visual', 'audio', 'text']).
        """
        self.data_dir = data_dir
        self.transform = transform
        self.enabled_modalities = enabled_modalities
        self.data = self._load_data()

    def _load_data(self):
        """
        Traverse the data directory and collect file paths and labels.
        """
        data = []
        for class_name in os.listdir(self.data_dir):
            class_dir = os.path.join(self.data_dir, class_name)
            # Ignore non-directory files like .DS_Store
            if os.path.isdir(class_dir):
                for file_name in os.listdir(class_dir):
                    if file_name.endswith('.mp4'):
                        file_path = os.path.join(class_dir, file_name)
                        data.append((file_path, class_name))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, class_name = self.data[idx]
        features = {'visual': None, 'audio': None, 'text': None}

        # Map class_name to an integer label
        label = self._get_label(class_name)

        if 'visual' in self.enabled_modalities:
            features['visual'] = self._extract_visual_features(path)
        if 'audio' in self.enabled_modalities:
            features['audio'] = self._extract_audio_features(path)
        if 'text' in self.enabled_modalities:
            features['text'] = self._extract_text_features(path)

        return features, label

    def _get_label(self, class_name):
        """
        Convert class names to numeric labels. Ensures consistency by using sorted order.
        """
        # Filter out non-directory entries when building the class list
        classes = [
            cls for cls in sorted(os.listdir(self.data_dir))
            if os.path.isdir(os.path.join(self.data_dir, cls))
        ]
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        return class_to_idx[class_name]

    def _extract_visual_features(self, path):
        video, _, _ = read_video(path, pts_unit='sec')
        # Take the middle frame (shape: H x W x C)
        frame = video[len(video) // 2]  # Middle frame of the video

        # Ensure frame is in (C, H, W) format for PyTorch
        frame = frame.permute(2, 0, 1)  # Convert from (H, W, C) to (C, H, W)

        if self.transform:
            # Convert to PIL.Image if transform expects it
            frame = ToPILImage()(frame)
            frame = self.transform(frame)  # Apply transformations (resize, normalize, etc.)

        return frame  # Tensor with shape (3, H, W)

    def _extract_audio_features(self, path):
        """
        Placeholder for audio feature extraction.
        """
        # Extract audio features (e.g., using pretrained models like Wav2Vec2)
        # For now, we return random features for simplicity.
        return torch.randn(768)

    def _extract_text_features(self, path):
        """
        Placeholder for text feature extraction.
        """
        # Extract text features from subtitles or other sources.
        # For now, we return random features for simplicity.
        return torch.randn(768)
