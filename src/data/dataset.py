import os
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from src.data.transforms import get_video_transforms, get_audio_transforms
from src.preprocessing.video_preprocess import preprocess_video
from src.preprocessing.audio_preprocess import preprocess_audio
from src.preprocessing.text_preprocess import preprocess_text


class MultimodalEmotionDataset(Dataset):
    """
    Dataset for loading multimodal emotion classification data.

    Supports video, audio, and text modalities with distinct transforms for train, val, and test splits.

    Args:
        data_dir (str): Root directory of the dataset (e.g., 'data/train').
        split (str): Data split - 'train', 'val', or 'test'.
        enabled_modalities (list): List of enabled modalities (e.g., ['video', 'audio', 'text']).
    """
    def __init__(self, data_dir, split='train', enabled_modalities=['video', 'audio', 'text']):
        self.data_dir = data_dir
        self.split = split
        self.enabled_modalities = enabled_modalities
        self.data = self._load_data()

        self.video_transform = get_video_transforms(split)
        self.audio_transform = get_audio_transforms(split)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: Dictionary containing modality-specific features.
            int: Numeric label corresponding to the sample.
        """
        file_path, label_name = self.data[idx]
        features = {modality: None for modality in self.enabled_modalities}

        # Convert label name to numeric label
        label = self._get_label(label_name)

        # Process each enabled modality
        if 'video' in self.enabled_modalities:
            features['video'] = self._extract_video_features(file_path)
        if 'audio' in self.enabled_modalities:
            features['audio'] = self._extract_audio_features(file_path)
        if 'text' in self.enabled_modalities:
            features['text'] = self._extract_text_features(file_path)

        return features, label
    
    def _load_data(self):
        """
        Loads file paths and corresponding labels from the dataset directory.

        Returns:
            list: A list of tuples (file_path, label_name).
        """
        data = []
        for label_name in os.listdir(self.data_dir):
            label_dir = os.path.join(self.data_dir, label_name)
            if os.path.isdir(label_dir):
                for file_name in os.listdir(label_dir):
                    file_path = os.path.join(label_dir, file_name)
                    data.append((file_path, label_name))
        return data

    def _get_label(self, label_name):
        """
        Converts label names to numeric labels.

        Returns:
            int: Numeric label corresponding to the label name.
        """
        classes = sorted([
            cls for cls in os.listdir(self.data_dir)
            if os.path.isdir(os.path.join(self.data_dir, cls))
        ])
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        return class_to_idx[label_name]

    def _extract_video_features(self, file_path):
        """
        Extracts and preprocesses video features.

        Args:
            file_path (str): Path to the video file.

        Returns:
            torch.Tensor: Preprocessed video features.
        """
        video_tensors = preprocess_video(file_path)
        # video_features = self.video_transform(video_tensors)
        video_features = video_tensors
        return video_features

    def _extract_audio_features(self, file_path):
        """
        Extracts and preprocesses audio features.

        Args:
            file_path (str): Path to the video file.

        Returns:
            torch.Tensor: Preprocessed audio features.
        """
        audio_tensors = preprocess_audio(file_path)
        audio_features = self.aduio_transform(audio_tensors)
        return audio_features(file_path, self.audio_transform)

    def _extract_text_features(self, file_path):
        """
        Extracts and preprocesses text features.

        Args:
            file_path (str): Path to the video file.

        Returns:
            torch.Tensor: Preprocessed text features.
        """
        text_features = preprocess_text(file_path)
        return text_features


class MockMultimodalDataset(Dataset):
    """
    Mock dataset for testing the train function with random tensors.

    Args:
    - num_samples (int): Number of samples in the dataset.
    - enabled_modalities (list): List of enabled modalities (e.g., ['video', 'audio', 'text']).
    - num_classes (int): Number of classes for classification.
    - video_shape (tuple): Shape of video input tensors (default: (3, 224, 224)).
    - audio_shape (tuple): Shape of audio input tensors (default: (3, 224, 224)).
    - text_length (int): Length of tokenized text sequences (default: 50).
    """
    def __init__(self, num_samples, enabled_modalities, num_classes, video_shape=(3, 224, 224), audio_shape=(3, 224, 224), text_length=50):
        self.num_samples = num_samples
        self.enabled_modalities = enabled_modalities
        self.num_classes = num_classes
        self.video_shape = video_shape
        self.audio_shape = audio_shape
        self.text_length = text_length

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        data = {}
        
        # Generate random tensors for enabled modalities
        if 'video' in self.enabled_modalities:
            data['video'] = torch.rand(self.video_shape)  # Simulate video frame tensor
        
        if 'audio' in self.enabled_modalities:
            data['audio'] = torch.rand(self.audio_shape)  # Simulate mel spectrogram tensor
        
        if 'text' in self.enabled_modalities:
            data['text'] = {
                'input_ids': torch.randint(0, 30522, (self.text_length,)),  # Simulate tokenized input IDs
                'attention_mask': torch.ones(self.text_length, dtype=torch.int64)  # Simulate attention mask
            }
        
        # Generate a random label
        data['label'] = torch.randint(0, self.num_classes, (1,)).item()

        return data

def collate_fn(batch):
    """
    Custom collate function to pad video tensors to the same length.
    
    Args:
        batch (list): List of tuples (features, label), where features is a dictionary.
    
    Returns:
        dict: Batch of padded features.
        torch.Tensor: Batch of labels.
    """
    features, labels = zip(*batch)
    video_tensors = [f['video'] for f in features]

    # Pad video tensors along the temporal dimension (frames)
    padded_videos = pad_sequence(video_tensors, batch_first=True)

    # Combine features into a dictionary
    batch_features = {
        'video': padded_videos,
        # 'audio': torch.stack([f['audio'] for f in features]),
        # 'text': torch.stack([f['text'] for f in features]),
    }

    batch_labels = torch.tensor(labels)

    return batch_features, batch_labels