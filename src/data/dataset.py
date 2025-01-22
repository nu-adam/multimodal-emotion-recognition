import os
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
import numpy as np

from src.data.transforms import get_video_transforms, get_audio_transforms
from src.preprocessing.video_preprocess import preprocess_video, tensor_to_pil
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
            if features['video'] is None:
                # Skip sample if no video features are extracted
                return self.__getitem__((idx + 1) % len(self))

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
        video_tensors = preprocess_video(file_path)
        
        # Handle the case where no faces are detected
        if video_tensors is None or video_tensors.size(0) == 0:
            print(f"No faces detected in video: {file_path}")
            return None

        # Convert the tensor to PIL.Image for transforms
        if isinstance(video_tensors, torch.Tensor):
            # Assume video_tensors is a batch of frames (T, C, H, W)
            frames = [tensor_to_pil(frame) for frame in video_tensors]
            video_features = torch.stack([self.video_transform(frame) for frame in frames])
        else:
            print(f"Unexpected type for video_tensors: {type(video_tensors)}")
            return None

        return video_features

    def _extract_audio_features(self, file_path):
        """
        Extracts and preprocesses audio features.

        Args:
            file_path (str): Path to the video file.

        Returns:
            torch.Tensor: Preprocessed audio features.
        """
        # Preprocess the audio to get a spectrogram tensor
        audio_tensors = preprocess_audio(file_path)

        # Convert torch.Tensor to numpy.ndarray for compatibility with transforms
        if isinstance(audio_tensors, torch.Tensor):
            audio_array = audio_tensors.squeeze().numpy()  # Remove extra dimensions, if any
            audio_array = (audio_array * 255).astype(np.uint8)  # Scale to [0, 255]
            
            # Ensure the array has the correct shape for grayscale or RGB
            if audio_array.ndim == 2:  # Grayscale
                audio_image = Image.fromarray(audio_array)
            elif audio_array.ndim == 3 and audio_array.shape[0] == 3:  # RGB-like tensor
                audio_image = Image.fromarray(np.transpose(audio_array, (1, 2, 0)))  # Convert (C, H, W) to (H, W, C)
            else:
                print(f"Unexpected shape for audio_array: {audio_array.shape}")
                return None

            # Apply the audio transform
            audio_features = self.audio_transform(audio_image)
        else:
            print(f"Unexpected type for audio_tensors: {type(audio_tensors)}")
            return None

        return audio_features



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

def collate_fn(batch, enabled_modalities):
    """
    Custom collate function to pad tensors for enabled modalities.
    
    Args:
        batch (list): List of tuples (features, label), where features is a dictionary.
        enabled_modalities (list): List of enabled modalities, e.g., ['video', 'audio', 'text'].
    
    Returns:
        dict: Batch of padded features for enabled modalities.
        torch.Tensor: Batch of labels.
    """
    features, labels = zip(*batch)
    batch_features = {}

    # Process each enabled modality
    if 'video' in enabled_modalities:
        video_tensors = [f['video'] for f in features if f['video'] is not None]
        if video_tensors:
            batch_features['video'] = pad_sequence(video_tensors, batch_first=True)
        else:
            batch_features['video'] = None

    if 'audio' in enabled_modalities:
        audio_tensors = [f['audio'] for f in features if f['audio'] is not None]
        if audio_tensors:
            batch_features['audio'] = torch.stack(audio_tensors)
        else:
            batch_features['audio'] = None

    if 'text' in enabled_modalities:
        text_tensors = [f['text'] for f in features if f['text'] is not None]
        if text_tensors:
            # Stack input_ids and attention_mask separately
            input_ids = torch.stack([t['input_ids'].squeeze(0) for t in text_tensors])
            attention_mask = torch.stack([t['attention_mask'].squeeze(0) for t in text_tensors])
            batch_features['text'] = {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }
        else:
            batch_features['text'] = None

    batch_labels = torch.tensor(labels)

    return batch_features, batch_labels
