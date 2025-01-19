# import os
# import torch
# from torch.utils.data import Dataset
# from torchvision.io import read_video
# from torchvision.transforms import ToPILImage
# from src.preprocessing.video_preprocess import extract_visual_features
# from src.preprocessing.text_preprocess import preprocess_text


# class MultimodalEmotionDataset(Dataset):
#     def __init__(self, data_dir, transform=None, enabled_modalities=['visual', 'audio', 'text']):
#         """
#         Dataset for loading emotion classification data.

#         Args:
#             data_dir (str): Root directory of the dataset (e.g., 'data/train').
#             transform (callable, optional): Transformations for visual data.
#             enabled_modalities (list): List of enabled modalities (e.g., ['visual', 'audio', 'text']).
#         """
#         self.data_dir = data_dir
#         self.transform = transform
#         self.enabled_modalities = enabled_modalities
#         self.data = self._load_data()

#     def _load_data(self):
#         """
#         Traverse the data directory and collect file paths and labels.
#         """
#         data = []
#         for class_name in os.listdir(self.data_dir):
#             class_dir = os.path.join(self.data_dir, class_name)
#             # Ignore non-directory files like .DS_Store
#             if os.path.isdir(class_dir):
#                 for file_name in os.listdir(class_dir):
#                     if file_name.endswith('.mp4'):
#                         file_path = os.path.join(class_dir, file_name)
#                         data.append((file_path, class_name))
#         return data

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         path, class_name = self.data[idx]
#         features = {'visual': None, 'audio': None, 'text': None}

#         label = self._get_label(class_name)

#         if 'visual' in self.enabled_modalities:
#             features['visual'] = extract_visual_features(path, self.transform)
#         if 'audio' in self.enabled_modalities:
#             features['audio'] = self._extract_audio_features(path)
#         if 'text' in self.enabled_modalities:
#             # TODO: provide text extraction in preprocessing/text_preprocess and use here
#             text_data = 'Today is the great day for training in the my favourite gym'
#             features['text'] = preprocess_text(text_data)

#         return features, label

#     def _get_label(self, class_name):
#         """
#         Convert class names to numeric labels. Ensures consistency by using sorted order.
#         """
#         # Filter out non-directory entries when building the class list
#         classes = [
#             cls for cls in sorted(os.listdir(self.data_dir))
#             if os.path.isdir(os.path.join(self.data_dir, cls))
#         ]
#         class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
#         return class_to_idx[class_name]

#     def _extract_visual_features(self, path):
#         video, _, _ = read_video(path, pts_unit='sec')
#         # Take the middle frame (shape: H x W x C)
#         frame = video[len(video) // 2]  # Middle frame of the video

#         # Ensure frame is in (C, H, W) format for PyTorch
#         frame = frame.permute(2, 0, 1)  # Convert from (H, W, C) to (C, H, W)

#         if self.transform:
#             # Convert to PIL.Image if transform expects it
#             frame = ToPILImage()(frame)
#             frame = self.transform(frame)  # Apply transformations (resize, normalize, etc.)

#         return frame  # Tensor with shape (3, H, W)

#     def _extract_audio_features(self, path):
#         """
#         Placeholder for audio feature extraction.
#         """
#         # Extract audio features (e.g., using pretrained models like Wav2Vec2)
#         # For now, we return random features for simplicity.
#         return torch.randn(768)

#     def _extract_text_features(self, path):
#         """
#         Placeholder for text feature extraction.
#         """
#         # Extract text features from subtitles or other sources.
#         # For now, we return random features for simplicity.
#         return torch.randn(768)


import torch
from torch.utils.data import Dataset, DataLoader

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


if __name__ == '__main__':
    # Configuration
    ENABLED_MODALITIES = ['video', 'audio', 'text']
    NUM_CLASSES = 7
    NUM_SAMPLES = 100  # Number of mock samples
    BATCH_SIZE = 8

    # Create mock dataset and DataLoader
    mock_dataset = MockMultimodalDataset(
        num_samples=NUM_SAMPLES,
        enabled_modalities=ENABLED_MODALITIES,
        num_classes=NUM_CLASSES
    )
    mock_loader = DataLoader(mock_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Example batch
    for batch in mock_loader:
        print(batch)
        break
