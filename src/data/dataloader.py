from data.dataset import MultimodalDataset
from data.transforms import get_transforms

from torch.utils.data import DataLoader


def get_dataloader(data_dir, batch_size=16, shuffle=True, num_workers=4, frame_rate=1, sr=16000):
    """
    Creates a DataLoader for the multimodal dataset.

    Args:
        data_dir (str): Directory containing data.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the data.
        num_workers (int): Number of worker processes for data loading.
        frame_rate (int): Frames per second to extract from videos.
        sr (int): Sampling rate for audio.

    Returns:
        DataLoader: DataLoader object for the dataset.
    """
    transforms = get_transforms()
    train_dataset = MultimodalDataset(data_dir, transforms=transforms, frame_rate=frame_rate, sr=sr)
    train_dataset.dataset.transform = transforms['train']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return train_loader
