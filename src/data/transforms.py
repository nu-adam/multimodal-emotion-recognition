from torchvision.transforms import Compose, Resize, Normalize, ToTensor, RandomHorizontalFlip, RandomRotation


def get_video_transforms(split):
    """
    Returns transforms for video data based on the data split.

    Args:
        split (str): 'train', 'val', or 'test'.

    Returns:
        callable: Composed transformations.
    """
    if split == 'train':
        return Compose([
            Resize((224, 224)),
            RandomHorizontalFlip(),
            RandomRotation(10),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


def get_audio_transforms(split):
    """
    Returns transforms for audio data (mel spectrogram images) based on the data split.

    Args:
        split (str): 'train', 'val', or 'test'.

    Returns:
        callable: Audio transformation pipeline for the specified split.
    """
    if split == 'train':
        return Compose([
            Resize((224, 224)),
            RandomHorizontalFlip(p=0.5),
            ToTensor(),
            Normalize(mean=[0.5], std=[0.5])
        ])
    else:
        return Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=[0.5], std=[0.5])
        ])
