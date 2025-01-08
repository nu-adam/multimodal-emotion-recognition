from torchvision import transforms


def get_transforms():
    """
    Returns transformations for video frames, mel spectrograms, and text.

    Returns:
        dict: Dictionary with 'video_frames' and 'mel_spectrogram' transforms.
    """
    return {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor()
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ]),
    }
