import torch
from torchvision.io import read_video
from torchvision.transforms import ToPILImage

def extract_visual_features(path, transform=None):
    """
    Extract visual features from a video file.

    Args:
        path (str): Path to the video file.
        transform (callable, optional): Transformations for visual data.

    Returns:
        torch.Tensor: Tensor with visual features, shape (3, H, W).
    """
    video, _, _ = read_video(path, pts_unit='sec')
    # Take the middle frame (shape: H x W x C)
    frame = video[len(video) // 2]  # Middle frame of the video

    # Ensure frame is in (C, H, W) format for PyTorch
    frame = frame.permute(2, 0, 1)  # Convert from (H, W, C) to (C, H, W)

    if transform:
        # Convert to PIL.Image if transform expects it
        frame = ToPILImage()(frame)
        frame = transform(frame)  # Apply transformations (resize, normalize, etc.)

    return frame  # Tensor with shape (3, H, W)