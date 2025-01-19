import os
import logging
import torch
from torch.utils.data import DataLoader

from src.data.dataset import MultimodalEmotionDataset
from src.training.test_utils import evaluate_model
from src.models.multimodal_emotion_recognition import MultimodalEmotionRecognition
from src.utils.logger import setup_logger


def test(enabled_modalities, data_dir, num_classes, batch_size, checkpoint_dir, log_dir):
    """
    Testing for the Face Emotion Recognition model on the specified dataset.

    Args:
    - enabled_modalities (list): List of enabled modalities (e.g., ['video', 'audio', 'text']).
    - data_dir (str): Path to the root directory containing the test subdirectory.
    - num_classes (int): Number of emotion classes for classification.
    - batch_size (int, optional): Number of samples per batch to load. Default is 32.
    - checkpoint_dir (str, optional): Directory to load model checkpoints from. Default is 'results/checkpoints/'.
    - log_dir (str, optional): Directory to save testing logs. Default is 'results/logs/'.

    Returns:
    - None
    """
    # Set up logging
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger(log_dir=log_dir, log_file='test', log_level=logging.INFO)

    # Set up the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    # Load the dataset
    test_dataset = MultimodalEmotionDataset(
        data_dir=f"{data_dir}/test",
        enabled_modalities=enabled_modalities
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    logger.info(f'Test dataset loaded from {data_dir}.')

    # Initialize the model
    model = MultimodalEmotionRecognition(
        enabled_modalities=enabled_modalities,
        embed_dim=256,
        num_heads=4,
        num_layers=2,
        num_classes=num_classes).to(device)

    # Load the checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint file not found: {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f'Model checkpoint loaded from {checkpoint_path}.')

    # Evaluate the model
    metrics = evaluate_model(model, test_loader, device, logger, enabled_modalities)

    # Log the results
    logger.info(f'Testing completed. Metrics:\n'
                f'Accuracy: {metrics["accuracy"]:.4f}\n'
                f'Precision: {metrics["precision"]:.4f}\n'
                f'Recall: {metrics["recall"]:.4f}\n'
                f'F1-score: {metrics["f1_score"]:.4f}')


if __name__ == '__main__':
    # Configuration parameters
    DATA_DIR = r'C:\dev\her-emotion-recognition\data'
    NUM_CLASSES = 7
    BATCH_SIZE = 32
    CHECKPOINT_DIR = 'results/checkpoints/'
    LOG_DIR = 'results/logs/'
    ENABLED_MODALITIES = ['video', 'audio', 'text']

    test(ENABLED_MODALITIES, DATA_DIR, NUM_CLASSES, BATCH_SIZE, CHECKPOINT_DIR, LOG_DIR)
