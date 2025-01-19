import os
import logging
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.multimodal_emotion_recognition import MultimodalEmotionRecognition
from src.data.dataset import MockMultimodalDataset
from src.training.train_utils import train_model
from src.utils.logger import setup_logger


def train(enabled_modalities, data_dir, num_classes, batch_size, learning_rate, num_epochs, checkpoint_dir, log_dir):
    """
    Training for the Face Emotion Recognition model using transfer learning on the specified dataset.

    Args:
    - enabled_modalities (list): List of enabled modalities (e.g., ['video', 'audio', 'text']).
    - data_dir (str): Path to the root directory containing 'train' and 'test' subdirectories.
    - num_classes (int): Number of emotion classes for classification.
    - batch_size (int, optional): Number of samples per batch to load. Default is 32.
    - learning_rate (float, optional): Learning rate for the optimizer. Default is 0.001.
    - num_epochs (int, optional): Number of epochs to train the model. Default is 10.
    - checkpoint_dir (str, optional): Directory to save model checkpoints. Default is 'results/checkpoints/'.
    - log_dir (str, optional): Directory to save training logs. Default is 'results/logs/'.

    Returns:
    - None
    """
    # Set up logging
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger(log_dir=log_dir, log_file='train', log_level=logging.INFO)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Set up the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    # Load the dataset
    train_dataset = MockMultimodalDataset(
        num_samples=1,
        enabled_modalities=ENABLED_MODALITIES,
        num_classes=NUM_CLASSES
    )
    val_dataset = MockMultimodalDataset(
        num_samples=1,
        enabled_modalities=ENABLED_MODALITIES,
        num_classes=NUM_CLASSES
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    logger.info(f'Dataset loaded from {data_dir}.')

    # Initialize the model
    model = MultimodalEmotionRecognition(
        enabled_modalities=enabled_modalities,
        embed_dim=256,
        num_heads=4,
        num_layers=2,
        num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    max_grad = 1.0
    scaler = torch.GradScaler(str(device))

    logger.info(f'Configuration:\n'
                f'Batch Size: {batch_size}\n'
                f'Learning Rate: {learning_rate}\n'
                f'Epochs: {num_epochs}\n'
                f'Criterion: {criterion}\n'
                f'Optimizer: {optimizer}\n'
                f'Model architecture:\n{model}')

    # Train the model
    train_model(
        model, enabled_modalities, train_loader, val_loader, 
        criterion, optimizer, scheduler, max_grad, scaler,
        num_epochs, device, checkpoint_dir, logger
        )


if __name__ == '__main__':
    # Configuration parameters
    DATA_DIR = r'C:\dev\her-emotion-recognition\data'
    NUM_CLASSES = 7
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0001
    NUM_EPOCHS = 1
    CHECKPOINT_DIR = 'results/checkpoints/'
    LOG_DIR = 'results/logs/'
    ENABLED_MODALITIES = ['video', 'audio', 'text']

    train(ENABLED_MODALITIES, DATA_DIR, NUM_CLASSES, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, CHECKPOINT_DIR, LOG_DIR)
