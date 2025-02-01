import os
import sys

# Add the root directory to the Python path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(root_dir)

import logging
import torch
import torch.optim as optim
import torch.nn as nn
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from functools import partial
import bitsandbytes as bnb

from src.models.multimodal_emotion_recognition import MultimodalEmotionRecognition
from src.data.dataset import MultimodalDataset, custom_collate_fn
from src.training.train_utils import train_model
from src.utils.logger import setup_logger


def train(enabled_modalities, data_dir, num_classes, batch_size, 
          learning_rate, weight_decay, accumulation_steps, 
          max_grad, num_epochs, checkpoint_dir, log_dir):
    """
    Training for the Face Emotion Recognition model using transfer learning on the specified dataset.

    Args:
    - enabled_modalities (list): List of enabled modalities (e.g., ['video', 'audio', 'text']).
    - data_dir (str): Path to the root directory containing 'train' and 'test' subdirectories.
    - num_classes (int): Number of emotion classes for classification.
    - batch_size (int, optional): Number of samples per batch to load.
    - learning_rate (float, optional): Learning rate for the optimizer.
    - accumulation_steps (int): Number of steps for gradient accumulation.
    - max_grad (float, optional): Maximum gradient norm for gradient clipping.
    - num_epochs (int, optional): Number of epochs to train the model.
    - checkpoint_dir (str, optional): Directory to save model checkpoints. Default is 'results/checkpoints/'.
    - log_dir (str, optional): Directory to save training logs. Default is 'results/logs/'.

    Returns:
    - None
    """    
    # Set up logging
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger(log_dir=log_dir, log_file=f'train', log_level=logging.INFO)
    
    # Set up the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    # Dataset initialization
    train_dataset = MultimodalDataset(
        data_dir=data_dir,
        modalities=enabled_modalities,
        split='train'
    )
    val_dataset = MultimodalDataset(
        data_dir=data_dir,
        modalities=enabled_modalities,
        split='val'
    )

    # Collate function initialization
    collate_fn = partial(custom_collate_fn, modalities=enabled_modalities)

    # DataLoader initialization
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=False,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=False,
        shuffle=False,
        collate_fn=collate_fn
    )
    logger.info(f'Dataset loaded from {data_dir}.')

    # Initialize the model
    model = MultimodalEmotionRecognition(
        enabled_modalities=enabled_modalities,
        embed_dim=256,
        num_heads=4,
        num_layers=2,
        num_classes=num_classes).to(device)
    model = nn.DataParallel(model)
    
    # Initialize hyperparameters
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    scaler = GradScaler()

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
        criterion, optimizer, accumulation_steps, scheduler, 
        max_grad, scaler, num_epochs, device, checkpoint_dir, logger
        )


if __name__ == '__main__':
    # Configuration parameters
    DATA_DIR = 'data/processed/'
    NUM_CLASSES = 7
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-2
    WEIGHT_DECAY = 1e-4
    ACCUMULATION_STEPS = 2
    MAX_GRAD = 1
    NUM_EPOCHS = 3
    CHECKPOINT_DIR = 'results/checkpoints/'
    LOG_DIR = 'results/logs/'
    ENABLED_MODALITIES = ['video', 'audio', 'text']

    train(ENABLED_MODALITIES, DATA_DIR, NUM_CLASSES, BATCH_SIZE, LEARNING_RATE, 
          WEIGHT_DECAY, ACCUMULATION_STEPS, MAX_GRAD, NUM_EPOCHS, CHECKPOINT_DIR, LOG_DIR)
