import sys
import os

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

from src.data.dataset import EmotionDataset
from src.models.multimodal_emotion_recognition import MultimodalEmotionRecognition
from src.training.trainer import Trainer
from src.config import data_dir, batch_size, num_epochs, enabled_modalities, learning_rate

def main():
    # Define transformations for the visual modality
    transform = Compose([
        Resize((224, 224)),  # Resize frames to fit the model input size
        ToTensor(),          # Convert PIL/numpy images to tensors
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
    ])

    # Prepare datasets and dataloaders
    train_dataset = EmotionDataset(
        data_dir=f"{data_dir}/train", 
        transform=transform, 
        enabled_modalities=enabled_modalities
    )
    val_dataset = EmotionDataset(
        data_dir=f"{data_dir}/val", 
        transform=transform, 
        enabled_modalities=enabled_modalities
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    dataloaders = {'train': train_loader, 'val': val_loader}

    # Initialize the model
    model = MultimodalEmotionRecognition(
        enabled_modalities=enabled_modalities, 
        num_classes=7  # Adjust the number of classes as needed
    )
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    trainer = Trainer(model, dataloaders, criterion, optimizer, device)
    trainer.train(num_epochs)

if __name__ == "__main__":
    main()
