import sys
import os

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.data.dataset import EmotionDataset
from src.models.fuser import MultimodalTransformer
from src.config import data_dir, batch_size, enabled_modalities

def main():
    checkpoint_path = os.path.join('checkpoints', 'best_model.pth')
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    # Define the model
    model = MultimodalTransformer(enabled_modalities=enabled_modalities, num_classes=7)
    model.to(device)

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract the model state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']} with validation loss {checkpoint['best_loss']:.4f}")

    # Define the test dataset and dataloader
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    test_dataset = EmotionDataset(os.path.join(data_dir, 'test'), transform, enabled_modalities=enabled_modalities)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Evaluate the model
    model.eval()
    running_corrects = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Testing loop'):
            inputs = {modality: inputs[modality].to(device) for modality in inputs}
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            running_corrects += (preds == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = running_corrects / total_samples
    print(f"Test Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
