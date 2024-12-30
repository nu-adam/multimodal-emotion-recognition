import os
import logging
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from src.models.her_model import HERModel
from src.datasets.dataset import MELTDataset

# Define constants
NUM_EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
MAX_GRAD_NORM = 1.0
CHECKPOINT_DIR = "checkpoints"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train_logger")

# Save model function
def save_model(state, checkpoint_dir):
    filename = f'{checkpoint_dir}/best_model.pth'
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(state, filename)

# Training function
def train_one_epoch(model, train_loader, criterion, optimizer, max_grad, scaler, epoch, num_epochs, device):
    model.train()
    running_loss = 0.0

    for batch in tqdm(train_loader, unit="batch", desc=f"Epoch {epoch+1}/{num_epochs} - Training", ncols=100):
        video, audio, text_input_ids, text_attention_mask, labels = (
            batch['video'].to(device),
            batch['audio'].to(device),
            batch['text_input_ids'].to(device),
            batch['text_attention_mask'].to(device),
            batch['label'].to(device)
        )

        optimizer.zero_grad()

        # Forward pass
        with autocast(device_type=str(device), dtype=torch.float16):
            outputs = model(video, audio, text_input_ids, text_attention_mask)
            loss = criterion(outputs, labels)

        # Backward pass
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * video.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss

# Validation function
def validate_one_epoch(model, val_loader, criterion, epoch, num_epochs, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, unit="batch", desc=f"Epoch {epoch+1}/{num_epochs} - Validation", ncols=100):
            video, audio, text_input_ids, text_attention_mask, labels = (
                batch['video'].to(device),
                batch['audio'].to(device),
                batch['text_input_ids'].to(device),
                batch['text_attention_mask'].to(device),
                batch['label'].to(device)
            )

            outputs = model(video, audio, text_input_ids, text_attention_mask)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * video.size(0)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(val_loader.dataset)
    accuracy = correct / total
    return epoch_loss, accuracy

# Training pipeline
def train_model(train_loader, val_loader, model, criterion, optimizer, scheduler, max_grad, scaler, num_epochs, device, checkpoint_dir, logger):
    logger.info(f"Training the model for {num_epochs} epochs...")

    best_loss = float('inf')

    for epoch in range(num_epochs):
        # Train for one epoch
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, max_grad, scaler, epoch, num_epochs, device)
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Training Loss: {train_loss:.4f}")

        # Validate after one epoch
        val_loss, val_accuracy = validate_one_epoch(model, val_loader, criterion, epoch, num_epochs, device)
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy*100:.2f}%")

        # Step the scheduler
        scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Learning Rate: {current_lr:.6f}")

        # Save the best model checkpoint
        if val_loss < best_loss:
            best_loss = val_loss
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy
            }
            save_model(checkpoint, checkpoint_dir=checkpoint_dir)
            logger.info(f"Model checkpoint saved at epoch {epoch+1} to {checkpoint_dir}")

        logger.info(f"Epoch {epoch+1}/{num_epochs} - Best Validation Loss: {best_loss:.4f}")

    logger.info("Training finished successfully.")

# Main script
def main():
    # Load dataset and DataLoader
    train_dataset = MELTDataset(data_dir="data/train", face_detector=None)  # Update face detector as needed
    val_dataset = MELTDataset(data_dir="data/val", face_detector=None)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Initialize model, optimizer, criterion, scheduler, and scaler
    model = HERModel().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    scaler = GradScaler()

    # Train the model
    train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        max_grad=MAX_GRAD_NORM,
        scaler=scaler,
        num_epochs=NUM_EPOCHS,
        device=DEVICE,
        checkpoint_dir=CHECKPOINT_DIR,
        logger=logger
    )

if __name__ == "__main__":
    main()
