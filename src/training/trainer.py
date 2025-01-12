import os
import torch
from tqdm import tqdm

class Trainer:
    def __init__(self, model, dataloaders, criterion, optimizer, device, checkpoint_dir='checkpoints'):
        self.model = model
        self.dataloaders = dataloaders
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = checkpoint_dir

        # Create checkpoint directory if it doesn't exist
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # To track the best model
        self.best_loss = float('inf')
        self.best_model_state = None
        self.best_epoch = -1

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            self.model.train()
            running_loss = 0.0

            for inputs, labels in tqdm(self.dataloaders['train'], desc='Training loop'):                
                inputs = {modality: inputs[modality].to(self.device) for modality in inputs}
                labels = labels.to(self.device)

                # Debug input shapes
                # for modality, tensor in inputs.items():
                #     print(f"Train {modality} input shape: {tensor.shape}")

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                # print(f"Model output shape: {outputs.shape}")  # Debug shape
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * labels.size(0)

            epoch_loss = running_loss / len(self.dataloaders['train'].dataset)
            print(f"Training Loss: {epoch_loss:.4f}")

            # Validation
            val_loss = self.validate()

            # Update the best model if this epoch has the lowest validation loss
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_model_state = self.model.state_dict()
                self.best_epoch = epoch + 1
                print(f"New best model found at epoch {self.best_epoch} with validation loss {self.best_loss:.4f}")

        # Save the best model after all epochs
        self._log_best_model()

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for inputs, labels in tqdm(self.dataloaders['val'], desc='Validation loop'):
                inputs = {modality: inputs[modality].to(self.device) for modality in inputs}
                labels = labels.to(self.device)

                # Debug input shapes
                for modality, tensor in inputs.items():
                    print(f"Validation {modality} input shape: {tensor.shape}")

                outputs = self.model(inputs)
                print(f"Validation output shape: {outputs.shape}")  # Debug shape
                loss = self.criterion(outputs, labels)
                running_loss += loss.item() * labels.size(0)

        val_loss = running_loss / len(self.dataloaders['val'].dataset)
        print(f"Validation Loss: {val_loss:.4f}")
        return val_loss

    def _log_best_model(self):
        if self.best_model_state:
            checkpoint_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save({
                'epoch': self.best_epoch,
                'model_state_dict': self.best_model_state,
                'best_loss': self.best_loss
            }, checkpoint_path)
            print(f"Best model saved at {checkpoint_path} from epoch {self.best_epoch} with validation loss {self.best_loss:.4f}")
        else:
            print("No best model found during training.")
