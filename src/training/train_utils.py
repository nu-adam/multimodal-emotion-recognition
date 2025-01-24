from tqdm import tqdm
import torch


def filter_enabled_modalities(inputs, enabled_modalities):
    """
    Filters the inputs dictionary to include only the enabled modalities.

    Args:
    - inputs (dict): Dictionary containing data for all modalities.
    - enabled_modalities (list): List of enabled modalities (e.g., ['video', 'audio', 'text']).

    Returns:
    - dict: Filtered inputs containing only the enabled modalities.
    """
    return {modality: inputs[modality] for modality in enabled_modalities if modality in inputs}


def save_model(state, checkpoint_dir):
    """
    Saves the model state to a checkpoint file.

    Args:
    - state (dict): Dictionary containing model state, optimizer state, and other metadata.
    - checkpoint_dir (str): Directory where the checkpoints are saved.
    """
    filename = f'{checkpoint_dir}/best_model.pth'
    torch.save(state, filename)


def train_one_epoch(model, enabled_modalities, train_loader, criterion, optimizer, max_grad, scaler, epoch, num_epochs, device):
    """
    Trains the model on the train dataset for one epoch.

    Args:
    - model (torch.nn.Module): The model to be trained.
    - enabled_modalities (list): List of enabled modalities (e.g., ['video', 'audio', 'text']).
    - train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
    - criterion (torch.nn.Module): Loss function used for training.
    - optimizer (torch.optim.Optimizer): Optimizer to update model weights.
    - epoch (int): Current epoch number.
    - num_epochs (int): Total number of epochs to train.
    - device (torch.device): Device to use for training ('cuda' or 'cpu').

    Returns:
    - epoch_loss (float): Average training loss over the epoch.
    """
    model.train()
    running_loss = 0.0

    for batch in tqdm(train_loader, unit="batch", desc=f"Epoch {epoch+1}/{num_epochs} - Training", ncols=100):
        inputs, labels = batch
        inputs = filter_enabled_modalities(inputs, enabled_modalities)
        inputs = {
            modality: (
                data.to(device) if isinstance(data, torch.Tensor) else 
                {k: v.to(device) for k, v in data.items()}
            )
            for modality, data in inputs.items()
        }
        labels = labels.to(device)
        
        optimizer.zero_grad()

        # Forward pass
        with torch.autocast(device_type=str(device), dtype=torch.float16):
            outputs = model(
                video=inputs['video'] if 'video' in inputs else None,
                audio=inputs['audio'] if 'audio' in inputs else None,
                text_input_ids=inputs['text']['input_ids'] if 'text' in inputs else None,
                text_attention_mask=inputs['text']['attention_mask'] if 'text' in inputs else None
            )
            loss = criterion(outputs, labels)
        
        # Backward pass
        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad)
        
        # Update the weights
        scaler.step(optimizer)
        scaler.update()
        
        batch_size = next(iter(inputs.values())).size(0) if inputs else 0
        running_loss += loss.item() * batch_size
        del inputs, labels, outputs, loss  # Free up memory
        torch.cuda.empty_cache()  # Clear unused GPU memory

    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss


def validate_one_epoch(model, enabled_modalities, val_loader, criterion, epoch, num_epochs, device):
    """
    Validates the model on the validation dataset for one epoch.

    Args:
    - model (torch.nn.Module): The model to be validated.
    - enabled_modalities (list): List of enabled modalities (e.g., ['video', 'audio', 'text']).
    - val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
    - criterion (torch.nn.Module): Loss function used for validation.
    - epoch (int): Current epoch number.
    - num_epochs (int): Total number of epochs to validate.
    - device (torch.device): Device to use for validation ('cuda' or 'cpu').

    Returns:
    - tuple: (float) Average validation loss over the epoch, (float) Validation accuracy.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, unit="batch", desc=f"Epoch {epoch+1}/{num_epochs} - Validation", ncols=100):
            inputs, labels = batch
            inputs = filter_enabled_modalities(inputs, enabled_modalities)
            inputs = {
                modality: (
                    data.to(device) if isinstance(data, torch.Tensor) else 
                    {k: v.to(device) for k, v in data.items()}
                )
                for modality, data in inputs.items()
            }
            labels = labels.to(device)

            outputs = model(
                video=inputs['video'] if 'video' in inputs else None,
                audio=inputs['audio'] if 'audio' in inputs else None,
                text_input_ids=inputs['text']['input_ids'] if 'text' in inputs else None,
                text_attention_mask=inputs['text']['attention_mask'] if 'text' in inputs else None
            )

            loss = criterion(outputs, labels)
            batch_size = next(iter(inputs.values())).size(0) if inputs else 0
            running_loss += loss.item() * batch_size
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            del inputs, labels, outputs, loss  # Free up memory
            torch.cuda.empty_cache()  # Clear unused GPU memory 

    epoch_loss = running_loss / len(val_loader.dataset)
    accuracy = correct / total

    return epoch_loss, accuracy


def train_model(
        model, enabled_modalities, train_loader, val_loader,
        criterion, optimizer, scheduler, max_grad, scaler,
        num_epochs, device, checkpoint_dir, logger
        ):
    """
    Trains and validates the Face Emotion Recognition model using specified model.
    
    Args:
    - model (nn.Module): Face Emotion Recognition model for training.
    - enabled_modalities (list): List of enabled modalities (e.g., ['video', 'audio', 'text']).
    - train_loader (DataLoader): DataLoader for the training dataset.
    - val_loader (DataLoader): DataLoader for the validation dataset.
    - criterion (nn.Module): Loss function for training.
    - optimizer (torch.optim.Optimizer): Optimizer for training.
    - num_epochs (int): Number of epochs to train the model.
    - device (str): Device to use for training ('cuda' or 'cpu').
    - checkpoint_dir (str): Directory to save the model checkpoints.
    """
    logger.info(f'Training the model for {num_epochs} epochs with modalities: {enabled_modalities}...')
    
    best_loss = float('inf')

    for epoch in range(num_epochs):
        # Train for one epoch
        train_loss = train_one_epoch(model, enabled_modalities, train_loader, criterion, optimizer, max_grad, scaler, epoch, num_epochs, device)

        logger.info(f'Epoch {epoch+1}/{num_epochs} - Training Loss: {train_loss:.4f}')

        # Validate after one epoch
        val_loss, val_accuracy = validate_one_epoch(model, enabled_modalities, val_loader, criterion, epoch, num_epochs, device)

        logger.info(f'Epoch {epoch+1}/{num_epochs} - Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy*100:.2f}%')

        # Step the scheduler
        scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f'Epoch {epoch+1}/{num_epochs} - Learning Rate: {current_lr:.6f}')

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

            logger.info(f'Model checkpoint saved at epoch {epoch+1} to {checkpoint_dir}')

        logger.info(f'Epoch {epoch+1}/{num_epochs} - Best Validation Loss: {best_loss:.4f}')
    
    logger.info('Training finished successfully.')
