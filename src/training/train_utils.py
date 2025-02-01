from tqdm import tqdm
import torch


def filter_enabled_modalities(inputs, modalities):
    """
    Filters the inputs dictionary to include only the enabled modalities.

    Args:
    - inputs (dict): Dictionary containing data for all modalities.
    - enabled_modalities (list): List of enabled modalities (e.g., ['video', 'audio', 'text']).

    Returns:
    - dict: Filtered inputs containing only the enabled modalities.
    """
    return {modality: inputs[modality] for modality in modalities if modality in inputs}


def save_model(state, checkpoint_dir):
    """
    Saves the model state to a checkpoint file.

    Args:
    - state (dict): Dictionary containing model state, optimizer state, and other metadata.
    - checkpoint_dir (str): Directory where the checkpoints are saved.
    """
    filename = f'{checkpoint_dir}/best_model.pth'
    torch.save(state, filename)


def train_one_epoch(model, enabled_modalities, train_loader, criterion, optimizer, accumulation_steps, max_grad, scaler, epoch, num_epochs, device):
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
    optimizer.zero_grad()

    for idx, batch in enumerate(tqdm(train_loader, unit="batch", desc=f"Epoch {epoch+1}/{num_epochs} - Training", ncols=100)):
        # Extract labels
        labels = batch["label"].to(device)

        # Extract inputs, filtering by enabled modalities
        inputs = {
            modality: (
                batch[modality].to(device) if modality in ["video", "audio"] else
                {k: v.to(device) for k, v in batch[modality].items()}
            )
            for modality in enabled_modalities
        }

        # Forward pass
        with torch.amp.autocast(str(device), dtype=torch.float16):
            outputs = model(
                video=inputs.get("video"),
                audio=inputs.get("audio"),
                text_input_ids=inputs["text"]["input_ids"] if "text" in inputs else None,
                text_attention_mask=inputs["text"]["attention_mask"] if "text" in inputs else None
            )
            loss = criterion(outputs, labels) / accumulation_steps

        # Backward pass
        scaler.scale(loss).backward()
        
        if (idx + 1) % accumulation_steps == 0 or idx == len(train_loader) - 1:
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad)
            
            # Step optimizer
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        running_loss += loss.item() * labels.size(0) * accumulation_steps
        
        del inputs, labels, outputs, loss
        torch.cuda.empty_cache()

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
            labels = batch["label"].to(device)

            inputs = {
                modality: (
                    batch[modality].to(device) if modality in ["video", "audio"] else
                    {k: v.to(device) for k, v in batch[modality].items()}
                )
                for modality in enabled_modalities
            }
            
            # Forward pass
            with torch.amp.autocast(str(device), dtype=torch.float16):
                outputs = model(
                    video=inputs.get("video"),
                    audio=inputs.get("audio"),
                    text_input_ids=inputs["text"]["input_ids"] if "text" in inputs else None,
                    text_attention_mask=inputs["text"]["attention_mask"] if "text" in inputs else None
                )

                loss = criterion(outputs, labels)
                
            running_loss += loss.item() * labels.size(0)

             # Compute accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            del inputs, labels, outputs, loss
            torch.cuda.empty_cache()

    epoch_loss = running_loss / len(val_loader.dataset)
    accuracy = correct / total
    return epoch_loss, accuracy


def train_model(
        model, enabled_modalities, train_loader, val_loader,
        criterion, optimizer, accumulation_steps, scheduler, 
        max_grad, scaler, num_epochs, device, checkpoint_dir, logger
        ):
    """
    Trains and validates the Face Emotion Recognition model using specified model.
    
    Args:
    - rank (int): Process rank in distributed training.
    - model (torch.nn.parallel.DistributedDataParallel): The DDP-wrapped model.
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
        train_loss = train_one_epoch(model, enabled_modalities, train_loader, criterion, 
                                     optimizer, accumulation_steps, max_grad, 
                                     scaler, epoch, num_epochs, device)
        logger.info(f'Epoch {epoch+1}/{num_epochs} - Training Loss: {train_loss:.4f}')

        # Validate after one epoch
        val_loss, val_accuracy = validate_one_epoch(model, enabled_modalities, val_loader, 
                                                    criterion, epoch, num_epochs, device)
        logger.info(f'Epoch {epoch+1}/{num_epochs} - Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy*100:.2f}%')

        # Step the scheduler
        scheduler.step()

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
