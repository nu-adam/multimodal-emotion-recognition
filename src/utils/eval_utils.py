from tqdm import tqdm
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def compute_metrics(all_labels, all_preds):
    """
    Computes accuracy, precision, recall, and F1-score from predictions and ground truth labels.

    Args:
    - all_labels (list or np.array): True labels.
    - all_preds (list or np.array): Model predictions.

    Returns:
    - dict: A dictionary containing accuracy, precision, recall, and F1-score.
    """
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }

    return metrics


def evaluate_model(model, data_loader, device, logger):
    """
    Evaluates the performance of the Face Emotion Recognition model on the test dataset.

    Args:
    - model (torch.nn.Module): The trained Face Emotion Recognition model.
    - data_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
    - device (torch.device): Device to perform computations on ('cuda' or 'cpu').

    Returns:
    - dict: A dictionary containing accuracy, precision, recall, and F1-score.
    """
    logger.info('Evaluating the model...')
    
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(data_loader, unit="batch", desc="Evaluation", ncols=100):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    metrics = compute_metrics(all_labels, all_preds)

    logger.info(f'Evaluation Metrics:\n'
                f'Accuracy: {metrics["accuracy"]:.4f}\n'
                f'Precision: {metrics["precision"]:.4f}\n'
                f'Recall: {metrics["recall"]:.4f}\n'
                f'F1-score: {metrics["f1_score"]:.4f}')

    logger.info('Evaluation finished successfully.')
