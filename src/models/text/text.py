import torch.nn as nn
from transformers import RobertaModel

class TextFeatureExtractor(nn.Module):
    """
    Feature Extractor based on the RoBERTa architecture for the text modality.
    """
    def __init__(self):
        super(TextFeatureExtractor, self).__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base")

    def forward(self, input_ids, attention_mask):
        """
        Args:
            x (dict): Tokenized inputs containing 'input_ids' and 'attention_mask'.

        Returns:
            torch.Tensor: Feature embeddings of shape (batch_size, hidden_dim).
        """
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        # Use the [CLS] token embedding as the feature representation
        return outputs.last_hidden_state[:, 0, :]  # Output: (batch_size, hidden_dim)

class ProjectionNetwork(nn.Module):
    """
    Projection Network for dimensionality reduction.
    """
    def __init__(self, input_dim=768, output_dim=256):
        super(ProjectionNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Reduced-dimensionality tensor of shape (batch_size, output_dim).
        """
        x = self.fc(x)
        x = self.relu(x)
        return x  # Output: (batch_size, output_dim)
