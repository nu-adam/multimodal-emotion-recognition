import torch.nn as nn
from transformers import RobertaModel


class TextEncoder(nn.Module):
    """
    Extracts text features using a pre-trained RoBERTa model.

    Methods:
        forward(input_ids, attention_mask): Extracts and projects text features.
    """
    def __init__(self, model_name='roberta-base', embed_dim=256):
        super(TextEncoder, self).__init__()
        self.text_model = RobertaModel.from_pretrained(model_name)
        self.projection_network = nn.Linear(self.text_model.config.hidden_size, embed_dim)

    def forward(self, input_ids, attention_mask):
        """
        Forward pass through the text feature extractor.

        Args:
            input_ids (torch.Tensor): Tokenized input IDs of shape (batch_size, seq_len).
            attention_mask (torch.Tensor): Attention mask of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Encoded text features of shape (batch_size, embed_dim).
        """
        x = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        x = x.last_hidden_state[:, 0, :]  # Shape: (batch_size, hidden_size)
        x = self.projection_network(x)  # Shape: (batch_size, embed_dim)
        return x
