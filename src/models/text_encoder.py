import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


class BERTEncoder(nn.Module):
    def __init__(self, pretrained_model="FacebookAI/roberta-base", output_dim=768):
        """
        Text encoder using BioClinicalBERT.

        Args:
            pretrained_model (str): Hugging Face model identifier for BioClinicalBERT.
            output_dim (int): Dimensionality of the output embedding.
        """
        super(BERTEncoder, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)
        self.output_dim = output_dim
        self.fc = nn.Linear(self.bert_model.config.hidden_size, output_dim)
        self.activation = nn.ReLU()

    def forward(self, text_inputs):
        """
        Forward pass for the BioClinicalBERT encoder.

        Args:
            text_inputs (list[str]): Batch of questions (as strings).

        Returns:
            torch.Tensor: Batch of text embeddings (shape: [batch_size, output_dim]).
        """
        # Tokenize and prepare inputs for the model
        encoded_inputs = self.tokenizer(
            text_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128  # Adjust max_length based on your data
        )
        
        # Move inputs to the same device as the model
        for key, value in encoded_inputs.items():
            encoded_inputs[key] = value.to(next(self.bert_model.parameters()).device)

        # Forward pass through BioClinicalBERT
        outputs = self.bert_model(**encoded_inputs)

        # Use the [CLS] token embedding or pooler output
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token embedding

        # Apply a fully connected layer and activation for dimensionality reduction
        cls_embedding = self.fc(cls_embedding)
        cls_embedding = self.activation(cls_embedding)

        return cls_embedding
