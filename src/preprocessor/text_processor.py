from transformers import RobertaTokenizer
import torch
import sys
import os

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.models.text.text import TextFeatureExtractor, ProjectionNetwork


# Define the string
input_text = "Today is the great day for training in the my favourite gym"

# Initialize the tokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# Preprocess the text
encoded_inputs = tokenizer(
    input_text,
    padding="max_length",  # Pad to the maximum sequence length
    truncation=True,       # Truncate if the text exceeds max length
    max_length=50,         # Define the maximum length of tokens
    return_tensors="pt"   # Return PyTorch tensors
)

# Extract input_ids and attention_mask
input_ids = encoded_inputs["input_ids"]
attention_mask = encoded_inputs["attention_mask"]

# Print the results
print("Input Text:", input_text)
print("Input IDs:", input_ids)
print("Attention Mask:", attention_mask)


# Initialize the text model and projection network
text_encoder = TextFeatureExtractor()
proj = ProjectionNetwork()

# Forward pass
output = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
output = proj(output)

# Print the output
print("Output shape:", output.shape)
print("Output values:", output)
