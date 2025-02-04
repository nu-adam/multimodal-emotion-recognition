from transformers import RobertaTokenizer

def preprocess_text(input_text, max_length=50):
    """
    Preprocess text for input to a transformer model.

    Args:
        input_text (str): The input text string.
        max_length (int): Maximum sequence length for tokenization.

    Returns:
        dict: A dictionary containing 'input_ids' and 'attention_mask' as PyTorch tensors.
    """
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    encoded_inputs = tokenizer(
        input_text,
        padding="max_length",  # Pad to the maximum sequence length
        truncation=True,       # Truncate if the text exceeds max length
        max_length=max_length, # Define the maximum length of tokens
        return_tensors="pt"   # Return PyTorch tensors
    )
    return {
        "input_ids": encoded_inputs["input_ids"],
        "attention_mask": encoded_inputs["attention_mask"]
    }