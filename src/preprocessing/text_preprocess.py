from transformers import RobertaTokenizer
from src.preprocessing.audio_preprocess import extract_audio
import whisper
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# model = whisper.load_model("light")
model = whisper.load_model("tiny")

def transcribe_audio(audio_path): 
    """
    Transcribes audio into subtitles in SRT format using OpenAI Whisper.

    Args:
        audio_path (str): Path to the audio file.

    Returns:
        str: Path to the transcript file.
    """
    result = model.transcribe(audio_path)
    return result['text']


def tokenize_text(input_text, max_length=50):
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
        return_tensors="pt"    # Return PyTorch tensors
    )
    tokenized_text = {
        "input_ids": encoded_inputs["input_ids"],
        "attention_mask": encoded_inputs["attention_mask"]
    }
    return tokenized_text


import os

def preprocess_text(input_video):
    """
    Preprocess text for input to a transformer model.

    Args:
        input_video (str): Path to the video file.

    Returns:
        dict: A dictionary containing 'input_ids' and 'attention_mask' as PyTorch tensors.
    """
    # Generate the audio path by replacing the video extension with `.wav`
    audio_path = os.path.splitext(input_video)[0] + ".wav"
    
    # Extract audio from the video
    audio = extract_audio(input_video, audio_path)
    
    # Transcribe the audio to text
    input_text = transcribe_audio(audio)
    
    # Tokenize the transcribed text
    text_tensors = tokenize_text(input_text)
    
    return text_tensors
