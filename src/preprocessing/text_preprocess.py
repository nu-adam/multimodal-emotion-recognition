import torch
from transformers import BertTokenizer
import speech_recognition as sr


# Constants
MAX_TEXT_LENGTH = 50

# Initialize Tokenizer
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def extract_text_from_audio(audio_path):
    """Transcribe audio to text using SpeechRecognition."""
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio_data)
    except sr.UnknownValueError:
        text = ""
    return text


def preprocess_text(text):
    """Preprocess text for embedding."""
    tokens = bert_tokenizer(text, padding="max_length", truncation=True, max_length=MAX_TEXT_LENGTH, return_tensors="pt")
    return tokens['input_ids'], tokens['attention_mask']
