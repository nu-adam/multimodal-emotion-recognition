from moviepy.editor import VideoFileClip
import torchaudio
from torchaudio.transforms import MelSpectrogram
from torchvision import transforms
from transformers import BertTokenizer
import torch
from PIL import Image
import os
import librosa
import speech_recognition as sr

# Constants
FRAME_SIZE = (64, 64)
SAMPLE_RATE = 16000
MAX_TEXT_LENGTH = 50

# Initialize Tokenizer
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Video Preprocessing
def preprocess_video(video_path):
    video = VideoFileClip(video_path)
    frames = [Image.fromarray(frame) for frame in video.iter_frames(fps=1)]  # 1 frame per second
    transform = transforms.Compose([
        transforms.Resize(FRAME_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return torch.stack([transform(frame) for frame in frames])

# Audio Preprocessing
def extract_audio_from_video(video_path):
    audio_path = video_path.replace('.mp4', '.wav')
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path)
    return audio_path

def preprocess_audio(audio_path):
    waveform, sample_rate = librosa.load(audio_path, sr=SAMPLE_RATE)
    mel_spec_transform = MelSpectrogram(sample_rate=SAMPLE_RATE)
    waveform_tensor = torch.tensor(waveform).unsqueeze(0)  # Add channel dimension
    mel_spectrogram = mel_spec_transform(waveform_tensor)
    return torch.log1p(mel_spectrogram)  # Log scaling for numerical stability

# Text Preprocessing
def extract_text_from_audio(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio_data)
    except sr.UnknownValueError:
        text = ""
    return text

def preprocess_text(text):
    tokens = bert_tokenizer(text, padding="max_length", truncation=True, max_length=MAX_TEXT_LENGTH, return_tensors="pt")
    return tokens['input_ids'], tokens['attention_mask']
