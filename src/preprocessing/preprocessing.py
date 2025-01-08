from src.preprocessing.audio_preprocess import extract_audio_from_video, preprocess_audio
from src.preprocessing.video_preprocess import extract_video_frames
from src.preprocessing.text_preprocess import transcribe_audio_to_srt, preprocess_text


def preprocess_multimodal_input(video_path, audio_save_path, transcript_save_path, frame_rate=1, sr=16000):
    """
    Preprocesses video, audio, and text inputs for multimodal emotion recognition.

    Args:
        video_path (str): Path to the video file.
        audio_save_path (str): Path to save the extracted audio file.
        transcript_save_path (str): Path to save the transcribed subtitles.
        frame_rate (int): Number of frames to extract per second.
        sr (int): Target sampling rate for audio.

    Returns:
        dict: Preprocessed data including video frames, audio spectrogram, and tokenized text.
    """
    # Process video frames
    video_frames = extract_video_frames(video_path, frame_rate=frame_rate)

    # Process audio
    audio_path = extract_audio_from_video(video_path, audio_save_path)
    mel_spectrogram = preprocess_audio(audio_path, sr=sr)

    # Transcribe audio
    transcript_path = transcribe_audio_to_srt(audio_path, transcript_save_path)

    # Read and tokenize text
    with open(transcript_path, "r", encoding="utf-8") as f:
        text = " ".join(line.strip() for line in f if not line.strip().isdigit() and "-->" not in line)

    tokenized_text = preprocess_text(text)

    return {
        'video_frames': video_frames,
        'mel_spectrogram': mel_spectrogram,
        'tokenized_text': tokenized_text
    }
