from transformers import AutoTokenizer
import whisper


def transcribe_audio_to_srt(audio_path, transcript_save_path, model_name='base'): 
    """
    Transcribes audio into subtitles in SRT format using OpenAI Whisper.

    Args:
        audio_path (str): Path to the audio file.
        transcript_save_path (str): Path to save the transcribed subtitles.
        model_name (str): Whisper model name ('tiny', 'base', 'small', 'medium', 'large').

    Returns:
        str: Path to the SRT file.
    """
    model = whisper.load_model(model_name)
    result = model.transcribe(audio_path, task="transcribe")
    with open(transcript_save_path, "w", encoding="utf-8") as srt_file:
        for segment in result["segments"]:
            start = segment["start"]
            end = segment["end"]
            text = segment["text"]
            srt_file.write(f"{segment['id'] + 1}\n")
            srt_file.write(f"{start:.3f} --> {end:.3f}\n")
            srt_file.write(f"{text}\n\n")
    return transcript_save_path


def preprocess_text(text, tokenizer_name='roberta-base', max_length=50):
    """
    Tokenizes text using a specified tokenizer.

    Args:
        text (str): Text to tokenize.
        tokenizer_name (str): Pretrained tokenizer name.
        max_length (int): Maximum sequence length.

    Returns:
        dict: Tokenized inputs with 'input_ids' and 'attention_mask'.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokens = tokenizer(
        text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    return tokens
