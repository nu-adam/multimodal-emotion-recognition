import ray
import torch
import whisper
from transformers import RobertaTokenizer
import os
from glob import glob

from src.preprocessing.audio_preprocess import extract_audio_as_waveform


@ray.remote(num_gpus=1)
class TextProcessor:
    """
    Ray Actor for audio transcription and text tokenization.
    """

    def __init__(self, whisper_model="base", roberta_model="roberta-base"):
        """
        Initializes the TextProcessor with Whisper and RoBERTa models.

        Args:
            whisper_model (str): Whisper model to use for transcription.
            roberta_model (str): RoBERTa model to use for tokenization.
        """
        self.whisper_model = whisper.load_model(whisper_model)
        self.tokenizer = RobertaTokenizer.from_pretrained(roberta_model)

    def process_text(self, video_path):
        """
        Transcribes audio from a video and tokenizes the text.

        Args:
            video_path (str): Path to the video file.

        Returns:
            dict or None: Tokenized data containing "input_ids" and "attention_mask",
                          or None if no audio exists.
        """
        try:
            # Extract audio
            waveform, sr = extract_audio_as_waveform(video_path)
            if waveform is None:
                return None  # No valid audio

            # Transcribe audio to text
            result = self.whisper_model.transcribe(video_path)
            text = result["text"]

            # Tokenize the text
            encoded = self.tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=512,
            )
            return {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}
        except Exception as e:
            print(f"[ERROR] Failed to process text for video {video_path}: {e}")
            return None


@ray.remote
def preprocess_text_task(video_path, text_processor_actor):
    """
    Processes a video to generate tokenized text data and saves it.

    Args:
        video_path (str): Path to the video file.
        text_processor_actor (ray.actor.ActorHandle): A Ray Actor for text processing.

    Returns:
        str: Path to the saved tokenized data file, or None if processing failed.
    """
    tokenized_data = ray.get(text_processor_actor.process_text.remote(video_path))
    if tokenized_data is not None:
        # Save tokenized data to processed/text directory
        output_path = video_path.replace("raw", "processed").replace(".mp4", ".pth").replace("/video/", "/text/")
        torch.save(tokenized_data, output_path)
        return output_path
    else:
        print(f"[INFO] No tokenized data saved for video: {video_path}")
        return None
    

def get_video_paths(raw_dir):
    """
    Recursively finds all video files in the dataset directory.

    Args:
        raw_dir (str): Path to the raw dataset directory.

    Returns:
        List[str]: A list of paths to video files.
    """
    # return glob(os.path.join(raw_dir, "**", "*.mp4"), recursive=True)
    return glob(os.path.join(raw_dir, "*.mp4"), recursive=True)


def main():
    """
    Main function to preprocess all text in the dataset using Ray Actors.
    """
    ray.init()  # Initialize Ray

    # Step 1: Start the TextProcessor Actor
    text_processor_actor = TextProcessor.remote()

    # Step 2: Get all video paths
    raw_dir = "data/raw/test/disgust"
    video_paths = get_video_paths(raw_dir)

    # Step 3: Launch preprocessing tasks in parallel
    futures = [preprocess_text_task.remote(video_path, text_processor_actor) for video_path in video_paths]

    # Step 4: Wait for all tasks to complete and get results
    results = ray.get(futures)
    print("All text processed. Saved to:")
    for result in results:
        if result:
            print(result)


if __name__ == "__main__":
    main()
