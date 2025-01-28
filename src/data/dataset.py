import os
import torch
from torch.utils.data import Dataset


class MultimodalDataset(Dataset):
    def __init__(self, data_dir, modalities=["video", "audio", "text"], split="train"):
        """
        Initializes the multimodal dataset.

        Args:
            data_dir (str): Base directory containing preprocessed data (e.g., "data/processed").
            modalities (list[str]): List of modalities to include (e.g., ["video", "audio", "text"]).
            split (str): Dataset split to load ("train", "val", or "test").
        """
        self.modalities = modalities
        self.data = {}  # Store tensors and labels for each modality
        self.video_ids = set()  # Keep track of video IDs for alignment

        # Load data for each modality
        for modality in modalities:
            modality_dir = os.path.join(data_dir, split, modality)
            self.data[modality] = self._load_modality_data(modality_dir)

        # Align modalities based on video IDs
        self._align_modalities()

    def _load_modality_data(self, modality_dir):
        """
        Loads tensors, labels, and video IDs for a specific modality.

        Args:
            modality_dir (str): Path to the directory containing preprocessed data for the modality.

        Returns:
            dict: Dictionary containing "tensors", "labels", and "video_ids".
        """
        tensors, labels, video_ids = [], [], []
        for file in sorted(os.listdir(modality_dir)):  # Ensure deterministic order
            if file.endswith(".pth"):
                chunk = torch.load(os.path.join(modality_dir, file))
                tensors.extend(chunk["tensors"])
                labels.extend(chunk["labels"])
                video_ids.extend(chunk["video_ids"])
        return {"tensors": tensors, "labels": labels, "video_ids": video_ids}

    def _align_modalities(self):
        """
        Aligns tensors across modalities based on shared video IDs.
        """
        # Intersect video IDs across all included modalities
        if self.modalities:
            self.video_ids = set(self.data[self.modalities[0]]["video_ids"])
            for modality in self.modalities[1:]:
                self.video_ids &= set(self.data[modality]["video_ids"])

        # Filter tensors and labels for each modality based on aligned video IDs
        aligned_data = {}
        for modality in self.modalities:
            modality_data = self.data[modality]
            indices = [i for i, vid in enumerate(modality_data["video_ids"]) if vid in self.video_ids]
            aligned_data[modality] = {
                "tensors": [modality_data["tensors"][i] for i in indices],
                "labels": [modality_data["labels"][i] for i in indices],
                "video_ids": [modality_data["video_ids"][i] for i in indices],
            }
        self.data = aligned_data
        self.video_ids = sorted(self.video_ids)  # Sort for consistent indexing

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.video_ids)

    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: Dictionary containing tensors for selected modalities and the label.
        """
        sample = {}
        for modality in self.modalities:
            sample[modality] = self.data[modality]["tensors"][idx]
        sample["label"] = self.data[self.modalities[0]]["labels"][idx]  # Labels are assumed consistent
        sample["video_id"] = self.video_ids[idx]  # Include video ID for reference
        return sample


def main():
    data_dir = "data/processed"

    # Initialize dataset with all modalities for training
    train_dataset = MultimodalDataset(data_dir, modalities=["video", "audio", "text"], split="train")

    # Initialize dataset with only video and audio for validation
    val_dataset = MultimodalDataset(data_dir, modalities=["video", "audio"], split="val")

    sample = train_dataset[0]
    print("Sample keys:", sample.keys())
    print("Video tensor shape:", sample["video"].shape)
    print("Audio tensor shape:", sample["audio"].shape)
    print("Text tensor shape:", sample["text"].shape)
    print("Label:", sample["label"])
    print("Video ID:", sample["video_id"])

    from torch.utils.data import DataLoader

    # Training DataLoader
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # Validation DataLoader
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    for batch in train_loader:
        video = batch.get("video")  # Video tensors
        audio = batch.get("audio")  # Audio tensors
        text = batch.get("text")    # Text tensors
        labels = batch["label"]     # Labels
        video_ids = batch["video_id"]  # Video IDs
        print(video.shape, audio.shape, text.shape, labels, video_ids)


if __name__ == "__main__":
    main()
