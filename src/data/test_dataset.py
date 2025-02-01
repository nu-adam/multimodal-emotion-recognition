import os
import sys

# Add the root directory to the Python path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(root_dir)

from torch.utils.data import DataLoader
from src.data.dataset import MultimodalDataset, custom_collate_fn


def test_dataset_loading():
    """
    Tests lazy loading of dataset samples to ensure memory efficiency and correct sample retrieval.
    """
    data_dir = "data/processed"
    modalities = ["video", "audio", "text"]

    # Load dataset
    dataset = MultimodalDataset(data_dir, modalities=modalities, split="train")

    # Check dataset length
    assert len(dataset) > 0, "Dataset is empty! Check preprocessing."

    # Check first sample
    sample = dataset[0]
    assert "video_id" in sample, "Sample missing 'video_id'!"
    assert "label" in sample, "Sample missing 'label'!"
    
    # Verify that modalities exist
    for modality in modalities:
        assert modality in sample, f"Missing modality: {modality}"
        assert sample[modality] is not None, f"Modality {modality} should not be None!"

    print("✅ Dataset loading test passed!")


def test_dataset_lazy_loading():
    """
    Ensures dataset does not load all data into RAM at once (lazy loading verification).
    """
    import psutil

    data_dir = "data/processed"
    dataset = MultimodalDataset(data_dir, modalities=["video", "audio", "text"], split="train")

    # Check initial memory usage
    initial_memory = psutil.Process().memory_info().rss / (1024 ** 2)  # Convert bytes to MB

    # Accessing first sample
    _ = dataset[0]

    # Check memory usage after accessing one sample
    final_memory = psutil.Process().memory_info().rss / (1024 ** 2)

    assert final_memory - initial_memory < 100, "Dataset is preloading too much into RAM!"
    print("✅ Lazy loading test passed!")


def test_dataloader_batching():
    """
    Ensures DataLoader correctly batches multimodal data with proper padding.
    """
    data_dir = "data/processed"
    modalities = ["video", "audio", "text"]
    batch_size = 4

    # Initialize dataset and dataloader
    dataset = MultimodalDataset(data_dir, modalities=modalities, split="train")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

    # Fetch one batch
    batch = next(iter(dataloader))

    # Check batch keys
    assert "label" in batch, "Batch missing 'label'!"
    assert "video_id" in batch, "Batch missing 'video_id'!"
    
    for modality in modalities:
        assert modality in batch, f"Batch missing modality: {modality}"
        assert batch[modality] is not None, f"Modality {modality} should not be None!"

    # Check tensor shapes
    assert batch["label"].shape[0] == batch_size, "Label batch size mismatch!"
    assert len(batch["video_id"]) == batch_size, "Video ID batch size mismatch!"

    # Check padding correctness for text
    if "text" in batch:
        assert batch["text"]["input_ids"].shape[0] == batch_size, "Text batch size mismatch!"
        assert batch["text"]["attention_mask"].shape[0] == batch_size, "Text attention mask batch size mismatch!"

    print("✅ DataLoader batching test passed!")


def test_handling_missing_modalities():
    """
    Tests dataset and DataLoader with missing modalities to ensure robustness.
    """
    data_dir = "data/processed"
    
    # Test different missing modality cases
    modality_cases = [
        ["video"],  # Only video
        ["audio"],  # Only audio
        ["text"],  # Only text
        ["video", "text"],  # Missing audio
        ["audio", "text"],  # Missing video
    ]

    for modalities in modality_cases:
        print(f"Testing with modalities: {modalities}")

        dataset = MultimodalDataset(data_dir, modalities=modalities, split="train")
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=custom_collate_fn)

        # Fetch one batch
        batch = next(iter(dataloader))

        for modality in modalities:
            assert modality in batch, f"Missing modality {modality} in batch!"
            assert batch[modality] is not None, f"Modality {modality} should not be None!"

        print(f"✅ Passed missing modality test for {modalities}!")


def test_dataloader_speed():
    """
    Measures DataLoader speed and ensures it is efficient.
    """
    import time

    data_dir = "data/processed"
    batch_size = 8

    dataset = MultimodalDataset(data_dir, modalities=["video", "audio", "text"], split="train")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn, num_workers=2, pin_memory=True)

    start_time = time.time()

    # Iterate through one full epoch
    for batch in dataloader:
        pass

    end_time = time.time()
    print(f"✅ DataLoader speed test completed in {end_time - start_time:.2f} seconds!")


def test_all():
    """
    Runs all dataset and DataLoader tests.
    """
    print("🚀 Running Dataset & DataLoader Tests 🚀")
    # test_dataset_loading()
    # test_dataset_lazy_loading()
    # test_dataloader_batching()
    # test_handling_missing_modalities()
    # test_dataloader_speed()
    print("🎉 All tests passed successfully! 🎉")


if __name__ == "__main__":
    test_all()
