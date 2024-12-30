from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder


def get_transforms():
    """
    Returns data transformations for training and testing datasets.
    
    The training transformations include resizing, random horizontal flipping, color jitter, 
    and normalization. The testing transformations include resizing and normalization.

    Returns:
    - transforms (dict): A dictionary containing 'train' and 'test' transforms.
    """
    return {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
    }


def load_data(data_dir, batch_size=32):
    """
    Loads the FER dataset, applies data transformations, and returns DataLoader objects for training and testing.
    
    Args:
    - data_dir (str): Path to the root directory containing 'train' and 'test' subdirectories.
    - batch_size (int): Number of samples per batch to load.

    Returns:
    - train_loader (DataLoader): DataLoader for the training dataset.
    - val_loader (DataLoader): DataLoader for the validation dataset.
    - test_loader (DataLoader): DataLoader for the testing dataset.
    """
    transforms = get_transforms()
    
    full_train_dataset = ImageFolder(root=f'{data_dir}/train', transform=transforms['train'])
    test_dataset = ImageFolder(root=f'{data_dir}/test', transform=transforms['test'])

    train_dataset, val_dataset = random_split(full_train_dataset, [0.8, 0.2])
    val_dataset.dataset.transform = transforms['test']
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, test_loader
