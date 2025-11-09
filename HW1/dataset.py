"""
Food-101-LT Dataset Loader
Generated from Food-101-LT annotation files
"""

import os
import json
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class Food101LTDataset(Dataset):
    """
    Food-101 Long-Tail Dataset
    
    This dataset loads images from Food-101-LT train_lt.txt annotations.
    The dataset has an imbalanced (long-tail) distribution of classes.
    """
    
    def __init__(self, 
                 root_dir='data/food-101',
                 split='train_lt',
                 transform=None,
                 target_transform=None):
        """
        Args:
            root_dir (str): Root directory containing 'images' folder and 'meta' folder
            split (str): Which split to use ('train_lt', 'train', 'test')
            transform (callable, optional): Transform to be applied on images
            target_transform (callable, optional): Transform to be applied on labels
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        
        # Load class names
        classes_file = os.path.join(root_dir, 'meta', 'classes.txt')
        with open(classes_file, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        # Create class to index mapping
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        # Load image paths and labels
        self.samples = self._load_samples()
        
    def _load_samples(self):
        """Load image paths and labels from annotation file"""
        samples = []
        
        if self.split == 'train_lt':
            # Load from Food-101-LT train_lt.txt
            annotation_file = os.path.join(self.root_dir, '..', 'Food-101-LT', 'train_lt.txt')
        elif self.split == 'train':
            # Load from Food-101 train.txt
            annotation_file = os.path.join(self.root_dir, 'meta', 'train.txt')
        elif self.split == 'test':
            # Load from Food-101 test.txt
            annotation_file = os.path.join(self.root_dir, 'meta', 'test.txt')
        else:
            raise ValueError(f"Unknown split: {self.split}")
        
        with open(annotation_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Parse class_name/image_id format
                class_name, image_id = line.split('/')
                
                # Construct full image path
                image_path = os.path.join(self.root_dir, 'images', class_name, f'{image_id}.jpg')
                
                # Get class index
                class_idx = self.class_to_idx[class_name]
                
                samples.append((image_path, class_idx))
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset
        
        Returns:
            tuple: (image, label) where image is a PIL Image or tensor
        """
        image_path, label = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), color='black')
        
        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)
        
        if self.target_transform is not None:
            label = self.target_transform(label)
        
        return image, label
    
    def get_class_distribution(self):
        """Get the distribution of classes in the dataset"""
        from collections import Counter
        labels = [label for _, label in self.samples]
        return Counter(labels)
    
    def get_class_counts(self):
        """Get the count of samples per class"""
        counts = {}
        for _, label in self.samples:
            class_name = self.classes[label]
            counts[class_name] = counts.get(class_name, 0) + 1
        return counts


def get_default_transforms(image_size=224, augment=False):
    """
    Get default transforms for training and validation
    
    Args:
        image_size (int): Target image size
        augment (bool): Whether to use data augmentation
        
    Returns:
        dict: Dictionary with 'train' and 'val' transforms
    """
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return {
        'train': train_transform,
        'val': val_transform
    }


if __name__ == '__main__':
    # Example usage
    print("Creating Food-101-LT dataset...")
    
    # Create dataset
    transforms_dict = get_default_transforms(augment=True)
    dataset = Food101LTDataset(
        root_dir='data/food-101',
        split='train_lt',
        transform=transforms_dict['train']
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of classes: {len(dataset.classes)}")
    
    # Get class distribution
    class_counts = dataset.get_class_counts()
    print("\nClass distribution (first 10 classes):")
    for i, (class_name, count) in enumerate(list(class_counts.items())[:10]):
        print(f"  {class_name}: {count} samples")
    
    # Test loading a sample
    if len(dataset) > 0:
        image, label = dataset[0]
        print(f"\nSample image shape: {image.shape}")
        print(f"Sample label: {label} ({dataset.classes[label]})")

