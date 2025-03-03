"""
This module contains a dataset class for the UCF101 video dataset along with helper functions 
for data transformation, splitting, collating, and loading.

Author: yumemonzo@gmail.com
Date: 2025-03-03
"""

import os
import random
from typing import List, Tuple, Dict, Optional, Callable

import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, Subset, DataLoader


class UCF101Dataset(Dataset):
    """
    Dataset class for handling the UCF101 video dataset.
    
    This class loads .avi video files from a given directory, processes the video frames,
    applies optional transformations, and provides video samples along with their class labels.
    """
    def __init__(self, data_dir: str, transform: Optional[Callable] = None, max_samples_per_class: Optional[int] = None) -> None:
        """
        Initialize the dataset with the directory path and optional parameters.
        
        Parameters:
            data_dir (str): Path to the directory containing video files.
            transform (Optional[Callable]): Transformation to apply to each video frame.
            max_samples_per_class (Optional[int]): Maximum number of samples to include per class.
        """
        self.data_dir: str = data_dir
        self.transform: Optional[Callable] = transform
        self.max_samples_per_class: Optional[int] = max_samples_per_class
        self.samples, self.classes, self.class_to_idx = self._load_samples()

    def _load_samples(self) -> Tuple[List[Tuple[str, int]], List[str], Dict[str, int]]:
        """
        Load and process video file paths and their corresponding class labels.
        
        Returns:
            Tuple[List[Tuple[str, int]], List[str], Dict[str, int]]:
                - A list of tuples where each tuple contains the video file path and the corresponding class index.
                - A sorted list of class names.
                - A dictionary mapping each class name to its index.
        """
        # Retrieve all .avi files from the specified directory
        file_names: List[str] = [f for f in os.listdir(self.data_dir) if f.lower().endswith('.avi')]
        
        # Group files by class name extracted from the filename
        samples_dict: Dict[str, List[str]] = {}
        for file_name in file_names:
            # Assumes filename format "xxx_classname_yyy.avi"
            tokens: List[str] = file_name.split('_')
            class_name: str = tokens[1] if len(tokens) > 1 else "Unknown"
            samples_dict.setdefault(class_name, []).append(os.path.join(self.data_dir, file_name))
        
        # Sort class names and limit number of samples per class if specified
        classes: List[str] = sorted(samples_dict.keys())
        filtered_samples: List[Tuple[str, str]] = []
        for cls in classes:
            file_list: List[str] = samples_dict[cls]
            if self.max_samples_per_class is not None:
                file_list = file_list[:self.max_samples_per_class]
            for file_path in file_list:
                filtered_samples.append((file_path, cls))
        
        # Create a mapping from class names to indices and form the sample list
        class_to_idx: Dict[str, int] = {cls: idx for idx, cls in enumerate(classes)}
        samples: List[Tuple[str, int]] = [(file_path, class_to_idx[label]) for file_path, label in filtered_samples]
        
        return samples, classes, class_to_idx

    def __len__(self) -> int:
        """
        Return the total number of video samples in the dataset.
        
        Returns:
            int: Number of samples.
        """
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Retrieve a video sample and its class label by index.
        
        Parameters:
            idx (int): Index of the desired sample.
        
        Returns:
            Tuple[torch.Tensor, int]:
                - A tensor representing the video (stack of frames).
                - The integer class label corresponding to the video.
        """
        file_path, label = self.samples[idx]
        cap = cv2.VideoCapture(file_path)

        frames: List[torch.Tensor] = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame from BGR to RGB and convert to a PIL image
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            if self.transform:
                frame = self.transform(frame)

            frames.append(frame)

        cap.release()
        video_tensor: torch.Tensor = torch.stack(frames)

        return video_tensor, label
    

def get_transform(resize: int) -> transforms.Compose:
    """
    Create a transformation pipeline for processing video frames.
    
    Parameters:
        resize (int): The size to which each frame should be resized (height and width).
    
    Returns:
        transforms.Compose: The composed transformation pipeline.
    """
    transform = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return transform


def get_dataset(data_dir: str, transform: transforms.Compose, max_samples_per_class: Optional[int] = None) -> UCF101Dataset:
    """
    Initialize the UCF101 dataset.
    
    Parameters:
        data_dir (str): Path to the directory containing video files.
        transform (transforms.Compose): Transformation pipeline to apply to video frames.
        max_samples_per_class (Optional[int]): Maximum number of samples to include per class.
    
    Returns:
        UCF101Dataset: An instance of the UCF101 dataset.
    """
    return UCF101Dataset(data_dir, transform, max_samples_per_class)


def split_dataset(dataset: UCF101Dataset) -> Tuple[Subset, Subset, Subset]:
    """
    Split the dataset into training, validation, and test subsets.
    
    The split is done per class to ensure balanced distribution:
      - 60% for training
      - 20% for validation
      - 20% for testing
    
    Parameters:
        dataset (UCF101Dataset): The complete dataset to split.
    
    Returns:
        Tuple[Subset, Subset, Subset]: The training, validation, and test subsets.
    """
    indices_by_class: Dict[int, List[int]] = {}
    for idx, (_, label) in enumerate(dataset.samples):
        indices_by_class.setdefault(label, []).append(idx)

    train_indices: List[int] = []
    valid_indices: List[int] = []
    test_indices: List[int] = []

    # Shuffle and split indices for each class
    for indices in indices_by_class.values():
        random.shuffle(indices)
        n: int = len(indices)
        n_train: int = int(n * 0.6)
        n_valid: int = int(n * 0.2)
        train_indices.extend(indices[:n_train])
        valid_indices.extend(indices[n_train:n_train + n_valid])
        test_indices.extend(indices[n_train + n_valid:])

    return (
        Subset(dataset, train_indices),
        Subset(dataset, valid_indices),
        Subset(dataset, test_indices)
    )


def fixed_length_collate_fn(batch: List[Tuple[torch.Tensor, int]], max_frame: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Custom collate function to ensure each video has a fixed number of frames.
    
    If a video has more frames than 'max_frame', it is truncated.
    If it has fewer, it is padded with zeros.
    
    Parameters:
        batch (List[Tuple[torch.Tensor, int]]): A list of tuples, each containing a video tensor and its label.
        max_frame (int): The fixed number of frames each video should have.
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - A tensor containing the batch of videos, each with exactly 'max_frame' frames.
            - A tensor of corresponding labels.
    """
    videos, labels = zip(*batch)

    processed_videos: List[torch.Tensor] = []
    for video in videos:
        num_frames: int = video.size(0)
        if num_frames >= max_frame:
            # Truncate the video to 'max_frame' frames
            video = video[:max_frame]
        else:
            # Pad the video with zeros to have 'max_frame' frames
            pad_frames: int = max_frame - num_frames
            pad_tensor = torch.zeros((pad_frames, *video.shape[1:]), dtype=video.dtype)
            video = torch.cat([video, pad_tensor], dim=0)
        processed_videos.append(video)
    videos_tensor: torch.Tensor = torch.stack(processed_videos)
    labels_tensor: torch.Tensor = torch.tensor(labels, dtype=torch.long)

    return videos_tensor, labels_tensor


def get_loaders(
    train_dataset: Subset, 
    valid_dataset: Subset, 
    test_dataset: Subset, 
    max_frame: int, 
    batch_size: int, 
    num_workers: int
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for training, validation, and testing datasets.
    
    Each loader uses a custom collate function to enforce a fixed number of frames per video.
    
    Parameters:
        train_dataset (Subset): Subset for training.
        valid_dataset (Subset): Subset for validation.
        test_dataset (Subset): Subset for testing.
        max_frame (int): Fixed number of frames for each video.
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of worker processes for data loading.
    
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: Data loaders for training, validation, and testing.
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda batch: fixed_length_collate_fn(batch, max_frame)
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda batch: fixed_length_collate_fn(batch, max_frame)
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda batch: fixed_length_collate_fn(batch, max_frame)
    )

    return train_loader, valid_loader, test_loader
