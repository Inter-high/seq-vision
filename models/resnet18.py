"""
This module provides a CNN-based classifier that uses a pretrained ResNet18 as a feature extractor.
It defines a function to retrieve a pretrained ResNet18 model and a CNNClassifier class for classification tasks.

Author: yumemonzo@gmail.com
Date: 2025-03-03
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def get_resnet18() -> nn.Module:
    """
    Retrieve a pretrained ResNet18 model with default weights.
    
    Returns:
        nn.Module: A pretrained ResNet18 model.
    """
    return resnet18(weights=ResNet18_Weights.DEFAULT)


class CNNClassifier(nn.Module):
    """
    CNN-based classifier using ResNet18 as the feature extractor.
    
    Attributes:
        feature_extractor (nn.Sequential): Sequential module containing ResNet18 layers excluding the final classification layer.
        output_layer (nn.Linear): Linear layer that maps extracted features to the number of classes.
    """
    def __init__(self, num_classes: int) -> None:
        """
        Initialize the CNNClassifier with the specified number of classes.
        
        Args:
            num_classes (int): Number of classes for classification.
        """
        super(CNNClassifier, self).__init__()
        # Load pretrained ResNet18 and remove its final fully connected layer.
        resnet18_model = get_resnet18()
        self.feature_extractor = nn.Sequential(*list(resnet18_model.children())[:-1])
        # Define a new output layer to map features to the desired number of classes.
        self.output_layer = nn.Linear(512, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the classifier.
        
        The input tensor is expected to have shape (batch_size, sequence_length, channels, height, width).
        Each frame in the sequence is processed independently by the feature extractor, and the features
        are averaged across the sequence before being passed to the output layer.
        
        Args:
            x (torch.Tensor): Input tensor with shape (B, T, C, H, W).
        
        Returns:
            torch.Tensor: Output logits with shape (B, num_classes).
        """
        batch_size, seq_len, C, H, W = x.shape
        features = []

        # Process each frame in the sequence individually.
        for t in range(seq_len):
            # Extract features for the t-th frame.
            x_t = self.feature_extractor(x[:, t, :, :, :])
            # Flatten the extracted features.
            x_t = x_t.view(batch_size, -1)
            features.append(x_t)

        # Stack the features from all frames and compute their average.
        features = torch.stack(features, dim=1)
        avg_features = torch.mean(features, dim=1)
        # Pass the averaged features through the output layer.
        output = self.output_layer(avg_features)

        return output
