"""
This module provides a CNN-LSTM classifier for video classification.
It combines CNN-based feature extraction with an LSTM for processing temporal features.

Author: yumemonzo@gmail.com
Date: 2025-03-03
"""

import torch
import torch.nn as nn
from models import get_resnet18


class CNNLSTMClassifier(nn.Module):
    def __init__(self, hidden_size: int, num_classes: int, num_layers: int = 1, dropout: float = 0.5) -> None:
        """
        Initialize the CNN-LSTM classifier.

        This classifier extracts features from video frames using a pretrained ResNet18 and
        processes the sequence of features using an LSTM.

        Args:
            hidden_size (int): Dimension of the LSTM hidden state.
            num_classes (int): Number of output classes.
            num_layers (int): Number of stacked LSTM layers. Defaults to 1.
            dropout (float): Dropout rate applied between LSTM layers. Defaults to 0.5.
        """
        super(CNNLSTMClassifier, self).__init__()
        resnet18_model = get_resnet18()
        # Remove the final fully connected layer of ResNet18.
        self.feature_extractor = nn.Sequential(*list(resnet18_model.children())[:-1])
        # LSTM for sequence modeling. batch_first=True makes the input shape (batch, seq, feature).
        self.lstm = nn.LSTM(input_size=512, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout)
        # Final classifier layer: use the last hidden state for classification.
        self.classifier = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CNN-LSTM classifier.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, seq_len, C, H, W).

        Returns:
            torch.Tensor: Output logits with shape (batch_size, num_classes).
        """
        batch_size, seq_len, C, H, W = x.shape
        features = []
        for t in range(seq_len):
            # Extract features for each frame.
            x_t = self.feature_extractor(x[:, t])  # Expected shape: (batch_size, 512, 1, 1)
            x_t = x_t.view(batch_size, -1)          # Flatten to shape: (batch_size, 512)
            features.append(x_t)
        # Stack features along the time dimension: (batch_size, seq_len, 512)
        features = torch.stack(features, dim=1)
        # Process the sequence with LSTM.
        lstm_out, (hn, cn) = self.lstm(features)   # hn shape: (num_layers, batch_size, hidden_size)
        # Use the last layer's hidden state for classification.
        out = self.classifier(hn[-1])
        return out
