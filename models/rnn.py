"""
This module provides RNN and CNN-RNN classifiers for video classification.
It includes:
  - VanilaRNN: A basic recurrent neural network for processing sequential data.
  - RNNClassifier: A simple wrapper around VanilaRNN.
  - CNNRNNClassifier: A classifier that combines CNN-based feature extraction with a recurrent network.

Author: yumemonzo@gmail.com
Date: 2025-03-03
"""

import torch
import torch.nn as nn
from models import get_resnet18


class VanilaRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        """
        Initialize a basic RNN with a single hidden layer.

        Args:
            input_size (int): Dimension of the input feature vector.
            hidden_size (int): Dimension of the hidden state.
            output_size (int): Dimension of the output vector.
        """
        super(VanilaRNN, self).__init__()
        self.hidden_size: int = hidden_size
        self.input_layer: nn.Linear = nn.Linear(input_size, hidden_size)
        self.hidden_layer: nn.Linear = nn.Linear(hidden_size, hidden_size)
        self.output_layer: nn.Linear = nn.Linear(hidden_size, output_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process an input sequence through the RNN.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, seq_len, input_size).

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, output_size).
        """
        batch_size, seq_len, _ = x.shape
        h_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
        for t in range(seq_len):
            h_t = torch.tanh(self.input_layer(x[:, t, :]) + self.hidden_layer(h_t))
        y_t = self.output_layer(h_t)
        return y_t
    

class RNNClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int) -> None:
        """
        Initialize the RNN classifier.

        Args:
            input_size (int): Dimension of the input feature vector.
            hidden_size (int): Dimension of the hidden state.
            num_classes (int): Number of output classes.
        """
        super(RNNClassifier, self).__init__()
        self.rnn: VanilaRNN = VanilaRNN(input_size, hidden_size, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the RNN classifier.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, seq_len, input_size).

        Returns:
            torch.Tensor: Output logits with shape (batch_size, num_classes).
        """
        return self.rnn(x)
    

class CNNRNNClassifier(nn.Module):
    def __init__(self, hidden_size: int, num_classes: int) -> None:
        """
        Initialize the CNN-RNN classifier.

        This classifier extracts features from video frames using a pretrained ResNet18 and
        processes the sequence of features using a basic RNN.

        Args:
            hidden_size (int): Dimension of the RNN hidden state.
            num_classes (int): Number of output classes.
        """
        super(CNNRNNClassifier, self).__init__()
        resnet18_model = get_resnet18()
        self.feature_extractor: nn.Sequential = nn.Sequential(*list(resnet18_model.children())[:-1])
        self.rnn: VanilaRNN = VanilaRNN(512, hidden_size, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CNN-RNN classifier.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, seq_len, C, H, W).

        Returns:
            torch.Tensor: Output logits with shape (batch_size, num_classes).
        """
        batch_size, seq_len, C, H, W = x.shape
        features = []
        for t in range(seq_len):
            x_t = self.feature_extractor(x[:, t])
            x_t = x_t.view(batch_size, -1)
            features.append(x_t)
        features = torch.stack(features, dim=1)
        return self.rnn(features)
