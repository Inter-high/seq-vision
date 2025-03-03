"""
This module implements LSTM-based classifiers for video classification.
It includes:
  - VanilaLSTM: A custom LSTM cell implemented with separate gate linear layers.
  - LSTMClassifier: A simple wrapper using VanilaLSTM for sequence classification.
  - CNNLSTMClassifier: A classifier that extracts features from video frames using a CNN (ResNet18)
    and processes the sequence of features with a custom LSTM.

Author: yumemonzo@gmail.com
Date: 2025-03-03
"""

import torch
import torch.nn as nn
from models import get_resnet18


class VanilaLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        """
        Initialize a custom LSTM cell with separate gate linear layers.

        Args:
            input_size (int): Dimension of the input feature vector.
            hidden_size (int): Dimension of the hidden state.
            output_size (int): Dimension of the output vector.
        """
        super(VanilaLSTM, self).__init__()
        self.hidden_size: int = hidden_size

        self.W_xf = nn.Linear(input_size, hidden_size, bias=False)
        self.W_hf = nn.Linear(hidden_size, hidden_size, bias=True)

        self.W_xi = nn.Linear(input_size, hidden_size, bias=False)
        self.W_hi = nn.Linear(hidden_size, hidden_size, bias=True)

        self.W_xc = nn.Linear(input_size, hidden_size, bias=False)
        self.W_hc = nn.Linear(hidden_size, hidden_size, bias=True)

        self.W_xo = nn.Linear(input_size, hidden_size, bias=False)
        self.W_ho = nn.Linear(hidden_size, hidden_size, bias=True)

        self.output_layer = nn.Linear(hidden_size, output_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process an input sequence through the LSTM cell.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        batch_size, seq_len, _ = x.shape

        h_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
        c_t = torch.zeros(batch_size, self.hidden_size, device=x.device)

        for t in range(seq_len):
            f_t = torch.sigmoid(self.W_xf(x[:, t, :]) + self.W_hf(h_t))
            i_t = torch.sigmoid(self.W_xi(x[:, t, :]) + self.W_hi(h_t))
            o_t = torch.sigmoid(self.W_xo(x[:, t, :]) + self.W_ho(h_t))
            c_t_hat = torch.tanh(self.W_xc(x[:, t, :]) + self.W_hc(h_t))

            c_t = f_t * c_t + i_t * c_t_hat
            h_t = o_t * torch.tanh(c_t)

        y_t = self.output_layer(h_t)
        return y_t
    

class LSTMClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int) -> None:
        """
        Initialize the LSTM classifier using VanilaLSTM.

        Args:
            input_size (int): Dimension of the input feature vector.
            hidden_size (int): Dimension of the hidden state.
            num_classes (int): Number of output classes.
        """
        super(LSTMClassifier, self).__init__()
        self.lstm = VanilaLSTM(input_size, hidden_size, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the LSTM classifier.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes).
        """
        return self.lstm(x)
    

class CNNLSTMClassifier(nn.Module):
    def __init__(self, hidden_size: int, num_classes: int) -> None:
        """
        Initialize the CNN-LSTM classifier.

        This classifier uses a CNN (ResNet18) to extract features from video frames,
        and then processes the sequence of features with a custom LSTM.

        Args:
            hidden_size (int): Dimension of the LSTM hidden state.
            num_classes (int): Number of output classes.
        """
        super(CNNLSTMClassifier, self).__init__()
        resnet18_model = get_resnet18()
        self.feature_extractor = nn.Sequential(*list(resnet18_model.children())[:-1])
        self.lstm = VanilaLSTM(512, hidden_size, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CNN-LSTM classifier.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, C, H, W).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes).
        """
        batch_size, seq_len, C, H, W = x.shape

        features = []
        for t in range(seq_len):
            # Extract features from each frame using the CNN feature extractor.
            x_t = self.feature_extractor(x[:, t])
            x_t = x_t.view(batch_size, -1)
            features.append(x_t)

        features = torch.stack(features, dim=1)
        return self.lstm(features)
