"""
This module implements LSTM-based classifiers for video classification.
It includes:
  - VanilaLSTM: A custom multi-layer LSTM cell implemented with separate gate linear layers.
  - CNNLSTMClassifier: A classifier that extracts features from video frames using a CNN (ResNet18)
    and processes the sequence of features with a custom LSTM.

Author: yumemonzo@gmail.com
Date: 2025-03-03
"""

import torch
import torch.nn as nn
from models import get_resnet18


class VanilaLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 1) -> None:
        """
        Initialize a custom multi-layer LSTM cell with separate gate linear layers.

        Args:
            input_size (int): Dimension of the input feature vector.
            hidden_size (int): Dimension of the hidden state.
            output_size (int): Dimension of the output vector.
            num_layers (int): Number of LSTM layers. Defaults to 1.
        """
        super(VanilaLSTM, self).__init__()
        self.hidden_size: int = hidden_size
        self.num_layers: int = num_layers

        # Create ModuleLists for each gate for each layer.
        self.W_xf_layers = nn.ModuleList()
        self.W_hf_layers = nn.ModuleList()
        self.W_xi_layers = nn.ModuleList()
        self.W_hi_layers = nn.ModuleList()
        self.W_xc_layers = nn.ModuleList()
        self.W_hc_layers = nn.ModuleList()
        self.W_xo_layers = nn.ModuleList()
        self.W_ho_layers = nn.ModuleList()

        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            self.W_xf_layers.append(nn.Linear(layer_input_size, hidden_size, bias=False))
            self.W_hf_layers.append(nn.Linear(hidden_size, hidden_size, bias=True))

            self.W_xi_layers.append(nn.Linear(layer_input_size, hidden_size, bias=False))
            self.W_hi_layers.append(nn.Linear(hidden_size, hidden_size, bias=True))

            self.W_xc_layers.append(nn.Linear(layer_input_size, hidden_size, bias=False))
            self.W_hc_layers.append(nn.Linear(hidden_size, hidden_size, bias=True))

            self.W_xo_layers.append(nn.Linear(layer_input_size, hidden_size, bias=False))
            self.W_ho_layers.append(nn.Linear(hidden_size, hidden_size, bias=True))

        self.output_layer = nn.Linear(hidden_size, output_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process an input sequence through the multi-layer LSTM.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        batch_size, seq_len, _ = x.shape

        # Initialize hidden state and cell state for each layer.
        h = [torch.zeros(batch_size, self.hidden_size, device=x.device) for _ in range(self.num_layers)]
        c = [torch.zeros(batch_size, self.hidden_size, device=x.device) for _ in range(self.num_layers)]

        for t in range(seq_len):
            layer_input = x[:, t, :]
            for layer in range(self.num_layers):
                f_t = torch.sigmoid(self.W_xf_layers[layer](layer_input) + self.W_hf_layers[layer](h[layer]))
                i_t = torch.sigmoid(self.W_xi_layers[layer](layer_input) + self.W_hi_layers[layer](h[layer]))
                o_t = torch.sigmoid(self.W_xo_layers[layer](layer_input) + self.W_ho_layers[layer](h[layer]))
                c_t_hat = torch.tanh(self.W_xc_layers[layer](layer_input) + self.W_hc_layers[layer](h[layer]))

                c[layer] = f_t * c[layer] + i_t * c_t_hat
                h[layer] = o_t * torch.tanh(c[layer])

                # The output of the current layer becomes the input to the next layer.
                layer_input = h[layer]

        y = self.output_layer(h[-1])
        return y
    

class CNNLSTMClassifier(nn.Module):
    def __init__(self, hidden_size: int, num_classes: int, num_layers: int = 1) -> None:
        """
        Initialize the CNN-LSTM classifier.

        This classifier uses a CNN (ResNet18) to extract features from video frames,
        and then processes the sequence of features with a custom multi-layer LSTM.

        Args:
            hidden_size (int): Dimension of the LSTM hidden state.
            num_classes (int): Number of output classes.
            num_layers (int): Number of LSTM layers. Defaults to 1.
        """
        super(CNNLSTMClassifier, self).__init__()
        resnet18_model = get_resnet18()
        # Remove the final fully connected layer from ResNet18.
        self.feature_extractor = nn.Sequential(*list(resnet18_model.children())[:-1])
        # Input to LSTM is the flattened feature map of size 512.
        self.lstm = VanilaLSTM(512, hidden_size, num_classes, num_layers)
    
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
            # Flatten the feature map.
            x_t = x_t.view(batch_size, -1)
            features.append(x_t)

        features = torch.stack(features, dim=1)
        return self.lstm(features)
