"""
This module provides a function to retrieve a classifier model based on the configuration.
It selects and returns a classifier (CNN, RNN, CNN-RNN, LSTM, or CNN-LSTM) according to the model name specified in the configuration.

Author: yumemonzo@gmail.com
Date: 2025-03-03
"""

from models.resnet18 import CNNClassifier
from models.rnn import RNNClassifier, CNNRNNClassifier
from models.lstm import LSTMClassifier, CNNLSTMClassifier
from torch import nn


def get_classifier(cfg: dict, input_size: int) -> nn.Module:
    """
    Retrieve a classifier model based on the configuration.

    Args:
        cfg (dict): Configuration dictionary containing keys 'model_name', 'num_classes', and optionally 'hidden_size'.
        input_size (int): The input feature dimension for models that require it (e.g., RNNClassifier, LSTMClassifier).

    Returns:
        nn.Module: An instance of the selected classifier model.
    """
    # Select CNNClassifier if specified in the configuration.
    if cfg['model_name'] == 'CNNClassifier':
        model = CNNClassifier(cfg['num_classes'])
    # Select RNNClassifier for sequence modeling using an RNN.
    elif cfg['model_name'] == 'RNNClassifier':
        model = RNNClassifier(input_size, cfg['hidden_size'], cfg['num_classes'])
    # Select CNNRNNClassifier that combines CNN-based feature extraction with an RNN.
    elif cfg['model_name'] == 'CNNRNNClassifier':
        model = CNNRNNClassifier(cfg['hidden_size'], cfg['num_classes'])
    # Select LSTMClassifier for sequence modeling using an LSTM.
    elif cfg['model_name'] == 'LSTMClassifier':
        model = LSTMClassifier(input_size, cfg['hidden_size'], cfg['num_classes'])
    # Select CNNLSTMClassifier that combines CNN-based feature extraction with an LSTM.
    elif cfg['model_name'] == 'CNNLSTMClassifier':
        model = CNNLSTMClassifier(cfg['hidden_size'], cfg['num_classes'])
    else:
        raise ValueError(f"Unsupported model name: {cfg['model_name']}")

    return model
