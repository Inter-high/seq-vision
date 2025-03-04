"""
This module aggregates all model functions and classes for the video classification project.
It includes utilities for ResNet18-based models, RNN and LSTM classifiers, as well as a function to retrieve a classifier.

Author: yumemonzo@gmail.com
Date: 2025-03-03
"""

from models.resnet18 import get_resnet18, CNNClassifier
from models.rnn import CNNRNNClassifier
from models.lstm import CNNLSTMClassifier
from models.classifier import get_classifier

__all__ = [
    "get_resnet18", "CNNClassifier", "CNNRNNClassifier",
    "CNNLSTMClassifier", "get_classifier"
]
