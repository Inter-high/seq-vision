import torch
import torch.nn as nn
from models import get_resnet18

class CNNRNNClassifier(nn.Module):
    def __init__(self, hidden_size: int, num_classes: int, num_layers: int = 1, dropout: float = 0.5) -> None:
        """
        Initialize the CNN-RNN classifier.

        This classifier extracts features from video frames using a pretrained ResNet18 and
        processes the sequence of features using an RNN.

        Args:
            hidden_size (int): Dimension of the RNN hidden state.
            num_classes (int): Number of output classes.
            num_layers (int): Number of stacked RNN layers. Defaults to 1.
            dropout (float): Dropout rate applied between RNN layers (if num_layers > 1). Defaults to 0.5.
        """
        super(CNNRNNClassifier, self).__init__()
        resnet18_model = get_resnet18()
        # Remove the final fully connected layer of ResNet18.
        self.feature_extractor = nn.Sequential(*list(resnet18_model.children())[:-1])
        # Built-in RNN module; dropout only works if num_layers > 1.
        self.rnn = nn.RNN(input_size=512, hidden_size=hidden_size, num_layers=num_layers,
                          batch_first=True, nonlinearity="tanh", dropout=dropout if num_layers > 1 else 0)
        # Classifier using the last layer's hidden state.
        self.classifier = nn.Linear(hidden_size, num_classes)
    
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
            # Extract features for each frame.
            x_t = self.feature_extractor(x[:, t])
            # Flatten the feature map: expected shape becomes (batch_size, 512)
            x_t = x_t.view(batch_size, -1)
            features.append(x_t)
        # Stack features along the time dimension -> shape: (batch_size, seq_len, 512)
        features = torch.stack(features, dim=1)
        # Process the sequence with the RNN.
        rnn_out, hidden = self.rnn(features)  # hidden shape: (num_layers, batch_size, hidden_size)
        # Use the last layer's hidden state for classification.
        logits = self.classifier(hidden[-1])
        return logits