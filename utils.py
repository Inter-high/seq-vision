"""
This module provides utility functions for reproducibility, counting model parameters,
plotting performance comparisons, and sending email notifications.

Author: yumemonzo@gmail.com
Date: 2025-02-24
"""

import random
import torch
import matplotlib.pyplot as plt
import numpy as np
import smtplib
from typing import List, Any, Tuple
from torch.utils.data import TensorDataset, DataLoader
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


def seed_everything(seed: int = 42) -> None:
    """
    Set random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): Seed value to use. Default is 42.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_model_parameters(model: torch.nn.Module) -> int:
    """
    Count the total number of trainable parameters in a PyTorch model.

    Args:
        model (torch.nn.Module): The model to count parameters for.

    Returns:
        int: Total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_compare_loss(data: List[List[float]], labels: List[str], save_path: str, title: str = 'Loss Comparison') -> None:
    """
    Plot loss curves from multiple experiments and save the figure.

    Args:
        data (List[List[float]]): List of loss values for each experiment.
        labels (List[str]): Corresponding labels for each experiment.
        save_path (str): Path to save the plotted figure.
        title (str): Title for the plot. Default is 'Loss Comparison'.

    Raises:
        ValueError: If lengths of data and labels do not match.
    """
    if len(data) != len(labels):
        raise ValueError("The lengths of data and labels must be equal.")
    
    plt.figure(figsize=(10, 6))
    for i, losses in enumerate(data):
        epochs = range(1, len(losses) + 1)
        plt.plot(epochs, losses, label=labels[i])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.clf()


def plot_compare_acc(data: List[List[float]], labels: List[str], save_path: str, title: str = 'Accuracy Comparison') -> None:
    """
    Plot accuracy curves from multiple experiments and save the figure.

    Args:
        data (List[List[float]]): List of accuracy values for each experiment.
        labels (List[str]): Corresponding labels for each experiment.
        save_path (str): Path to save the plotted figure.
        title (str): Title for the plot. Default is 'Accuracy Comparison'.

    Raises:
        ValueError: If lengths of data and labels do not match.
    """
    if len(data) != len(labels):
        raise ValueError("The lengths of data and labels must be equal.")
    
    plt.figure(figsize=(10, 6))
    for i, accuracies in enumerate(data):
        epochs = range(1, len(accuracies) + 1)
        plt.plot(epochs, accuracies, label=labels[i])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.clf()


def plot_compare_error(errors: List[float], labels: List[str], save_path: str, title: str = 'Error Comparison') -> None:
    """
    Plot a bar chart comparing error values from multiple experiments and save the figure.

    Args:
        errors (List[float]): Error values for each experiment.
        labels (List[str]): Corresponding labels for each experiment.
        save_path (str): Path to save the plotted figure.
        title (str): Title for the plot. Default is 'Error Comparison'.

    Raises:
        ValueError: If lengths of errors and labels do not match.
    """
    if len(errors) != len(labels):
        raise ValueError("The lengths of errors and labels must be equal.")
    
    n = len(errors)
    x = np.arange(n)
    width = 0.5
    colors = [f'C{i}' for i in range(n)]
    
    plt.figure(figsize=(8, 6))
    plt.bar(x, errors, width, color=colors)
    plt.xticks(x, labels)
    plt.ylabel('Error')
    plt.title(title)
    plt.savefig(save_path)
    plt.clf()


def send_email(subject: str, body: str, to_email: str, from_email: str, password: str,
               smtp_server: str = "smtp.gmail.com", port: int = 587) -> None:
    """
    Send an email with the specified subject and body using SMTP.

    Args:
        subject (str): Email subject.
        body (str): Email body content.
        to_email (str): Recipient email address.
        from_email (str): Sender email address.
        password (str): Password for the sender's email account.
        smtp_server (str): SMTP server address. Default is "smtp.gmail.com".
        port (int): SMTP server port. Default is 587.
    """
    msg = MIMEMultipart()
    msg["From"] = from_email
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))
    
    try:
        server = smtplib.SMTP(smtp_server, port)
        server.starttls()  # Secure the connection using TLS
        server.login(from_email, password)
        server.sendmail(from_email, to_email, msg.as_string())
        server.quit()
        print("Email sent successfully!")
    except Exception as e:
        print(f"Failed to send email: {e}")


def get_dummy_loaders(batch_size, max_frame, resize) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create dummy DataLoaders for testing purposes.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: DataLoaders for training, validation, and test datasets.
    """
    # Create dummy tensors for input data and labels.
    train_data = torch.zeros((batch_size * 2, max_frame, 3, resize, resize))
    train_labels = torch.zeros(batch_size * 2, dtype=torch.long)
    valid_data = torch.zeros((batch_size * 2, max_frame, 3, resize, resize))
    valid_labels = torch.zeros(batch_size * 2, dtype=torch.long)
    test_data  = torch.zeros((batch_size * 2, max_frame, 3, resize, resize))
    test_labels  = torch.zeros(batch_size * 2, dtype=torch.long)

    # Create TensorDatasets for each set.
    train_dataset = TensorDataset(train_data, train_labels)
    valid_dataset = TensorDataset(valid_data, valid_labels)
    test_dataset  = TensorDataset(test_data, test_labels)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader
