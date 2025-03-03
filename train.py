"""
This script trains a video classification model by setting up the training environment, loading and splitting data,
building the model, configuring the optimizer, scheduler, and loss function, training the model, saving training results,
and sending an email notification upon completion.

Author: yumemonzo@gmail.com
Date: 2025-03-03
"""

import os
import logging
import pickle
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
import hydra
from omegaconf import DictConfig

# Import utility functions and modules for data handling and training.
from data import get_transform, get_dataset, split_dataset, get_loaders
from models import get_classifier
from trainer import Trainer
from utils import seed_everything, count_model_parameters, send_email


@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg: DictConfig) -> None:
    """
    Run the training pipeline for the video classification model.
    
    Parameters:
        cfg (DictConfig): Configuration object containing hyperparameters and settings.
    """
    # Set up logging configuration.
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Set random seed for reproducibility.
    seed_everything(cfg['seed'])

    # Create a transformation pipeline for video frames.
    transform = get_transform(cfg['data']['resize'])

    # Load the dataset using the specified directory and transformation.
    dataset = get_dataset(cfg['data']['data_dir'], transform, cfg['data']['max_samples_per_class'])
    
    # Split the dataset into training, validation, and test subsets.
    train_dataset, valid_dataset, test_dataset = split_dataset(dataset)
    logger.info(f"Dataset | Train: {len(train_dataset)} | Valid: {len(valid_dataset)} | Test: {len(test_dataset)}")

    # Create data loaders for each subset.
    train_loader, valid_loader, test_loader = get_loaders(
        train_dataset, valid_dataset, test_dataset,
        cfg['data']['max_frame'], cfg['data']['batch_size'], cfg['data']['num_workers']
    )
    logger.info(f"DataLoader | Train: {len(train_loader)} | Valid: {len(valid_loader)} | Test: {len(test_loader)}")

    # Select the device: use GPU if available, otherwise CPU.
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # Initialize the classifier model and move it to the selected device.
    model = get_classifier(cfg['model'], 3 * cfg['data']['resize'] * cfg['data']['resize']).to(device)
    logger.info(f"Model| Name: {cfg['model']['model_name']} | Parameters: {count_model_parameters(model):3,}")

    # Set up the optimizer and learning rate scheduler.
    optimizer = optim.SGD(model.parameters(), lr=cfg['train']['lr'])
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg['train']['milestones'], gamma=cfg['train']['gamma']
    )
    
    # Define the loss function.
    criterion: nn.Module = nn.CrossEntropyLoss().to(device)

    # Determine the output directory from Hydra configuration.
    output_dir: str = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    # Define the path to save the best model weights.
    weight_path: str = os.path.join(output_dir, f"best_{cfg['model']['model_name']}.pth")

    # Initialize the Trainer with model, optimizer, scheduler, loss function, device, logger, and output directory.
    trainer = Trainer(model, optimizer, scheduler, criterion, device, logger, output_dir)

    # Check if the model requires flattening (if it does not contain any Conv2d layers).
    flatten: bool = not any(isinstance(layer, torch.nn.Conv2d) for layer in model.modules())

    # Start training and retrieve training statistics.
    train_losses, train_accs, valid_losses, valid_accs, top1_error, top5_error = trainer.training(
        cfg['train']['epochs'], train_loader, valid_loader, test_loader, flatten, weight_path, cfg['train']['patience']
    )

    # Save the training results to a pickle file.
    pickle_path: str = os.path.join(output_dir, "training_results.pkl")
    with open(pickle_path, "wb") as f:
        pickle.dump({
            "train_losses": train_losses,
            "train_accs": train_accs,
            "valid_losses": valid_losses,
            "valid_accs": valid_accs,
            "top1_error": top1_error,
            "top5_error": top5_error
        }, f)

    # Prepare email notification details.
    subject: str = "Training Completed"
    body: str = (
        f"Training job has completed successfully.\n"
        f"Final top1_error: {top1_error:.4f} | top5_error: {top5_error:.4f}"
    )
    # Send an email notification with the training results.
    send_email(subject, body, cfg['email']['to'], cfg['email']['from'], cfg['email']['password'])


if __name__ == "__main__":
    my_app()
