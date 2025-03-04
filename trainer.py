"""
This module implements the Trainer class for managing the training, validation, and testing processes of a PyTorch model.
It logs metrics using TensorBoard and saves the best model based on validation loss.

Author: yumemonzo@gmail.com
Date: 2025-03-03
"""

import time
from typing import Tuple, List, Any

import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    """
    Trainer class to manage the training, validation, and testing processes.

    Attributes:
        model (torch.nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        scheduler: Learning rate scheduler.
        criterion (torch.nn.Module): Loss function.
        device (str): Device to perform computations ('cuda' or 'cpu').
        logger: Logger for recording training progress.
        writer (SummaryWriter): TensorBoard writer for logging metrics.
        lowest_loss (float): Tracks the lowest validation loss for saving the best model.
    """
    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: Any, 
                 criterion: torch.nn.Module, device: str, logger: Any, log_dir: str) -> None:
        """
        Initialize the Trainer with model, optimizer, scheduler, loss function, device, logger, and log directory.

        Args:
            model (torch.nn.Module): Model to be trained.
            optimizer (torch.optim.Optimizer): Optimizer.
            scheduler: Learning rate scheduler.
            criterion (torch.nn.Module): Loss function.
            device (str): Device to use ('cuda' or 'cpu').
            logger: Logger for logging information.
            log_dir (str): Directory for TensorBoard logs.
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.lowest_loss: float = float("inf")
        self.device: str = device
        self.logger = logger
        self.writer = SummaryWriter(log_dir=log_dir)

    def train(self, train_loader: torch.utils.data.DataLoader, flatten: bool) -> Tuple[float, float]:
        """
        Train the model for one epoch.

        Args:
            train_loader (torch.utils.data.DataLoader): DataLoader for training data.
            flatten (bool): Whether to flatten the input data.

        Returns:
            Tuple[float, float]: Average training loss and accuracy.
        """
        self.model.train()
        total_loss: float = 0.0
        total_acc: float = 0.0
        total_samples: int = 0
        progress_bar = tqdm(train_loader, desc="Training", leave=True)

        for x, y in progress_bar:
            x, y = x.to(self.device), y.to(self.device)
            if flatten:
                x = x.view(x.size(0), x.size(1), -1)

            self.optimizer.zero_grad()
            y_hat = self.model(x)

            loss = self.criterion(y_hat, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_samples += y.size(0)
            pred = y_hat.argmax(dim=1)
            total_acc += (pred == y).sum().item()

            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        avg_acc = total_acc / total_samples
        return avg_loss, avg_acc

    def valid(self, valid_loader: torch.utils.data.DataLoader, flatten: bool) -> Tuple[float, float]:
        """
        Validate the model.

        Args:
            valid_loader (torch.utils.data.DataLoader): DataLoader for validation data.
            flatten (bool): Whether to flatten the input data.

        Returns:
            Tuple[float, float]: Average validation loss and accuracy.
        """
        self.model.eval()
        total_loss: float = 0.0
        total_acc: float = 0.0
        total_samples: int = 0
        progress_bar = tqdm(valid_loader, desc="Validating", leave=True)

        with torch.no_grad():
            for x, y in progress_bar:
                x, y = x.to(self.device), y.to(self.device)
                if flatten:
                    x = x.view(x.size(0), x.size(1), -1)

                y_hat = self.model(x)
                loss = self.criterion(y_hat, y)

                total_loss += loss.item()
                total_samples += y.size(0)
                pred = y_hat.argmax(dim=1)
                total_acc += (pred == y).sum().item()

                progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(valid_loader)
        avg_acc = total_acc / total_samples
        return avg_loss, avg_acc

    def test(self, test_loader: torch.utils.data.DataLoader, flatten: bool) -> Tuple[float, float]:
        """
        Test the model and compute the top-1 and top-5 error rates.

        Args:
            test_loader (torch.utils.data.DataLoader): DataLoader for test data.
            flatten (bool): Whether to flatten the input data.

        Returns:
            Tuple[float, float]: Top-1 error and top-5 error.
        """
        self.model.eval()
        top1_correct: float = 0.0
        top5_correct: float = 0.0
        total_samples: int = 0
        progress_bar = tqdm(test_loader, desc="Testing", leave=True)

        with torch.no_grad():
            for x, y in progress_bar:
                x, y = x.to(self.device), y.to(self.device)
                if flatten:
                    x = x.view(x.size(0), x.size(1), -1)

                y_hat = self.model(x)

                _, top5_preds = torch.topk(y_hat, k=5, dim=1)
                top1_pred = top5_preds[:, 0]
                total_samples += y.size(0)

                top1_correct += (top1_pred == y).sum().item()
                top5_correct += sum(y[i].item() in top5_preds[i].tolist() for i in range(y.size(0)))

                progress_bar.set_postfix(top1_acc=top1_correct / total_samples, top5_acc=top5_correct / total_samples)

        top1_error: float = 1 - (top1_correct / total_samples)
        top5_error: float = 1 - (top5_correct / total_samples)
        return top1_error, top5_error

    def training(
        self, 
        epochs: int, 
        train_loader: torch.utils.data.DataLoader, 
        valid_loader: torch.utils.data.DataLoader, 
        test_loader: torch.utils.data.DataLoader, 
        flatten: bool, 
        weight_path: str, 
        patience: int
    ) -> Tuple[List[float], List[float], List[float], List[float], float, float]:
        """
        Manage the overall training process, including training, validation, and testing.

        Args:
            epochs (int): Number of training epochs.
            train_loader (torch.utils.data.DataLoader): DataLoader for training data.
            valid_loader (torch.utils.data.DataLoader): DataLoader for validation data.
            test_loader (torch.utils.data.DataLoader): DataLoader for test data.
            flatten (bool): Whether to flatten the input data.
            weight_path (str): File path to save the best model weights.
            patience (int): Number of evaluations to wait for improvement before early stopping.

        Returns:
            Tuple[List[float], List[float], List[float], List[float], float, float]:
                - List of training losses per epoch.
                - List of training accuracies per epoch.
                - List of validation losses per evaluation.
                - List of validation accuracies per evaluation.
                - Final top-1 error.
                - Final top-5 error.
        """
        train_losses: List[float] = []
        train_accs: List[float] = []
        valid_losses: List[float] = []
        valid_accs: List[float] = []

        start_time = time.time()  # Record overall training start time.
        patience_counter = 0      # Counter for evaluations without improvement.

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train(train_loader, flatten)
            train_losses.append(train_loss)
            train_accs.append(train_acc)

            loss_dict = {"train": train_loss}
            acc_dict = {"train": train_acc}

            if epoch % 5 == 0:
                valid_loss, valid_acc = self.valid(valid_loader, flatten)
                valid_losses.append(valid_loss)
                valid_accs.append(valid_acc)
                loss_dict["valid"] = valid_loss
                acc_dict["valid"] = valid_acc

                self.logger.info(f"Epoch: {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f}")

                # Check for improvement in validation loss.
                if valid_loss < self.lowest_loss:
                    self.lowest_loss = valid_loss
                    torch.save(self.model.state_dict(), weight_path)
                    self.logger.info(f"New lowest valid loss: {valid_loss:.4f}. Model weights saved to {weight_path}")
                    patience_counter = 0  # Reset counter if improvement is observed.
                else:
                    patience_counter += 1
                    self.logger.info(f"No improvement in validation loss. Patience counter: {patience_counter}/{patience}")
                    if patience_counter >= patience:
                        self.logger.info(f"Early stopping triggered. No improvement for {patience} consecutive evaluations.")
                        break

            self.writer.add_scalars("Loss", loss_dict, epoch)
            self.writer.add_scalars("Acc", acc_dict, epoch)
            self.scheduler.step()

            # Estimate remaining training time.
            elapsed_time = time.time() - start_time  # Elapsed time so far (seconds).
            avg_epoch_time = elapsed_time / epoch    # Average time per epoch.
            remaining_epochs = epochs - epoch         # Number of remaining epochs.
            est_remaining_time = avg_epoch_time * remaining_epochs  # Estimated remaining time (seconds).
            hours, rem = divmod(est_remaining_time, 3600)
            minutes, _ = divmod(rem, 60)
            self.logger.info(f"Epoch {epoch}/{epochs} completed. Estimated remaining time: {int(hours)}h {int(minutes)}m")

        top1_error, top5_error = self.test(test_loader, flatten)
        self.logger.info(f"Top1 Error: {top1_error:.4f} | Top5 Error: {top5_error:.4f}")

        return train_losses, train_accs, valid_losses, valid_accs, top1_error, top5_error
