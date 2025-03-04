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
from data import load_ucf101_samples, split_ucf101_samples, get_train_transform, get_test_transform, UCF101Dataset, get_loaders
from models import get_classifier
from trainer import Trainer
from utils import seed_everything, count_model_parameters, send_email, plot_compare_loss, plot_compare_acc


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

    samples, classes, class_to_idx = load_ucf101_samples(
        data_dir=cfg['data']['data_dir'],
        max_samples_per_class=cfg['data']['max_samples_per_class']
    )
    logger.info(f"Total samples loaded: {len(samples)} (Classes: {len(classes)})")

    # 2) 샘플 분할 (클래스별로 섞어서 60:20:20)
    train_samples, valid_samples, test_samples = split_ucf101_samples(
        samples,
        split_ratio=(0.6, 0.2, 0.2)  # 원하는 비율로 조정
    )
    logger.info(
        f"Split samples | Train: {len(train_samples)} | "
        f"Valid: {len(valid_samples)} | Test: {len(test_samples)}"
    )

    # 3) Transform 정의
    train_transform = get_train_transform(cfg['data']['resize'])
    test_transform = get_test_transform(cfg['data']['resize'])

    # 4) 각각의 Dataset 생성 (여기서 transform 다르게)
    train_dataset = UCF101Dataset(train_samples, transform=train_transform)
    valid_dataset = UCF101Dataset(valid_samples, transform=test_transform)
    test_dataset  = UCF101Dataset(test_samples,  transform=test_transform)

    logger.info(
        f"Dataset | Train: {len(train_dataset)} | "
        f"Valid: {len(valid_dataset)} | Test: {len(test_dataset)}"
    )

    # 5) DataLoader 생성
    train_loader, valid_loader, test_loader = get_loaders(
        train_dataset, valid_dataset, test_dataset,
        max_frame=cfg['data']['max_frame'],
        batch_size=cfg['data']['batch_size'],
        num_workers=cfg['data']['num_workers']
    )
    logger.info(
        f"DataLoader | Train: {len(train_loader)} | "
        f"Valid: {len(valid_loader)} | Test: {len(test_loader)}"
    )

    # Select the device: use GPU if available, otherwise CPU.
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # Initialize the classifier model and move it to the selected device.
    model = get_classifier(cfg['model']).to(device)
    logger.info(f"Model| Name: {cfg['model']['model_name']} | Parameters: {count_model_parameters(model):3,}")

    # Set up the optimizer and learning rate scheduler.
    optimizer = optim.Adam(model.parameters(), lr=cfg['train']['lr'])
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

    loss_image = plot_compare_loss([train_losses, valid_losses], ["Train Loss", "Valid Loss"], os.path.join(output_dir, f"loss.jpg"))
    acc_image = plot_compare_acc([train_accs, valid_accs], ["Train Acc", "Valid Acc"], os.path.join(output_dir, f"acc.jpg"))

    # Prepare email notification details.
    subject: str = "Training Completed"
    body: str = (
        f"Training job has completed successfully.\n"
        f"Final top1_error: {top1_error:.4f} | top5_error: {top5_error:.4f}"
    )

    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.image import MIMEImage
    from email.mime.text import MIMEText

    # 예시 변수들 (실제 값으로 대체하세요)
    sender = "yumemonzo@gmail.com"
    recipient = "yumemonzo@gmail.com"
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    smtp_user = "yumemonzo@gmail.com"
    smtp_password = "gbnl jjqb uetw oahk"

    # 이미지 데이터 (loss_image, acc_image)는 앞서 생성한 이미지 바이트 데이터라고 가정합니다.
    # 예:
    # loss_image = plot_compare_loss([train_losses, valid_losses], ["Train Loss", "Valid Loss"], ...)
    # acc_image = plot_compare_acc([train_accs, valid_accs], ["Train Acc", "Valid Acc"], ...)

    # 이메일 제목과 본문 내용
    subject = "Training Completed"

    # MIMEMultipart 객체 생성 (related: 본문과 인라인 이미지 포함)
    msg = MIMEMultipart('related')
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = recipient

    # HTML 형식의 본문을 alternative 파트로 추가
    msg_alternative = MIMEMultipart('alternative')
    msg.attach(msg_alternative)

    html_content = f"""
    <html>
    <body>
        <p>{cfg['model']['model_name']} Training job has completed successfully.<br>
        Final top1_error: {top1_error:.4f} | top5_error: {top5_error:.4f}
        </p>
        <p>Loss Curve:<br>
        <img src="cid:loss_image">
        </p>
        <p>Accuracy Curve:<br>
        <img src="cid:acc_image">
        </p>
    </body>
    </html>
    """
    msg_alternative.attach(MIMEText(html_content, 'html'))

    # 이미지 데이터 첨부 (Content-ID 지정)
    img_loss = MIMEImage(loss_image, _subtype="jpeg")
    img_loss.add_header('Content-ID', '<loss_image>')
    msg.attach(img_loss)

    img_acc = MIMEImage(acc_image, _subtype="jpeg")
    img_acc.add_header('Content-ID', '<acc_image>')
    msg.attach(img_acc)


    # 이메일 전송 예제 (SMTP 서버 사용)
    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.sendmail(sender, recipient, msg.as_string())

    # Send an email notification with the training results.
    # send_email(subject, body, cfg['email']['to'], cfg['email']['from'], cfg['email']['password'])


if __name__ == "__main__":
    my_app()
