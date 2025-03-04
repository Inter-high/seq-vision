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
from data import *
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

    folder = "/workspace/seq-vision/UCF101"
    num_samples = 4       # 각 클래스별 사용할 영상 수 (예: 4)
    num_frame = 25       # 각 영상에서 추출할 프레임 수
    split_ratio = (6,2,2) # 예: 4개의 영상이면 train:2, valid:1, test:1 (비율은 상대적으로 적용됨)
    batch_size = 4
    size = 128
    num_workers = 0

     # 1. 파일 읽어오기
    files = load_avi_files(folder)
    # 2. 파일명에서 label 추출 및 그룹화
    class_to_files = group_files_by_label(folder, files)
    # 2-2. label mapping 생성 (101개 클래스)
    label2idx = create_label_mapping(class_to_files)
    # 3. 각 클래스별 num_samples 영상 선택 및 num_frame 프레임 추출
    class_to_videos = sample_videos_for_each_class(class_to_files, num_samples, num_frame)
    # 4. 각 클래스별로 영상들을 split_ratio에 따라 train/valid/test로 분할
    train_list, valid_list, test_list = split_videos(class_to_videos, ratio=split_ratio)
    print(f"Train samples: {len(train_list)}, Valid samples: {len(valid_list)}, Test samples: {len(test_list)}")
    
    # 5. Dataset 생성 (transform: ToPILImage, Resize((224,244)), ToTensor)
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
         # 1) Resize: 원본 크기가 너무 클 경우 미리 줄임
        transforms.Resize((size, size)),

        # 2) RandomResizedCrop: 일정 비율로 랜덤 크롭 (기본 0.8~1.0 범위)
        transforms.RandomResizedCrop(size=size, scale=(0.8, 1.0)),

        # 3) RandomHorizontalFlip: 좌우 뒤집기
        transforms.RandomHorizontalFlip(p=0.5),

        # 4) RandomVerticalFlip: 상하 뒤집기
        transforms.RandomVerticalFlip(p=0.5),

        # 5) ColorJitter: 밝기/대비/채도/색조 변환
        transforms.ColorJitter(
            brightness=0.2,   # 밝기
            contrast=0.2,     # 대비
            saturation=0.2,   # 채도
            hue=0.1           # 색조
        ),

        # 6) RandomRotation: 임의 회전
        transforms.RandomRotation(degrees=15),

        # 7) ToTensor: [0,255] → [0,1] 범위의 Tensor로
        transforms.ToTensor(),

        # 8) Normalize: 평균과 표준편차로 정규화
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = VideoDataset(train_list, label2idx, transform=train_transform)
    valid_dataset = VideoDataset(valid_list, label2idx, transform=test_transform)
    test_dataset  = VideoDataset(test_list,  label2idx, transform=test_transform)
    
    # 6. DataLoader 생성
    train_loader, valid_loader, test_loader = create_dataloaders(train_dataset, valid_dataset, test_dataset, batch_size, num_workers)
    
    # 예시: 각 DataLoader에서 몇 개의 배치 출력
    print("Train Loader:")
    for i, (videos, labels) in enumerate(train_loader):
        print(f"Batch {i+1}: Videos shape: {videos.shape}, Labels shape: {labels.shape}")
        if i == 4:
            break

    print("Valid Loader:")
    for i, (videos, labels) in enumerate(valid_loader):
        print(f"Batch {i+1}: Videos shape: {videos.shape}, Labels shape: {labels.shape}")
        if i == 4:
            break

    print("Test Loader:")
    for i, (videos, labels) in enumerate(test_loader):
        print(f"Batch {i+1}: Videos shape: {videos.shape}, Labels shape: {labels.shape}")
        if i == 4:
            break

    # Select the device: use GPU if available, otherwise CPU.
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # Initialize the classifier model and move it to the selected device.
    model = get_classifier(cfg['model']).to(device)
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

    loss_image = plot_compare_loss([train_losses, valid_losses], ["Train Loss", "Valid Loss"], os.path.join(output_dir, f"loss.jpg"))
    acc_image = plot_compare_acc([train_accs, valid_accs], ["Train Acc", "Valid Acc"], os.path.join(output_dir, f"acc.jpg"))

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
