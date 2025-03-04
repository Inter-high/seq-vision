import os
import random
from typing import List, Tuple, Dict, Optional, Callable

import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class UCF101Dataset(Dataset):
    """
    UCF101 비디오 데이터셋 클래스 (transform을 __getitem__에서 적용).
    samples: (file_path, label) 형태의 리스트.
    transform: 각 frame(PIL Image)에 적용할 transform 함수(또는 Compose).
    """
    def __init__(self, samples: List[Tuple[str, int]], transform: Optional[Callable] = None) -> None:
        """
        Parameters:
            samples (List[Tuple[str, int]]): 비디오 경로와 라벨 인덱스를 담은 리스트
            transform (Optional[Callable]): 각 frame에 적용할 transform
        """
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        file_path, label = self.samples[idx]
        cap = cv2.VideoCapture(file_path)

        frames: List[torch.Tensor] = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # BGR -> RGB 변환
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)  # PIL Image로 변환

            # (중요) 여기서 transform 적용
            if self.transform:
                frame = self.transform(frame)

            frames.append(frame)

        cap.release()

        # (C,H,W) shape을 가진 프레임들이 들어있는 리스트를 쌓아서 (T,C,H,W) 만들기
        video_tensor = torch.stack(frames) if len(frames) > 0 else torch.empty(0)

        return video_tensor, label


def load_ucf101_samples(
    data_dir: str,
    max_samples_per_class: Optional[int] = None
) -> Tuple[List[Tuple[str, int]], List[str], Dict[str, int]]:
    """
    UCF101 데이터셋의 (파일 경로, 클래스) 정보를 전부 로드한 뒤 반환.
    (여기서 max_samples_per_class 적용)

    Returns:
        samples (List[Tuple[str, int]]): (비디오 경로, 클래스 인덱스) 리스트
        classes (List[str]): 정렬된 클래스명 리스트
        class_to_idx (Dict[str, int]): 클래스명 -> 인덱스 매핑
    """
    # 1) data_dir에서 .avi 파일 전부 찾기
    file_names: List[str] = [f for f in os.listdir(data_dir) if f.lower().endswith('.avi')]

    # 2) 클래스별로 파일을 분류
    samples_dict: Dict[str, List[str]] = {}
    for file_name in file_names:
        tokens: List[str] = file_name.split('_')
        class_name: str = tokens[1] if len(tokens) > 1 else "Unknown"
        samples_dict.setdefault(class_name, []).append(os.path.join(data_dir, file_name))

    # 3) 클래스 이름 정렬 & max_samples_per_class 적용
    classes: List[str] = sorted(samples_dict.keys())
    filtered_samples: List[Tuple[str, str]] = []
    for cls in classes:
        file_list: List[str] = samples_dict[cls]
        if max_samples_per_class is not None:
            file_list = file_list[:max_samples_per_class]
        for file_path in file_list:
            filtered_samples.append((file_path, cls))

    # 4) class_to_idx 매핑
    class_to_idx: Dict[str, int] = {cls: idx for idx, cls in enumerate(classes)}

    # 5) (파일 경로, 클래스 인덱스) 형태로 변환
    samples: List[Tuple[str, int]] = [
        (file_path, class_to_idx[label]) for file_path, label in filtered_samples
    ]

    return samples, classes, class_to_idx


def split_ucf101_samples(
    samples: List[Tuple[str, int]],
    split_ratio: Tuple[float, float, float] = (0.6, 0.2, 0.2)
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]], List[Tuple[str, int]]]:
    """
    UCF101의 (파일 경로, 라벨 인덱스) 샘플들을
    클래스별로 섞어서 주어진 비율대로 (train, valid, test)로 나눈다.

    Returns:
        train_samples, valid_samples, test_samples
    """
    # 클래스별 인덱스 분류
    from collections import defaultdict
    samples_by_class = defaultdict(list)
    for i, (file_path, label) in enumerate(samples):
        samples_by_class[label].append((file_path, label))

    train_samples, valid_samples, test_samples = [], [], []

    train_ratio, valid_ratio, test_ratio = split_ratio
    for label, sample_list in samples_by_class.items():
        random.shuffle(sample_list)
        n = len(sample_list)

        n_train = int(n * train_ratio)
        n_valid = int(n * valid_ratio)
        # 나머지를 test로
        n_test = n - (n_train + n_valid)

        train_samples.extend(sample_list[:n_train])
        valid_samples.extend(sample_list[n_train:n_train + n_valid])
        test_samples.extend(sample_list[n_train + n_valid : n_train + n_valid + n_test])

    return train_samples, valid_samples, test_samples


def fixed_length_collate_fn(
    batch: List[Tuple[torch.Tensor, int]],
    max_frame: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    비디오 프레임을 max_frame 길이에 맞춰 truncate/pad하는 collate 함수
    """
    videos, labels = zip(*batch)
    processed_videos: List[torch.Tensor] = []

    for video in videos:
        num_frames: int = video.size(0)
        if num_frames >= max_frame:
            # Truncate
            video = video[:max_frame]
        else:
            # Pad
            pad_frames: int = max_frame - num_frames
            pad_tensor = torch.zeros((pad_frames, *video.shape[1:]), dtype=video.dtype)
            video = torch.cat([video, pad_tensor], dim=0)
        processed_videos.append(video)

    videos_tensor: torch.Tensor = torch.stack(processed_videos)
    labels_tensor: torch.Tensor = torch.tensor(labels, dtype=torch.long)
    return videos_tensor, labels_tensor


def get_loaders(
    train_dataset: Dataset,
    valid_dataset: Dataset,
    test_dataset: Dataset,
    max_frame: int,
    batch_size: int,
    num_workers: int
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    세 개의 Dataset 각각에 대해 DataLoader를 생성
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda batch: fixed_length_collate_fn(batch, max_frame)
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda batch: fixed_length_collate_fn(batch, max_frame)
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda batch: fixed_length_collate_fn(batch, max_frame)
    )
    return train_loader, valid_loader, test_loader


def get_train_transform(resize: int) -> transforms.Compose:
    """
    학습용 Transform (강한 Augmentation 버전)
    """
    return transforms.Compose([
        # 1) Resize: 원본 크기가 너무 클 경우 미리 줄임
        transforms.Resize((resize, resize)),

        # 2) RandomResizedCrop: 일정 비율로 랜덤 크롭 (기본 0.8~1.0 범위)
        transforms.RandomResizedCrop(size=resize, scale=(0.8, 1.0)),

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
    

def get_test_transform(resize: int) -> transforms.Compose:
    """
    검증/테스트용 Transform
    """
    return transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
