import os
import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# 1. 특정 폴더 내 .avi 파일 읽어오기
def load_avi_files(folder):
    files = [f for f in os.listdir(folder) if f.lower().endswith('.avi') and os.path.isfile(os.path.join(folder, f))]
    return files

# 2. 파일명에서 1번째 "_"와 2번째 "_" 사이의 문자열 추출
def extract_label_from_filename(filename):
    basename = os.path.splitext(filename)[0]
    first_us = basename.find('_')
    if first_us == -1:
        return None
    second_us = basename.find('_', first_us + 1)
    if second_us == -1:
        return None
    return basename[first_us+1:second_us]

def group_files_by_label(folder, files):
    class_to_files = {}
    for file in files:
        label_str = extract_label_from_filename(file)
        if label_str is None:
            print(f"파일 '{file}'에 올바른 '_' 구분자가 없습니다. 건너뜁니다.")
            continue
        path = os.path.join(folder, file)
        class_to_files.setdefault(label_str, []).append(path)
    return class_to_files

def create_label_mapping(class_to_files):
    unique_labels = sorted(class_to_files.keys())
    if len(unique_labels) != 101:
        print(f"경고: 고유 label의 개수가 101개가 아닙니다. (총 {len(unique_labels)}개)")
    mapping = {label: idx for idx, label in enumerate(unique_labels)}
    return mapping

# 3. 각 클래스별로 num_samples 개의 영상을 선택하고, 각 영상에서 num_frame 프레임 추출 (부족 시 제로 패딩)
def sample_videos_for_each_class(class_to_files, num_samples, num_frame):
    class_to_videos = {}
    for label, file_list in class_to_files.items():
        if len(file_list) < num_samples:
            print(f"클래스 '{label}'의 영상 수가 {num_samples}개 미만입니다 (총 {len(file_list)}개). 건너뜁니다.")
            continue
        # 무작위로 num_samples개 선택
        selected_files = random.sample(file_list, num_samples) if len(file_list) > num_samples else file_list
        videos = []
        for video_path in selected_files:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"영상 '{video_path}' 열기 실패. 건너뜁니다.")
                continue
            frames = []
            for _ in range(num_frame):
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()
            if len(frames) == 0:
                print(f"영상 '{video_path}'에서 프레임을 하나도 읽지 못했습니다.")
                continue
            # 부족한 경우 제로 패딩 (첫 프레임 기준)
            if len(frames) < num_frame:
                pad_frame = np.zeros_like(frames[0])
                frames += [pad_frame] * (num_frame - len(frames))
            if len(frames) > num_frame:
                frames = frames[:num_frame]
            videos.append(frames)
        if len(videos) != num_samples:
            print(f"클래스 '{label}'의 {num_samples}개 샘플을 모으지 못했습니다. (모은 샘플 수: {len(videos)})")
            continue
        class_to_videos[label] = videos
    return class_to_videos

# 4. 각 클래스별로 선택된 영상들을 ratio에 따라 train, valid, test로 분할
def split_videos(class_to_videos, ratio=(6,2,2)):
    train_list, valid_list, test_list = [], [], []
    total_ratio = sum(ratio)
    for label, videos in class_to_videos.items():
        # 총 영상 수 = num_samples
        n = len(videos)
        n_train = int(n * ratio[0] / total_ratio)
        n_valid = int(n * ratio[1] / total_ratio)
        # 나머지를 test에 할당
        n_test = n - n_train - n_valid
        # 만약 n_train or n_valid or n_test가 0이면 최소 1개씩 할당하도록 조정 (필요시)
        if n_train < 1:
            n_train = 1
        if n_valid < 1:
            n_valid = 1
        if n_test < 1:
            n_test = 1
        # 영상 순서를 무작위 섞은 후 분할
        random.shuffle(videos)
        train_videos = videos[:n_train]
        valid_videos = videos[n_train:n_train+n_valid]
        test_videos = videos[n_train+n_valid:n_train+n_valid+n_test]
        for video in train_videos:
            train_list.append((label, video))
        for video in valid_videos:
            valid_list.append((label, video))
        for video in test_videos:
            test_list.append((label, video))
    return train_list, valid_list, test_list

# 5. Dataset 만들기 (transform: ToPILImage, Resize, ToTensor)
def get_video_transform(single_transform):
    def video_transform(video_tensor):
        # video_tensor: (num_frame, C, H, W) – 각 프레임에 대해 변환 적용 후 stack
        transformed = [single_transform(frame) for frame in video_tensor]
        return torch.stack(transformed)
    return video_transform

class VideoDataset(Dataset):
    def __init__(self, video_list, label2idx, transform=None):
        """
        video_list: 각 항목이 (label_str, video) 형태, video는 num_frame개의 프레임 리스트 (numpy arrays)
        label2idx: label mapping (문자열 -> int)
        transform: 영상에 적용할 transform (default: get_video_transform)
        """
        self.video_list = video_list
        self.label2idx = label2idx
        self.transform = get_video_transform(transform)
    
    def __len__(self):
        return len(self.video_list)
    
    def __getitem__(self, idx):
        label_str, video_frames = self.video_list[idx]
        # 각 프레임: numpy array (H, W, C) -> torch tensor (C, H, W)
        video_tensor = torch.stack([torch.from_numpy(frame).permute(2,0,1) for frame in video_frames])
        # transform: 개별 프레임 변환 후 stack (변환된 video_tensor의 shape: (num_frame, C, new_H, new_W))
        if self.transform:
            video_tensor = self.transform(video_tensor)
        label = torch.tensor(self.label2idx[label_str], dtype=torch.long)
        return video_tensor, label

# 6. DataLoader 생성
def create_dataloaders(train_dataset, valid_dataset, test_dataset, batch_size=4, num_workers=0):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, valid_loader, test_loader
