# seq-vision

This repository experiments with whether combining a sequence model that learns temporal features and a convolutional model that learns spatial features can improve video processing performance compared to using either a sequence model or a convolutional model alone.

## Environment

### Software
- **Host OS:** Windows 11
- **CUDA:** 12.4
- **Docker:** Ubuntu 22.04
- **Python:** 3.10.12
- **Libraries:** See `requirements.txt`

### Hardware
- **CPU:** AMD Ryzen 5 7500F 6-Core Processor
- **GPU:** RTX 4070 Ti Super
- **RAM:** 32GB

## Dataset & Augmentation

This project uses the [UCF101 dataset](https://www.crcv.ucf.edu/data/UCF101.php).

To preprocess the video data, the following pipeline is used:

```python
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224 as specified in config.yaml
    transforms.ToTensor(),          # Convert images to tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),  # Normalize the images
])
```

### config.yaml

```yaml
resize: 224
max_frame: 100
max_samples_per_class: 10
```

- **resize:** Resizes all videos to 224×224.
- **max_frame:** Uses up to 100 frames per video.
- **max_samples_per_class:** Limits the dataset to a maximum of 10 samples per class to maintain data balance.

## Training Environment
- **Optimizer:** SGD
- **Scheduler:** MultiStepLR
- **Loss Function:** Cross-Entropy Loss (CELoss)
- **Hyper-parameters:** See `conf/config.yaml` for details.

## Verification List
1. Whether CNNs outperform RNNs for video processing.
2. Whether LSTMs perform better than RNNs.
3. Whether the combination of CNN and RNN outperforms using either model alone.
4. Whether the combination of CNN and LSTM outperforms the CNN + RNN configuration.

## Results
To be announced.

## How to Run the Experiment
To be announced.
