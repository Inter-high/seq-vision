#!/bin/bash

# 실험에 사용할 모델 이름 배열
models=("CNNClassifier" "RNNClassifier" "CNNRNNClassifier" "LSTMClassifier" "CNNLSTMClassifier")

# config 파일 경로 (필요시 수정)
config_file="./conf/config.yaml"

for model in "${models[@]}"; do
  echo "=============================="
  echo "Running experiment with model: $model"
  
  # config.yaml 내의 model_name 값을 변경 (들여쓰기를 고려하여 정규표현식 사용)
  sed -i.bak -E "s/(^[[:space:]]*model_name:[[:space:]]*\").*(\")/\1${model}\2/" "$config_file"
  
  # 변경된 model_name 확인
  grep "model_name:" "$config_file"
  
  # training 실행
  python3 train.py
  
  echo "Finished experiment for model: $model"
  echo "=============================="
done
