from datetime import datetime
from pathlib import Path
from sklearn.metrics import accuracy_score
import precision_score
import recall_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
import ModelCheckpoint
import TensorBoard
from tensorflow.keras.layers import LSTM
import Dropout
import Dense
from tensorflow.keras.models import Sequential
from typing import Tuple
import Any
import numpy as np
import os
import pandas as pd
import tensorflow as tf
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: swing_lstm_train.py
목적: 주식 스윙매매 자동매매 LSTM 모델 훈련 및 평가
작성일: 2025-07-08
Author: AI Assistant
"""

# 경로 설정
DATA_PATH = r"C:\data\daily_stock.csv"
TEST_PATH = r"C:\data\daily_stock_test.csv"
MODEL_PATH = r"C:\models\swing_lstm_best.h5"
LOG_DIR = r"C:\logs\swing_lstm"
RESULT_PATH = r"C:\results\swing_pred.csv"

# 경로 자동 생성
for p in [Path(MODEL_PATH).parent, Path(LOG_DIR), Path(RESULT_PATH).parent]:
    p.mkdir(parents=True, exist_ok=True)

SEQUENCE_LENGTH = 20  # 입력 시퀀스 길이
PREDICT_WINDOW = 5    # 5일 내 목표/손실 판정
BATCH_SIZE = 128
EPOCHS = 50
VAL_SPLIT = 0.2
LEARNING_RATE = 0.001

FEATURES = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'MA5', 'MA20', 'MA60', 'BB_Upper', 'BB_Lower', 'BB_Width', 'MACD'
]


def load_data(path: str) -> pd.DataFrame:
    """CSV 데이터 로드 및 전처리"""
    df = pd.read_csv(path)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
    return df


def create_swing_labels(df: pd.DataFrame, seq_len: int, predict_window: int, up_thresh: float = 0.10, down_thresh: float = -0.05) -> np.ndarray:
    """스윙매매용 라벨 생성: 5일 내 10% 이상 상승(1), 최대 하락률 -5% 이하(0)"""
    labels = []
    for i in range(len(df) - seq_len - predict_window):
        base = df['Close'].iloc[i + seq_len - 1]
        future = df['Close'].iloc[i + seq_len : i + seq_len + predict_window].values
        max_up = (future.max() - base) / base
        max_down = (future.min() - base) / base
        if max_up >= up_thresh:
            labels.append(1)
        elif max_down <= down_thresh:
            labels.append(0)
        else:
            labels.append(-1)  # 중립(학습 제외)
    return np.array(labels)


def create_sequences(df: pd.DataFrame, features: list, seq_len: int, predict_window: int) -> Tuple[np.ndarray, np.ndarray]:
    """시퀀스 데이터 및 라벨 생성 (중립 제외)"""
    X, y = [], []
    labels = create_swing_labels(df, seq_len, predict_window)
    for i in range(len(labels)):
        if labels[i] == -1:
            continue
        seq_x = df[features].iloc[i : i + seq_len].values
        X.append(seq_x)
        y.append(labels[i])
    return np.array(X), np.array(y)


def scale_features(train_df: pd.DataFrame, test_df: pd.DataFrame, features: list) -> Tuple[pd.DataFrame, pd.DataFrame, Any]:
    """특성 표준화"""
    scaler = StandardScaler()
    train_scaled = train_df.copy()
    test_scaled = test_df.copy()
    train_scaled[features] = scaler.fit_transform(train_df[features])
    test_scaled[features] = scaler.transform(test_df[features])
    return train_scaled, test_scaled, scaler


def build_lstm_model(input_shape: Tuple[int, int]) -> tf.keras.Model:
    """LSTM 2층 모델 생성"""
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    return model


def train_and_evaluate():
    pass
    # ... (기존 코드)
    # ...
