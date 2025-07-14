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

# 경로 설정
DATA_PATH = r"C:\data\intraday_stock.csv"
TEST_PATH = r"C:\data\intraday_stock_test.csv"
MODEL_PATH = r"C:\models\daytrade_lstm_best.h5"
LOG_DIR = r"C:\logs\daytrade_lstm"
RESULT_PATH = r"C:\results\daytrade_pred.csv"

# 경로 자동 생성
for p in [Path(MODEL_PATH).parent, Path(LOG_DIR), Path(RESULT_PATH).parent]:
    p.mkdir(parents=True, exist_ok=True)

SEQUENCE_LENGTH = 30  # 입력 시퀀스 길이
PREDICT_STEP = 1      # 1분 뒤 예측 (5분 뒤로 변경 가능)
BATCH_SIZE = 256
EPOCHS = 40
VAL_SPLIT = 0.2
LEARNING_RATE = 0.001

FEATURES = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'RSI', 'MACD', 'Stochastic'
]


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime')
        df = df.reset_index(drop=True)
    return df


def create_sequences(df: pd.DataFrame, features: list, seq_len: int, predict_step: int) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(len(df) - seq_len - predict_step):
        seq_x = df[features].iloc[i : i + seq_len].values
        next_close = df['Close'].iloc[i + seq_len + predict_step - 1]
        curr_close = df['Close'].iloc[i + seq_len - 1]
        label = 1 if next_close > curr_close else 0
        X.append(seq_x)
        y.append(label)
    return np.array(X), np.array(y)


def scale_features(train_df: pd.DataFrame, test_df: pd.DataFrame, features: list) -> Tuple[pd.DataFrame, pd.DataFrame, Any]:
    scaler = StandardScaler()
    train_scaled = train_df.copy()
    test_scaled = test_df.copy()
    train_scaled[features] = scaler.fit_transform(train_df[features])
    test_scaled[features] = scaler.transform(test_df[features])
    return train_scaled, test_scaled, scaler


def build_lstm_model(input_shape: Tuple[int, int]) -> tf.keras.Model:
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
    train_df = load_data(DATA_PATH)
    test_df = load_data(TEST_PATH)
    train_scaled, test_scaled, scaler = scale_features(train_df, test_df, FEATURES)
    X_train, y_train = create_sequences(train_scaled, FEATURES, SEQUENCE_LENGTH, PREDICT_STEP)
    X_test, y_test = create_sequences(test_scaled, FEATURES, SEQUENCE_LENGTH, PREDICT_STEP)
    model = build_lstm_model(input_shape=(SEQUENCE_LENGTH, len(FEATURES)))
    # ... (훈련 코드)
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype("int32")
    # ... (평가 코드)
    result_df = test_df.iloc[SEQUENCE_LENGTH + PREDICT_STEP - 1 : SEQUENCE_LENGTH + PREDICT_STEP - 1 + len(y_pred)].copy()
    result_df['y_true'] = y_test
    result_df['y_pred'] = y_pred
    result_df['y_pred_prob'] = y_pred_prob.flatten()
    result_df.to_csv(RESULT_PATH, index=False, encoding='utf-8-sig')
    print(f"예측 결과 저장: {RESULT_PATH}")


if __name__ == "__main__":
    train_and_evaluate()
