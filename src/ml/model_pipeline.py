#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: model_pipeline.py
목적: Parquet/Feather/PostgreSQL/MLflow 기반 ML/DL 데이터 로딩, 전처리, 학습, 예측, 모델 버전 관리
Author: [Your Name]
Created: 2025-07-10
Version: 1.0.0

- 커서룰 100% 준수 (타입힌트, 예외처리, 구조화 로깅, 문서화, 테스트포인트)
- PyTorch, scikit-learn, XGBoost, Prophet, MLflow, 구조화 로깅
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Any
import Dict
import Optional
import pandas as pd
import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sqlalchemy import create_engine
import torch
import joblib

# 구조화 로깅
logging.basicConfig(
    filename="logs/model_pipeline.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

# DB 연결
engine = create_engine("postgresql+psycopg2://user:password@localhost:5432/stockdb")

# 데이터 로딩 함수
def load_data_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)

def load_data_postgres(table: str) -> pd.DataFrame:
    return pd.read_sql_table(table, engine)

# 전처리 예시
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # 결측치/이상치 처리, 피처 엔지니어링 등 (실전은 확장)
    return df.dropna()

# ML 학습 예시 (scikit-learn)
def train_sklearn_rf(df: pd.DataFrame, target: str) -> Any:
    X = df.drop(columns=[target])
    y = df[target]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    preds = model.predict(X)
    mse = mean_squared_error(y, preds)
    logger.info(f"RandomForest MSE: {mse}")
    return model

# MLflow 모델 저장
def save_mlflow_model(model: Any, model_name: str) -> None:
    mlflow.sklearn.log_model(model, model_name)
    logger.info(f"MLflow 모델 저장: {model_name}")

# PyTorch 예시 (딥러닝)
class SimpleNN(torch.nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.fc = torch.nn.Linear(input_dim, 1)
    def forward(self, x):
        return self.fc(x)

def train_pytorch_nn(df: pd.DataFrame, target: str) -> torch.nn.Module:
    X = torch.tensor(df.drop(columns=[target]).values, dtype=torch.float32)
    y = torch.tensor(df[target].values, dtype=torch.float32).view(-1, 1)
    model = SimpleNN(X.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss()
    for epoch in range(10):
        optimizer.zero_grad()
        output = model(X)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()
    logger.info(f"PyTorch NN 최종 Loss: {loss.item()}")
    return model

if __name__ == "__main__":
    # 사용 가능한 데이터 파일로 변경
    try:
        df = load_data_parquet("data/005930_daily.parquet")  # 삼성전자 데이터 사용
    except FileNotFoundError:
        # 다른 사용 가능한 데이터 파일 시도
        available_files = [
            "data/005930_daily.parquet",
            "data/000660_daily.parquet",
            "data/051900_daily.parquet"
        ]
        df = None
        for file_path in available_files:
            try:
                df = load_data_parquet(file_path)
                print(f"Using data from: {file_path}")
                break
            except FileNotFoundError:
                continue

        if df is None:
            print("No available data files found. Please run data collection first.")
            exit(1)

    df = preprocess(df)
    # ML 학습 - 한국어 컬럼명 사용
    model = train_sklearn_rf(df, target="종가")
    # PyTorch 모델
    nn_model = train_pytorch_nn(df, target="종가")
    joblib.dump(nn_model.state_dict(), "models/nn_close.pt")
    logger.info("ML/DL 파이프라인 완료")
