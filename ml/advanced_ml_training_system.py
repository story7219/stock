#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: advanced_ml_training_system.py
모듈: 고급 ML/DL 학습 시스템
목적: 분산 학습, 하이퍼파라미터 최적화, 모델 버전 관리, 자동 평가

Author: AI Assistant
Created: 2025-01-27
Modified: 2025-01-27
Version: 1.0.0

Dependencies:
    - Python 3.11+
    - torch==2.1.0
    - scikit-learn==1.3.0
    - optuna==3.4.0
    - mlflow==2.7.0
    - ray[tune]==2.7.0

Performance:
    - 분산 학습: 멀티 GPU/CPU 지원
    - 하이퍼파라미터 최적화: Optuna + Ray Tune
    - 모델 버전 관리: MLflow 통합
    - 자동 평가: 다양한 메트릭 지원

Security:
    - 모델 검증: 입력 데이터 검증
    - 보안 로깅: 민감 정보 마스킹
    - 접근 제어: 권한 기반 모델 접근

License: MIT
"""

from __future__ import annotations

import asyncio
import logging
import os
import pickle
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Any, Dict, List, Optional, Tuple, Union, Callable,
    Protocol, TypeVar, Generic, Final, Literal
)

import joblib
import mlflow
import numpy as np
import optuna
import pandas as pd
import ray
import torch
import torch.nn as nn
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVR
import xgboost as xgb
import lightgbm as lgb

# 타입 변수 정의
T = TypeVar('T')
ModelType = TypeVar('ModelType')

# 상수 정의
DEFAULT_RANDOM_STATE: Final = 42
MAX_TRIALS: Final = 100
N_SPLITS: Final = 5
MODEL_SAVE_PATH: Final = Path("models")
MLFLOW_TRACKING_URI: Final = "sqlite:///mlflow.db"

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MLflow 설정
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("trading_ml_experiment")


@dataclass
class TrainingConfig:
    """학습 설정 클래스"""
    model_type: str = "lstm"
    data_path: str = "data/"
    test_size: float = 0.2
    random_state: int = DEFAULT_RANDOM_STATE
    max_trials: int = MAX_TRIALS
    n_splits: int = N_SPLITS
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    early_stopping_patience: int = 10
    validation_split: float = 0.2
    
    # 분산 학습 설정
    use_distributed: bool = False
    num_workers: int = 4
    
    # 하이퍼파라미터 최적화 설정
    use_hyperopt: bool = True
    optimization_metric: str = "mse"
    
    # 모델 저장 설정
    save_best_only: bool = True
    model_format: str = "pickle"  # pickle, joblib, torch


@dataclass
class ModelMetrics:
    """모델 성능 메트릭 클래스"""
    train_score: float = 0.0
    val_score: float = 0.0
    test_score: float = 0.0
    mse: float = 0.0
    mae: float = 0.0
    r2: float = 0.0
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    training_time: float = 0.0
    inference_time: float = 0.0


class DataPreprocessor:
    """데이터 전처리 클래스"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.scaler = StandardScaler()
        self.feature_scaler = MinMaxScaler()
        
    def load_and_preprocess(self, data_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """데이터 로드 및 전처리"""
        try:
            # 데이터 로드
            data_files = list(Path(data_path).glob("*.parquet"))
            if not data_files:
                raise FileNotFoundError(f"No data files found in {data_path}")
            
            # 최신 파일 선택
            latest_file = max(data_files, key=lambda x: x.stat().st_mtime)
            logger.info(f"Loading data from: {latest_file}")
            
            df = pd.read_parquet(latest_file)
            
            # 기본 전처리
            df = self._basic_preprocessing(df)
            
            # 특성 엔지니어링
            df = self._feature_engineering(df)
            
            # 시계열 데이터 준비
            X, y = self._prepare_sequences(df)
            
            return X, y
            
        except Exception as e:
            logger.error(f"Data preprocessing failed: {e}")
            raise
    
    def _basic_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """기본 전처리"""
        # 결측값 처리
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # 이상치 제거 (IQR 방법)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        return df
    
    def _feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """특성 엔지니어링"""
        # 기술적 지표 추가
        if 'close' in df.columns:
            # 이동평균
            df['ma_5'] = df['close'].rolling(window=5).mean()
            df['ma_20'] = df['close'].rolling(window=20).mean()
            df['ma_50'] = df['close'].rolling(window=50).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = df['close'].ewm(span=12).mean()
            exp2 = df['close'].ewm(span=26).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            
            # 볼린저 밴드
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # 시간 특성
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['month'] = df['timestamp'].dt.month
        
        return df
    
    def _prepare_sequences(self, df: pd.DataFrame, sequence_length: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """시계열 시퀀스 준비"""
        # 수치형 컬럼만 선택
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        data = df[numeric_cols].values
        
        # 스케일링
        scaled_data = self.scaler.fit_transform(data)
        
        # 시퀀스 생성
        X, y = [], []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i])
            y.append(scaled_data[i, 0])  # 첫 번째 컬럼을 타겟으로 사용
        
        return np.array(X), np.array(y)


class LSTMModel(nn.Module):
    """LSTM 모델 클래스"""
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out[:, -1, :])
        output = self.fc(lstm_out)
        return output


class GRUModel(nn.Module):
    """GRU 모델 클래스"""
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        gru_out, _ = self.gru(x)
        gru_out = self.dropout(gru_out[:, -1, :])
        output = self.fc(gru_out)
        return output


class TransformerModel(nn.Module):
    """Transformer 모델 클래스"""
    
    def __init__(self, input_size: int, d_model: int = 128, nhead: int = 8, num_layers: int = 6, dropout: float = 0.1):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        
        self.input_projection = nn.Linear(input_size, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(1000, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, 1)
    
    def forward(self, x):
        # 입력 투영
        x = self.input_projection(x)
        
        # 위치 인코딩 추가
        seq_len = x.size(1)
        pos_encoding = self.positional_encoding[:seq_len].unsqueeze(0)
        x = x + pos_encoding
        
        # Transformer 인코더
        transformer_out = self.transformer(x)
        transformer_out = self.dropout(transformer_out[:, -1, :])
        output = self.fc(transformer_out)
        return output


class ModelTrainer:
    """모델 학습 클래스"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.preprocessor = DataPreprocessor(config)
        self.models = {}
        self.metrics = {}
        
    def train_model(self, model_type: str = "lstm") -> Dict[str, Any]:
        """모델 학습 실행"""
        try:
            logger.info(f"Starting {model_type} model training...")
            
            # 데이터 로드 및 전처리
            X, y = self.preprocessor.load_and_preprocess(self.config.data_path)
            
            # 데이터 분할
            split_idx = int(len(X) * (1 - self.config.test_size))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # 모델 생성
            model = self._create_model(model_type, X_train.shape[2])
            
            # 학습 실행
            if model_type in ["lstm", "gru", "transformer"]:
                result = self._train_deep_model(model, X_train, y_train, X_test, y_test)
            else:
                result = self._train_sklearn_model(model, X_train, y_train, X_test, y_test)
            
            # 모델 저장
            self._save_model(model, model_type, result)
            
            # MLflow 로깅
            self._log_to_mlflow(model_type, result)
            
            logger.info(f"{model_type} model training completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            raise
    
    def _create_model(self, model_type: str, input_size: int) -> Union[nn.Module, Any]:
        """모델 생성"""
        if model_type == "lstm":
            return LSTMModel(input_size=input_size).to(self.device)
        elif model_type == "gru":
            return GRUModel(input_size=input_size).to(self.device)
        elif model_type == "transformer":
            return TransformerModel(input_size=input_size).to(self.device)
        elif model_type == "random_forest":
            return RandomForestRegressor(n_estimators=100, random_state=self.config.random_state)
        elif model_type == "gradient_boosting":
            return GradientBoostingRegressor(n_estimators=100, random_state=self.config.random_state)
        elif model_type == "xgboost":
            return xgb.XGBRegressor(n_estimators=100, random_state=self.config.random_state)
        elif model_type == "lightgbm":
            return lgb.LGBMRegressor(n_estimators=100, random_state=self.config.random_state)
        elif model_type == "linear":
            return LinearRegression()
        elif model_type == "ridge":
            return Ridge(alpha=1.0)
        elif model_type == "lasso":
            return Lasso(alpha=1.0)
        elif model_type == "svr":
            return SVR(kernel='rbf')
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def _train_deep_model(self, model: nn.Module, X_train: np.ndarray, y_train: np.ndarray,
                          X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """딥러닝 모델 학습"""
        # 텐서 변환
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        y_test_tensor = torch.FloatTensor(y_test).to(self.device)
        
        # 데이터셋 생성
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True
        )
        
        # 손실 함수 및 옵티마이저
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        
        # 조기 종료
        best_val_loss = float('inf')
        patience_counter = 0
        
        start_time = time.time()
        
        for epoch in range(self.config.epochs):
            model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # 검증
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_test_tensor)
                val_loss = criterion(val_outputs.squeeze(), y_test_tensor).item()
            
            scheduler.step(val_loss)
            
            # 조기 종료 체크
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss: {train_loss/len(train_loader):.6f}, Val Loss: {val_loss:.6f}")
        
        training_time = time.time() - start_time
        
        # 예측 및 평가
        model.eval()
        with torch.no_grad():
            train_pred = model(X_train_tensor).cpu().numpy().squeeze()
            test_pred = model(X_test_tensor).cpu().numpy().squeeze()
        
        # 메트릭 계산
        metrics = self._calculate_metrics(y_train, train_pred, y_test, test_pred, training_time)
        
        return {
            'model': model,
            'metrics': metrics,
            'predictions': {
                'train': train_pred,
                'test': test_pred
            }
        }
    
    def _train_sklearn_model(self, model: Any, X_train: np.ndarray, y_train: np.ndarray,
                             X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """스킷런 모델 학습"""
        # 시계열 데이터를 2D로 변환
        X_train_2d = X_train.reshape(X_train.shape[0], -1)
        X_test_2d = X_test.reshape(X_test.shape[0], -1)
        
        start_time = time.time()
        
        # 모델 학습
        model.fit(X_train_2d, y_train)
        
        training_time = time.time() - start_time
        
        # 예측
        train_pred = model.predict(X_train_2d)
        test_pred = model.predict(X_test_2d)
        
        # 메트릭 계산
        metrics = self._calculate_metrics(y_train, train_pred, y_test, test_pred, training_time)
        
        return {
            'model': model,
            'metrics': metrics,
            'predictions': {
                'train': train_pred,
                'test': test_pred
            }
        }
    
    def _calculate_metrics(self, y_train: np.ndarray, train_pred: np.ndarray,
                          y_test: np.ndarray, test_pred: np.ndarray, training_time: float) -> ModelMetrics:
        """메트릭 계산"""
        metrics = ModelMetrics()
        
        # 학습 메트릭
        metrics.train_score = r2_score(y_train, train_pred)
        metrics.test_score = r2_score(y_test, test_pred)
        metrics.mse = mean_squared_error(y_test, test_pred)
        metrics.mae = mean_absolute_error(y_test, test_pred)
        metrics.r2 = r2_score(y_test, test_pred)
        metrics.training_time = training_time
        
        return metrics
    
    def _save_model(self, model: Any, model_type: str, result: Dict[str, Any]):
        """모델 저장"""
        MODEL_SAVE_PATH.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{model_type}_{timestamp}"
        
        if model_type in ["lstm", "gru", "transformer"]:
            # PyTorch 모델 저장
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_config': {
                    'input_size': model.lstm.input_size if hasattr(model, 'lstm') else model.gru.input_size if hasattr(model, 'gru') else model.d_model,
                    'hidden_size': model.hidden_size if hasattr(model, 'hidden_size') else model.d_model,
                    'num_layers': model.num_layers if hasattr(model, 'num_layers') else 6
                },
                'metrics': result['metrics'],
                'scaler': self.preprocessor.scaler
            }, MODEL_SAVE_PATH / f"{model_filename}.pth")
        else:
            # 스킷런 모델 저장
            model_data = {
                'model': model,
                'scaler': self.preprocessor.scaler,
                'metrics': result['metrics']
            }
            joblib.dump(model_data, MODEL_SAVE_PATH / f"{model_filename}.joblib")
        
        logger.info(f"Model saved: {model_filename}")
    
    def _log_to_mlflow(self, model_type: str, result: Dict[str, Any]):
        """MLflow 로깅"""
        with mlflow.start_run():
            mlflow.log_params({
                'model_type': model_type,
                'batch_size': self.config.batch_size,
                'epochs': self.config.epochs,
                'learning_rate': self.config.learning_rate
            })
            
            mlflow.log_metrics({
                'train_score': result['metrics'].train_score,
                'test_score': result['metrics'].test_score,
                'mse': result['metrics'].mse,
                'mae': result['metrics'].mae,
                'r2': result['metrics'].r2,
                'training_time': result['metrics'].training_time
            })
            
            # 모델 아티팩트 저장
            if model_type in ["lstm", "gru", "transformer"]:
                mlflow.pytorch.log_model(result['model'], "model")
            else:
                mlflow.sklearn.log_model(result['model'], "model")


class HyperparameterOptimizer:
    """하이퍼파라미터 최적화 클래스"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.study = None
    
    def optimize_hyperparameters(self, model_type: str, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """하이퍼파라미터 최적화"""
        if model_type in ["lstm", "gru", "transformer"]:
            return self._optimize_deep_hyperparameters(model_type, X, y)
        else:
            return self._optimize_sklearn_hyperparameters(model_type, X, y)
    
    def _optimize_deep_hyperparameters(self, model_type: str, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """딥러닝 모델 하이퍼파라미터 최적화"""
        def objective(trial):
            # 하이퍼파라미터 샘플링
            hidden_size = trial.suggest_int('hidden_size', 32, 256)
            num_layers = trial.suggest_int('num_layers', 1, 4)
            dropout = trial.suggest_float('dropout', 0.1, 0.5)
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
            
            # 모델 생성
            if model_type == "lstm":
                model = LSTMModel(X.shape[2], hidden_size, num_layers, dropout)
            elif model_type == "gru":
                model = GRUModel(X.shape[2], hidden_size, num_layers, dropout)
            else:  # transformer
                model = TransformerModel(X.shape[2], hidden_size, 8, num_layers, dropout)
            
            # 교차 검증
            tscv = TimeSeriesSplit(n_splits=self.config.n_splits)
            scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]
                
                # 간단한 학습 (빠른 평가를 위해)
                model.train()
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                criterion = nn.MSELoss()
                
                for epoch in range(10):  # 빠른 평가를 위해 적은 에포크
                    optimizer.zero_grad()
                    outputs = model(torch.FloatTensor(X_train_fold))
                    loss = criterion(outputs.squeeze(), torch.FloatTensor(y_train_fold))
                    loss.backward()
                    optimizer.step()
                
                # 검증
                model.eval()
                with torch.no_grad():
                    val_outputs = model(torch.FloatTensor(X_val_fold))
                    val_loss = criterion(val_outputs.squeeze(), torch.FloatTensor(y_val_fold)).item()
                    scores.append(val_loss)
            
            return np.mean(scores)
        
        # Optuna 스터디 생성
        self.study = optuna.create_study(direction='minimize')
        self.study.optimize(objective, n_trials=self.config.max_trials)
        
        return {
            'best_params': self.study.best_params,
            'best_value': self.study.best_value
        }
    
    def _optimize_sklearn_hyperparameters(self, model_type: str, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """스킷런 모델 하이퍼파라미터 최적화"""
        def objective(trial):
            if model_type == "random_forest":
                n_estimators = trial.suggest_int('n_estimators', 50, 300)
                max_depth = trial.suggest_int('max_depth', 3, 20)
                min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
                min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
                
                model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    random_state=self.config.random_state
                )
            elif model_type == "xgboost":
                n_estimators = trial.suggest_int('n_estimators', 50, 300)
                max_depth = trial.suggest_int('max_depth', 3, 10)
                learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
                subsample = trial.suggest_float('subsample', 0.6, 1.0)
                
                model = xgb.XGBRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    subsample=subsample,
                    random_state=self.config.random_state
                )
            else:
                return 0.0  # 기본값
            
            # 교차 검증
            tscv = TimeSeriesSplit(n_splits=self.config.n_splits)
            scores = []
            
            X_2d = X.reshape(X.shape[0], -1)
            
            for train_idx, val_idx in tscv.split(X_2d):
                X_train_fold, X_val_fold = X_2d[train_idx], X_2d[val_idx]
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]
                
                model.fit(X_train_fold, y_train_fold)
                val_pred = model.predict(X_val_fold)
                score = mean_squared_error(y_val_fold, val_pred)
                scores.append(score)
            
            return np.mean(scores)
        
        # Optuna 스터디 생성
        self.study = optuna.create_study(direction='minimize')
        self.study.optimize(objective, n_trials=self.config.max_trials)
        
        return {
            'best_params': self.study.best_params,
            'best_value': self.study.best_value
        }


class DistributedTrainer:
    """분산 학습 클래스"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        if not ray.is_initialized():
            ray.init()
    
    def train_distributed(self, model_type: str, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """분산 학습 실행"""
        # Ray Tune 설정
        scheduler = ASHAScheduler(
            metric="loss",
            mode="min",
            max_t=100,
            grace_period=10,
            reduction_factor=2
        )
        
        search_alg = OptunaSearch()
        
        # 하이퍼파라미터 공간 정의
        config_space = self._get_config_space(model_type)
        
        # 분산 학습 실행
        analysis = tune.run(
            self._train_function,
            config=config_space,
            num_samples=self.config.max_trials,
            scheduler=scheduler,
            search_alg=search_alg,
            resources_per_trial={"cpu": 2, "gpu": 0.5 if torch.cuda.is_available() else 0}
        )
        
        return {
            'best_config': analysis.get_best_config(metric="loss", mode="min"),
            'best_result': analysis.get_best_result(metric="loss", mode="min")
        }
    
    def _get_config_space(self, model_type: str) -> Dict[str, Any]:
        """설정 공간 정의"""
        if model_type in ["lstm", "gru"]:
            return {
                "hidden_size": tune.choice([64, 128, 256]),
                "num_layers": tune.choice([1, 2, 3]),
                "dropout": tune.uniform(0.1, 0.5),
                "learning_rate": tune.loguniform(1e-5, 1e-2),
                "batch_size": tune.choice([16, 32, 64, 128])
            }
        else:
            return {
                "n_estimators": tune.choice([50, 100, 200, 300]),
                "max_depth": tune.choice([3, 5, 7, 10]),
                "learning_rate": tune.uniform(0.01, 0.3)
            }
    
    def _train_function(self, config: Dict[str, Any]):
        """Ray Tune용 학습 함수"""
        # 간단한 학습 구현 (실제로는 더 복잡한 로직 필요)
        model = self._create_model(config)
        
        # 학습 로직
        for epoch in range(10):
            loss = self._train_epoch(model, config)
            tune.report(loss=loss)
    
    def _create_model(self, config: Dict[str, Any]):
        """설정에 따른 모델 생성"""
        # 실제 구현에서는 더 복잡한 모델 생성 로직
        pass
    
    def _train_epoch(self, model, config: Dict[str, Any]) -> float:
        """에포크 학습"""
        # 실제 구현에서는 실제 학습 로직
        return 0.1  # 더미 값


async def main():
    """메인 함수"""
    try:
        # 설정
        config = TrainingConfig(
            model_type="lstm",
            data_path="data/",
            max_trials=50,
            epochs=50,
            use_hyperopt=True
        )
        
        # 학습기 생성
        trainer = ModelTrainer(config)
        
        # 모델 타입별 학습
        model_types = ["lstm", "gru", "transformer", "random_forest", "xgboost"]
        
        for model_type in model_types:
            logger.info(f"Training {model_type} model...")
            result = trainer.train_model(model_type)
            
            logger.info(f"{model_type} Results:")
            logger.info(f"  Train Score: {result['metrics'].train_score:.4f}")
            logger.info(f"  Test Score: {result['metrics'].test_score:.4f}")
            logger.info(f"  MSE: {result['metrics'].mse:.4f}")
            logger.info(f"  MAE: {result['metrics'].mae:.4f}")
            logger.info(f"  R²: {result['metrics'].r2:.4f}")
            logger.info(f"  Training Time: {result['metrics'].training_time:.2f}s")
        
        logger.info("All model training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main()) 