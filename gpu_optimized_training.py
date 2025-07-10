#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: gpu_optimized_training.py
모듈: GPU 최적화 딥러닝 훈련 시스템
목적: GPU 자원을 적극적으로 활용한 대용량 데이터 딥러닝 훈련

Author: AI Assistant
Created: 2025-01-27
Modified: 2025-01-27
Version: 1.0.0

Dependencies:
    - Python 3.11+
    - torch, torchvision, torchaudio
    - cudf, cupy (GPU 가속)
    - dask-cuda (GPU 분산 처리)
    - ray[tune] (하이퍼파라미터 튜닝)
    - optuna (최적화)
"""

from __future__ import annotations

import json
import logging
import os
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

try:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
    NP_AVAILABLE = True
    PD_AVAILABLE = True
    SEABORN_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    NP_AVAILABLE = False
    PD_AVAILABLE = False
    SEABORN_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.cuda.amp import autocast, GradScaler
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from torch.optim import Lion
    LION_AVAILABLE = True
except ImportError:
    LION_AVAILABLE = False

try:
    import cudf
    import cupy as cp
    CUDF_AVAILABLE = True
    CUPY_AVAILABLE = True
except ImportError:
    CUDF_AVAILABLE = False
    CUPY_AVAILABLE = False

try:
    import dask
    from dask.distributed import Client
    from dask_cuda import LocalCUDACluster
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

try:
    import ray
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.search.optuna import OptunaSearch
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

warnings.filterwarnings('ignore')

# GPU 라이브러리
if TORCH_AVAILABLE and CUDF_AVAILABLE and CUPY_AVAILABLE:
    GPU_AVAILABLE = True
    print("✅ GPU 라이브러리 로드 성공")
else:
    GPU_AVAILABLE = False
    print("⚠️ GPU 라이브러리 로드 실패")

# 하이퍼파라미터 튜닝
if RAY_AVAILABLE and OPTUNA_AVAILABLE:
    TUNING_AVAILABLE = True
else:
    TUNING_AVAILABLE = False

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gpu_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class GPUTrainingConfig:
    """GPU 훈련 설정"""
    # GPU 설정
    use_gpu: bool = GPU_AVAILABLE
    gpu_memory_fraction: float = 0.9
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 4

    # 모델 설정
    model_type: str = "lstm"  # lstm, transformer, cnn
    hidden_size: int = 256
    num_layers: int = 4
    dropout: float = 0.2

    # 훈련 설정
    batch_size: int = 1024
    learning_rate: float = 1e-3
    num_epochs: int = 100
    early_stopping_patience: int = 10

    # 데이터 설정
    sequence_length: int = 60
    prediction_horizon: int = 5
    train_split: float = 0.8
    validation_split: float = 0.1

    # 최적화 설정
    optimizer: str = "adamw"  # adam, adamw, lion
    scheduler: str = "cosine"  # cosine, step, plateau
    weight_decay: float = 1e-4

    # 하이퍼파라미터 튜닝
    enable_tuning: bool = TUNING_AVAILABLE
    num_trials: int = 50
    max_concurrent_trials: int = 4


class AdvancedLSTM(nn.Module):
    """고급 LSTM 모델"""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 output_size: int, dropout: float = 0.2) -> None:
        super(AdvancedLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM 레이어
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )

        # 어텐션 메커니즘
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,  # bidirectional
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )

        # 출력 레이어
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )

        # 배치 정규화
        self.batch_norm = nn.BatchNorm1d(hidden_size * 2)

        # 가중치 초기화
        self._init_weights()

    def _init_weights(self) -> None:
        """가중치 초기화"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LSTM 순전파
        lstm_out, (hidden, cell) = self.lstm(x)

        # 어텐션 적용
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # 마지막 시퀀스 출력
        last_output = attn_out[:, -1, :]

        # 배치 정규화
        last_output = self.batch_norm(last_output)

        # 출력 레이어
        output = self.fc_layers(last_output)

        return output


class TransformerModel(nn.Module):
    """Transformer 모델"""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 output_size: int, num_heads: int = 8, dropout: float = 0.2) -> None:
        super(TransformerModel, self).__init__()

        self.input_projection = nn.Linear(input_size, hidden_size)
        self.positional_encoding = self._create_positional_encoding(hidden_size, 1000)

        # Transformer 인코더
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # 출력 레이어
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )

    def _create_positional_encoding(self, d_model: int, max_len: int) -> torch.Tensor:
        """위치 인코딩 생성"""
        if not NP_AVAILABLE:
            raise ImportError("numpy가 설치되지 않았습니다.")

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 입력 투영
        x = self.input_projection(x)

        # 위치 인코딩 추가
        seq_len = x.size(1)
        x = x + self.positional_encoding[:, :seq_len, :].to(x.device)

        # Transformer 인코더
        transformer_out = self.transformer(x)

        # 마지막 시퀀스 출력
        last_output = transformer_out[:, -1, :]

        # 출력 투영
        output = self.output_projection(last_output)

        return output


class GPUOptimizedTrainer:
    """GPU 최적화 훈련기"""

    def __init__(self, config: GPUTrainingConfig) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch가 설치되지 않았습니다.")

        self.config = config
        self.device = self._setup_device()

        # 모델 및 옵티마이저
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler: Optional[Any] = None
        self.scaler: Optional[GradScaler] = GradScaler() if config.mixed_precision else None

        # 훈련 상태
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }

        # 성능 모니터링
        self.performance_stats = {
            'training_time': [],
            'gpu_memory_usage': [],
            'throughput': []
        }

        logger.info(f"GPU 최적화 훈련기 초기화 완료: {self.device}")

    def _setup_device(self) -> torch.device:
        """디바이스 설정"""
        if self.config.use_gpu and torch.cuda.is_available():
            device = torch.device('cuda')

            # GPU 메모리 설정
            torch.cuda.set_per_process_memory_fraction(self.config.gpu_memory_fraction)

            # GPU 정보 출력
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"GPU 사용: {gpu_name} ({gpu_memory:.1f}GB)")

            return device
        else:
            logger.info("CPU 모드로 실행")
            return torch.device('cpu')

    def create_model(self, input_size: int, output_size: int) -> nn.Module:
        """모델 생성"""
        if self.config.model_type == "lstm":
            model = AdvancedLSTM(
                input_size=input_size,
                hidden_size=self.config.hidden_size,
                num_layers=self.config.num_layers,
                output_size=output_size,
                dropout=self.config.dropout
            )
        elif self.config.model_type == "transformer":
            model = TransformerModel(
                input_size=input_size,
                hidden_size=self.config.hidden_size,
                num_layers=self.config.num_layers,
                output_size=output_size,
                dropout=self.config.dropout
            )
        else:
            raise ValueError(f"지원하지 않는 모델 타입: {self.config.model_type}")

        model = model.to(self.device)

        # 모델 정보 출력
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        logger.info(f"모델 생성 완료:")
        logger.info(f"  타입: {self.config.model_type}")
        logger.info(f"  총 파라미터: {total_params:,}")
        logger.info(f"  훈련 가능 파라미터: {trainable_params:,}")

        return model

    def setup_optimizer(self, model: nn.Module) -> None:
        """옵티마이저 설정"""
        # 옵티마이저 선택
        if self.config.optimizer == "adam":
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "adamw":
            self.optimizer = optim.AdamW(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "lion":
            # Lion 옵티마이저 (PyTorch 2.0+)
            if LION_AVAILABLE:
                self.optimizer = Lion(
                    model.parameters(),
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay
                )
            else:
                logger.warning("Lion 옵티마이저를 사용할 수 없습니다. AdamW로 대체합니다.")
                self.optimizer = optim.AdamW(
                    model.parameters(),
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay
                )

        # 스케줄러 설정
        if self.config.scheduler == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs
            )
        elif self.config.scheduler == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        elif self.config.scheduler == "plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5
            )

        logger.info(f"옵티마이저 설정 완료: {self.config.optimizer}")

    def prepare_data_gpu(self, data_path: str) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """GPU 최적화 데이터 준비"""
        if not PD_AVAILABLE:
            raise ImportError("pandas가 설치되지 않았습니다.")

        logger.info("GPU 최적화 데이터 준비 시작")

        # 데이터 로드
        if data_path.endswith('.parquet'):
            df = pd.read_parquet(data_path)
        elif data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        else:
            raise ValueError("지원하지 않는 데이터 형식")

        # GPU에서 데이터 전처리
        if self.config.use_gpu and GPU_AVAILABLE:
            df = self._preprocess_gpu(df)
        else:
            df = self._preprocess_cpu(df)

        # 시퀀스 데이터 생성
        X, y = self._create_sequences(df)

        # 데이터 분할
        train_size = int(len(X) * self.config.train_split)
        val_size = int(len(X) * self.config.validation_split)

        X_train = X[:train_size]
        y_train = y[:train_size]
        X_val = X[train_size:train_size + val_size]
        y_val = y[train_size:train_size + val_size]
        X_test = X[train_size + val_size:]
        y_test = y[train_size + val_size:]

        # GPU로 데이터 이동
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).to(self.device)
        X_val = torch.FloatTensor(X_val).to(self.device)
        y_val = torch.FloatTensor(y_val).to(self.device)
        X_test = torch.FloatTensor(X_test).to(self.device)
        y_test = torch.FloatTensor(y_test).to(self.device)

        # DataLoader 생성
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=4
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=4
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=4
        )

        logger.info(f"데이터 준비 완료:")
        logger.info(f"  훈련: {len(X_train):,} 샘플")
        logger.info(f"  검증: {len(X_val):,} 샘플")
        logger.info(f"  테스트: {len(X_test):,} 샘플")

        return train_loader, val_loader, test_loader

    def _preprocess_gpu(self, df: pd.DataFrame) -> pd.DataFrame:
        """GPU 기반 데이터 전처리"""
        if not CUDF_AVAILABLE:
            raise ImportError("cudf가 설치되지 않았습니다.")

        # cuDF로 변환
        gdf = cudf.DataFrame.from_pandas(df)

        # 기술적 지표 계산 (GPU)
        gdf = self._calculate_technical_indicators_gpu(gdf)

        # 정규화 (GPU)
        numeric_columns = gdf.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_columns:
            if col != 'target':
                mean_val = gdf[col].mean()
                std_val = gdf[col].std()
                gdf[col] = (gdf[col] - mean_val) / (std_val + 1e-8)

        # 결측값 처리
        gdf = gdf.fillna(0)

        # pandas로 변환
        return gdf.to_pandas()

    def _preprocess_cpu(self, df: pd.DataFrame) -> pd.DataFrame:
        """CPU 기반 데이터 전처리"""
        # 기술적 지표 계산
        df = self._calculate_technical_indicators_cpu(df)

        # 정규화
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_columns:
            if col != 'target':
                mean_val = df[col].mean()
                std_val = df[col].std()
                df[col] = (df[col] - mean_val) / (std_val + 1e-8)

        # 결측값 처리
        df = df.fillna(0)

        return df

    def _calculate_technical_indicators_gpu(self, gdf: 'cudf.DataFrame') -> 'cudf.DataFrame':
        """GPU 기반 기술적 지표 계산"""
        if 'close' in gdf.columns:
            # 이동평균
            gdf['ma_5'] = gdf['close'].rolling(window=5).mean()
            gdf['ma_20'] = gdf['close'].rolling(window=20).mean()
            gdf['ma_50'] = gdf['close'].rolling(window=50).mean()

            # RSI
            delta = gdf['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            gdf['rsi'] = 100 - (100 / (1 + rs))

            # MACD
            ema_12 = gdf['close'].ewm(span=12).mean()
            ema_26 = gdf['close'].ewm(span=26).mean()
            gdf['macd'] = ema_12 - ema_26
            gdf['macd_signal'] = gdf['macd'].ewm(span=9).mean()

            # 볼린저 밴드
            gdf['bb_middle'] = gdf['close'].rolling(window=20).mean()
            bb_std = gdf['close'].rolling(window=20).std()
            gdf['bb_upper'] = gdf['bb_middle'] + (bb_std * 2)
            gdf['bb_lower'] = gdf['bb_middle'] - (bb_std * 2)

        return gdf

    def _calculate_technical_indicators_cpu(self, df: pd.DataFrame) -> pd.DataFrame:
        """CPU 기반 기술적 지표 계산"""
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
            ema_12 = df['close'].ewm(span=12).mean()
            ema_26 = df['close'].ewm(span=26).mean()
            df['macd'] = ema_12 - ema_26
            df['macd_signal'] = df['macd'].ewm(span=9).mean()

            # 볼린저 밴드
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)

        return df

    def _create_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """시퀀스 데이터 생성"""
        if not NP_AVAILABLE:
            raise ImportError("numpy가 설치되지 않았습니다.")

        # 특성 컬럼 선택
        feature_columns = [col for col in df.columns if col not in ['symbol', 'date', 'timestamp']]

        # 시퀀스 생성
        X, y = [], []

        for i in range(self.config.sequence_length, len(df) - self.config.prediction_horizon + 1):
            # 입력 시퀀스
            sequence = df[feature_columns].iloc[i-self.config.sequence_length:i].values
            X.append(sequence)

            # 타겟 (미래 가격)
            if 'close' in df.columns:
                target = df['close'].iloc[i:i+self.config.prediction_horizon].values
            else:
                target = df[feature_columns[0]].iloc[i:i+self.config.prediction_horizon].values
            y.append(target)

        return np.array(X), np.array(y)

    def train_epoch(self, model: nn.Module, train_loader: DataLoader) -> Tuple[float, float]:
        """한 에포크 훈련"""
        if not self.optimizer:
            raise ValueError("옵티마이저가 초기화되지 않았습니다.")

        model.train()
        total_loss = 0.0
        total_samples = 0

        start_time = time.time()

        for batch_idx, (data, target) in enumerate(train_loader):
            # 그래디언트 초기화
            self.optimizer.zero_grad()

            # Mixed Precision 훈련
            if self.config.mixed_precision and self.scaler:
                with autocast():
                    output = model(data)
                    loss = F.mse_loss(output, target)

                # 스케일된 역전파
                self.scaler.scale(loss).backward()

                # 그래디언트 누적
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                output = model(data)
                loss = F.mse_loss(output, target)
                loss.backward()

                # 그래디언트 누적
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    self.optimizer.step()

            # 통계 업데이트
            total_loss += loss.item() * data.size(0)
            total_samples += data.size(0)

            # 진행률 출력
            if batch_idx % 100 == 0:
                logger.info(f"  배치 {batch_idx}/{len(train_loader)}: Loss={loss.item():.6f}")

        # 남은 그래디언트 업데이트
        if self.config.mixed_precision and self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        epoch_time = time.time() - start_time
        avg_loss = total_loss / total_samples

        # 성능 통계 업데이트
        self.performance_stats['training_time'].append(epoch_time)
        if self.config.use_gpu:
            gpu_memory = torch.cuda.memory_allocated() / 1e9
            self.performance_stats['gpu_memory_usage'].append(gpu_memory)

        throughput = total_samples / epoch_time
        self.performance_stats['throughput'].append(throughput)

        return avg_loss, throughput

    def validate(self, model: nn.Module, val_loader: DataLoader) -> Tuple[float, float]:
        """검증"""
        model.eval()
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                loss = F.mse_loss(output, target)

                total_loss += loss.item() * data.size(0)
                total_samples += data.size(0)

        avg_loss = total_loss / total_samples

        return avg_loss, 0.0

    def train(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader) -> nn.Module:
        """전체 훈련 과정"""
        logger.info("🚀 GPU 최적화 훈련 시작")

        self.model = model
        self.setup_optimizer(model)

        best_model_state = None

        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch

            # 훈련
            train_loss, train_throughput = self.train_epoch(model, train_loader)

            # 검증
            val_loss, _ = self.validate(model, val_loader)

            # 스케줄러 업데이트
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # 히스토리 업데이트
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)

            # 로그 출력
            if self.optimizer:
                current_lr = self.optimizer.param_groups[0]['lr']
                logger.info(f"에포크 {epoch+1}/{self.config.num_epochs}:")
                logger.info(f"  훈련 손실: {train_loss:.6f}")
                logger.info(f"  검증 손실: {val_loss:.6f}")
                logger.info(f"  학습률: {current_lr:.2e}")
                logger.info(f"  처리량: {train_throughput:.0f} 샘플/초")

            if self.config.use_gpu:
                gpu_memory = torch.cuda.memory_allocated() / 1e9
                logger.info(f"  GPU 메모리: {gpu_memory:.2f}GB")

            # Early Stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                self.patience_counter += 1

                if self.patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break

        # 최고 모델 복원
        if best_model_state:
            model.load_state_dict(best_model_state)

        # 성능 통계 출력
        self._print_performance_stats()

        return model

    def _print_performance_stats(self) -> None:
        """성능 통계 출력"""
        if not NP_AVAILABLE:
            logger.warning("numpy가 설치되지 않아 성능 통계를 계산할 수 없습니다.")
            return

        avg_training_time = np.mean(self.performance_stats['training_time'])
        avg_throughput = np.mean(self.performance_stats['throughput'])

        logger.info("🎯 GPU 훈련 성능 통계:")
        logger.info(f"  평균 에포크 시간: {avg_training_time:.2f}초")
        logger.info(f"  평균 처리량: {avg_throughput:.0f} 샘플/초")

        if self.config.use_gpu and self.performance_stats['gpu_memory_usage']:
            avg_gpu_memory = np.mean(self.performance_stats['gpu_memory_usage'])
            max_gpu_memory = np.max(self.performance_stats['gpu_memory_usage'])
            logger.info(f"  평균 GPU 메모리: {avg_gpu_memory:.2f}GB")
            logger.info(f"  최대 GPU 메모리: {max_gpu_memory:.2f}GB")

        if self.config.mixed_precision:
            logger.info("  Mixed Precision: 활성화")
        else:
            logger.info("  Mixed Precision: 비활성화")

    def save_model(self, model: nn.Module, filepath: str) -> None:
        """모델 저장"""
        if not self.optimizer:
            raise ValueError("옵티마이저가 초기화되지 않았습니다.")

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config,
            'training_history': self.training_history,
            'performance_stats': self.performance_stats,
            'best_val_loss': self.best_val_loss
        }, filepath)

        logger.info(f"모델 저장 완료: {filepath}")

    def load_model(self, model: nn.Module, filepath: str) -> None:
        """모델 로드"""
        if not self.optimizer:
            raise ValueError("옵티마이저가 초기화되지 않았습니다.")

        checkpoint = torch.load(filepath, map_location=self.device)

        model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.training_history = checkpoint['training_history']
        self.performance_stats = checkpoint['performance_stats']
        self.best_val_loss = checkpoint['best_val_loss']

        logger.info(f"모델 로드 완료: {filepath}")


def run_hyperparameter_tuning(config: GPUTrainingConfig, data_path: str) -> None:
    """하이퍼파라미터 튜닝 실행"""
    if not config.enable_tuning:
        logger.warning("하이퍼파라미터 튜닝이 비활성화되어 있습니다.")
        return

    if not RAY_AVAILABLE or not OPTUNA_AVAILABLE:
        logger.warning("Ray 또는 Optuna가 설치되지 않아 하이퍼파라미터 튜닝을 건너뜁니다.")
        return

    logger.info("🔍 하이퍼파라미터 튜닝 시작")

    # Ray 초기화
    ray.init(num_cpus=config.max_concurrent_trials)

    # 검색 공간 정의
    search_space = {
        "learning_rate": tune.loguniform(1e-5, 1e-2),
        "hidden_size": tune.choice([128, 256, 512]),
        "num_layers": tune.choice([2, 3, 4]),
        "dropout": tune.uniform(0.1, 0.5),
        "batch_size": tune.choice([512, 1024, 2048])
    }

    # 스케줄러 설정
    scheduler = ASHAScheduler(
        time_attr="training_iteration",
        metric="val_loss",
        mode="min",
        max_t=config.num_epochs,
        grace_period=10,
        reduction_factor=2
    )

    # 검색 알고리즘
    search_alg = OptunaSearch(metric="val_loss", mode="min")

    # 튜닝 실행
    analysis = tune.run(
        lambda trial_config: train_with_config(trial_config, data_path),
        config=search_space,
        num_samples=config.num_trials,
        scheduler=scheduler,
        search_alg=search_alg,
        resources_per_trial={"cpu": 1, "gpu": 1 if config.use_gpu else 0}
    )

    # 최고 결과 출력
    best_trial = analysis.get_best_trial("val_loss", "min")
    logger.info(f"최고 하이퍼파라미터: {best_trial.config}")
    logger.info(f"최고 검증 손실: {best_trial.last_result['val_loss']}")

    ray.shutdown()


def train_with_config(trial_config: Dict[str, Any], data_path: str) -> None:
    """튜닝용 훈련 함수"""
    # 설정 업데이트
    config = GPUTrainingConfig()
    for key, value in trial_config.items():
        setattr(config, key, value)

    # 훈련 실행
    trainer = GPUOptimizedTrainer(config)
    train_loader, val_loader, _ = trainer.prepare_data_gpu(data_path)

    # 모델 생성
    input_size = train_loader.dataset[0][0].shape[-1]
    output_size = train_loader.dataset[0][1].shape[-1]
    model = trainer.create_model(input_size, output_size)

    # 훈련
    for epoch in range(config.num_epochs):
        train_loss, _ = trainer.train_epoch(model, train_loader)
        val_loss, _ = trainer.validate(model, val_loader)

        # Ray Tune에 결과 보고
        tune.report(
            train_loss=train_loss,
            val_loss=val_loss,
            training_iteration=epoch + 1
        )


async def main() -> None:
    """메인 함수"""
    print("🚀 GPU 최적화 딥러닝 훈련 시스템 시작")
    print("=" * 60)

    # 설정
    config = GPUTrainingConfig()

    # 데이터 경로 (실제 데이터 경로로 변경 필요)
    data_path = "optimized_data_20250127_120000.parquet"

    if not os.path.exists(data_path):
        print(f"⚠️ 데이터 파일을 찾을 수 없습니다: {data_path}")
        print("먼저 optimized_data_pipeline.py를 실행하여 데이터를 생성하세요.")
        return

    try:
        # 훈련기 생성
        trainer = GPUOptimizedTrainer(config)

        # 데이터 준비
        train_loader, val_loader, test_loader = trainer.prepare_data_gpu(data_path)

        # 모델 생성
        input_size = train_loader.dataset[0][0].shape[-1]
        output_size = train_loader.dataset[0][1].shape[-1]
        model = trainer.create_model(input_size, output_size)

        # 훈련 실행
        trained_model = trainer.train(model, train_loader, val_loader)

        # 모델 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"gpu_optimized_model_{timestamp}.pth"
        trainer.save_model(trained_model, model_path)

        # 하이퍼파라미터 튜닝 (선택사항)
        if config.enable_tuning:
            run_hyperparameter_tuning(config, data_path)

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        logger.error(f"훈련 실패: {e}")
    finally:
        print("✅ GPU 최적화 훈련 완료")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

