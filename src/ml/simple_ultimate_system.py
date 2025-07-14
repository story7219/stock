#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Ultimate Trading AI System
RTX 5080 + i9-14900KF 환경 최적화 버전

목표:
- Sharpe Ratio 5.0+
- 승률 85%+
- 연수익률 300%+
- GPU/CPU 95%+ 활용
"""

import asyncio
import logging
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset
import tensorflow as tf
from tensorflow import keras
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import mean_absolute_error
import optuna
from concurrent.futures import ThreadPoolExecutor
import ProcessPoolExecutor
import multiprocessing as mp
import time
import gc
import os
import psutil
import GPUtil
from pathlib import Path
from typing import Dict
import List, Tuple, Optional, Any
import pickle
import joblib

# GPU 메모리 최적화
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

# TensorFlow GPU 설정
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU 설정 오류: {e}")

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ultimate_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SystemMonitor:
    """시스템 리소스 모니터링"""

    def __init__(self):
        self.start_time = time.time()

    def get_system_info(self) -> Dict[str, Any]:
        """시스템 정보 수집"""
        info = {
            'cpu_count': mp.cpu_count(),
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_total': psutil.virtual_memory().total / (1024**3),
            'memory_used': psutil.virtual_memory().used / (1024**3),
            'memory_percent': psutil.virtual_memory().percent,
        }

        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                info.update({
                    'gpu_name': gpu.name,
                    'gpu_memory_total': gpu.memoryTotal,
                    'gpu_memory_used': gpu.memoryUsed,
                    'gpu_memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100,
                    'gpu_temperature': gpu.temperature,
                    'gpu_load': gpu.load * 100
                })
        except Exception as e:
            logger.warning(f"GPU 정보 수집 실패: {e}")

        return info

    def log_system_status(self):
        """시스템 상태 로깅"""
        info = self.get_system_info()
        logger.info(f"🖥️  CPU: {info['cpu_percent']:.1f}% | "
                   f"RAM: {info['memory_percent']:.1f}% | "
                   f"GPU: {info.get('gpu_load', 0):.1f}%")


class DataProcessor:
    """데이터 로딩 및 전처리"""

    def __init__(self, data_path: str = "data/"):
        self.data_path = Path(data_path)
        self.scalers = {}

    async def load_data(self) -> pd.DataFrame:
        """비동기 데이터 로딩"""
        logger.info("📊 데이터 로딩 시작...")

        # 샘플 데이터 생성 (실제 환경에서는 실제 데이터 로드)
        dates = pd.date_range('2020-01-01', '2024-12-31', freq='D')
        n = len(dates)

        # 가격 데이터 시뮬레이션
        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.02, n)
        price = 100 * np.exp(np.cumsum(returns))

        data = pd.DataFrame({
            'date': dates,
            'open': price * np.random.uniform(0.99, 1.01, n),
            'high': price * np.random.uniform(1.00, 1.05, n),
            'low': price * np.random.uniform(0.95, 1.00, n),
            'close': price,
            'volume': np.random.randint(1000000, 10000000, n),
        })

        logger.info(f"✅ 데이터 로딩 완료: {len(data):,} 행")
        return data

    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """고급 피처 생성"""
        logger.info("🔧 피처 엔지니어링 시작...")

        df = data.copy()

        # 기본 가격 피처
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['price_change'] = df['close'] - df['open']
        df['price_range'] = df['high'] - df['low']
        df['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
        df['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']

        # 이동평균
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'ma_{period}'] = df['close'].rolling(period).mean()
            df[f'ma_ratio_{period}'] = df['close'] / df[f'ma_{period}']

        # 볼린저 밴드
        for period in [20, 50]:
            ma = df['close'].rolling(period).mean()
            std = df['close'].rolling(period).std()
            df[f'bb_upper_{period}'] = ma + (2 * std)
            df[f'bb_lower_{period}'] = ma - (2 * std)
            df[f'bb_ratio_{period}'] = (df['close'] - ma) / std

        # RSI
        for period in [14, 30]:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / loss
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        # 변동성 지표
        for period in [10, 20, 30]:
            df[f'volatility_{period}'] = df['returns'].rolling(period).std()
            df[f'atr_{period}'] = df['price_range'].rolling(period).mean()

        # 모멘텀 지표
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
            df[f'roc_{period}'] = df['close'].pct_change(period)

        # 거래량 지표
        df['volume_ma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma_20']
        df['price_volume'] = df['close'] * df['volume']

        # 고급 피처
        df['williams_r'] = ((df['high'].rolling(14).max() - df['close']) /
                           (df['high'].rolling(14).max() - df['low'].rolling(14).min())) * -100

        # 시간 기반 피처
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter

        # 타겟 변수 (다음 날 수익률)
        df['target'] = df['returns'].shift(-1)

        # NaN 제거
        df = df.dropna()

        logger.info(f"✅ 피처 생성 완료: {df.shape[1]} 개 피처")
        return df

    def scale_features(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """피처 스케일링"""
        feature_cols = [col for col in train_data.columns if col not in ['date', 'target']]

        scaler = RobustScaler()
        train_scaled = train_data.copy()
        test_scaled = test_data.copy()

        train_scaled[feature_cols] = scaler.fit_transform(train_data[feature_cols])
        test_scaled[feature_cols] = scaler.transform(test_data[feature_cols])

        self.scalers['features'] = scaler
        return train_scaled, test_scaled


class AdvancedLSTM(nn.Module):
    """고급 LSTM 모델"""

    def __init__(self, input_size: int, hidden_size: int = 256, num_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout, bidirectional=True
        )

        self.attention = nn.MultiheadAttention(hidden_size * 2, num_heads=8, dropout=dropout)
        self.layer_norm = nn.LayerNorm(hidden_size * 2)

        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)

        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.layer_norm(attn_out + lstm_out)

        # 마지막 시퀀스만 사용
        last_out = attn_out[:, -1, :]
        output = self.fc_layers(last_out)

        return output


class TransformerModel(nn.Module):
    """Transformer 기반 모델"""

    def __init__(self, input_size: int, d_model: int = 256, nhead: int = 8, num_layers: int = 6):
        super().__init__()
        self.d_model = d_model

        self.input_projection = nn.Linear(input_size, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(1000, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x):
        seq_len = x.size(1)
        x = self.input_projection(x)
        x += self.positional_encoding[:seq_len, :].unsqueeze(0)

        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling

        return self.output_projection(x)


class EnsembleTrainer:
    """앙상블 모델 트레이너"""

    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.models = {}
        self.monitor = SystemMonitor()

    async def train_pytorch_models(self, X_train: np.ndarray, y_train: np.ndarray,
                                 X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """PyTorch 모델들 훈련"""
        logger.info("🔥 PyTorch 모델 훈련 시작...")

        # 데이터를 텐서로 변환
        sequence_length = 60
        X_train_seq = self._create_sequences(X_train, sequence_length)
        X_val_seq = self._create_sequences(X_val, sequence_length)
        y_train_seq = y_train[sequence_length:]
        y_val_seq = y_val[sequence_length:]

        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_seq).to(self.device),
            torch.FloatTensor(y_train_seq).to(self.device)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val_seq).to(self.device),
            torch.FloatTensor(y_val_seq).to(self.device)
        )

        train_loader = TorchDataLoader(train_dataset, batch_size=512, shuffle=True)
        val_loader = TorchDataLoader(val_dataset, batch_size=512, shuffle=False)

        input_size = X_train.shape[1]
        models_config = {
            'lstm': AdvancedLSTM(input_size),
            'transformer': TransformerModel(input_size)
        }

        results = {}

        for name, model in models_config.items():
            logger.info(f"🚀 {name.upper()} 모델 훈련 중...")
            model = model.to(self.device)

            optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
            criterion = nn.MSELoss()

            best_val_loss = float('inf')
            patience_counter = 0

            for epoch in range(100):
                # 훈련
                model.train()
                train_loss = 0
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X).squeeze()
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    train_loss += loss.item()

                # 검증
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = model(batch_X).squeeze()
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()

                train_loss /= len(train_loader)
                val_loss /= len(val_loader)
                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # 모델 저장
                    torch.save(model.state_dict(), f'models/{name}_best.pth')
                else:
                    patience_counter += 1

                if epoch % 10 == 0:
                    logger.info(f"{name} Epoch {epoch}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
                    self.monitor.log_system_status()

                if patience_counter >= 20:
                    logger.info(f"{name} 조기 종료: {epoch} 에포크")
                    break

            self.models[name] = model
            results[name] = {'best_val_loss': best_val_loss}

            # GPU 메모리 정리
            torch.cuda.empty_cache()
            gc.collect()

        return results

    def _create_sequences(self, data: np.ndarray, sequence_length: int) -> np.ndarray:
        """시계열 시퀀스 생성"""
        sequences = []
        for i in range(sequence_length, len(data)):
            sequences.append(data[i-sequence_length:i])
        return np.array(sequences)

    async def train_tree_models(self, X_train: np.ndarray, y_train: np.ndarray,
                              X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """트리 기반 모델들 훈련"""
        logger.info("🌳 트리 모델 훈련 시작...")

        models_config = {
            'xgboost': xgb.XGBRegressor(
                n_estimators=1000,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                tree_method='gpu_hist' if torch.cuda.is_available() else 'hist',
                gpu_id=0 if torch.cuda.is_available() else None
            ),
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=1000,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                device='gpu' if torch.cuda.is_available() else 'cpu',
                verbose=-1
            ),
            'random_forest': RandomForestRegressor(
                n_estimators=500,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        }

        results = {}

        for name, model in models_config.items():
            logger.info(f"🚀 {name.upper()} 모델 훈련 중...")
            start_time = time.time()

            # 훈련
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)] if hasattr(model, 'fit') and name in ['xgboost', 'lightgbm'] else None,
                early_stopping_rounds=50 if name in ['xgboost', 'lightgbm'] else None,
                verbose=False
            )

            # 예측 및 평가
            val_pred = model.predict(X_val)
            val_mse = mean_squared_error(y_val, val_pred)
            val_mae = mean_absolute_error(y_val, val_pred)

            training_time = time.time() - start_time
            logger.info(f"{name} 훈련 완료: MSE={val_mse:.6f}, MAE={val_mae:.6f}, 시간={training_time:.1f}초")

            # 모델 저장
            joblib.dump(model, f'models/{name}_model.pkl')

            self.models[name] = model
            results[name] = {
                'val_mse': val_mse,
                'val_mae': val_mae,
                'training_time': training_time
            }

            self.monitor.log_system_status()

        return results

    async def optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """Optuna를 사용한 하이퍼파라미터 최적화"""
        logger.info("🎯 하이퍼파라미터 최적화 시작...")

        def objective_xgb(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'random_state': 42
            }

            tscv = TimeSeriesSplit(n_splits=5)
            scores = []

            for train_idx, val_idx in tscv.split(X_train):
                X_tr, X_vl = X_train[train_idx], X_train[val_idx]
                y_tr, y_vl = y_train[train_idx], y_train[val_idx]

                model = xgb.XGBRegressor(**params)
                model.fit(X_tr, y_tr)
                pred = model.predict(X_vl)
                score = mean_squared_error(y_vl, pred)
                scores.append(score)

            return np.mean(scores)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective_xgb, n_trials=50)

        logger.info(f"✅ 최적화 완료: 최적 스코어 = {study.best_value:.6f}")
        return {'best_params': study.best_params, 'best_score': study.best_value}


class TradingSystem:
    """메인 트레이딩 시스템"""

    def __init__(self):
        self.data_loader = DataProcessor()
        self.trainer = EnsembleTrainer()
        self.monitor = SystemMonitor()

        # 모델 저장 디렉토리 생성
        os.makedirs('models', exist_ok=True)

    async def run(self):
        """시스템 실행"""
        logger.info("🚀 Ultimate Trading AI System 시작!")
        self.monitor.log_system_status()

        # 1. 데이터 로딩
        data = await self.data_loader.load_data()

        # 2. 피처 엔지니어링
        featured_data = self.data_loader.create_features(data)

        # 3. 훈련/검증 분할
        split_idx = int(len(featured_data) * 0.8)
        train_data = featured_data[:split_idx]
        test_data = featured_data[split_idx:]

        # 4. 데이터 스케일링
        train_scaled, test_scaled = self.data_loader.scale_features(train_data, test_data)

        # 5. 피처/타겟 분리
        feature_cols = [col for col in train_scaled.columns if col not in ['date', 'target']]
        X_train, y_train = train_scaled[feature_cols].values, train_scaled['target'].values
        X_test, y_test = test_scaled[feature_cols].values, test_scaled['target'].values

        # 6. 하이퍼파라미터 최적화
        optimization_results = await self.trainer.optimize_hyperparameters(X_train, y_train)
        logger.info(f"🎯 최적화 결과: {optimization_results}")

        # 7. 모델 훈련
        split_val = int(len(X_train) * 0.9)
        X_train_final, X_val = X_train[:split_val], X_train[split_val:]
        y_train_final, y_val = y_train[:split_val], y_train[split_val:]

        # PyTorch 모델 훈련
        pytorch_results = await self.trainer.train_pytorch_models(
            X_train_final, y_train_final, X_val, y_val
        )

        # 트리 모델 훈련
        tree_results = await self.trainer.train_tree_models(
            X_train_final, y_train_final, X_val, y_val
        )

        # 8. 결과 평가
        logger.info("📊 최종 결과:")
        logger.info(f"PyTorch 모델: {pytorch_results}")
        logger.info(f"트리 모델: {tree_results}")

        # 9. 시스템 통계
        self.monitor.log_system_status()
        total_time = time.time() - self.monitor.start_time
        logger.info(f"⏱️  총 실행 시간: {total_time:.1f}초")

        logger.info("🎉 Ultimate Trading AI System 완료!")


async def main():
    """메인 함수"""
    system = TradingSystem()
    await system.run()


def run_system():
    """시스템 실행 함수"""
    asyncio.run(main())


if __name__ == "__main__":
    # GPU가 사용 가능한지 확인
    if torch.cuda.is_available():
        logger.info(f"🔥 GPU 감지: {torch.cuda.get_device_name(0)}")
        logger.info(f"💾 GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    else:
        logger.info("⚠️  CPU 모드로 실행")

    try:
        run_system()
    except Exception as e:
        logger.error(f"❌ 시스템 오류: {e}")
        import traceback
        traceback.print_exc()
