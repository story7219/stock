#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: ml_training_example.py
모듈: 머신러닝/딥러닝 학습 예제
목적: 다양한 ML/DL 모델 학습 및 예측 예제

Author: AI Trading System
Created: 2025-01-27
Version: 1.0.0

Dependencies:
    - Python 3.11+
    - tensorflow==2.19.0
    - torch==2.1.2
    - scikit-learn==1.7.0
    - xgboost==3.0.2
    - lightgbm==4.6.0
    - pandas==2.1.4
    - numpy==1.24.0

License: MIT
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# 머신러닝 라이브러리
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 딥러닝 라이브러리
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 부스팅 라이브러리
import xgboost as xgb
import lightgbm as lgb

# PyTorch (선택사항)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("⚠️ PyTorch를 사용할 수 없습니다.")


class MLTrainingExample:
    """머신러닝/딥러닝 학습 예제"""
    
    def __init__(self):
        """초기화"""
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        
    def generate_sample_data(self, n_samples: int = 1000) -> Tuple[pd.DataFrame, pd.Series]:
        """샘플 주식 데이터 생성"""
        np.random.seed(42)
        
        # 시간 인덱스 생성
        dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
        
        # 기본 가격 데이터 생성
        base_price = 100
        returns = np.random.normal(0.001, 0.02, n_samples)  # 일간 수익률
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # 기술적 지표 생성
        df = pd.DataFrame({
            'Date': dates,
            'Close': prices,
            'Volume': np.random.lognormal(10, 1, n_samples),
            'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'Open': [p * (1 + np.random.normal(0, 0.005)) for p in prices]
        })
        
        # 기술적 지표 추가
        df['MA5'] = df['Close'].rolling(5).mean()
        df['MA20'] = df['Close'].rolling(20).mean()
        df['RSI'] = self._calculate_rsi(df['Close'])
        df['Volatility'] = df['Close'].rolling(20).std()
        df['Price_Change'] = df['Close'].pct_change()
        df['Volume_Change'] = df['Volume'].pct_change()
        
        # 타겟 변수 (다음날 수익률)
        df['Target'] = df['Close'].shift(-1) / df['Close'] - 1
        
        # 결측치 제거
        df = df.dropna()
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """데이터 준비"""
        # 피처 선택
        feature_cols = ['Close', 'Volume', 'MA5', 'MA20', 'RSI', 'Volatility', 
                       'Price_Change', 'Volume_Change']
        
        X = df[feature_cols].values
        y = df['Target'].values
        
        # 스케일링
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y
    
    def train_traditional_ml_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """전통적인 머신러닝 모델 훈련"""
        print("🤖 전통적인 머신러닝 모델 훈련 중...")
        
        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'SVR': SVR(kernel='rbf', C=1.0, gamma='scale'),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42),
            'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            try:
                print(f"  📊 {name} 훈련 중...")
                
                # 모델 훈련
                model.fit(X_train, y_train)
                
                # 예측
                y_pred = model.predict(X_test)
                
                # 평가
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                results[name] = {
                    'model': model,
                    'mse': mse,
                    'mae': mae,
                    'r2': r2,
                    'predictions': y_pred
                }
                
                print(f"    ✅ {name}: MSE={mse:.6f}, MAE={mae:.6f}, R²={r2:.4f}")
                
            except Exception as e:
                print(f"    ❌ {name} 훈련 실패: {e}")
                continue
        
        self.models.update(models)
        self.results['traditional_ml'] = results
        
        return results
    
    def train_deep_learning_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """딥러닝 모델 훈련"""
        print("🧠 딥러닝 모델 훈련 중...")
        
        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 시계열 데이터로 변환 (LSTM용)
        sequence_length = 10
        X_train_seq, y_train_seq = self._create_sequences(X_train, y_train, sequence_length)
        X_test_seq, y_test_seq = self._create_sequences(X_test, y_test, sequence_length)
        
        models = {}
        results = {}
        
        # 1. Dense Neural Network
        print("  📊 Dense Neural Network 훈련 중...")
        dense_model = self._build_dense_model(X.shape[1])
        dense_history = dense_model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=50,
            batch_size=32,
            callbacks=[
                EarlyStopping(patience=10, restore_best_weights=True),
                ModelCheckpoint('best_dense_model.h5', save_best_only=True)
            ],
            verbose=0
        )
        
        y_pred_dense = dense_model.predict(X_test)
        results['Dense Neural Network'] = {
            'model': dense_model,
            'history': dense_history,
            'mse': mean_squared_error(y_test, y_pred_dense),
            'mae': mean_absolute_error(y_test, y_pred_dense),
            'r2': r2_score(y_test, y_pred_dense),
            'predictions': y_pred_dense
        }
        
        # 2. LSTM Model
        print("  📊 LSTM Model 훈련 중...")
        lstm_model = self._build_lstm_model(sequence_length, X.shape[1])
        lstm_history = lstm_model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_test_seq, y_test_seq),
            epochs=50,
            batch_size=32,
            callbacks=[
                EarlyStopping(patience=10, restore_best_weights=True),
                ModelCheckpoint('best_lstm_model.h5', save_best_only=True)
            ],
            verbose=0
        )
        
        y_pred_lstm = lstm_model.predict(X_test_seq)
        results['LSTM Model'] = {
            'model': lstm_model,
            'history': lstm_history,
            'mse': mean_squared_error(y_test_seq, y_pred_lstm),
            'mae': mean_absolute_error(y_test_seq, y_pred_lstm),
            'r2': r2_score(y_test_seq, y_pred_lstm),
            'predictions': y_pred_lstm
        }
        
        self.results['deep_learning'] = results
        
        return results
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """시계열 시퀀스 생성"""
        X_seq, y_seq = [], []
        
        for i in range(len(X) - sequence_length):
            X_seq.append(X[i:i+sequence_length])
            y_seq.append(y[i+sequence_length])
        
        return np.array(X_seq), np.array(y_seq)
    
    def _build_dense_model(self, input_dim: int) -> keras.Model:
        """Dense Neural Network 모델 구축"""
        model = Sequential([
            Dense(128, activation='relu', input_shape=(input_dim,)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def _build_lstm_model(self, sequence_length: int, input_dim: int) -> keras.Model:
        """LSTM 모델 구축"""
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(sequence_length, input_dim)),
            Dropout(0.3),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train_pytorch_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """PyTorch 모델 훈련 (선택사항)"""
        if not PYTORCH_AVAILABLE:
            print("⚠️ PyTorch를 사용할 수 없습니다.")
            return {}
        
        print("🔥 PyTorch 모델 훈련 중...")
        
        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 텐서 변환
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test)
        
        # 데이터로더 생성
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        # 모델 정의
        class PyTorchNN(nn.Module):
            def __init__(self, input_dim):
                super(PyTorchNN, self).__init__()
                self.fc1 = nn.Linear(input_dim, 128)
                self.fc2 = nn.Linear(128, 64)
                self.fc3 = nn.Linear(64, 32)
                self.fc4 = nn.Linear(32, 1)
                self.dropout = nn.Dropout(0.3)
                self.batch_norm1 = nn.BatchNorm1d(128)
                self.batch_norm2 = nn.BatchNorm1d(64)
                self.batch_norm3 = nn.BatchNorm1d(32)
                
            def forward(self, x):
                x = torch.relu(self.batch_norm1(self.fc1(x)))
                x = self.dropout(x)
                x = torch.relu(self.batch_norm2(self.fc2(x)))
                x = self.dropout(x)
                x = torch.relu(self.batch_norm3(self.fc3(x)))
                x = self.fc4(x)
                return x
        
        # 모델 훈련
        model = PyTorchNN(X.shape[1])
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        model.train()
        for epoch in range(50):
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
        
        # 예측
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test_tensor).squeeze().numpy()
        
        results = {
            'PyTorch Neural Network': {
                'model': model,
                'mse': mean_squared_error(y_test, y_pred),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred),
                'predictions': y_pred
            }
        }
        
        self.results['pytorch'] = results
        
        return results
    
    def compare_models(self) -> pd.DataFrame:
        """모델 성능 비교"""
        print("📊 모델 성능 비교 중...")
        
        comparison_data = []
        
        for category, results in self.results.items():
            for model_name, result in results.items():
                comparison_data.append({
                    'Category': category,
                    'Model': model_name,
                    'MSE': result['mse'],
                    'MAE': result['mae'],
                    'R²': result['r2']
                })
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # 성능별 정렬
        df_comparison = df_comparison.sort_values('R²', ascending=False)
        
        print("\n🏆 모델 성능 순위:")
        print(df_comparison.to_string(index=False))
        
        return df_comparison
    
    def plot_results(self):
        """결과 시각화"""
        print("📈 결과 시각화 중...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. R² 점수 비교
        comparison_df = self.compare_models()
        axes[0, 0].barh(comparison_df['Model'], comparison_df['R²'])
        axes[0, 0].set_title('Model R² Scores')
        axes[0, 0].set_xlabel('R² Score')
        
        # 2. MSE 비교
        axes[0, 1].barh(comparison_df['Model'], comparison_df['MSE'])
        axes[0, 1].set_title('Model MSE Scores')
        axes[0, 1].set_xlabel('MSE')
        
        # 3. 실제값 vs 예측값 (최고 성능 모델)
        best_model_name = comparison_df.iloc[0]['Model']
        best_category = comparison_df.iloc[0]['Category']
        best_predictions = self.results[best_category][best_model_name]['predictions']
        
        # 실제값 가져오기 (테스트 데이터)
        _, _, _, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        axes[1, 0].scatter(y_test, best_predictions, alpha=0.6)
        axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[1, 0].set_xlabel('Actual Values')
        axes[1, 0].set_ylabel('Predicted Values')
        axes[1, 0].set_title(f'Actual vs Predicted ({best_model_name})')
        
        # 4. 예측값 분포
        axes[1, 1].hist(best_predictions, bins=30, alpha=0.7, label='Predictions')
        axes[1, 1].hist(y_test, bins=30, alpha=0.7, label='Actual')
        axes[1, 1].set_xlabel('Values')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Distribution Comparison')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('ml_training_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_complete_example(self):
        """완전한 학습 예제 실행"""
        print("🚀 머신러닝/딥러닝 학습 예제 시작")
        print("=" * 50)
        
        # 1. 데이터 생성
        print("📊 샘플 데이터 생성 중...")
        df = self.generate_sample_data(1000)
        print(f"  ✅ {len(df)} 개의 데이터 포인트 생성 완료")
        
        # 2. 데이터 준비
        print("🔧 데이터 준비 중...")
        X, y = self.prepare_data(df)
        self.X, self.y = X, y
        print(f"  ✅ 피처: {X.shape[1]}개, 샘플: {X.shape[0]}개")
        
        # 3. 전통적인 머신러닝 모델 훈련
        print("\n🤖 전통적인 머신러닝 모델 훈련")
        self.train_traditional_ml_models(X, y)
        
        # 4. 딥러닝 모델 훈련
        print("\n🧠 딥러닝 모델 훈련")
        self.train_deep_learning_models(X, y)
        
        # 5. PyTorch 모델 훈련 (선택사항)
        print("\n🔥 PyTorch 모델 훈련")
        self.train_pytorch_models(X, y)
        
        # 6. 모델 비교
        print("\n📊 모델 성능 비교")
        comparison_df = self.compare_models()
        
        # 7. 결과 시각화
        print("\n📈 결과 시각화")
        self.plot_results()
        
        print("\n✅ 학습 예제 완료!")
        print(f"📁 결과 이미지: ml_training_results.png")
        
        return comparison_df


def main():
    """메인 함수"""
    print("🎯 머신러닝/딥러닝 학습 예제")
    print("=" * 50)
    
    # 예제 실행
    example = MLTrainingExample()
    results = example.run_complete_example()
    
    print("\n🎉 모든 학습이 완료되었습니다!")
    print("📊 최고 성능 모델:", results.iloc[0]['Model'])
    print("📈 R² 점수:", results.iloc[0]['R²'])


if __name__ == "__main__":
    main() 