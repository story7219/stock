#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: test_ml_system.py
모듈: ML 시스템 종합 테스트
목적: 단위 테스트, 통합 테스트, 성능 테스트

Author: AI Assistant
Created: 2025-01-27
Modified: 2025-01-27
Version: 1.0.0

Dependencies:
    - Python 3.11+
    - pytest==7.4.0
    - pytest-asyncio==0.21.0
    - pytest-cov==4.1.0
    - hypothesis==6.82.0
    - torch==2.1.0
    - scikit-learn==1.3.0

Performance:
    - 테스트 커버리지: 95% 이상
    - 실행 시간: 5분 이내
    - 메모리 사용량: 2GB 이내
    - 병렬 실행: pytest-xdist 지원

Security:
    - 테스트 격리: 독립적인 테스트 환경
    - 데이터 검증: 입력 데이터 검증
    - 보안 테스트: API 키 검증

License: MIT
"""

import asyncio
import json
import os
import pickle
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from unittest.mock import Mock, patch, AsyncMock

import numpy as np
import pandas as pd
import pytest
import torch
from hypothesis import given, strategies as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 테스트 대상 모듈들
import sys
sys.path.append('.')

from ml.advanced_ml_training_system import (
    TrainingConfig, ModelMetrics, DataPreprocessor,
    LSTMModel, GRUModel, TransformerModel, ModelTrainer,
    HyperparameterOptimizer, DistributedTrainer
)


class TestTrainingConfig:
    """TrainingConfig 테스트 클래스"""
    
    def test_default_config(self):
        """기본 설정 테스트"""
        config = TrainingConfig()
        
        assert config.model_type == "lstm"
        assert config.data_path == "data/"
        assert config.test_size == 0.2
        assert config.random_state == 42
        assert config.max_trials == 100
        assert config.n_splits == 5
        assert config.batch_size == 32
        assert config.epochs == 100
        assert config.learning_rate == 0.001
        assert config.early_stopping_patience == 10
        assert config.validation_split == 0.2
        assert config.use_distributed == False
        assert config.num_workers == 4
        assert config.use_hyperopt == True
        assert config.optimization_metric == "mse"
        assert config.save_best_only == True
        assert config.model_format == "pickle"
    
    def test_custom_config(self):
        """사용자 정의 설정 테스트"""
        config = TrainingConfig(
            model_type="transformer",
            data_path="custom_data/",
            test_size=0.3,
            random_state=123,
            max_trials=50,
            n_splits=3,
            batch_size=64,
            epochs=200,
            learning_rate=0.0001,
            early_stopping_patience=20,
            validation_split=0.3,
            use_distributed=True,
            num_workers=8,
            use_hyperopt=False,
            optimization_metric="mae",
            save_best_only=False,
            model_format="joblib"
        )
        
        assert config.model_type == "transformer"
        assert config.data_path == "custom_data/"
        assert config.test_size == 0.3
        assert config.random_state == 123
        assert config.max_trials == 50
        assert config.n_splits == 3
        assert config.batch_size == 64
        assert config.epochs == 200
        assert config.learning_rate == 0.0001
        assert config.early_stopping_patience == 20
        assert config.validation_split == 0.3
        assert config.use_distributed == True
        assert config.num_workers == 8
        assert config.use_hyperopt == False
        assert config.optimization_metric == "mae"
        assert config.save_best_only == False
        assert config.model_format == "joblib"


class TestModelMetrics:
    """ModelMetrics 테스트 클래스"""
    
    def test_default_metrics(self):
        """기본 메트릭 테스트"""
        metrics = ModelMetrics()
        
        assert metrics.train_score == 0.0
        assert metrics.val_score == 0.0
        assert metrics.test_score == 0.0
        assert metrics.mse == 0.0
        assert metrics.mae == 0.0
        assert metrics.r2 == 0.0
        assert metrics.accuracy == 0.0
        assert metrics.precision == 0.0
        assert metrics.recall == 0.0
        assert metrics.f1 == 0.0
        assert metrics.training_time == 0.0
        assert metrics.inference_time == 0.0
    
    def test_custom_metrics(self):
        """사용자 정의 메트릭 테스트"""
        metrics = ModelMetrics(
            train_score=0.85,
            val_score=0.82,
            test_score=0.80,
            mse=0.15,
            mae=0.12,
            r2=0.80,
            accuracy=0.85,
            precision=0.83,
            recall=0.87,
            f1=0.85,
            training_time=120.5,
            inference_time=0.05
        )
        
        assert metrics.train_score == 0.85
        assert metrics.val_score == 0.82
        assert metrics.test_score == 0.80
        assert metrics.mse == 0.15
        assert metrics.mae == 0.12
        assert metrics.r2 == 0.80
        assert metrics.accuracy == 0.85
        assert metrics.precision == 0.83
        assert metrics.recall == 0.87
        assert metrics.f1 == 0.85
        assert metrics.training_time == 120.5
        assert metrics.inference_time == 0.05


class TestDataPreprocessor:
    """DataPreprocessor 테스트 클래스"""
    
    @pytest.fixture
    def sample_data(self):
        """샘플 데이터 생성"""
        dates = pd.date_range('2023-01-01', periods=1000, freq='D')
        np.random.seed(42)
        
        data = {
            'timestamp': dates,
            'open': np.random.randn(1000).cumsum() + 100,
            'high': np.random.randn(1000).cumsum() + 105,
            'low': np.random.randn(1000).cumsum() + 95,
            'close': np.random.randn(1000).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 1000)
        }
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def config(self):
        """테스트용 설정"""
        return TrainingConfig()
    
    @pytest.fixture
    def preprocessor(self, config):
        """전처리기 인스턴스"""
        return DataPreprocessor(config)
    
    def test_basic_preprocessing(self, preprocessor, sample_data):
        """기본 전처리 테스트"""
        # 결측값 추가
        sample_data.loc[100:110, 'close'] = np.nan
        
        processed_data = preprocessor._basic_preprocessing(sample_data)
        
        # 결측값이 제거되었는지 확인
        assert processed_data['close'].isna().sum() == 0
        assert len(processed_data) < len(sample_data)  # 이상치 제거로 인한 감소
    
    def test_feature_engineering(self, preprocessor, sample_data):
        """특성 엔지니어링 테스트"""
        processed_data = preprocessor._feature_engineering(sample_data)
        
        # 기술적 지표가 추가되었는지 확인
        expected_features = ['ma_5', 'ma_20', 'ma_50', 'rsi', 'macd', 'macd_signal', 'bb_middle', 'bb_upper', 'bb_lower']
        
        for feature in expected_features:
            assert feature in processed_data.columns
    
    def test_prepare_sequences(self, preprocessor, sample_data):
        """시퀀스 준비 테스트"""
        processed_data = preprocessor._feature_engineering(sample_data)
        X, y = preprocessor._prepare_sequences(processed_data, sequence_length=60)
        
        # 시퀀스 형태 확인
        assert X.shape[0] == y.shape[0]
        assert X.shape[1] == 60  # sequence_length
        assert X.shape[2] == len(processed_data.select_dtypes(include=[np.number]).columns)
        assert len(y.shape) == 1
    
    def test_load_and_preprocess(self, preprocessor, sample_data, tmp_path):
        """데이터 로드 및 전처리 테스트"""
        # 임시 파일 생성
        data_file = tmp_path / "test_data.parquet"
        sample_data.to_parquet(data_file)
        
        # 설정 수정
        preprocessor.config.data_path = str(tmp_path)
        
        X, y = preprocessor.load_and_preprocess(str(tmp_path))
        
        # 결과 확인
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert len(X) > 0
        assert len(y) > 0


class TestLSTMModel:
    """LSTM 모델 테스트 클래스"""
    
    @pytest.fixture
    def model(self):
        """LSTM 모델 인스턴스"""
        return LSTMModel(input_size=10, hidden_size=64, num_layers=2, dropout=0.2)
    
    def test_model_creation(self, model):
        """모델 생성 테스트"""
        assert isinstance(model, LSTMModel)
        assert model.hidden_size == 64
        assert model.num_layers == 2
    
    def test_forward_pass(self, model):
        """순전파 테스트"""
        batch_size = 32
        sequence_length = 60
        input_size = 10
        
        x = torch.randn(batch_size, sequence_length, input_size)
        output = model(x)
        
        assert output.shape == (batch_size, 1)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_model_parameters(self, model):
        """모델 파라미터 테스트"""
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0
    
    @pytest.mark.parametrize("input_size,hidden_size,num_layers", [
        (5, 32, 1),
        (10, 64, 2),
        (20, 128, 3)
    ])
    def test_different_configurations(self, input_size, hidden_size, num_layers):
        """다양한 설정 테스트"""
        model = LSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        
        batch_size = 16
        sequence_length = 50
        
        x = torch.randn(batch_size, sequence_length, input_size)
        output = model(x)
        
        assert output.shape == (batch_size, 1)


class TestGRUModel:
    """GRU 모델 테스트 클래스"""
    
    @pytest.fixture
    def model(self):
        """GRU 모델 인스턴스"""
        return GRUModel(input_size=10, hidden_size=64, num_layers=2, dropout=0.2)
    
    def test_model_creation(self, model):
        """모델 생성 테스트"""
        assert isinstance(model, GRUModel)
        assert model.hidden_size == 64
        assert model.num_layers == 2
    
    def test_forward_pass(self, model):
        """순전파 테스트"""
        batch_size = 32
        sequence_length = 60
        input_size = 10
        
        x = torch.randn(batch_size, sequence_length, input_size)
        output = model(x)
        
        assert output.shape == (batch_size, 1)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_model_parameters(self, model):
        """모델 파라미터 테스트"""
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0


class TestTransformerModel:
    """Transformer 모델 테스트 클래스"""
    
    @pytest.fixture
    def model(self):
        """Transformer 모델 인스턴스"""
        return TransformerModel(input_size=10, d_model=64, nhead=8, num_layers=2, dropout=0.1)
    
    def test_model_creation(self, model):
        """모델 생성 테스트"""
        assert isinstance(model, TransformerModel)
        assert model.d_model == 64
    
    def test_forward_pass(self, model):
        """순전파 테스트"""
        batch_size = 32
        sequence_length = 60
        input_size = 10
        
        x = torch.randn(batch_size, sequence_length, input_size)
        output = model(x)
        
        assert output.shape == (batch_size, 1)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_model_parameters(self, model):
        """모델 파라미터 테스트"""
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0


class TestModelTrainer:
    """ModelTrainer 테스트 클래스"""
    
    @pytest.fixture
    def config(self):
        """테스트용 설정"""
        return TrainingConfig(
            model_type="lstm",
            data_path="test_data/",
            epochs=5,  # 빠른 테스트를 위해 적은 에포크
            batch_size=16,
            learning_rate=0.001
        )
    
    @pytest.fixture
    def trainer(self, config):
        """트레이너 인스턴스"""
        return ModelTrainer(config)
    
    @pytest.fixture
    def sample_data(self):
        """샘플 데이터 생성"""
        # 간단한 시계열 데이터 생성
        np.random.seed(42)
        n_samples = 1000
        sequence_length = 60
        n_features = 10
        
        X = np.random.randn(n_samples, sequence_length, n_features)
        y = np.random.randn(n_samples)
        
        return X, y
    
    def test_create_model(self, trainer):
        """모델 생성 테스트"""
        # LSTM 모델
        lstm_model = trainer._create_model("lstm", 10)
        assert isinstance(lstm_model, LSTMModel)
        
        # GRU 모델
        gru_model = trainer._create_model("gru", 10)
        assert isinstance(gru_model, GRUModel)
        
        # Transformer 모델
        transformer_model = trainer._create_model("transformer", 10)
        assert isinstance(transformer_model, TransformerModel)
        
        # 스킷런 모델들
        rf_model = trainer._create_model("random_forest", 10)
        assert isinstance(rf_model, RandomForestRegressor)
    
    def test_calculate_metrics(self, trainer, sample_data):
        """메트릭 계산 테스트"""
        X, y = sample_data
        
        # 분할
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # 더미 예측
        train_pred = y_train + np.random.randn(len(y_train)) * 0.1
        test_pred = y_test + np.random.randn(len(y_test)) * 0.1
        
        metrics = trainer._calculate_metrics(y_train, train_pred, y_test, test_pred, 10.5)
        
        assert isinstance(metrics, ModelMetrics)
        assert metrics.training_time == 10.5
        assert metrics.train_score >= 0  # R² score
        assert metrics.test_score >= 0
        assert metrics.mse >= 0
        assert metrics.mae >= 0
    
    @pytest.mark.asyncio
    async def test_train_model_integration(self, trainer, tmp_path):
        """모델 학습 통합 테스트"""
        # 테스트 데이터 생성
        np.random.seed(42)
        n_samples = 500
        sequence_length = 30
        n_features = 5
        
        X = np.random.randn(n_samples, sequence_length, n_features)
        y = np.random.randn(n_samples)
        
        # 데이터 저장
        data_file = tmp_path / "test_data.parquet"
        df = pd.DataFrame({
            'feature_1': X[:, -1, 0],
            'feature_2': X[:, -1, 1],
            'feature_3': X[:, -1, 2],
            'feature_4': X[:, -1, 3],
            'feature_5': X[:, -1, 4],
            'target': y
        })
        df.to_parquet(data_file)
        
        # 설정 수정
        trainer.config.data_path = str(tmp_path)
        trainer.config.epochs = 3  # 빠른 테스트
        
        # 학습 실행
        result = trainer.train_model("lstm")
        
        assert isinstance(result, dict)
        assert 'model' in result
        assert 'metrics' in result
        assert 'predictions' in result
        assert isinstance(result['metrics'], ModelMetrics)


class TestHyperparameterOptimizer:
    """HyperparameterOptimizer 테스트 클래스"""
    
    @pytest.fixture
    def config(self):
        """테스트용 설정"""
        return TrainingConfig(max_trials=5)  # 빠른 테스트
    
    @pytest.fixture
    def optimizer(self, config):
        """옵티마이저 인스턴스"""
        return HyperparameterOptimizer(config)
    
    @pytest.fixture
    def sample_data(self):
        """샘플 데이터"""
        np.random.seed(42)
        n_samples = 200
        n_features = 10
        
        X = np.random.randn(n_samples, n_features)
        y = np.random.randn(n_samples)
        
        return X, y
    
    def test_optimize_sklearn_hyperparameters(self, optimizer, sample_data):
        """스킷런 하이퍼파라미터 최적화 테스트"""
        X, y = sample_data
        
        result = optimizer.optimize_hyperparameters("random_forest", X, y)
        
        assert isinstance(result, dict)
        assert 'best_params' in result
        assert 'best_value' in result
        assert isinstance(result['best_params'], dict)
        assert isinstance(result['best_value'], float)
    
    def test_optimize_deep_hyperparameters(self, optimizer, sample_data):
        """딥러닝 하이퍼파라미터 최적화 테스트"""
        X, y = sample_data
        
        # 시계열 데이터로 변환
        sequence_length = 20
        X_seq = np.array([X[i:i+sequence_length] for i in range(len(X) - sequence_length)])
        y_seq = y[sequence_length:]
        
        result = optimizer.optimize_hyperparameters("lstm", X_seq, y_seq)
        
        assert isinstance(result, dict)
        assert 'best_params' in result
        assert 'best_value' in result


class TestDistributedTrainer:
    """DistributedTrainer 테스트 클래스"""
    
    @pytest.fixture
    def config(self):
        """테스트용 설정"""
        return TrainingConfig(use_distributed=True, max_trials=3)
    
    @pytest.fixture
    def trainer(self, config):
        """트레이너 인스턴스"""
        return DistributedTrainer(config)
    
    def test_trainer_initialization(self, trainer):
        """트레이너 초기화 테스트"""
        assert trainer.config.use_distributed == True
        assert trainer.config.max_trials == 3
    
    def test_get_config_space(self, trainer):
        """설정 공간 테스트"""
        # LSTM 설정 공간
        lstm_config = trainer._get_config_space("lstm")
        assert 'hidden_size' in lstm_config
        assert 'num_layers' in lstm_config
        assert 'dropout' in lstm_config
        assert 'learning_rate' in lstm_config
        assert 'batch_size' in lstm_config
        
        # 스킷런 설정 공간
        sklearn_config = trainer._get_config_space("random_forest")
        assert 'n_estimators' in sklearn_config
        assert 'max_depth' in sklearn_config
        assert 'learning_rate' in sklearn_config


# Property-based 테스트
class TestPropertyBased:
    """Property-based 테스트 클래스"""
    
    @given(
        st.integers(min_value=1, max_value=100),
        st.integers(min_value=1, max_value=50),
        st.integers(min_value=1, max_value=10)
    )
    def test_lstm_model_properties(self, batch_size, sequence_length, input_size):
        """LSTM 모델 속성 테스트"""
        model = LSTMModel(input_size=input_size)
        
        x = torch.randn(batch_size, sequence_length, input_size)
        output = model(x)
        
        # 출력 형태 확인
        assert output.shape == (batch_size, 1)
        
        # NaN/Inf 확인
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    @given(
        st.lists(st.floats(min_value=-10, max_value=10), min_size=10, max_size=100),
        st.lists(st.floats(min_value=-10, max_value=10), min_size=10, max_size=100)
    )
    def test_metrics_calculation_properties(self, y_true, y_pred):
        """메트릭 계산 속성 테스트"""
        if len(y_true) != len(y_pred):
            return  # 길이가 다르면 건너뛰기
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # 메트릭 계산
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # 속성 확인
        assert mse >= 0  # MSE는 항상 양수
        assert r2 <= 1  # R²는 최대 1
        assert not np.isnan(mse)
        assert not np.isnan(r2)


# 성능 테스트
class TestPerformance:
    """성능 테스트 클래스"""
    
    def test_lstm_training_performance(self):
        """LSTM 학습 성능 테스트"""
        model = LSTMModel(input_size=20, hidden_size=128, num_layers=2)
        
        batch_size = 64
        sequence_length = 100
        input_size = 20
        
        x = torch.randn(batch_size, sequence_length, input_size)
        
        start_time = time.time()
        
        # 순전파 100회 실행
        for _ in range(100):
            with torch.no_grad():
                output = model(x)
        
        end_time = time.time()
        inference_time = end_time - start_time
        
        # 성능 기준: 100회 추론이 10초 이내
        assert inference_time < 10.0
        assert output.shape == (batch_size, 1)
    
    def test_memory_usage(self):
        """메모리 사용량 테스트"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 대용량 모델 생성
        model = TransformerModel(input_size=50, d_model=256, nhead=8, num_layers=6)
        
        # 메모리 사용량 확인
        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = current_memory - initial_memory
        
        # 메모리 증가가 1GB 이내여야 함
        assert memory_increase < 1024  # MB
    
    def test_batch_processing_performance(self):
        """배치 처리 성능 테스트"""
        model = LSTMModel(input_size=10, hidden_size=64)
        
        batch_sizes = [16, 32, 64, 128]
        sequence_length = 50
        input_size = 10
        
        processing_times = []
        
        for batch_size in batch_sizes:
            x = torch.randn(batch_size, sequence_length, input_size)
            
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(10):  # 10회 반복
                    output = model(x)
            
            end_time = time.time()
            processing_time = end_time - start_time
            processing_times.append(processing_time)
        
        # 배치 크기가 클수록 처리 시간이 증가하지만 선형적이어야 함
        for i in range(1, len(processing_times)):
            # 처리 시간이 급격히 증가하지 않아야 함
            assert processing_times[i] < processing_times[i-1] * 4


# 통합 테스트
class TestIntegration:
    """통합 테스트 클래스"""
    
    @pytest.fixture
    def temp_data_dir(self, tmp_path):
        """임시 데이터 디렉토리"""
        data_dir = tmp_path / "test_data"
        data_dir.mkdir()
        
        # 테스트 데이터 생성
        np.random.seed(42)
        n_samples = 1000
        
        dates = pd.date_range('2023-01-01', periods=n_samples, freq='D')
        data = {
            'timestamp': dates,
            'open': np.random.randn(n_samples).cumsum() + 100,
            'high': np.random.randn(n_samples).cumsum() + 105,
            'low': np.random.randn(n_samples).cumsum() + 95,
            'close': np.random.randn(n_samples).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, n_samples)
        }
        
        df = pd.DataFrame(data)
        df.to_parquet(data_dir / "test_data.parquet")
        
        return data_dir
    
    def test_full_training_pipeline(self, temp_data_dir):
        """전체 학습 파이프라인 테스트"""
        config = TrainingConfig(
            model_type="lstm",
            data_path=str(temp_data_dir),
            epochs=3,  # 빠른 테스트
            batch_size=16,
            learning_rate=0.001
        )
        
        trainer = ModelTrainer(config)
        
        # 학습 실행
        result = trainer.train_model("lstm")
        
        # 결과 검증
        assert isinstance(result, dict)
        assert 'model' in result
        assert 'metrics' in result
        assert 'predictions' in result
        
        # 모델 검증
        model = result['model']
        assert isinstance(model, LSTMModel)
        
        # 메트릭 검증
        metrics = result['metrics']
        assert isinstance(metrics, ModelMetrics)
        assert metrics.training_time > 0
        
        # 예측 검증
        predictions = result['predictions']
        assert 'train' in predictions
        assert 'test' in predictions
    
    def test_hyperparameter_optimization_integration(self, temp_data_dir):
        """하이퍼파라미터 최적화 통합 테스트"""
        config = TrainingConfig(max_trials=3)  # 빠른 테스트
        optimizer = HyperparameterOptimizer(config)
        
        # 데이터 로드
        df = pd.read_parquet(temp_data_dir / "test_data.parquet")
        
        # 데이터 준비
        data = df.select_dtypes(include=[np.number]).values
        X = data[:-1]
        y = data[1:, 0]
        
        # 최적화 실행
        result = optimizer.optimize_hyperparameters("random_forest", X, y)
        
        # 결과 검증
        assert isinstance(result, dict)
        assert 'best_params' in result
        assert 'best_value' in result
        assert isinstance(result['best_params'], dict)
        assert isinstance(result['best_value'], float)


# API 테스트
class TestAPI:
    """API 테스트 클래스"""
    
    @pytest.fixture
    def api_client(self):
        """API 클라이언트"""
        from fastapi.testclient import TestClient
        from api.ml_api_server import app
        return TestClient(app)
    
    def test_health_endpoint(self, api_client):
        """헬스 체크 엔드포인트 테스트"""
        response = api_client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
    
    def test_models_endpoint(self, api_client):
        """모델 목록 엔드포인트 테스트"""
        response = api_client.get("/models")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
    
    def test_predict_endpoint(self, api_client):
        """예측 엔드포인트 테스트"""
        # 테스트 데이터
        test_data = {
            "data": [[1.0, 2.0, 3.0] for _ in range(60)],  # 60개 시퀀스
            "model_type": "lstm",
            "api_key": "your-secret-api-key"
        }
        
        response = api_client.post("/predict", json=test_data)
        
        # API 키가 유효하지 않으므로 401 에러 예상
        assert response.status_code == 401
    
    def test_metrics_endpoint(self, api_client):
        """메트릭 엔드포인트 테스트"""
        response = api_client.get("/metrics")
        assert response.status_code == 200
        
        data = response.json()
        assert "system" in data
        assert "models" in data
        assert "timestamp" in data


# 실행
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"]) 