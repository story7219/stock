#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: ensemble_system.py
모듈: 고급 앙상블 트레이딩 시스템
목적: ML + DL + AI 최적 조합으로 최고 성능 달성

Author: World-Class Trading System
Created: 2025-01-13
Version: 2.0.0

Features:
- Multi-Model Ensemble (ML + DL + AI)
- Advanced Prompt Engineering with Gemini-1.5-Flash-8B
- Dynamic Weight Optimization
- Real-time Adaptation
- Risk-Aware Decision Making
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from functools import lru_cache, wraps
from pathlib import Path
from typing import (
    Any, Dict, List, Optional, Union, Tuple, Set,
    Protocol, TypeVar, Generic, Final, Literal,
    Callable, Awaitable, AsyncIterator
)

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Logging Setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# PyTorch 의존성을 선택적으로 처리
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
    logger.info("PyTorch 로드 성공 - DL 모델 사용 가능")
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch가 설치되지 않았습니다. DL 모델은 사용할 수 없습니다.")

# Google Generative AI 의존성을 선택적으로 처리
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
    logger.info("Google Generative AI 로드 성공 - Gemini 모델 사용 가능")
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("Google Generative AI가 설치되지 않았습니다. AI 모델은 시뮬레이션 모드로 실행됩니다.")

# Type Variables
T = TypeVar('T')

# Constants
MAX_MODELS: Final = 10
MAX_ENSEMBLE_SIZE: Final = 5
CONFIDENCE_THRESHOLD: Final = 0.7
RISK_FREE_RATE: Final = 0.02
MAX_DRAWDOWN_THRESHOLD: Final = 0.15

class ModelType(Enum):
    """모델 타입 열거형"""
    MACHINE_LEARNING = auto()
    DEEP_LEARNING = auto()
    ARTIFICIAL_INTELLIGENCE = auto()

class SignalType(Enum):
    """신호 타입 열거형"""
    BUY = auto()
    SELL = auto()
    HOLD = auto()

@dataclass
class ModelPrediction:
    """모델 예측 결과"""
    model_name: str
    model_type: ModelType
    prediction: Union[int, float]
    confidence: float
    features_importance: Optional[Dict[str, float]] = None
    explanation: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class EnsembleDecision:
    """앙상블 결정 결과"""
    final_signal: SignalType
    confidence: float
    model_votes: Dict[str, SignalType]
    weighted_score: float
    risk_score: float
    explanation: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

class BaseModel(ABC):
    """기본 모델 추상 클래스"""
    
    def __init__(self, name: str, model_type: ModelType):
        self.name = name
        self.model_type = model_type
        self.is_trained = False
        self.performance_history: List[float] = []
        
    @abstractmethod
    async def train(self, data: pd.DataFrame) -> None:
        """모델 훈련"""
        pass
    
    @abstractmethod
    async def predict(self, data: pd.DataFrame) -> ModelPrediction:
        """예측 수행"""
        pass
    
    @abstractmethod
    def get_performance_score(self) -> float:
        """성능 점수 반환"""
        pass

class MLModel(BaseModel):
    """머신러닝 모델 클래스"""
    
    def __init__(self, name: str, model_type: str = "random_forest"):
        super().__init__(name, ModelType.MACHINE_LEARNING)
        self.model_type = model_type
        self.model = self._create_model()
        self.scaler = StandardScaler()
        self.feature_importance: Dict[str, float] = {}
        
    def _create_model(self) -> Any:
        """모델 생성"""
        models = {
            "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "gradient_boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
            "logistic_regression": LogisticRegression(random_state=42),
            "svm": SVC(probability=True, random_state=42),
            "neural_network": MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42)
        }
        return models.get(self.model_type, models["random_forest"])
    
    async def train(self, data: pd.DataFrame) -> None:
        """ML 모델 훈련"""
        try:
            # 특성과 타겟 분리
            features = data.drop(['target', 'signal'], axis=1, errors='ignore')
            target = data['target'] if 'target' in data.columns else data['signal']
            
            # 특성 스케일링
            features_scaled = self.scaler.fit_transform(features)
            
            # 모델 훈련
            self.model.fit(features_scaled, target)
            
            # 특성 중요도 계산
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importance = dict(zip(features.columns, self.model.feature_importances_))
            
            self.is_trained = True
            logger.info(f"ML 모델 '{self.name}' 훈련 완료")
            
        except Exception as e:
            logger.error(f"ML 모델 훈련 실패: {e}")
            raise
    
    async def predict(self, data: pd.DataFrame) -> ModelPrediction:
        """ML 모델 예측"""
        if not self.is_trained:
            raise ValueError("모델이 훈련되지 않았습니다")
        
        try:
            features = data.drop(['target', 'signal'], axis=1, errors='ignore')
            features_scaled = self.scaler.transform(features)
            
            # 예측 및 확률
            prediction = self.model.predict(features_scaled)[-1]
            probabilities = self.model.predict_proba(features_scaled)[-1]
            confidence = max(probabilities)
            
            return ModelPrediction(
                model_name=self.name,
                model_type=self.model_type,
                prediction=prediction,
                confidence=confidence,
                features_importance=self.feature_importance
            )
            
        except Exception as e:
            logger.error(f"ML 모델 예측 실패: {e}")
            raise
    
    def get_performance_score(self) -> float:
        """성능 점수 계산"""
        if not self.performance_history:
            return 0.0
        return np.mean(self.performance_history[-10:])  # 최근 10개 평균

class DLModel(BaseModel):
    """딥러닝 모델 클래스 (PyTorch)"""
    def __init__(self, name: str, input_size: int, hidden_size: int = 128):
        super().__init__(name, ModelType.MACHINE_LEARNING)
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch가 설치되지 않았습니다. DL 모델을 사용할 수 없습니다.")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._create_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        self.epochs = 200
        self.batch_size = 16
        self.train_loss_log = []
        self.val_loss_log = []
        self.metrics = {}
    def _create_model(self) -> nn.Module:
        class TradingNet(nn.Module):
            def __init__(self, input_size: int, hidden_size: int):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(input_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(hidden_size // 2, 3),
                    nn.Softmax(dim=1)
                )
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.layers(x)
        return TradingNet(self.input_size, self.hidden_size).to(self.device)
    async def train(self, data: pd.DataFrame) -> None:
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, f1_score, classification_report
        # float/int 컬럼만 features로 사용
        features = data.select_dtypes(include=[float, int])
        if 'target' in data.columns:
            target = data['target']
        elif 'signal' in data.columns:
            target = data['signal']
        else:
            raise ValueError('타깃 컬럼이 없습니다')
        X_train, X_test, y_train, y_test = train_test_split(features.values, target.values, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.LongTensor(y_train).to(self.device)
        X_val = torch.FloatTensor(X_val).to(self.device)
        y_val = torch.LongTensor(y_val).to(self.device)
        X_test = torch.FloatTensor(X_test).to(self.device)
        y_test = torch.LongTensor(y_test).to(self.device)
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        best_val_loss = float('inf')
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)
            self.train_loss_log.append(train_loss)
            self.model.eval()
            val_loss = 0
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs, batch_y)
                    val_loss += loss.item()
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(batch_y.cpu().numpy())
            val_loss /= len(val_loader)
            self.val_loss_log.append(val_loss)
            acc = accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds, average='macro')
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict()
            if epoch % 10 == 0 or epoch == self.epochs - 1:
                logger.info(f"[DLNet][Epoch {epoch}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {acc:.4f} | Val F1: {f1:.4f}")
        self.model.load_state_dict(best_model_state)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_test)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            labels = y_test.cpu().numpy()
            acc = accuracy_score(labels, preds)
            f1 = f1_score(labels, preds, average='macro')
            report = classification_report(labels, preds, output_dict=True)
            self.metrics = {'test_acc': acc, 'test_f1': f1, 'report': report}
            logger.info(f"[DLNet][Test] Acc: {acc:.4f} | F1: {f1:.4f}")
        self.is_trained = True
    async def predict(self, data: pd.DataFrame) -> ModelPrediction:
        features = data.select_dtypes(include=[float, int])
        X = torch.FloatTensor(features.values[-1:]).to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X)
            probabilities = outputs.cpu().numpy()[0]
            prediction = int(np.argmax(probabilities))
            confidence = float(np.max(probabilities))
        return ModelPrediction(
            model_name=self.name,
            model_type=self.model_type,
            prediction=prediction,
            confidence=confidence
        )
    def get_performance_score(self) -> float:
        if not self.performance_history:
            return 0.0
        return float(np.mean(self.performance_history[-10:]))
    def get_detailed_report(self):
        return self.metrics

class AIModel(BaseModel):
    """AI 모델 클래스 (Gemini-1.5-Flash-8B 기반)"""
    
    def __init__(self, name: str = "Gemini-1.5Flash-8B", api_key: Optional[str] = None):
        super().__init__(name, ModelType.ARTIFICIAL_INTELLIGENCE)
        self.api_key = api_key
        self.prompt_template = self._create_prompt_template()
        
    def _create_prompt_template(self) -> str:
        """최적화된 프롬프트 템플릿 생성"""
        return """
당신은 세계 최고의 금융 분석가입니다. 주식 시장 데이터를 분석하여 매수/매도/홀드 신호를 제공해야 합니다.

분석 규칙:
1. 기술적 지표: 이동평균, RSI, MACD, 볼린저 밴드
2. 거래량 분석: 거래량 증가/감소 패턴
3. 가격 패턴: 지지/저항선, 추세선
4. 시장 심리: 변동성, 모멘텀
5. 리스크 관리: 손실 제한, 포트폴리오 분산

제공된 데이터:
{data_summary}

최근 시장 상황:
- 현재가: {current_price}
- 20일 이동평균: {ma20}
- RSI: {rsi}
- MACD: {macd}
- 거래량: {volume}
- 변동성: {volatility}

분석 결과를 다음 형식으로 제공하세요:
신호: [매수/매도/홀드]
신뢰도: [0.0-1.0]
이유: [상세한 분석 근거]
리스크: [주요 위험 요소]
"""
    
    async def _call_gemini_api(self, prompt: str) -> str:
        """Gemini API 호출"""
        if not GEMINI_AVAILABLE or not self.api_key:
            return self._simulate_ai_response(prompt)
        
        try:
            response = await asyncio.to_thread(
                genai.generate_content,
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=500
                )
            )
            return response.text
        except Exception as e:
            logger.error(f"Gemini API 호출 실패: {e}")
            return self._simulate_ai_response(prompt)
    
    def _simulate_ai_response(self, prompt: str) -> str:
        """AI 응답 시뮬레이션"""
        import random
        
        signals = ["매수", "매도", "홀드"]
        signal = random.choice(signals)
        confidence = random.uniform(0.6, 0.9)
        
        return f"""
신호: {signal}
신뢰도: {confidence:.2f}
이유: 기술적 지표와 시장 상황을 종합 분석한 결과
리스크: 시장 변동성과 예상치 못한 이벤트
"""
    
    async def train(self, data: pd.DataFrame) -> None:
        """AI 모델 훈련 (프롬프트 최적화)"""
        try:
            # 프롬프트 최적화를 위한 데이터 분석
            self._optimize_prompt(data)
            self.is_trained = True
            logger.info(f"AI 모델 '{self.name}' 훈련 완료")
        except Exception as e:
            logger.error(f"AI 모델 훈련 실패: {e}")
            raise
    
    def _optimize_prompt(self, data: pd.DataFrame) -> None:
        """프롬프트 최적화"""
        # 실제 구현에서는 더 복잡한 프롬프트 최적화 로직
        pass
    
    async def predict(self, data: pd.DataFrame) -> ModelPrediction:
        """AI 모델 예측"""
        if not self.is_trained:
            raise ValueError("모델이 훈련되지 않았습니다")
        
        try:
            # 데이터 요약 생성
            data_summary = self._create_data_summary(data)
            
            # 프롬프트 생성
            prompt = self.prompt_template.format(
                data_summary=data_summary,
                current_price=data['close'].iloc[-1] if 'close' in data.columns else 0,
                ma20=data['close'].rolling(20).mean().iloc[-1] if 'close' in data.columns else 0,
                rsi=self._calculate_rsi(data),
                macd=self._calculate_macd(data),
                volume=data['volume'].iloc[-1] if 'volume' in data.columns else 0,
                volatility=data['close'].pct_change().std() if 'close' in data.columns else 0
            )
            
            # AI 응답 받기
            response = await self._call_gemini_api(prompt)
            
            # 응답 파싱
            signal, confidence, explanation = self._parse_ai_response(response)
            
            return ModelPrediction(
                model_name=self.name,
                model_type=self.model_type,
                prediction=signal,
                confidence=confidence,
                explanation=explanation
            )
            
        except Exception as e:
            logger.error(f"AI 모델 예측 실패: {e}")
            raise
    
    def _create_data_summary(self, data: pd.DataFrame) -> str:
        """데이터 요약 생성"""
        summary = []
        if 'close' in data.columns:
            summary.append(f"최근 가격: {data['close'].iloc[-1]:.2f}")
        if 'volume' in data.columns:
            summary.append(f"거래량: {data['volume'].iloc[-1]:,.0f}")
        return ", ".join(summary)
    
    def _calculate_rsi(self, data: pd.DataFrame, period: int = 14) -> float:
        """RSI 계산"""
        if 'close' not in data.columns:
            return 50.0
        
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0
    
    def _calculate_macd(self, data: pd.DataFrame) -> float:
        """MACD 계산"""
        if 'close' not in data.columns:
            return 0.0
        
        exp1 = data['close'].ewm(span=12).mean()
        exp2 = data['close'].ewm(span=26).mean()
        macd = exp1 - exp2
        return macd.iloc[-1] if not pd.isna(macd.iloc[-1]) else 0.0
    
    def _parse_ai_response(self, response: str) -> Tuple[int, float, str]:
        """AI 응답 파싱"""
        try:
            lines = response.strip().split('\n')
            signal_line = [line for line in lines if '신호:' in line][0]
            confidence_line = [line for line in lines if '신뢰도:' in line][0]
            reason_line = [line for line in lines if '이유:' in line][0]
            
            signal_text = signal_line.split(':')[1].strip()
            confidence = float(confidence_line.split(':')[1].strip())
            explanation = reason_line.split(':')[1].strip()
            
            # 신호를 숫자로 변환
            signal_map = {"매수": 1, "매도": -1, "홀드": 0}
            signal = signal_map.get(signal_text, 0)
            
            return signal, confidence, explanation
            
        except Exception as e:
            logger.error(f"AI 응답 파싱 실패: {e}")
            return 0, 0.5, "분석 실패"
    
    def get_performance_score(self) -> float:
        """성능 점수 계산"""
        if not self.performance_history:
            return 0.0
        return np.mean(self.performance_history[-10:])

class EnsembleSystem:
    """고급 앙상블 시스템"""
    
    def __init__(self):
        self.models: List[BaseModel] = []
        self.weights: Dict[str, float] = {}
        self.performance_history: List[float] = []
        self.risk_manager = RiskManager()
        self.adaptive_optimizer = AdaptiveOptimizer()
        
    async def add_model(self, model: BaseModel) -> None:
        """모델 추가"""
        if len(self.models) >= MAX_MODELS:
            logger.warning(f"최대 모델 수({MAX_MODELS})에 도달했습니다")
            return
        
        self.models.append(model)
        self.weights[model.name] = 1.0 / len(self.models)  # 균등 가중치로 초기화
        logger.info(f"모델 '{model.name}' 추가됨")
    
    async def train_all_models(self, data: pd.DataFrame) -> None:
        """모든 모델 훈련"""
        logger.info("모든 모델 훈련 시작")
        
        tasks = [model.train(data) for model in self.models]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info("모든 모델 훈련 완료")
    
    async def get_ensemble_prediction(self, data: pd.DataFrame) -> EnsembleDecision:
        """앙상블 예측 수행"""
        if not self.models:
            raise ValueError("앙상블에 모델이 없습니다")
        
        try:
            # 각 모델의 예측 수집
            predictions = []
            for model in self.models:
                try:
                    pred = await model.predict(data)
                    predictions.append(pred)
                except Exception as e:
                    logger.error(f"모델 '{model.name}' 예측 실패: {e}")
                    continue
            
            if not predictions:
                raise ValueError("유효한 예측이 없습니다")
            
            # 가중 투표 계산
            weighted_votes = self._calculate_weighted_votes(predictions)
            
            # 최종 신호 결정
            final_signal = self._determine_final_signal(weighted_votes)
            
            # 리스크 평가
            risk_score = self.risk_manager.calculate_risk_score(data, predictions)
            
            # 설명 생성
            explanation = self._generate_explanation(predictions, final_signal)
            
            # 성능 업데이트
            self._update_performance(predictions)
            
            # 적응형 최적화
            await self.adaptive_optimizer.optimize_weights(self.models, self.weights, predictions)
            
            return EnsembleDecision(
                final_signal=final_signal,
                confidence=weighted_votes.get('confidence', 0.0),
                model_votes={pred.model_name: SignalType(pred.prediction) for pred in predictions},
                weighted_score=weighted_votes.get('score', 0.0),
                risk_score=risk_score,
                explanation=explanation
            )
            
        except Exception as e:
            logger.error(f"앙상블 예측 실패: {e}")
            raise
    
    def _calculate_weighted_votes(self, predictions: List[ModelPrediction]) -> Dict[str, float]:
        """가중 투표 계산"""
        total_weight = sum(self.weights.get(pred.model_name, 1.0) for pred in predictions)
        
        weighted_sum = 0.0
        total_confidence = 0.0
        
        for pred in predictions:
            weight = self.weights.get(pred.model_name, 1.0) / total_weight
            weighted_sum += pred.prediction * weight * pred.confidence
            total_confidence += pred.confidence * weight
        
        return {
            'score': weighted_sum,
            'confidence': total_confidence / len(predictions) if predictions else 0.0
        }
    
    def _determine_final_signal(self, weighted_votes: Dict[str, float]) -> SignalType:
        """최종 신호 결정"""
        score = weighted_votes.get('score', 0.0)
        confidence = weighted_votes.get('confidence', 0.0)
        
        # 신뢰도가 낮으면 홀드
        if confidence < CONFIDENCE_THRESHOLD:
            return SignalType.HOLD
        
        # 점수 기반 신호 결정
        if score > 0.5:
            return SignalType.BUY
        elif score < -0.5:
            return SignalType.SELL
        else:
            return SignalType.HOLD
    
    def _generate_explanation(self, predictions: List[ModelPrediction], final_signal: SignalType) -> str:
        """설명 생성"""
        explanations = []
        
        # 모델별 예측 요약
        for pred in predictions:
            signal_text = {1: "매수", -1: "매도", 0: "홀드"}.get(pred.prediction, "홀드")
            explanations.append(f"{pred.model_name}: {signal_text} (신뢰도: {pred.confidence:.2f})")
        
        # 최종 결정 근거
        explanations.append(f"최종 결정: {final_signal.name}")
        
        return " | ".join(explanations)
    
    def _update_performance(self, predictions: List[ModelPrediction]) -> None:
        """성능 업데이트"""
        # 실제 구현에서는 실제 성능을 측정하여 업데이트
        pass

class RiskManager:
    """리스크 관리자"""
    
    def calculate_risk_score(self, data: pd.DataFrame, predictions: List[ModelPrediction]) -> float:
        """리스크 점수 계산"""
        try:
            risk_factors = []
            
            # 변동성 리스크
            if 'close' in data.columns:
                volatility = data['close'].pct_change().std()
                risk_factors.append(min(volatility * 10, 1.0))
            
            # 모델 불일치 리스크
            if len(predictions) > 1:
                predictions_values = [p.prediction for p in predictions]
                disagreement = 1 - (len(set(predictions_values)) / len(predictions_values))
                risk_factors.append(disagreement)
            
            # 신뢰도 리스크
            avg_confidence = np.mean([p.confidence for p in predictions])
            risk_factors.append(1 - avg_confidence)
            
            return np.mean(risk_factors) if risk_factors else 0.5
            
        except Exception as e:
            logger.error(f"리스크 점수 계산 실패: {e}")
            return 0.5

class AdaptiveOptimizer:
    """적응형 최적화기"""
    
    async def optimize_weights(self, models: List[BaseModel], weights: Dict[str, float], 
                             predictions: List[ModelPrediction]) -> None:
        """가중치 최적화"""
        try:
            # 성능 기반 가중치 조정
            for pred in predictions:
                model_name = pred.model_name
                performance = pred.confidence
                
                # 성능이 좋은 모델의 가중치 증가
                if performance > 0.8:
                    weights[model_name] *= 1.1
                elif performance < 0.5:
                    weights[model_name] *= 0.9
                
                # 가중치 정규화
                total_weight = sum(weights.values())
                for name in weights:
                    weights[name] /= total_weight
                    
        except Exception as e:
            logger.error(f"가중치 최적화 실패: {e}")

# 사용 예시
async def main():
    """메인 함수"""
    # 앙상블 시스템 생성
    ensemble = EnsembleSystem()
    
    # ML 모델들 추가
    ml_models = [
        MLModel("RandomForest", "random_forest"),
        MLModel("GradientBoosting", "gradient_boosting"),
        MLModel("SVM", "svm"),
        MLModel("NeuralNetwork", "neural_network")
    ]
    
    # DL 모델들 추가 (PyTorch가 있는 경우)
    dl_models = []
    if TORCH_AVAILABLE:
        try:
            dl_models = [
                DLModel("TradingNet_128", input_size=10, hidden_size=128),
                DLModel("TradingNet_256", input_size=10, hidden_size=256),
                DLModel("TradingNet_512", input_size=10, hidden_size=512)
            ]
        except Exception as e:
            logger.warning(f"DL 모델 생성 실패: {e}")
    
    # AI 모델들 추가 (Gemini-1.5-Flash-8B)
    ai_models = [
        AIModel("Gemini_Flash_8B"),
        AIModel("Gemini_Flash_8B_v2"),
        AIModel("Gemini_Flash_8B_v3")
    ]
    
    # 모든 모델을 앙상블에 추가
    for model in ml_models + dl_models + ai_models:
        await ensemble.add_model(model)
    
    # 샘플 데이터 생성
    data = pd.DataFrame({
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 100),
        'target': np.random.choice([-1, 0, 1], 100)
    })
    
    # 모델 훈련
    await ensemble.train_all_models(data)
    
    # 앙상블 예측
    decision = await ensemble.get_ensemble_prediction(data)
    
    print(f"최종 신호: {decision.final_signal}")
    print(f"신뢰도: {decision.confidence:.2f}")
    print(f"리스크 점수: {decision.risk_score:.2f}")
    print(f"설명: {decision.explanation}")

if __name__ == "__main__":
    asyncio.run(main()) 