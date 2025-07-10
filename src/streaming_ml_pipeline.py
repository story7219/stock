#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: streaming_ml_pipeline.py
모듈: 실시간 ML 스트리밍 파이프라인
목적: 실시간 데이터 → 피처 → 모델 추론 → 신호 → 액션 트리거 전체 파이프라인

Author: Trading AI System
Created: 2025-01-27
Version: 1.0.0

Dependencies:
    - Python 3.11+
    - numpy, pandas, torch, sklearn, cupy (선택), joblib
    - aiokafka, aioredis, cachetools
    - typing_extensions

Performance:
    - 모델 추론 100ms 이내
    - 피처 계산 최적화
    - 메모리 사용량 최소화
    - CPU/GPU 효율적 사용
"""

import asyncio
import time
import numpy as np
import pandas as pd
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
import logging

try:
    import torch
    import joblib
    from cachetools import LRUCache
    import cupy as cp
    GPU_AVAILABLE = torch.cuda.is_available()
except ImportError:
    GPU_AVAILABLE = False

logger = logging.getLogger(__name__)

# 1. 실시간 피처 계산기
class RealTimeFeatureCalculator:
    """롤링 윈도우 기반 실시간 피처 계산 및 정규화/스케일링"""
    def __init__(self, window: int = 60, feature_list: Optional[List[str]] = None):
        self.window = window
        self.buffer = deque(maxlen=window)
        self.feature_list = feature_list or ["close", "volume"]
        self.scaler = None
        self.last_features = None

    def update(self, data: Dict[str, Any]) -> Optional[np.ndarray]:
        self.buffer.append(data)
        if len(self.buffer) < self.window:
            return None
        df = pd.DataFrame(list(self.buffer))
        features = self._compute_features(df)
        self.last_features = features
        return features

    def _compute_features(self, df: pd.DataFrame) -> np.ndarray:
        feats = []
        # 기술적 지표 예시: 이동평균, RSI, 볼린저밴드 등
        close = df["close"].values
        volume = df["volume"].values
        feats.append(close[-1])
        feats.append(np.mean(close[-5:]))  # 5MA
        feats.append(np.std(close[-10:]))  # 10STD
        feats.append(np.max(close[-10:]) - np.min(close[-10:]))  # 10Range
        feats.append(np.sum(volume[-5:]))  # 5VolSum
        # 정규화/스케일링
        if self.scaler is None:
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            self.scaler.fit([feats])
        feats_scaled = self.scaler.transform([feats])[0]
        return feats_scaled

# 2. 모델 추론 파이프라인
class RealTimePredictor:
    """마이크로배치, GPU 가속, 앙상블, 캐싱 지원 실시간 추론"""
    def __init__(self, models: List[Any], batch_size: int = 16, cache_size: int = 1024):
        self.models = models
        self.batch_size = batch_size
        self.cache = LRUCache(maxsize=cache_size)
        self.device = torch.device("cuda" if GPU_AVAILABLE else "cpu")
        self.last_batch = []

    async def predict(self, features: np.ndarray) -> float:
        key = tuple(features)
        if key in self.cache:
            return self.cache[key]
        self.last_batch.append(features)
        if len(self.last_batch) < self.batch_size:
            return None
        batch = np.stack(self.last_batch)
        batch_tensor = torch.tensor(batch, dtype=torch.float32).to(self.device)
        preds = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(batch_tensor)
                preds.append(pred.cpu().numpy())
        ensemble_pred = np.mean(preds, axis=0)
        for i, f in enumerate(self.last_batch):
            self.cache[tuple(f)] = float(ensemble_pred[i])
        self.last_batch.clear()
        return float(ensemble_pred[-1])

# 3. 신호 생성 및 필터링
class SignalGenerator:
    """실시간 신호 강도 계산, 리스크 필터, 품질 검증, 우선순위 결정"""
    def __init__(self, risk_limit: float = 0.05):
        self.risk_limit = risk_limit

    def generate(self, pred: float, features: np.ndarray) -> Optional[Dict[str, Any]]:
        # 신호 강도 계산 (예: softmax, z-score 등)
        strength = float(np.tanh(pred))
        # 리스크 필터 (예: 변동성, drawdown 등)
        risk = float(np.std(features))
        if risk > self.risk_limit:
            return None
        # 품질 검증 (예: 신뢰구간, 이상치 등)
        if not np.isfinite(strength):
            return None
        # 우선순위 결정 (예: 강도, 최근성 등)
        priority = strength * (1 - risk)
        return {
            "signal": np.sign(strength),
            "strength": strength,
            "risk": risk,
            "priority": priority
        }

# 4. 액션 트리거
class ActionTrigger:
    """매매 신호 → 주문 실행, 포지션 관리, 리스크 한도 확인"""
    def __init__(self, order_executor: Callable, position_manager: Callable, risk_checker: Callable):
        self.order_executor = order_executor
        self.position_manager = position_manager
        self.risk_checker = risk_checker

    async def trigger(self, signal: Dict[str, Any], context: Dict[str, Any]):
        # 리스크 한도 확인
        if not self.risk_checker(signal, context):
            return False
        # 주문 실행
        order_result = await self.order_executor(signal, context)
        # 포지션 관리 업데이트
        await self.position_manager(signal, context, order_result)
        return True

# 5. 성능 최적화
class PerformanceOptimizer:
    """피처/추론/메모리/CPU/GPU 성능 최적화"""
    def __init__(self):
        self.latencies = deque(maxlen=1000)
        self.mem_usage = []
        self.gpu_usage = []

    def record_latency(self, latency: float):
        self.latencies.append(latency)

    def get_latency_stats(self) -> Dict[str, float]:
        arr = np.array(self.latencies)
        return {
            "mean": float(np.mean(arr)) if len(arr) else 0.0,
            "p95": float(np.percentile(arr, 95)) if len(arr) else 0.0,
            "max": float(np.max(arr)) if len(arr) else 0.0
        }

    def monitor_resources(self):
        import psutil
        self.mem_usage.append(psutil.virtual_memory().percent)
        if GPU_AVAILABLE:
            import torch
            self.gpu_usage.append(torch.cuda.memory_allocated() / 1e6)

    def get_resource_stats(self) -> Dict[str, Any]:
        return {
            "mem_usage": self.mem_usage[-1] if self.mem_usage else 0.0,
            "gpu_usage": self.gpu_usage[-1] if self.gpu_usage else 0.0
        }

# 전체 파이프라인
class StreamingMLPipeline:
    """실시간 데이터 → 피처 → 추론 → 신호 → 액션 전체 파이프라인"""
    def __init__(self, feature_calc: RealTimeFeatureCalculator, predictor: RealTimePredictor,
                 signal_gen: SignalGenerator, action_trigger: ActionTrigger, perf_opt: PerformanceOptimizer):
        self.feature_calc = feature_calc
        self.predictor = predictor
        self.signal_gen = signal_gen
        self.action_trigger = action_trigger
        self.perf_opt = perf_opt

    async def process(self, data: Dict[str, Any], context: Dict[str, Any]):
        t0 = time.time()
        features = self.feature_calc.update(data)
        if features is None:
            return None
        pred = await self.predictor.predict(features)
        if pred is None:
            return None
        signal = self.signal_gen.generate(pred, features)
        if signal is None:
            return None
        await self.action_trigger.trigger(signal, context)
        latency = (time.time() - t0) * 1000
        self.perf_opt.record_latency(latency)
        self.perf_opt.monitor_resources()
        return {
            "features": features,
            "prediction": pred,
            "signal": signal,
            "latency_ms": latency,
            "resources": self.perf_opt.get_resource_stats()
        }

# 사용 예시 (실제 배포 환경에서는 각 컴포넌트를 비동기 서비스로 통합)
# ... 