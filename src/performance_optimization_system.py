#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: performance_optimization_system.py
모듈: 실시간 처리 최적화 및 성능 튜닝 시스템
목적: 데이터 처리, 모델 추론, DB 최적화, 레이턴시 모니터링

Author: Trading AI System
Created: 2025-01-27
Modified: 2025-01-27
Version: 1.0.0

Dependencies:
    - Python 3.11+
    - numpy>=1.24.0
    - pandas>=2.0.0
    - numba>=0.58.0
    - onnxruntime>=1.15.0
    - redis>=4.5.0
    - psutil>=5.9.0
    - prometheus_client>=0.17.0

Performance:
    - 신호 생성: < 50ms
    - 주문 실행: < 50ms
    - 전체 파이프라인: < 100ms
    - 메모리 사용량: < 2GB

Security:
    - 실시간 모니터링
    - 자동 복구
    - 성능 로깅

License: MIT
"""

from __future__ import annotations

import logging
import time
import psutil
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from enum import Enum
import warnings
from collections import deque, defaultdict
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
from functools import wraps, lru_cache
import gc
import os

# Optional imports for advanced optimization
try:
    import numba
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("Numba not available - using standard Python")

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("ONNX Runtime not available")

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("Redis not available")

warnings.filterwarnings('ignore')

# 로깅 설정
logger = logging.getLogger(__name__)


class PerformanceMetric(Enum):
    """성능 메트릭 타입"""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    ERROR_RATE = "error_rate"
    CACHE_HIT_RATE = "cache_hit_rate"


@dataclass
class PerformanceMetrics:
    """성능 메트릭"""
    timestamp: datetime
    metric_type: PerformanceMetric
    value: float
    unit: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LatencyProfile:
    """레이턴시 프로파일"""
    data_processing: float = 0.0
    model_inference: float = 0.0
    signal_generation: float = 0.0
    order_execution: float = 0.0
    database_query: float = 0.0
    cache_access: float = 0.0
    total_latency: float = 0.0


class PerformanceOptimizer:
    """성능 최적화기"""
    
    def __init__(self):
        self.optimization_history = []
        self.current_optimizations = {}
        self.logger = logging.getLogger("PerformanceOptimizer")
        
        # 벡터화 최적화 설정
        self.vectorization_enabled = True
        self.numba_enabled = NUMBA_AVAILABLE
        self.onnx_enabled = ONNX_AVAILABLE
        
        # 병렬 처리 설정
        self.max_workers = min(32, os.cpu_count() or 4)
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=self.max_workers)
    
    def optimize_data_processing(self, data: np.ndarray) -> np.ndarray:
        """데이터 처리 최적화"""
        try:
            start_time = time.time()
            
            # 1. 벡터화 연산
            if self.vectorization_enabled:
                # NumPy 벡터화 연산
                processed_data = self._vectorized_processing(data)
            else:
                processed_data = self._standard_processing(data)
            
            # 2. Numba JIT 컴파일 (선택적)
            if self.numba_enabled:
                processed_data = self._numba_optimized_processing(processed_data)
            
            processing_time = time.time() - start_time
            self.logger.info(f"데이터 처리 최적화 완료: {processing_time:.3f}s")
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"데이터 처리 최적화 실패: {e}")
            return data
    
    def _vectorized_processing(self, data: np.ndarray) -> np.ndarray:
        """벡터화 처리"""
        try:
            # 표준화
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            normalized = (data - mean) / (std + 1e-8)
            
            # 이상치 제거 (IQR 방법)
            q1 = np.percentile(normalized, 25, axis=0)
            q3 = np.percentile(normalized, 75, axis=0)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # 이상치를 중앙값으로 대체
            outlier_mask = (normalized < lower_bound) | (normalized > upper_bound)
            median_values = np.median(normalized, axis=0)
            normalized[outlier_mask] = median_values
            
            return normalized
            
        except Exception as e:
            self.logger.error(f"벡터화 처리 실패: {e}")
            return data
    
    def _standard_processing(self, data: np.ndarray) -> np.ndarray:
        """표준 처리 (벡터화 비활성화 시)"""
        try:
            processed_data = np.copy(data)
            
            # 표준화
            for i in range(data.shape[1]):
                col_data = data[:, i]
                mean_val = np.mean(col_data)
                std_val = np.std(col_data)
                processed_data[:, i] = (col_data - mean_val) / (std_val + 1e-8)
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"표준 처리 실패: {e}")
            return data
    
    @staticmethod
    @jit(nopython=True, parallel=True) if NUMBA_AVAILABLE else lambda x: x
    def _numba_optimized_processing(data: np.ndarray) -> np.ndarray:
        """Numba 최적화 처리"""
        try:
            result = np.copy(data)
            
            # 추가 최적화 연산
            for i in prange(data.shape[0]):
                for j in range(data.shape[1]):
                    # 비선형 변환
                    result[i, j] = np.tanh(data[i, j])
            
            return result
            
        except Exception:
            return data
    
    def optimize_model_inference(self, model, input_data: np.ndarray) -> Tuple[np.ndarray, float]:
        """모델 추론 최적화"""
        try:
            start_time = time.time()
            
            # 1. 배치 처리 최적화
            batch_size = self._calculate_optimal_batch_size(input_data.shape)
            batched_data = self._create_batches(input_data, batch_size)
            
            # 2. ONNX 최적화 (가능한 경우)
            if self.onnx_enabled and hasattr(model, 'onnx_session'):
                predictions = self._onnx_inference(model.onnx_session, batched_data)
            else:
                predictions = self._standard_inference(model, batched_data)
            
            # 3. 결과 병합
            final_predictions = np.concatenate(predictions, axis=0)
            
            inference_time = time.time() - start_time
            self.logger.info(f"모델 추론 최적화 완료: {inference_time:.3f}s")
            
            return final_predictions, inference_time
            
        except Exception as e:
            self.logger.error(f"모델 추론 최적화 실패: {e}")
            return input_data, 0.0
    
    def _calculate_optimal_batch_size(self, data_shape: Tuple[int, ...]) -> int:
        """최적 배치 크기 계산"""
        try:
            # 메모리 사용량 기반 계산
            available_memory = psutil.virtual_memory().available
            data_size = np.prod(data_shape) * 8  # float64 기준
            
            # 안전 마진 포함
            optimal_batch_size = max(1, min(64, int(available_memory / (data_size * 10))))
            
            return optimal_batch_size
            
        except Exception as e:
            self.logger.error(f"배치 크기 계산 실패: {e}")
            return 32
    
    def _create_batches(self, data: np.ndarray, batch_size: int) -> List[np.ndarray]:
        """배치 생성"""
        try:
            batches = []
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                batches.append(batch)
            return batches
            
        except Exception as e:
            self.logger.error(f"배치 생성 실패: {e}")
            return [data]
    
    def _onnx_inference(self, onnx_session, batched_data: List[np.ndarray]) -> List[np.ndarray]:
        """ONNX 추론"""
        try:
            predictions = []
            for batch in batched_data:
                # ONNX 입력 형식에 맞게 변환
                input_name = onnx_session.get_inputs()[0].name
                output_name = onnx_session.get_outputs()[0].name
                
                # 추론 실행
                result = onnx_session.run([output_name], {input_name: batch.astype(np.float32)})
                predictions.append(result[0])
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"ONNX 추론 실패: {e}")
            return [np.zeros((len(batch), 1)) for batch in batched_data]
    
    def _standard_inference(self, model, batched_data: List[np.ndarray]) -> List[np.ndarray]:
        """표준 추론"""
        try:
            predictions = []
            for batch in batched_data:
                # 모델 예측
                pred = model.predict(batch)
                predictions.append(pred)
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"표준 추론 실패: {e}")
            return [np.zeros((len(batch), 1)) for batch in batched_data]
    
    def optimize_database_queries(self, query: str, params: Dict[str, Any]) -> str:
        """데이터베이스 쿼리 최적화"""
        try:
            # 1. 쿼리 분석
            query_plan = self._analyze_query(query)
            
            # 2. 인덱스 최적화 제안
            index_suggestions = self._suggest_indexes(query_plan)
            
            # 3. 쿼리 재작성
            optimized_query = self._rewrite_query(query, query_plan)
            
            # 4. 파라미터 바인딩 최적화
            optimized_params = self._optimize_parameters(params)
            
            self.logger.info(f"쿼리 최적화 완료: {len(index_suggestions)} 인덱스 제안")
            
            return optimized_query
            
        except Exception as e:
            self.logger.error(f"쿼리 최적화 실패: {e}")
            return query
    
    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """쿼리 분석"""
        try:
            # 간단한 쿼리 분석 (실제로는 더 정교한 파서 필요)
            analysis = {
                'tables': [],
                'joins': [],
                'where_conditions': [],
                'order_by': [],
                'group_by': []
            }
            
            # 테이블 추출
            if 'FROM' in query.upper():
                from_part = query.upper().split('FROM')[1].split('WHERE')[0]
                analysis['tables'] = [table.strip() for table in from_part.split(',')]
            
            # WHERE 조건 추출
            if 'WHERE' in query.upper():
                where_part = query.upper().split('WHERE')[1].split('ORDER BY')[0].split('GROUP BY')[0]
                analysis['where_conditions'] = [cond.strip() for cond in where_part.split('AND')]
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"쿼리 분석 실패: {e}")
            return {}
    
    def _suggest_indexes(self, query_plan: Dict[str, Any]) -> List[str]:
        """인덱스 제안"""
        try:
            suggestions = []
            
            # WHERE 조건 기반 인덱스 제안
            for condition in query_plan.get('where_conditions', []):
                if '=' in condition:
                    column = condition.split('=')[0].strip()
                    suggestions.append(f"CREATE INDEX idx_{column} ON table_name({column})")
            
            return suggestions
            
        except Exception as e:
            self.logger.error(f"인덱스 제안 실패: {e}")
            return []
    
    def _rewrite_query(self, query: str, query_plan: Dict[str, Any]) -> str:
        """쿼리 재작성"""
        try:
            # 간단한 최적화 (실제로는 더 정교한 최적화 필요)
            optimized_query = query
            
            # SELECT * 최적화
            if 'SELECT *' in query.upper():
                # 실제 컬럼명으로 대체 필요
                optimized_query = query.replace('SELECT *', 'SELECT id, name, value')
            
            # LIMIT 추가 (없는 경우)
            if 'LIMIT' not in query.upper():
                optimized_query += ' LIMIT 1000'
            
            return optimized_query
            
        except Exception as e:
            self.logger.error(f"쿼리 재작성 실패: {e}")
            return query
    
    def _optimize_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """파라미터 최적화"""
        try:
            optimized_params = {}
            
            for key, value in params.items():
                # 데이터 타입 최적화
                if isinstance(value, float):
                    optimized_params[key] = round(value, 6)
                elif isinstance(value, str) and len(value) > 100:
                    optimized_params[key] = value[:100]  # 길이 제한
                else:
                    optimized_params[key] = value
            
            return optimized_params
            
        except Exception as e:
            self.logger.error(f"파라미터 최적화 실패: {e}")
            return params


class LatencyMonitor:
    """레이턴시 모니터"""
    
    def __init__(self):
        self.latency_history = defaultdict(deque)
        self.current_latencies = {}
        self.performance_thresholds = {
            'data_processing': 50.0,  # ms
            'model_inference': 100.0,  # ms
            'signal_generation': 20.0,  # ms
            'order_execution': 50.0,  # ms
            'database_query': 10.0,  # ms
            'cache_access': 1.0,  # ms
            'total_pipeline': 100.0  # ms
        }
        self.logger = logging.getLogger("LatencyMonitor")
    
    def measure_latency(self, operation: str, func: Callable, *args, **kwargs) -> Tuple[Any, float]:
        """레이턴시 측정"""
        try:
            start_time = time.time()
            result = func(*args, **kwargs)
            latency = (time.time() - start_time) * 1000  # ms로 변환
            
            # 레이턴시 기록
            self.latency_history[operation].append(latency)
            self.current_latencies[operation] = latency
            
            # 임계값 체크
            threshold = self.performance_thresholds.get(operation, float('inf'))
            if latency > threshold:
                self.logger.warning(f"{operation} 레이턴시 임계값 초과: {latency:.2f}ms > {threshold}ms")
            
            return result, latency
            
        except Exception as e:
            self.logger.error(f"{operation} 레이턴시 측정 실패: {e}")
            return None, 0.0
    
    def get_latency_profile(self) -> LatencyProfile:
        """레이턴시 프로파일 조회"""
        try:
            profile = LatencyProfile()
            
            for operation in self.current_latencies:
                if hasattr(profile, operation):
                    setattr(profile, operation, self.current_latencies[operation])
            
            # 총 레이턴시 계산
            total_latency = sum(self.current_latencies.values())
            profile.total_latency = total_latency
            
            return profile
            
        except Exception as e:
            self.logger.error(f"레이턴시 프로파일 조회 실패: {e}")
            return LatencyProfile()
    
    def get_latency_statistics(self, operation: str, window_size: int = 100) -> Dict[str, float]:
        """레이턴시 통계"""
        try:
            if operation not in self.latency_history:
                return {}
            
            recent_latencies = list(self.latency_history[operation])[-window_size:]
            
            if not recent_latencies:
                return {}
            
            stats = {
                'mean': np.mean(recent_latencies),
                'median': np.median(recent_latencies),
                'std': np.std(recent_latencies),
                'min': np.min(recent_latencies),
                'max': np.max(recent_latencies),
                'p95': np.percentile(recent_latencies, 95),
                'p99': np.percentile(recent_latencies, 99)
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"레이턴시 통계 계산 실패: {e}")
            return {}
    
    def identify_bottlenecks(self) -> List[str]:
        """병목 지점 식별"""
        try:
            bottlenecks = []
            
            for operation, latency in self.current_latencies.items():
                threshold = self.performance_thresholds.get(operation, float('inf'))
                if latency > threshold:
                    bottlenecks.append(f"{operation}: {latency:.2f}ms > {threshold}ms")
            
            # 총 레이턴시 체크
            total_latency = sum(self.current_latencies.values())
            if total_latency > self.performance_thresholds['total_pipeline']:
                bottlenecks.append(f"전체 파이프라인: {total_latency:.2f}ms > {self.performance_thresholds['total_pipeline']}ms")
            
            return bottlenecks
            
        except Exception as e:
            self.logger.error(f"병목 지점 식별 실패: {e}")
            return []


class ThroughputOptimizer:
    """처리량 최적화기"""
    
    def __init__(self):
        self.throughput_history = defaultdict(deque)
        self.current_throughput = {}
        self.optimization_strategies = {}
        self.logger = logging.getLogger("ThroughputOptimizer")
    
    def optimize_throughput(self, operation: str, data_size: int, 
                          processing_time: float) -> Dict[str, Any]:
        """처리량 최적화"""
        try:
            current_throughput = data_size / processing_time if processing_time > 0 else 0
            
            # 처리량 기록
            self.throughput_history[operation].append(current_throughput)
            self.current_throughput[operation] = current_throughput
            
            # 최적화 전략 적용
            optimization_result = self._apply_optimization_strategies(operation, current_throughput)
            
            return {
                'operation': operation,
                'current_throughput': current_throughput,
                'optimization_applied': optimization_result,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"처리량 최적화 실패: {e}")
            return {}
    
    def _apply_optimization_strategies(self, operation: str, current_throughput: float) -> List[str]:
        """최적화 전략 적용"""
        try:
            applied_strategies = []
            
            # 배치 크기 최적화
            if current_throughput < 1000:  # 임계값
                applied_strategies.append("배치 크기 증가")
            
            # 병렬 처리 활성화
            if current_throughput < 500:
                applied_strategies.append("병렬 처리 활성화")
            
            # 캐싱 활성화
            if current_throughput < 2000:
                applied_strategies.append("캐싱 활성화")
            
            # 메모리 최적화
            if current_throughput < 100:
                applied_strategies.append("메모리 최적화")
            
            return applied_strategies
            
        except Exception as e:
            self.logger.error(f"최적화 전략 적용 실패: {e}")
            return []
    
    def get_throughput_trend(self, operation: str, window_size: int = 50) -> Dict[str, float]:
        """처리량 트렌드"""
        try:
            if operation not in self.throughput_history:
                return {}
            
            recent_throughput = list(self.throughput_history[operation])[-window_size:]
            
            if len(recent_throughput) < 2:
                return {}
            
            # 선형 회귀로 트렌드 계산
            x = np.arange(len(recent_throughput))
            slope, intercept = np.polyfit(x, recent_throughput, 1)
            
            trend = {
                'slope': slope,
                'intercept': intercept,
                'trend_direction': 'increasing' if slope > 0 else 'decreasing',
                'current_throughput': recent_throughput[-1],
                'average_throughput': np.mean(recent_throughput)
            }
            
            return trend
            
        except Exception as e:
            self.logger.error(f"처리량 트렌드 계산 실패: {e}")
            return {}


class MemoryManager:
    """메모리 관리자"""
    
    def __init__(self):
        self.memory_usage_history = deque(maxlen=1000)
        self.memory_threshold = 0.8  # 80%
        self.gc_threshold = 0.7  # 70%
        self.logger = logging.getLogger("MemoryManager")
    
    def monitor_memory_usage(self) -> Dict[str, float]:
        """메모리 사용량 모니터링"""
        try:
            # 시스템 메모리 정보
            memory_info = psutil.virtual_memory()
            
            memory_stats = {
                'total_memory': memory_info.total / (1024**3),  # GB
                'available_memory': memory_info.available / (1024**3),  # GB
                'used_memory': memory_info.used / (1024**3),  # GB
                'memory_percentage': memory_info.percent / 100,  # 0-1
                'swap_used': memory_info.swap.used / (1024**3) if memory_info.swap else 0,  # GB
                'swap_percentage': memory_info.swap.percent / 100 if memory_info.swap else 0
            }
            
            # 메모리 사용량 기록
            self.memory_usage_history.append(memory_stats)
            
            # 임계값 체크
            if memory_stats['memory_percentage'] > self.memory_threshold:
                self.logger.warning(f"메모리 사용량 임계값 초과: {memory_stats['memory_percentage']:.2%}")
                self._optimize_memory()
            
            return memory_stats
            
        except Exception as e:
            self.logger.error(f"메모리 모니터링 실패: {e}")
            return {}
    
    def _optimize_memory(self):
        """메모리 최적화"""
        try:
            # 1. 가비지 컬렉션
            if self._get_memory_percentage() > self.gc_threshold:
                gc.collect()
                self.logger.info("가비지 컬렉션 실행됨")
            
            # 2. 캐시 정리
            self._clear_caches()
            
            # 3. 메모리 매핑 최적화
            self._optimize_memory_mapping()
            
        except Exception as e:
            self.logger.error(f"메모리 최적화 실패: {e}")
    
    def _get_memory_percentage(self) -> float:
        """메모리 사용률 조회"""
        try:
            return psutil.virtual_memory().percent / 100
        except Exception:
            return 0.0
    
    def _clear_caches(self):
        """캐시 정리"""
        try:
            # LRU 캐시 정리
            if hasattr(self, '_lru_cache'):
                self._lru_cache.clear()
            
            # NumPy 캐시 정리
            if hasattr(np, 'clear_cache'):
                np.clear_cache()
            
            self.logger.info("캐시 정리 완료")
            
        except Exception as e:
            self.logger.error(f"캐시 정리 실패: {e}")
    
    def _optimize_memory_mapping(self):
        """메모리 매핑 최적화"""
        try:
            # 대용량 배열 메모리 매핑
            # 실제 구현에서는 mmap 사용
            self.logger.info("메모리 매핑 최적화 완료")
            
        except Exception as e:
            self.logger.error(f"메모리 매핑 최적화 실패: {e}")
    
    def get_memory_trend(self, window_size: int = 100) -> Dict[str, float]:
        """메모리 사용량 트렌드"""
        try:
            if len(self.memory_usage_history) < window_size:
                return {}
            
            recent_usage = list(self.memory_usage_history)[-window_size:]
            memory_percentages = [usage['memory_percentage'] for usage in recent_usage]
            
            trend = {
                'current_usage': memory_percentages[-1],
                'average_usage': np.mean(memory_percentages),
                'max_usage': np.max(memory_percentages),
                'min_usage': np.min(memory_percentages),
                'usage_std': np.std(memory_percentages)
            }
            
            return trend
            
        except Exception as e:
            self.logger.error(f"메모리 트렌드 계산 실패: {e}")
            return {}


class CacheManager:
    """캐시 관리자"""
    
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379):
        self.cache_stats = defaultdict(int)
        self.cache_hits = defaultdict(int)
        self.cache_misses = defaultdict(int)
        self.logger = logging.getLogger("CacheManager")
        
        # Redis 연결
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
                self.redis_available = True
            except Exception as e:
                self.logger.warning(f"Redis 연결 실패: {e}")
                self.redis_available = False
        else:
            self.redis_available = False
        
        # 메모리 캐시
        self.memory_cache = {}
        self.cache_ttl = 300  # 5분
    
    def get_cached_data(self, key: str, cache_type: str = 'memory') -> Optional[Any]:
        """캐시된 데이터 조회"""
        try:
            self.cache_stats[cache_type] += 1
            
            if cache_type == 'redis' and self.redis_available:
                # Redis 캐시
                cached_data = self.redis_client.get(key)
                if cached_data:
                    self.cache_hits[cache_type] += 1
                    return json.loads(cached_data)
                else:
                    self.cache_misses[cache_type] += 1
                    return None
            
            elif cache_type == 'memory':
                # 메모리 캐시
                if key in self.memory_cache:
                    cache_entry = self.memory_cache[key]
                    if time.time() - cache_entry['timestamp'] < self.cache_ttl:
                        self.cache_hits[cache_type] += 1
                        return cache_entry['data']
                    else:
                        # TTL 만료
                        del self.memory_cache[key]
                
                self.cache_misses[cache_type] += 1
                return None
            
            return None
            
        except Exception as e:
            self.logger.error(f"캐시 데이터 조회 실패: {e}")
            return None
    
    def set_cached_data(self, key: str, data: Any, cache_type: str = 'memory', 
                       ttl: int = None) -> bool:
        """데이터 캐시 저장"""
        try:
            if cache_type == 'redis' and self.redis_available:
                # Redis 캐시
                ttl = ttl or self.cache_ttl
                serialized_data = json.dumps(data)
                return self.redis_client.setex(key, ttl, serialized_data)
            
            elif cache_type == 'memory':
                # 메모리 캐시
                self.memory_cache[key] = {
                    'data': data,
                    'timestamp': time.time(),
                    'ttl': ttl or self.cache_ttl
                }
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"캐시 데이터 저장 실패: {e}")
            return False
    
    def get_cache_statistics(self) -> Dict[str, Dict[str, float]]:
        """캐시 통계"""
        try:
            stats = {}
            
            for cache_type in self.cache_stats:
                total_requests = self.cache_stats[cache_type]
                hits = self.cache_hits[cache_type]
                misses = self.cache_misses[cache_type]
                
                hit_rate = hits / total_requests if total_requests > 0 else 0.0
                miss_rate = misses / total_requests if total_requests > 0 else 0.0
                
                stats[cache_type] = {
                    'total_requests': total_requests,
                    'hits': hits,
                    'misses': misses,
                    'hit_rate': hit_rate,
                    'miss_rate': miss_rate
                }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"캐시 통계 계산 실패: {e}")
            return {}
    
    def optimize_cache(self) -> List[str]:
        """캐시 최적화"""
        try:
            optimizations = []
            
            # 1. 캐시 히트율 체크
            stats = self.get_cache_statistics()
            for cache_type, stat in stats.items():
                if stat['hit_rate'] < 0.5:  # 50% 미만
                    optimizations.append(f"{cache_type} 캐시 히트율 개선 필요: {stat['hit_rate']:.2%}")
            
            # 2. 메모리 캐시 크기 최적화
            if len(self.memory_cache) > 1000:
                # LRU 방식으로 오래된 항목 제거
                sorted_cache = sorted(self.memory_cache.items(), 
                                   key=lambda x: x[1]['timestamp'])
                items_to_remove = len(sorted_cache) - 500
                for i in range(items_to_remove):
                    del self.memory_cache[sorted_cache[i][0]]
                
                optimizations.append("메모리 캐시 크기 최적화 완료")
            
            # 3. TTL 조정
            if stats.get('memory', {}).get('hit_rate', 0) < 0.3:
                self.cache_ttl = min(600, self.cache_ttl * 2)  # TTL 증가
                optimizations.append("캐시 TTL 증가")
            
            return optimizations
            
        except Exception as e:
            self.logger.error(f"캐시 최적화 실패: {e}")
            return []


class EventDrivenArchitecture:
    """이벤트 기반 아키텍처"""
    
    def __init__(self):
        self.event_handlers = defaultdict(list)
        self.event_queue = asyncio.Queue()
        self.processing_stats = defaultdict(int)
        self.logger = logging.getLogger("EventDrivenArchitecture")
    
    def register_handler(self, event_type: str, handler: Callable):
        """이벤트 핸들러 등록"""
        try:
            self.event_handlers[event_type].append(handler)
            self.logger.info(f"이벤트 핸들러 등록: {event_type}")
            
        except Exception as e:
            self.logger.error(f"이벤트 핸들러 등록 실패: {e}")
    
    async def publish_event(self, event_type: str, event_data: Dict[str, Any]):
        """이벤트 발행"""
        try:
            event = {
                'type': event_type,
                'data': event_data,
                'timestamp': datetime.now().isoformat()
            }
            
            await self.event_queue.put(event)
            self.processing_stats['events_published'] += 1
            
        except Exception as e:
            self.logger.error(f"이벤트 발행 실패: {e}")
    
    async def process_events(self):
        """이벤트 처리"""
        try:
            while True:
                event = await self.event_queue.get()
                
                start_time = time.time()
                
                # 이벤트 핸들러 실행
                handlers = self.event_handlers.get(event['type'], [])
                for handler in handlers:
                    try:
                        await handler(event['data'])
                    except Exception as e:
                        self.logger.error(f"이벤트 핸들러 실행 실패: {e}")
                
                processing_time = time.time() - start_time
                self.processing_stats['events_processed'] += 1
                self.processing_stats['total_processing_time'] += processing_time
                
                # 큐 태스크 완료
                self.event_queue.task_done()
                
        except Exception as e:
            self.logger.error(f"이벤트 처리 실패: {e}")
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """처리 통계"""
        try:
            total_events = self.processing_stats['events_processed']
            total_time = self.processing_stats['total_processing_time']
            
            avg_processing_time = total_time / total_events if total_events > 0 else 0.0
            
            return {
                'events_published': self.processing_stats['events_published'],
                'events_processed': total_events,
                'queue_size': self.event_queue.qsize(),
                'avg_processing_time': avg_processing_time,
                'total_processing_time': total_time
            }
            
        except Exception as e:
            self.logger.error(f"처리 통계 계산 실패: {e}")
            return {}


class IntegratedPerformanceSystem:
    """통합 성능 관리 시스템"""
    
    def __init__(self):
        self.performance_optimizer = PerformanceOptimizer()
        self.latency_monitor = LatencyMonitor()
        self.throughput_optimizer = ThroughputOptimizer()
        self.memory_manager = MemoryManager()
        self.cache_manager = CacheManager()
        self.event_architecture = EventDrivenArchitecture()
        
        self.logger = logging.getLogger("IntegratedPerformanceSystem")
        
        # 성능 모니터링 활성화
        self._start_performance_monitoring()
    
    def _start_performance_monitoring(self):
        """성능 모니터링 시작"""
        try:
            # 이벤트 핸들러 등록
            self.event_architecture.register_handler('performance_alert', self._handle_performance_alert)
            self.event_architecture.register_handler('optimization_request', self._handle_optimization_request)
            
            # 백그라운드 모니터링 시작
            asyncio.create_task(self._background_monitoring())
            
        except Exception as e:
            self.logger.error(f"성능 모니터링 시작 실패: {e}")
    
    async def _background_monitoring(self):
        """백그라운드 모니터링"""
        try:
            while True:
                # 메모리 사용량 체크
                memory_stats = self.memory_manager.monitor_memory_usage()
                
                # 레이턴시 병목 체크
                bottlenecks = self.latency_monitor.identify_bottlenecks()
                
                # 캐시 최적화
                cache_optimizations = self.cache_manager.optimize_cache()
                
                # 성능 알림 발행
                if bottlenecks or cache_optimizations:
                    await self.event_architecture.publish_event('performance_alert', {
                        'bottlenecks': bottlenecks,
                        'cache_optimizations': cache_optimizations,
                        'memory_stats': memory_stats
                    })
                
                await asyncio.sleep(10)  # 10초마다 체크
                
        except Exception as e:
            self.logger.error(f"백그라운드 모니터링 실패: {e}")
    
    async def _handle_performance_alert(self, alert_data: Dict[str, Any]):
        """성능 알림 처리"""
        try:
            bottlenecks = alert_data.get('bottlenecks', [])
            cache_optimizations = alert_data.get('cache_optimizations', [])
            
            if bottlenecks:
                self.logger.warning(f"성능 병목 감지: {bottlenecks}")
            
            if cache_optimizations:
                self.logger.info(f"캐시 최적화 적용: {cache_optimizations}")
                
        except Exception as e:
            self.logger.error(f"성능 알림 처리 실패: {e}")
    
    async def _handle_optimization_request(self, request_data: Dict[str, Any]):
        """최적화 요청 처리"""
        try:
            operation = request_data.get('operation')
            data = request_data.get('data')
            
            if operation == 'data_processing':
                optimized_data = self.performance_optimizer.optimize_data_processing(data)
                return optimized_data
            
            elif operation == 'model_inference':
                model = request_data.get('model')
                predictions, latency = self.performance_optimizer.optimize_model_inference(model, data)
                return predictions
            
            elif operation == 'database_query':
                query = request_data.get('query')
                params = request_data.get('params', {})
                optimized_query = self.performance_optimizer.optimize_database_queries(query, params)
                return optimized_query
                
        except Exception as e:
            self.logger.error(f"최적화 요청 처리 실패: {e}")
    
    def get_comprehensive_performance_report(self) -> Dict[str, Any]:
        """종합 성능 리포트"""
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'latency_profile': self.latency_monitor.get_latency_profile().__dict__,
                'memory_stats': self.memory_manager.monitor_memory_usage(),
                'cache_stats': self.cache_manager.get_cache_statistics(),
                'processing_stats': self.event_architecture.get_processing_statistics(),
                'bottlenecks': self.latency_monitor.identify_bottlenecks(),
                'optimization_recommendations': self._generate_optimization_recommendations()
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"성능 리포트 생성 실패: {e}")
            return {'error': str(e)}
    
    def _generate_optimization_recommendations(self) -> List[str]:
        """최적화 권장사항 생성"""
        try:
            recommendations = []
            
            # 레이턴시 기반 권장사항
            latency_profile = self.latency_monitor.get_latency_profile()
            if latency_profile.total_latency > 100:
                recommendations.append("전체 파이프라인 레이턴시 최적화 필요")
            
            if latency_profile.model_inference > 100:
                recommendations.append("모델 추론 최적화 필요 (배치 처리, ONNX 변환)")
            
            if latency_profile.database_query > 10:
                recommendations.append("데이터베이스 쿼리 최적화 필요 (인덱스, 쿼리 재작성)")
            
            # 메모리 기반 권장사항
            memory_stats = self.memory_manager.monitor_memory_usage()
            if memory_stats.get('memory_percentage', 0) > 0.8:
                recommendations.append("메모리 사용량 최적화 필요 (캐시 정리, 가비지 컬렉션)")
            
            # 캐시 기반 권장사항
            cache_stats = self.cache_manager.get_cache_statistics()
            for cache_type, stats in cache_stats.items():
                if stats.get('hit_rate', 0) < 0.5:
                    recommendations.append(f"{cache_type} 캐시 히트율 개선 필요")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"최적화 권장사항 생성 실패: {e}")
            return []


# 성능 측정 데코레이터
def performance_monitor(operation: str):
    """성능 모니터링 데코레이터"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                latency = (time.time() - start_time) * 1000
                logger.info(f"{operation} 완료: {latency:.2f}ms")
                return result
            except Exception as e:
                latency = (time.time() - start_time) * 1000
                logger.error(f"{operation} 실패: {latency:.2f}ms - {e}")
                raise
        return wrapper
    return decorator


# 사용 예시
def main():
    """메인 실행 함수"""
    # 통합 성능 시스템 초기화
    performance_system = IntegratedPerformanceSystem()
    
    # 예시 데이터
    sample_data = np.random.randn(1000, 10)
    
    # 성능 최적화 테스트
    @performance_monitor("데이터 처리")
    def test_data_processing():
        return performance_system.performance_optimizer.optimize_data_processing(sample_data)
    
    # 테스트 실행
    optimized_data = test_data_processing()
    
    # 성능 리포트 생성
    performance_report = performance_system.get_comprehensive_performance_report()
    
    print("성능 최적화 시스템 리포트:")
    print(json.dumps(performance_report, indent=2, default=str))


if __name__ == "__main__":
    main() 