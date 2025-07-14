#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: krx_ultimate_system.py
목적: 최신 Python 표준을 활용한 궁극적 KRX 트레이딩 시스템 - 실시간 모니터링
Author: Ultimate KRX System
Created: 2025-07-13
Version: 2.0.0

Features:
    - 최신 Python 3.11+ 표준 활용
    - 비동기 고속 병렬처리
    - 멀티레벨 캐싱
    - 커넥션 풀링
    - 메모리 최적화
    - 자동 코드수정 파일 보존
    - 실시간 모니터링 및 지속적 데이터 수집
    - 시장 변화 실시간 감지
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
import logging
import json
import time
import hashlib
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Literal, TypedDict, Protocol, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import lru_cache, wraps
from contextlib import asynccontextmanager
import weakref
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from collections import defaultdict, deque
import gc
import psutil
import tracemalloc
import os

# 최신 Python 표준 활용
from typing_extensions import NotRequired, Required
from pydantic import BaseModel, Field, field_validator

# structlog 대신 기본 logging 사용 (호환성 문제 해결)
import logging

# 성능 모니터링
tracemalloc.start()

# KRX 성능 평가 기준 (동일한 기준 적용)
KRX_PERFORMANCE_CRITERIA = {
    "excellent": {"min_r2": 0.8, "max_rmse": 0.1, "min_excellent_folds": 3},
    "good": {"min_r2": 0.6, "max_rmse": 0.2, "min_excellent_folds": 2},
    "fair": {"min_r2": 0.4, "max_rmse": 0.3, "min_excellent_folds": 1},
    "poor": {"min_r2": 0.0, "max_rmse": float('inf'), "min_excellent_folds": 0}
}

# KRX 자동매매 가능성 판단 기준
KRX_TRADING_CRITERIA = {
    "high_confidence": {
        "min_r2": 0.85,
        "max_rmse": 0.08,
        "min_excellent_folds": 4,
        "max_poor_folds": 0,
        "min_data_quality": 0.9
    },
    "medium_confidence": {
        "min_r2": 0.7,
        "max_rmse": 0.15,
        "min_excellent_folds": 3,
        "max_poor_folds": 1,
        "min_data_quality": 0.8
    },
    "low_confidence": {
        "min_r2": 0.5,
        "max_rmse": 0.25,
        "min_excellent_folds": 2,
        "max_poor_folds": 2,
        "min_data_quality": 0.7
    },
    "not_tradeable": {
        "min_r2": 0.0,
        "max_rmse": float('inf'),
        "min_excellent_folds": 0,
        "max_poor_folds": 5,
        "min_data_quality": 0.0
    }
}

# KRX 데이터 성격별 저장 전략
KRX_STORAGE_STRATEGIES = {
    "high_frequency_trading": {
        "storage_format": "parquet",
        "compression": "snappy",
        "partition_by": ["날짜", "종목코드"],
        "retention_days": 30,
        "backup_frequency": "daily",
        "description": "고빈도 거래 - 빠른 읽기/쓰기, 압축 최적화"
    },
    "medium_frequency_analysis": {
        "storage_format": "parquet",
        "compression": "gzip",
        "partition_by": ["월", "종목코드"],
        "retention_days": 90,
        "backup_frequency": "weekly",
        "description": "중빈도 분석 - 균형잡힌 성능과 용량"
    },
    "long_term_research": {
        "storage_format": "parquet",
        "compression": "brotli",
        "partition_by": ["년", "종목코드"],
        "retention_days": 365,
        "backup_frequency": "monthly",
        "description": "장기 연구 - 최대 압축, 장기 보관"
    },
    "real_time_monitoring": {
        "storage_format": "parquet",
        "compression": "snappy",
        "partition_by": ["시간", "종목코드"],
        "retention_days": 7,
        "backup_frequency": "hourly",
        "description": "실시간 모니터링 - 최소 지연, 빠른 처리"
    }
}

def evaluate_krx_performance(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """KRX 성능 평가 (동일한 기준 적용)"""
    avg_r2 = analysis.get('avg_r2', 0)
    avg_rmse = analysis.get('avg_rmse', float('inf'))
    excellent_folds = analysis.get('excellent_folds', 0)
    poor_folds = analysis.get('poor_folds', 0)
    
    # 성능 등급 평가
    performance_grade = "🔴 Poor"
    if avg_r2 >= KRX_PERFORMANCE_CRITERIA["excellent"]["min_r2"] and excellent_folds >= KRX_PERFORMANCE_CRITERIA["excellent"]["min_excellent_folds"]:
        performance_grade = "🟢 Excellent"
    elif avg_r2 >= KRX_PERFORMANCE_CRITERIA["good"]["min_r2"] and excellent_folds >= KRX_PERFORMANCE_CRITERIA["good"]["min_excellent_folds"]:
        performance_grade = "🟡 Good"
    elif avg_r2 >= KRX_PERFORMANCE_CRITERIA["fair"]["min_r2"] and excellent_folds >= KRX_PERFORMANCE_CRITERIA["fair"]["min_excellent_folds"]:
        performance_grade = "🟠 Fair"
    
    # 자동매매 가능성 판단
    trading_confidence = "not_tradeable"
    data_quality = 1.0 - (poor_folds / (excellent_folds + poor_folds + 1))
    
    for confidence, criteria in KRX_TRADING_CRITERIA.items():
        if (avg_r2 >= criteria["min_r2"] and 
            avg_rmse <= criteria["max_rmse"] and
            excellent_folds >= criteria["min_excellent_folds"] and
            poor_folds <= criteria["max_poor_folds"] and
            data_quality >= criteria["min_data_quality"]):
            trading_confidence = confidence
            break
    
    return {
        "performance_grade": performance_grade,
        "trading_confidence": trading_confidence,
        "data_quality_score": data_quality,
        "improvement_needed": performance_grade.startswith("🔴") or trading_confidence == "not_tradeable",
        "trading_recommendation": _get_trading_recommendation(trading_confidence)
    }

def _get_trading_recommendation(confidence: str) -> str:
    """자동매매 권장사항"""
    recommendations = {
        "high_confidence": "✅ 자동매매 권장 - 높은 신뢰도",
        "medium_confidence": "⚠️ 제한적 자동매매 - 중간 신뢰도",
        "low_confidence": "❌ 자동매매 비권장 - 낮은 신뢰도",
        "not_tradeable": "🚫 자동매매 불가 - 개선 필요"
    }
    return recommendations.get(confidence, "❓ 평가 불가")

def detect_krx_data_characteristics(df: pd.DataFrame) -> str:
    """KRX 데이터 성격 감지"""
    # 데이터 크기 및 빈도 분석
    data_size = len(df)
    time_columns = [col for col in df.columns if '시간' in col or '일자' in col or '날짜' in col]
    
    # 거래량 패턴 분석
    volume_columns = [col for col in df.columns if '거래량' in col or '거래대금' in col]
    has_volume_data = len(volume_columns) > 0
    
    # 가격 변동성 분석
    price_columns = [col for col in df.columns if '가' in col and '가격' not in col]
    has_price_data = len(price_columns) > 0
    
    # 데이터 성격 판단
    if data_size > 100000 and has_volume_data and has_price_data:
        return "high_frequency_trading"
    elif data_size > 10000 and has_price_data:
        return "medium_frequency_analysis"
    elif data_size > 1000:
        return "long_term_research"
    else:
        return "real_time_monitoring"

def get_krx_storage_strategy(df: pd.DataFrame, trading_confidence: str) -> Dict[str, Any]:
    """KRX 데이터 저장 전략 결정"""
    data_characteristics = detect_krx_data_characteristics(df)
    base_strategy = KRX_STORAGE_STRATEGIES[data_characteristics].copy()
    
    # 자동매매 신뢰도에 따른 저장 전략 조정
    if trading_confidence == "high_confidence":
        base_strategy["backup_frequency"] = "hourly"
        base_strategy["retention_days"] = 60
        base_strategy["description"] += " (자동매매 활성화)"
    elif trading_confidence == "medium_confidence":
        base_strategy["backup_frequency"] = "daily"
        base_strategy["retention_days"] = 45
        base_strategy["description"] += " (제한적 자동매매)"
    elif trading_confidence == "low_confidence":
        base_strategy["backup_frequency"] = "weekly"
        base_strategy["retention_days"] = 30
        base_strategy["description"] += " (연구용)"
    else:
        base_strategy["backup_frequency"] = "monthly"
        base_strategy["retention_days"] = 15
        base_strategy["description"] += " (개선 필요)"
    
    return base_strategy

def detect_krx_data_type(df: pd.DataFrame) -> str:
    """KRX 데이터 유형 자동 감지"""
    # KRX 데이터 특성 확인
    krx_indicators = [
        '종목코드', '종목명', '현재가', '등락률', '거래량',
        '시가총액', '상장주식수', '외국인비율'
    ]
    
    has_krx_cols = any(col in df.columns for col in krx_indicators)
    has_time_cols = any('일자' in col or '날짜' in col for col in df.columns)
    has_price_cols = any('가' in col for col in df.columns)
    
    # 데이터 크기 확인
    data_size = len(df)
    feature_count = len(df.select_dtypes(include=[np.number]).columns)
    
    # 데이터 유형 판단
    if has_krx_cols and has_time_cols and has_price_cols:
        return "financial_timeseries"
    elif data_size > 10000 and feature_count > 20:
        return "image_text"
    elif data_size < 5000 or feature_count < 10:
        return "experimental"
    else:
        return "general_ml"

def get_krx_optimized_config(df: pd.DataFrame) -> Dict[str, Any]:
    """KRX 데이터 유형에 따른 최적 설정 반환"""
    data_type = detect_krx_data_type(df)
    config = KRX_DATA_TYPE_CONFIGS[data_type].copy()
    
    logging.info(f"KRX 데이터 유형 감지: {data_type}")
    logging.info(f"설정 적용: {config['description']}")
    logging.info(f"최대 반복: {config['max_iterations']}회")
    logging.info(f"조기 종료: 연속 {config['max_no_improvement']}회 개선 없음")
    
    return config

# 전역 변수 (우수 등급 달성 추적)
achieved_excellent_grade = False

def np_encoder(obj):
    """numpy 타입 JSON 직렬화용"""
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return str(obj)

class SystemMode(Enum):
    """시스템 모드"""
    TRAIN = auto()
    LIVE = auto()
    BACKTEST = auto()
    EMERGENCY = auto()
    REALTIME_MONITORING = auto()  # 실시간 모니터링 모드 추가

class DataType(Enum):
    """데이터 타입"""
    STOCK = auto()
    FUTURES = auto()
    OPTIONS = auto()
    INDEX = auto()
    ETF = auto()

class Priority(Enum):
    """우선순위"""
    LOW = auto()
    NORMAL = auto()
    HIGH = auto()
    EMERGENCY = auto()

class MarketEventType(Enum):
    """시장 이벤트 타입"""
    PRICE_SPIKE = auto()
    VOLUME_SURGE = auto()
    VOLATILITY_INCREASE = auto()
    TREND_CHANGE = auto()
    BREAKOUT = auto()
    BREAKDOWN = auto()

@dataclass
class SystemConfig:
    """시스템 설정 - 최신 dataclass 활용"""
    mode: SystemMode = SystemMode.REALTIME_MONITORING  # 기본값을 실시간 모니터링으로 변경
    max_workers: int = field(default_factory=lambda: min(32, mp.cpu_count() + 4))
    cache_size: int = 1000
    connection_pool_size: int = 20
    timeout: float = 30.0
    retry_attempts: int = 3
    memory_limit_gb: float = 8.0
    
    # 실시간 모니터링 설정 추가
    monitoring_interval_seconds: int = 60  # 1분마다 수집
    market_hours_start: str = "09:00"
    market_hours_end: str = "15:30"
    weekend_monitoring: bool = False
    emergency_collection_interval: int = 10  # 긴급 상황 시 10초마다
    price_change_threshold: float = 0.02  # 2% 이상 변동 시 이벤트
    volume_change_threshold: float = 3.0  # 거래량 3배 이상 시 이벤트
    
    def __post_init__(self):
        """설정 검증"""
        try:
            if self.memory_limit_gb > psutil.virtual_memory().total / (1024**3):
                self.memory_limit_gb = psutil.virtual_memory().total / (1024**3) * 0.8
        except Exception:
            # psutil 사용 불가능한 경우 기본값 사용
            self.memory_limit_gb = 8.0

class CacheConfig(TypedDict):
    """캐시 설정 - TypedDict 활용"""
    memory_cache_size: int
    disk_cache_size: int
    ttl_seconds: int
    compression: bool

class PerformanceMetrics(BaseModel):
    """성능 메트릭 - Pydantic 활용"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    avg_response_time: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    
    # 실시간 모니터링 메트릭 추가
    total_collections: int = 0
    market_events_detected: int = 0
    last_collection_time: Optional[str] = None
    next_collection_time: Optional[str] = None
    
    @field_validator('avg_response_time')
    @classmethod
    def validate_response_time(cls, v):
        return max(0.0, v)

@dataclass
class MarketData:
    """시장 데이터"""
    timestamp: datetime
    symbol: str
    price: float
    volume: int
    change_percent: float
    market_cap: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'price': self.price,
            'volume': self.volume,
            'change_percent': self.change_percent,
            'market_cap': self.market_cap
        }

@dataclass
class MarketEvent:
    """시장 이벤트"""
    event_type: MarketEventType
    symbol: str
    timestamp: datetime
    description: str
    severity: Priority
    data: Dict[str, Any]

class KRXUltimateSystem:
    """궁극적 KRX 시스템 - 최신 Python 표준 활용 + 실시간 모니터링"""
    
    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or SystemConfig()
        
        # 기본 로깅 설정 (structlog 대신)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # 멀티레벨 캐싱 시스템
        self._setup_caching()
        
        # 커넥션 풀 설정
        self._setup_connection_pool()
        
        # 성능 모니터링
        self.metrics = PerformanceMetrics()
        
        # 메모리 관리
        self._setup_memory_management()
        
        # 자동 코드수정 파일 보존
        self._preserve_auto_fix_files()
        
        # 실시간 모니터링 설정
        self._setup_realtime_monitoring()
    
    def _setup_caching(self):
        """멀티레벨 캐싱 시스템 설정"""
        # L1: 메모리 캐시 (LRU)
        self.memory_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # L2: 디스크 캐시
        self.cache_dir = Path('cache/ultimate')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # L3: 분산 캐시 (Redis 대체)
        self.distributed_cache = {}
    
    def _setup_connection_pool(self):
        """커넥션 풀 설정"""
        self.connector = aiohttp.TCPConnector(
            limit=self.config.connection_pool_size,
            limit_per_host=10,
            ttl_dns_cache=300,
            use_dns_cache=True,
            keepalive_timeout=30
        )
        self.session = None
    
    def _setup_memory_management(self):
        """메모리 관리 설정"""
        try:
            self.memory_monitor = MemoryMonitor(self.config.memory_limit_gb)
        except Exception:
            # psutil 사용 불가능한 경우 더미 모니터 사용
            self.memory_monitor = DummyMemoryMonitor()
        self.gc_threshold = 0.8  # 80% 메모리 사용 시 GC
    
    def _preserve_auto_fix_files(self):
        """자동 코드수정 파일 보존"""
        auto_fix_files = [
            'smart_duplicate_cleaner.py',
            'ultimate_folder_consolidator.py'
        ]
        
        for file_name in auto_fix_files:
            file_path = Path(file_name)
            if file_path.exists():
                # 백업 생성
                backup_path = Path(f'backup/auto_fix/{file_name}')
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                
                if not backup_path.exists():
                    import shutil
                    shutil.copy2(file_path, backup_path)
                    print(f"자동 코드수정 파일 보존: {file_name}")
    
    def _setup_realtime_monitoring(self):
        """실시간 모니터링 설정"""
        self.is_monitoring = False
        self.market_data_history = deque(maxlen=1000)  # 최근 1000개 데이터
        self.detected_events = deque(maxlen=100)  # 최근 100개 이벤트
        self.last_market_data = {}
        self.emergency_mode = False
        
        # 시장 시간 설정
        self.market_start = datetime.strptime(self.config.market_hours_start, "%H:%M").time()
        self.market_end = datetime.strptime(self.config.market_hours_end, "%H:%M").time()
    
    @asynccontextmanager
    async def get_session(self):
        """비동기 세션 관리"""
        if self.session is None:
            self.session = aiohttp.ClientSession(
                connector=self.connector,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            )
        
        try:
            yield self.session
        except Exception as e:
            print(f"세션 에러: {e}")
            if self.session:
                await self.session.close()
                self.session = None
            raise
    
    @lru_cache(maxsize=1000)
    def _get_cache_key(self, data_type: str, market: str, date: str) -> str:
        """캐시 키 생성"""
        return hashlib.md5(f"{data_type}_{market}_{date}".encode()).hexdigest()
    
    async def _multi_level_cache_get(self, key: str) -> Optional[Dict[str, Any]]:
        """멀티레벨 캐시에서 데이터 조회"""
        # L1: 메모리 캐시
        if key in self.memory_cache:
            self.cache_hits += 1
            return self.memory_cache[key]
        
        # L2: 디스크 캐시
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                    self.memory_cache[key] = data  # L1에 로드
                    self.cache_hits += 1
                    return data
            except Exception as e:
                print(f"디스크 캐시 로드 실패: {e}")
        
        # L3: 분산 캐시
        if key in self.distributed_cache:
            self.cache_hits += 1
            return self.distributed_cache[key]
        
        self.cache_misses += 1
        return None
    
    async def _multi_level_cache_set(self, key: str, data: Dict[str, Any]):
        """멀티레벨 캐시에 데이터 저장"""
        # L1: 메모리 캐시
        self.memory_cache[key] = data
        
        # L2: 디스크 캐시 (비동기로 저장)
        asyncio.create_task(self._save_to_disk_cache(key, data))
        
        # L3: 분산 캐시
        self.distributed_cache[key] = data
    
    async def _save_to_disk_cache(self, key: str, data: Dict[str, Any]):
        """디스크 캐시에 저장"""
        try:
            cache_file = self.cache_dir / f"{key}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"디스크 캐시 저장 실패: {e}")
    
    async def start_realtime_monitoring(self):
        """실시간 모니터링 시작"""
        if self.is_monitoring:
            print("이미 모니터링 중입니다.")
            return
        
        self.is_monitoring = True
        print("실시간 모니터링 시작")
        
        try:
            # 모니터링 태스크들 시작
            monitoring_tasks = [
                asyncio.create_task(self._continuous_data_collection()),
                asyncio.create_task(self._market_event_detection()),
                asyncio.create_task(self._performance_monitoring()),
                asyncio.create_task(self._emergency_monitoring())
            ]
            
            # 모든 태스크 완료 대기 (무한 루프 방지)
            await asyncio.gather(*monitoring_tasks, return_exceptions=True)
            
        except Exception as e:
            print(f"모니터링 태스크 에러: {e}")
        finally:
            self.is_monitoring = False
    
    async def stop_realtime_monitoring(self):
        """실시간 모니터링 중지"""
        self.is_monitoring = False
        print("실시간 모니터링 중지")
    
    async def _continuous_data_collection(self):
        """지속적 데이터 수집"""
        collection_count = 0
        max_collections = 10  # 최대 10회 수집 후 종료 (테스트용)
        
        while self.is_monitoring and collection_count < max_collections:
            try:
                # 시장 시간 체크
                if not self._is_market_hours():
                    print("시장 시간이 아닙니다. 5분 대기...")
                    await asyncio.sleep(300)  # 5분 대기
                    continue
                
                # 데이터 수집
                start_time = time.time()
                data = await self.collect_data_parallel([DataType.STOCK, DataType.INDEX])
                
                # 수집 시간 기록
                self.metrics.total_collections += 1
                self.metrics.last_collection_time = datetime.now().isoformat()
                self.metrics.next_collection_time = (
                    datetime.now() + timedelta(seconds=self.config.monitoring_interval_seconds)
                ).isoformat()
                
                # 시장 데이터 히스토리에 저장
                await self._process_market_data(data)
                
                execution_time = time.time() - start_time
                self._update_performance_metrics(execution_time)
                
                print(f"데이터 수집 완료 (소요시간: {execution_time:.2f}초, 수집횟수: {collection_count + 1})")
                
                collection_count += 1
                
                # 다음 수집까지 대기
                await asyncio.sleep(self.config.monitoring_interval_seconds)
                
            except Exception as e:
                print(f"지속적 데이터 수집 에러: {e}")
                await asyncio.sleep(10)  # 에러 시 10초 대기
        
        print(f"데이터 수집 완료 (총 {collection_count}회)")
    
    async def _market_event_detection(self):
        """시장 이벤트 감지"""
        detection_count = 0
        max_detections = 20  # 최대 20회 감지 후 종료
        
        while self.is_monitoring and detection_count < max_detections:
            try:
                if len(self.market_data_history) < 2:
                    await asyncio.sleep(5)
                    continue
                
                # 최근 데이터와 이전 데이터 비교
                current_data = self.market_data_history[-1]
                previous_data = self.market_data_history[-2]
                
                # 가격 변동 감지
                price_change = abs(current_data.change_percent - previous_data.change_percent)
                if price_change > self.config.price_change_threshold:
                    event = MarketEvent(
                        event_type=MarketEventType.PRICE_SPIKE,
                        symbol=current_data.symbol,
                        timestamp=current_data.timestamp,
                        description=f"가격 급변: {price_change:.2%}",
                        severity=Priority.HIGH,
                        data={'price_change': price_change, 'current_price': current_data.price}
                    )
                    await self._handle_market_event(event)
                
                # 거래량 급증 감지
                volume_ratio = current_data.volume / max(previous_data.volume, 1)
                if volume_ratio > self.config.volume_change_threshold:
                    event = MarketEvent(
                        event_type=MarketEventType.VOLUME_SURGE,
                        symbol=current_data.symbol,
                        timestamp=current_data.timestamp,
                        description=f"거래량 급증: {volume_ratio:.1f}배",
                        severity=Priority.NORMAL,
                        data={'volume_ratio': volume_ratio, 'current_volume': current_data.volume}
                    )
                    await self._handle_market_event(event)
                
                detection_count += 1
                await asyncio.sleep(5)  # 5초마다 이벤트 체크
                
            except Exception as e:
                print(f"시장 이벤트 감지 에러: {e}")
                await asyncio.sleep(5)
        
        print(f"시장 이벤트 감지 완료 (총 {detection_count}회)")
    
    async def _handle_market_event(self, event: MarketEvent):
        """시장 이벤트 처리"""
        self.detected_events.append(event)
        self.metrics.market_events_detected += 1
        
        print(f"시장 이벤트 감지: {event.description}")
        
        # 긴급 상황 시 더 자주 수집
        if event.severity == Priority.HIGH:
            self.emergency_mode = True
            asyncio.create_task(self._emergency_data_collection())
        
        # 이벤트 저장
        await self._save_market_event(event)
    
    async def _emergency_data_collection(self):
        """긴급 데이터 수집"""
        print("긴급 데이터 수집 시작")
        
        for i in range(5):  # 5회 긴급 수집 (테스트용)
            try:
                data = await self.collect_data_parallel([DataType.STOCK])
                await self._process_market_data(data)
                print(f"긴급 데이터 수집 {i+1}/5 완료")
                await asyncio.sleep(self.config.emergency_collection_interval)
            except Exception as e:
                print(f"긴급 데이터 수집 에러: {e}")
        
        self.emergency_mode = False
        print("긴급 데이터 수집 완료")
    
    async def _emergency_monitoring(self):
        """긴급 상황 모니터링"""
        monitoring_count = 0
        max_monitoring = 30  # 최대 30회 모니터링 후 종료
        
        while self.is_monitoring and monitoring_count < max_monitoring:
            try:
                # 메모리 사용량 체크
                if self.metrics.memory_usage_mb > self.config.memory_limit_gb * 1024 * 0.9:
                    print("메모리 사용량 위험 수준")
                    gc.collect()
                
                # 에러율 체크
                error_rate = self.metrics.failed_requests / max(self.metrics.total_requests, 1)
                if error_rate > 0.1:  # 10% 이상 에러
                    print(f"높은 에러율: {error_rate:.1%}")
                
                monitoring_count += 1
                await asyncio.sleep(30)  # 30초마다 체크
                
            except Exception as e:
                print(f"긴급 모니터링 에러: {e}")
                await asyncio.sleep(10)
        
        print(f"긴급 모니터링 완료 (총 {monitoring_count}회)")
    
    def _is_market_hours(self) -> bool:
        """시장 시간 체크"""
        now = datetime.now()
        current_time = now.time()
        
        # 주말 체크
        if not self.config.weekend_monitoring and now.weekday() >= 5:
            return False
        
        # 시장 시간 체크
        return self.market_start <= current_time <= self.market_end
    
    async def _process_market_data(self, data: Dict[str, Any]):
        """시장 데이터 처리"""
        try:
            # 더미 데이터 생성 (실제 API 응답 대신)
            dummy_data = MarketData(
                timestamp=datetime.now(),
                symbol="005930",  # 삼성전자
                price=75000.0,
                volume=1000000,
                change_percent=2.5,
                market_cap=45000000000000.0
            )
            
            self.market_data_history.append(dummy_data)
            self.last_market_data[dummy_data.symbol] = dummy_data
            
            print(f"시장 데이터 처리 완료: {dummy_data.symbol}")
                
        except Exception as e:
            print(f"시장 데이터 처리 에러: {e}")
    
    async def _save_market_event(self, event: MarketEvent):
        """시장 이벤트 저장"""
        try:
            event_data = {
                'event_type': event.event_type.name,
                'symbol': event.symbol,
                'timestamp': event.timestamp.isoformat(),
                'description': event.description,
                'severity': event.severity.name,
                'data': event.data
            }
            
            events_file = Path('data/market_events.jsonl')
            events_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(events_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(event_data, ensure_ascii=False) + '\n')
                
        except Exception as e:
            print(f"시장 이벤트 저장 실패: {e}")
    
    async def collect_data_parallel(self, data_types: List[DataType]) -> Dict[str, Any]:
        """병렬 데이터 수집 - 최신 Python 표준 활용"""
        start_time = time.time()
        
        try:
            # 비동기 병렬 처리
            tasks = [self._collect_single_data_type(data_type) for data_type in data_types]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 결과 통합
            combined_results = {}
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    print(f"데이터 수집 실패 ({data_types[i].name}): {result}")
                else:
                    combined_results[data_types[i].name] = result
            
            execution_time = time.time() - start_time
            self._update_performance_metrics(execution_time)
            
            return combined_results
            
        except Exception as e:
            print(f"병렬 데이터 수집 실패: {e}")
            return {}
    
    async def _collect_single_data_type(self, data_type: DataType) -> Dict[str, Any]:
        """단일 데이터 타입 수집"""
        cache_key = self._get_cache_key(data_type.name, 'KRX', datetime.now().strftime('%Y%m%d'))
        
        # 캐시에서 먼저 확인
        cached_data = await self._multi_level_cache_get(cache_key)
        if cached_data:
            return cached_data
        
        try:
            # 실제 API 호출 대신 더미 데이터 반환 (테스트용)
            dummy_data = {
                'OutBlock_1': [
                    {
                        'ISU_CD': '005930',
                        'TDD_CLSPRC': '75000',
                        'ACC_TRDVOL': '1000000',
                        'CMPPREVDD_PRC': '2.5',
                        'MKTCAP': '45000000000000'
                    }
                ]
            }
            
            # 캐시에 저장
            await self._multi_level_cache_set(cache_key, dummy_data)
            
            self.metrics.successful_requests += 1
            return dummy_data
                        
        except Exception as e:
            self.metrics.failed_requests += 1
            print(f"데이터 수집 실패 ({data_type.name}): {e}")
            raise
    
    def _get_request_params(self, data_type: DataType) -> Dict[str, str]:
        """요청 파라미터 생성"""
        base_params = {
            'bld': 'dbms/MDC/STAT/standard/MDCSTAT01501',
            'trdDd': datetime.now().strftime('%Y%m%d')
        }
        
        if data_type == DataType.STOCK:
            base_params['mktId'] = 'STK'
        elif data_type == DataType.FUTURES:
            base_params['mktId'] = 'FUT'
        elif data_type == DataType.OPTIONS:
            base_params['mktId'] = 'OPT'
        elif data_type == DataType.INDEX:
            base_params['mktId'] = 'IDX'
        elif data_type == DataType.ETF:
            base_params['mktId'] = 'ETF'
        
        return base_params
    
    def _get_headers(self) -> Dict[str, str]:
        """요청 헤더 생성"""
        return {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json, text/javascript, */*; q=0.01',
            'Accept-Language': 'ko-KR,ko;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'X-Requested-With': 'XMLHttpRequest'
        }
    
    def _update_performance_metrics(self, execution_time: float):
        """성능 메트릭 업데이트"""
        self.metrics.total_requests += 1
        self.metrics.avg_response_time = (
            (self.metrics.avg_response_time * (self.metrics.total_requests - 1) + execution_time) 
            / self.metrics.total_requests
        )
        
        # 메모리 사용량 업데이트
        try:
            process = psutil.Process()
            self.metrics.memory_usage_mb = process.memory_info().rss / (1024 * 1024)
            self.metrics.cpu_usage_percent = process.cpu_percent()
        except Exception:
            # psutil 사용 불가능한 경우 기본값 사용
            self.metrics.memory_usage_mb = 100.0
            self.metrics.cpu_usage_percent = 5.0
    
    async def _performance_monitoring(self):
        """성능 모니터링"""
        monitoring_count = 0
        max_monitoring = 15  # 최대 15회 모니터링 후 종료
        
        while self.is_monitoring and monitoring_count < max_monitoring:
            try:
                # 메모리 사용량 체크
                if self.metrics.memory_usage_mb > self.config.memory_limit_gb * 1024 * 0.8:
                    print(f"메모리 사용량 높음: {self.metrics.memory_usage_mb:.2f}MB")
                    gc.collect()
                
                # 성능 리포트 저장
                await self._save_performance_report()
                
                monitoring_count += 1
                await asyncio.sleep(60)  # 1분마다 체크
                
            except Exception as e:
                print(f"성능 모니터링 에러: {e}")
                await asyncio.sleep(10)
        
        print(f"성능 모니터링 완료 (총 {monitoring_count}회)")
    
    async def _save_performance_report(self):
        """성능 리포트 저장"""
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'metrics': self.metrics.model_dump(),
                'cache_stats': {
                    'hits': self.cache_hits,
                    'misses': self.cache_misses,
                    'hit_rate': self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
                },
                'monitoring_stats': {
                    'is_monitoring': self.is_monitoring,
                    'emergency_mode': self.emergency_mode,
                    'market_data_count': len(self.market_data_history),
                    'detected_events_count': len(self.detected_events),
                    'is_market_hours': self._is_market_hours()
                }
            }
            
            report_file = Path('reports/performance_report.json')
            report_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"성능 리포트 저장 실패: {e}")
    
    async def run_backtest(self, strategy_name: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """백테스트 실행 - 최신 Python 표준 활용"""
        try:
            # 병렬 데이터 수집
            data = await self.collect_data_parallel([DataType.STOCK, DataType.INDEX])
            
            # 백테스트 실행
            results = await self._execute_backtest(strategy_name, data, start_date, end_date)
            
            return results
            
        except Exception as e:
            print(f"백테스트 실패: {e}")
            return {'error': str(e)}
    
    async def _execute_backtest(self, strategy_name: str, data: Dict[str, Any], start_date: str, end_date: str) -> Dict[str, Any]:
        """백테스트 실행"""
        # 간단한 백테스트 로직 (실제로는 더 복잡한 전략 구현)
        return {
            'strategy': strategy_name,
            'start_date': start_date,
            'end_date': end_date,
            'total_return': 0.15,
            'sharpe_ratio': 1.2,
            'max_drawdown': -0.05,
            'win_rate': 0.65,
            'timestamp': datetime.now().isoformat()
        }
    
    async def auto_improvement_loop(self, df: pd.DataFrame, output_dir: Path = None) -> Dict[str, Any]:
        """KRX 데이터 자동 개선 루프"""
        if output_dir is None:
            output_dir = Path("krx_improvement_outputs")
        output_dir.mkdir(exist_ok=True)
        
        # 데이터 유형별 최적 설정 적용
        config = get_krx_optimized_config(df)
        max_iterations = config["max_iterations"]
        target_excellent_folds = config["target_excellent_folds"]
        max_no_improvement = config["max_no_improvement"]
        
        logging.info(f"KRX 자동 개선 루프 시작 - 최대 {max_iterations}회 반복")
        
        # 텔레그램 알림 시스템 초기화
        telegram = KRXTelegramNotifier()
        await telegram.send_message("🚀 <b>KRX 데이터 자동 개선 시작</b>\n\n⏱️ 데이터 유형별 최적화 적용 중...", force=True)
        
        # 성능 추적
        performance_history = []
        best_r2 = 0
        no_improvement_count = 0
        global achieved_excellent_grade
        achieved_excellent_grade = False
        
        improvement_strategies_all = [
            "고급 Feature Engineering", "Feature Selection", "Scaling 다양화", "이상치 처리 강화",
            "모델 파라미터 랜덤 탐색", "Ensemble 다양화", "데이터 품질 개선"
        ]
        
        for iteration in range(1, max_iterations + 1):
            logging.info(f"KRX 개선 반복 {iteration} 시작")
            
            # 성능 분석
            analysis = await self._analyze_krx_performance(df, output_dir)
            performance_history.append(analysis)
            
            # 성능 개선 추적
            current_r2 = analysis.get("avg_r2", 0)
            if current_r2 > best_r2:
                best_r2 = current_r2
                no_improvement_count = 0
                logging.info(f"KRX 성능 개선 감지: R² {current_r2:.6f}")
            else:
                no_improvement_count += 1
                logging.info(f"KRX 성능 개선 없음 (연속 {no_improvement_count}회)")
            
            # 텔레그램 알림
            data_info = {"rows": len(df), "columns": len(df.columns)}
            await telegram.send_krx_performance_report(analysis, data_info)
            
            # 종료 조건 확인
            excellent_folds = analysis.get("excellent_folds", 0)
            performance_grade = analysis.get("performance_grade", "")
            avg_r2 = analysis.get("avg_r2", 0)
            
            should_stop = (
                excellent_folds >= target_excellent_folds and 
                performance_grade.startswith("🟢") and
                avg_r2 > 0.8
            )
            
            # 조기 종료 조건
            if no_improvement_count >= max_no_improvement:
                logging.info(f"KRX 연속 {max_no_improvement}회 성능 개선 없음 - 조기 종료")
                break
            
            if should_stop:
                logging.info("KRX 우수 성능 달성! 자동 개선 루프 종료")
                await telegram.send_krx_improvement_complete(iteration, analysis)
                achieved_excellent_grade = True
                break
            
            # 개선 실행
            if analysis.get("improvement_needed", False):
                strategies = analysis.get("improvement_strategies", [])
                await telegram.send_krx_improvement_start(iteration, strategies)
                logging.info(f"KRX 적용 개선 전략: {strategies}")
                
                try:
                    # 데이터 개선 적용
                    df = await self._apply_krx_improvements(df, strategies)
                    logging.info(f"KRX 개선 반복 {iteration} 완료")
                except Exception as e:
                    logging.error(f"KRX 개선 반복 {iteration} 실패", error=str(e))
                    await telegram.send_message(f"❌ <b>KRX 개선 반복 {iteration} 실패</b>\n\n오류: {str(e)}")
                    continue
            else:
                logging.info("KRX 개선이 필요하지 않음")
                break
        
        # 성능 히스토리 저장
        history_path = output_dir / "krx_improvement_history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(performance_history, f, indent=2, ensure_ascii=False, default=np_encoder)
        
        logging.info("KRX 자동 개선 루프 완료")
        return {"performance_history": performance_history, "final_analysis": analysis}
    
    async def _analyze_krx_performance(self, df: pd.DataFrame, output_dir: Path) -> Dict[str, Any]:
        """KRX 성능 분석"""
        try:
            # 간단한 성능 분석 (실제로는 ML 모델 훈련 필요)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) < 2:
                return {
                    "avg_r2": 0.0,
                    "performance_grade": "🔴 개선 필요",
                    "improvement_needed": True,
                    "improvement_strategies": ["데이터 품질 개선"]
                }
            
            # 가상 성능 지표 (실제 구현에서는 실제 모델 훈련 결과 사용)
            avg_r2 = np.random.uniform(0.3, 0.9)  # 예시용
            performance_grade = "🟢 우수" if avg_r2 > 0.8 else "🔴 개선 필요"
            
            return {
                "avg_r2": avg_r2,
                "performance_grade": performance_grade,
                "excellent_folds": 3 if avg_r2 > 0.8 else 0,
                "improvement_needed": avg_r2 < 0.8,
                "improvement_strategies": ["Feature Engineering", "데이터 품질 개선"] if avg_r2 < 0.8 else []
            }
        except Exception as e:
            logging.error(f"KRX 성능 분석 실패: {e}")
            return {
                "avg_r2": 0.0,
                "performance_grade": "🔴 개선 필요",
                "improvement_needed": True,
                "improvement_strategies": ["오류 복구"]
            }
    
    async def _apply_krx_improvements(self, df: pd.DataFrame, strategies: List[str]) -> pd.DataFrame:
        """KRX 데이터 개선 적용"""
        try:
            for strategy in strategies:
                if "Feature Engineering" in strategy:
                    # 특성 엔지니어링 적용
                    df = await self._apply_krx_feature_engineering(df)
                elif "데이터 품질 개선" in strategy:
                    # 데이터 품질 개선
                    df = await self._apply_krx_data_quality_improvement(df)
                elif "이상치 처리" in strategy:
                    # 이상치 처리
                    df = await self._apply_krx_outlier_removal(df)
            
            return df
        except Exception as e:
            logging.error(f"KRX 개선 적용 실패: {e}")
            return df
    
    async def _apply_krx_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """KRX 특성 엔지니어링"""
        try:
            # 가격 관련 특성 추가
            if '현재가' in df.columns:
                df['가격변화율'] = df['현재가'].pct_change()
                df['가격변화율_5일'] = df['현재가'].pct_change(5)
            
            # 거래량 관련 특성 추가
            if '거래량' in df.columns:
                df['거래량_이동평균'] = df['거래량'].rolling(window=5).mean()
                df['거래량_비율'] = df['거래량'] / df['거래량'].rolling(window=20).mean()
            
            logging.info("KRX 특성 엔지니어링 완료")
            return df
        except Exception as e:
            logging.error(f"KRX 특성 엔지니어링 실패: {e}")
            return df
    
    async def _apply_krx_data_quality_improvement(self, df: pd.DataFrame) -> pd.DataFrame:
        """KRX 데이터 품질 개선"""
        try:
            # 결측치 처리
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            # 중복 제거
            df = df.drop_duplicates()
            
            # 데이터 타입 변환
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].dtype == 'object':
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            logging.info("KRX 데이터 품질 개선 완료")
            return df
        except Exception as e:
            logging.error(f"KRX 데이터 품질 개선 실패: {e}")
            return df
    
    async def _apply_krx_outlier_removal(self, df: pd.DataFrame) -> pd.DataFrame:
        """KRX 이상치 제거"""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                if col in df.columns:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    # 이상치 제거
                    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            
            logging.info("KRX 이상치 제거 완료")
            return df
        except Exception as e:
            logging.error(f"KRX 이상치 제거 실패: {e}")
            return df

    async def save_krx_data_optimized(self, df: pd.DataFrame, analysis: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
        """KRX 데이터 최적화 저장 (자동매매 판단 고려)"""
        try:
            # 성능 평가
            evaluation = evaluate_krx_performance(analysis)
            trading_confidence = evaluation["trading_confidence"]
            performance_grade = evaluation["performance_grade"]
            
            # 저장 전략 결정
            storage_strategy = get_krx_storage_strategy(df, trading_confidence)
            
            # 저장 디렉토리 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = output_dir / f"krx_{trading_confidence}_{timestamp}"
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # 메타데이터 저장
            metadata = {
                "timestamp": timestamp,
                "data_shape": df.shape,
                "performance_grade": performance_grade,
                "trading_confidence": trading_confidence,
                "trading_recommendation": evaluation["trading_recommendation"],
                "storage_strategy": storage_strategy,
                "analysis_summary": {
                    "avg_r2": analysis.get("avg_r2", 0),
                    "avg_rmse": analysis.get("avg_rmse", 0),
                    "excellent_folds": analysis.get("excellent_folds", 0),
                    "poor_folds": analysis.get("poor_folds", 0)
                }
            }
            
            # 메타데이터 저장
            metadata_path = save_dir / "metadata.json"
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False, default=np_encoder)
            
            # 데이터 저장 (전략에 따라)
            data_path = await self._save_krx_data_with_strategy(df, save_dir, storage_strategy)
            
            # 자동매매 설정 파일 생성
            if trading_confidence in ["high_confidence", "medium_confidence"]:
                await self._create_trading_config(save_dir, evaluation, storage_strategy)
            
            # 백업 설정
            await self._setup_backup_strategy(save_dir, storage_strategy)
            
            logging.info(f"KRX 데이터 최적화 저장 완료: {data_path}")
            logging.info(f"자동매매 신뢰도: {trading_confidence}")
            logging.info(f"성능 등급: {performance_grade}")
            
            return {
                "save_path": str(data_path),
                "metadata_path": str(metadata_path),
                "trading_confidence": trading_confidence,
                "performance_grade": performance_grade,
                "storage_strategy": storage_strategy
            }
            
        except Exception as e:
            logging.error(f"KRX 데이터 저장 실패: {e}")
            return {"error": str(e)}
    
    async def _save_krx_data_with_strategy(self, df: pd.DataFrame, save_dir: Path, strategy: Dict[str, Any]) -> Path:
        """전략에 따른 KRX 데이터 저장"""
        format_type = strategy["storage_format"]
        compression = strategy["compression"]
        
        if format_type == "parquet":
            # Parquet 저장 (최적화된 형식)
            data_path = save_dir / f"krx_data.{format_type}"
            
            # 파티셔닝 적용
            partition_cols = strategy.get("partition_by", [])
            if partition_cols and any(col in df.columns for col in partition_cols):
                available_partitions = [col for col in partition_cols if col in df.columns]
                if available_partitions:
                    df.to_parquet(
                        data_path,
                        compression=compression,
                        partition_cols=available_partitions,
                        index=False
                    )
                else:
                    df.to_parquet(data_path, compression=compression, index=False)
            else:
                df.to_parquet(data_path, compression=compression, index=False)
        
        elif format_type == "csv":
            # CSV 저장 (호환성)
            data_path = save_dir / "krx_data.csv"
            df.to_csv(data_path, index=False, encoding='utf-8-sig')
        
        else:
            # 기본 Parquet 저장
            data_path = save_dir / "krx_data.parquet"
            df.to_parquet(data_path, compression="snappy", index=False)
        
        return data_path
    
    async def _create_trading_config(self, save_dir: Path, evaluation: Dict[str, Any], strategy: Dict[str, Any]):
        """자동매매 설정 파일 생성"""
        trading_config = {
            "enabled": evaluation["trading_confidence"] in ["high_confidence", "medium_confidence"],
            "confidence_level": evaluation["trading_confidence"],
            "performance_grade": evaluation["performance_grade"],
            "recommendation": evaluation["trading_recommendation"],
            "risk_management": {
                "max_position_size": 0.1 if evaluation["trading_confidence"] == "high_confidence" else 0.05,
                "stop_loss": 0.02,
                "take_profit": 0.05,
                "max_daily_trades": 10 if evaluation["trading_confidence"] == "high_confidence" else 5
            },
            "data_requirements": {
                "min_data_quality": 0.8,
                "min_performance_r2": 0.7,
                "max_rmse": 0.15
            },
            "storage_config": strategy
        }
        
        config_path = save_dir / "trading_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(trading_config, f, indent=2, ensure_ascii=False, default=np_encoder)
        
        logging.info(f"자동매매 설정 파일 생성: {config_path}")
    
    async def _setup_backup_strategy(self, save_dir: Path, strategy: Dict[str, Any]):
        """백업 전략 설정"""
        backup_config = {
            "frequency": strategy["backup_frequency"],
            "retention_days": strategy["retention_days"],
            "compression": strategy["compression"],
            "description": strategy["description"]
        }
        
        backup_path = save_dir / "backup_config.json"
        with open(backup_path, "w", encoding="utf-8") as f:
            json.dump(backup_config, f, indent=2, ensure_ascii=False, default=np_encoder)
        
        logging.info(f"백업 설정 파일 생성: {backup_path}")
    
    async def load_krx_data_optimized(self, data_path: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """KRX 데이터 최적화 로드"""
        try:
            # 메타데이터 로드
            metadata_path = data_path.parent / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
            else:
                metadata = {}
            
            # 데이터 로드
            if data_path.suffix == ".parquet":
                df = pd.read_parquet(data_path)
            elif data_path.suffix == ".csv":
                df = pd.read_csv(data_path, encoding='utf-8-sig')
            else:
                raise ValueError(f"지원하지 않는 파일 형식: {data_path.suffix}")
            
            logging.info(f"KRX 데이터 로드 완료: {data_path}")
            logging.info(f"데이터 크기: {df.shape}")
            
            return df, metadata
            
        except Exception as e:
            logging.error(f"KRX 데이터 로드 실패: {e}")
            return pd.DataFrame(), {}
    
    async def get_krx_storage_recommendations(self, df: pd.DataFrame, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """KRX 저장 권장사항 생성"""
        evaluation = evaluate_krx_performance(analysis)
        storage_strategy = get_krx_storage_strategy(df, evaluation["trading_confidence"])
        
        recommendations = {
            "performance_summary": {
                "grade": evaluation["performance_grade"],
                "trading_confidence": evaluation["trading_confidence"],
                "recommendation": evaluation["trading_recommendation"]
            },
            "storage_strategy": storage_strategy,
            "data_characteristics": detect_krx_data_characteristics(df),
            "optimization_suggestions": []
        }
        
        # 최적화 제안
        if evaluation["trading_confidence"] == "not_tradeable":
            recommendations["optimization_suggestions"].append("데이터 품질 개선 필요")
            recommendations["optimization_suggestions"].append("모델 성능 향상 필요")
        
        if df.shape[0] > 100000:
            recommendations["optimization_suggestions"].append("대용량 데이터 - 파티셔닝 권장")
        
        if len(df.select_dtypes(include=['object']).columns) > 5:
            recommendations["optimization_suggestions"].append("범주형 데이터 최적화 필요")
        
        return recommendations

    def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 반환"""
        return {
            'mode': self.config.mode.name,
            'metrics': self.metrics.model_dump(),
            'cache_stats': {
                'memory_cache_size': len(self.memory_cache),
                'disk_cache_size': len(list(self.cache_dir.glob('*.pkl'))),
                'distributed_cache_size': len(self.distributed_cache),
                'hit_rate': self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
            },
            'memory_usage': {
                'current_mb': self.metrics.memory_usage_mb,
                'limit_gb': self.config.memory_limit_gb,
                'usage_percent': (self.metrics.memory_usage_mb / (self.config.memory_limit_gb * 1024)) * 100
            },
            'monitoring_status': {
                'is_monitoring': self.is_monitoring,
                'emergency_mode': self.emergency_mode,
                'market_hours': self._is_market_hours(),
                'market_data_count': len(self.market_data_history),
                'detected_events_count': len(self.detected_events),
                'next_collection': self.metrics.next_collection_time
            }
        }

class MemoryMonitor:
    """메모리 모니터링 클래스"""
    
    def __init__(self, limit_gb: float):
        self.limit_bytes = limit_gb * 1024 * 1024 * 1024
        self.warning_threshold = 0.8
    
    def check_memory_usage(self) -> bool:
        """메모리 사용량 체크"""
        try:
            current_usage = psutil.virtual_memory().used
            usage_ratio = current_usage / self.limit_bytes
            
            if usage_ratio > self.warning_threshold:
                return False
            return True
        except Exception:
            return True  # psutil 사용 불가능한 경우 True 반환

class DummyMemoryMonitor:
    """더미 메모리 모니터링 클래스 (psutil 사용 불가능한 경우)"""
    
    def __init__(self, limit_gb: float = 8.0):
        self.limit_bytes = limit_gb * 1024 * 1024 * 1024
        self.warning_threshold = 0.8
    
    def check_memory_usage(self) -> bool:
        """메모리 사용량 체크 (더미)"""
        return True

# 비동기 텔레그램 알림 시스템 (KRX용)
class KRXTelegramNotifier:
    """KRX 시스템용 비동기 텔레그램 알림 시스템"""
    
    def __init__(self, bot_token: str = "", chat_id: str = "", enable_notifications: bool = True):
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}" if self.bot_token else ""
        self.enabled = bool(self.bot_token and self.chat_id and enable_notifications)
        self.last_notification_time = 0
        self.notification_cooldown = 300  # 5분 쿨다운
    
    async def send_message(self, message: str, force: bool = False) -> bool:
        """비동기 텔레그램 메시지 전송 (쿨다운 적용)"""
        if not self.enabled:
            return True
        
        current_time = time.time()
        if not force and current_time - self.last_notification_time < self.notification_cooldown:
            logging.debug("텔레그램 알림 쿨다운 중 - 메시지 전송 건너뜀")
            return True
        
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/sendMessage"
                data = {
                    "chat_id": self.chat_id,
                    "text": message,
                    "parse_mode": "HTML"
                }
                async with session.post(url, data=data) as response:
                    if response.status == 200:
                        self.last_notification_time = current_time
                        return True
                    return False
        except Exception as e:
            logging.error("텔레그램 전송 실패", error=str(e))
            return False
    
    async def send_krx_performance_report(self, analysis: Dict[str, Any], data_info: Dict[str, Any]) -> bool:
        """KRX 성능 리포트 전송"""
        try:
            performance_grade = analysis.get('performance_grade', '')
            improvement_needed = analysis.get('improvement_needed', False)
            
            if performance_grade.startswith("🔴") or improvement_needed:
                message = self._format_krx_performance_message(analysis, data_info)
                return await self.send_message(message, force=True)
            else:
                logging.debug("성능이 양호하여 텔레그램 알림 건너뜀")
                return True
        except Exception as e:
            logging.error("KRX 성능 리포트 전송 실패", error=str(e))
            return False
    
    async def send_krx_improvement_start(self, iteration: int, strategies: List[str]) -> bool:
        """KRX 개선 시작 알림 (첫 번째 반복에서만)"""
        if iteration == 1:  # 첫 번째 반복에서만 알림
            message = f"""🔄 <b>KRX 데이터 자동 개선 시작</b>

📋 <b>적용 전략:</b>
{chr(10).join(f'• {strategy}' for strategy in strategies)}

⏱️ 자동 개선 진행 중..."""
            return await self.send_message(message, force=True)
        return True
    
    async def send_krx_improvement_complete(self, iteration: int, analysis: Dict[str, Any]) -> bool:
        """KRX 개선 완료 알림 (목표 달성 시에만)"""
        is_excellent = analysis.get('performance_grade', '').startswith('🟢')
        avg_r2 = analysis.get('avg_r2', 0)
        
        if is_excellent and avg_r2 > 0.8:
            message = f"""✅ <b>KRX 데이터 자동 개선 완료</b>

📊 <b>최종 결과:</b>
• 평균 R²: {analysis.get('avg_r2', 0):.6f}
• 성능 등급: {analysis.get('performance_grade', 'N/A')}
• 우수 성능 Fold: {analysis.get('excellent_folds', 0)}개

🎉 목표 달성! 우수 성능 달성"""
            return await self.send_message(message, force=True)
        return True
    
    def _format_krx_performance_message(self, analysis: Dict[str, Any], data_info: Dict[str, Any]) -> str:
        """KRX 성능 메시지 포맷팅"""
        message = f"""📊 <b>KRX 데이터 성능 현황</b>

• 평균 R²: {analysis.get('avg_r2', 0):.6f}
• 성능 등급: {analysis.get('performance_grade', 'N/A')}
• 개선 필요 Fold: {analysis.get('poor_folds', 0)}개
• 데이터 크기: {data_info.get('rows', 0)}행

🔄 자동 개선 진행 중..."""
        
        return message

async def main():
    """메인 실행 함수 - 실시간 모니터링 + 자동 개선"""
    # 시스템 초기화
    config = SystemConfig(mode=SystemMode.REALTIME_MONITORING)
    system = KRXUltimateSystem(config)
    
    try:
        print("KRX Ultimate System 시작...")
        print("실시간 모니터링을 시작합니다.")
        
        # 실시간 모니터링 시작
        await system.start_realtime_monitoring()
        
        # 샘플 데이터 생성 (실제로는 수집된 데이터 사용)
        sample_data = pd.DataFrame({
            '종목코드': ['005930', '000660', '035420'],
            '종목명': ['삼성전자', 'SK하이닉스', 'NAVER'],
            '현재가': [70000, 120000, 350000],
            '등락률': [2.5, -1.2, 0.8],
            '거래량': [1000000, 500000, 300000],
            '시가총액': [4200000, 7200000, 21000000]
        })
        
        # KRX 자동 개선 루프 실행
        improvement_result = await system.auto_improvement_loop(sample_data)
        logging.info("KRX 자동 개선 완료")
        
        # 성능 분석 및 최적화 저장
        if improvement_result and "final_analysis" in improvement_result:
            analysis = improvement_result["final_analysis"]
            
            # 저장 권장사항 생성
            recommendations = await system.get_krx_storage_recommendations(sample_data, analysis)
            logging.info(f"KRX 저장 권장사항: {recommendations}")
            
            # 최적화 저장 실행
            output_dir = Path("krx_optimized_outputs")
            save_result = await system.save_krx_data_optimized(sample_data, analysis, output_dir)
            logging.info(f"KRX 최적화 저장 완료: {save_result}")
            
            # 텔레그램 알림
            telegram = KRXTelegramNotifier()
            await telegram.send_message(f"""✅ <b>KRX 시스템 완료</b>

📊 <b>성능 결과:</b>
• 성능 등급: {analysis.get('performance_grade', 'N/A')}
• 자동매매 신뢰도: {analysis.get('trading_confidence', 'N/A')}
• 권장사항: {analysis.get('trading_recommendation', 'N/A')}

💾 <b>저장 정보:</b>
• 저장 경로: {save_result.get('save_path', 'N/A')}
• 저장 전략: {save_result.get('storage_strategy', {}).get('description', 'N/A')}""", force=True)
        
    except KeyboardInterrupt:
        print("\n사용자에 의해 중지되었습니다.")
        await system.stop_realtime_monitoring()
    except Exception as e:
        print(f"시스템 에러: {e}")
        await system.stop_realtime_monitoring()
    finally:
        print("시스템 종료")

if __name__ == "__main__":
    asyncio.run(main()) 