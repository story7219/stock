#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: kis_realtime_collector.py
목적: 한국투자증권 OpenAPI 기반 실시간 데이터 수집 시스템 - 최신 Python 표준 활용
Author: Ultimate KIS System
Created: 2025-07-13
Version: 2.0.0

Features:
    - 실시간 시세, 거래량, 호가창 수집
    - WebSocket + REST API 하이브리드
    - 비동기 고속 병렬처리
    - 멀티레벨 캐싱
    - 자동 재연결 및 에러 복구
    - 실시간 알림 및 이벤트 처리
"""

from __future__ import annotations

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
from typing import Dict, List, Optional, Any, Union, Literal, TypedDict, Protocol
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import lru_cache, wraps
from contextlib import asynccontextmanager
import weakref
from collections import defaultdict, deque
import gc
import psutil
import tracemalloc
import websockets
import ssl
import os
import multiprocessing as mp

# 최신 Python 표준 활용
from typing_extensions import NotRequired, Required
from pydantic import BaseModel, Field, field_validator
import structlog

# 성능 모니터링
tracemalloc.start()

class DataType(Enum):
    """데이터 타입"""
    REALTIME_PRICE = auto()
    REALTIME_VOLUME = auto()
    REALTIME_ORDERBOOK = auto()
    REALTIME_TRADE = auto()
    MINUTE_OHLCV = auto()
    DAILY_OHLCV = auto()

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
    LARGE_TRADE = auto()

@dataclass
class SystemConfig:
    """시스템 설정 - 최신 dataclass 활용"""
    # KIS API 설정
    kis_app_key: str = field(default_factory=lambda: os.getenv("LIVE_KIS_APP_KEY", ""))
    kis_app_secret: str = field(default_factory=lambda: os.getenv("LIVE_KIS_APP_SECRET", ""))
    kis_account_number: str = field(default_factory=lambda: os.getenv("LIVE_KIS_ACCOUNT_NUMBER", ""))
    
    # 실시간 수집 설정
    max_workers: int = field(default_factory=lambda: min(32, mp.cpu_count() + 4))
    cache_size: int = 1000
    connection_pool_size: int = 20
    timeout: float = 30.0
    retry_attempts: int = 3
    memory_limit_gb: float = 8.0
    
    # 실시간 모니터링 설정
    monitoring_interval_seconds: int = 1  # 1초마다 수집
    market_hours_start: str = "09:00"
    market_hours_end: str = "15:30"
    weekend_monitoring: bool = False
    emergency_collection_interval: int = 0.1  # 긴급 상황 시 0.1초마다
    price_change_threshold: float = 0.01  # 1% 이상 변동 시 이벤트
    volume_change_threshold: float = 2.0  # 거래량 2배 이상 시 이벤트
    
    # WebSocket 설정
    websocket_url: str = "wss://openapi.koreainvestment.com:9443/oauth2/ws"
    websocket_reconnect_interval: int = 5
    websocket_heartbeat_interval: int = 30
    
    def __post_init__(self):
        """설정 검증"""
        if self.memory_limit_gb > psutil.virtual_memory().total / (1024**3):
            self.memory_limit_gb = psutil.virtual_memory().total / (1024**3) * 0.8

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
    websocket_connected: bool = False
    websocket_reconnects: int = 0
    
    @field_validator('avg_response_time')
    @classmethod
    def validate_response_time(cls, v):
        return max(0.0, v)

@dataclass
class RealtimeData:
    """실시간 데이터"""
    timestamp: datetime
    symbol: str
    data_type: DataType
    price: Optional[float] = None
    volume: Optional[int] = None
    change_percent: Optional[float] = None
    bid_price: Optional[float] = None
    ask_price: Optional[float] = None
    bid_volume: Optional[int] = None
    ask_volume: Optional[int] = None
    trade_price: Optional[float] = None
    trade_volume: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'data_type': self.data_type.name,
            'price': self.price,
            'volume': self.volume,
            'change_percent': self.change_percent,
            'bid_price': self.bid_price,
            'ask_price': self.ask_price,
            'bid_volume': self.bid_volume,
            'ask_volume': self.ask_volume,
            'trade_price': self.trade_price,
            'trade_volume': self.trade_volume
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

class KISRealtimeCollector:
    """KIS 실시간 데이터 수집 시스템 - 최신 Python 표준 활용"""
    
    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or SystemConfig()
        
        # 구조화된 로깅 설정
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.JSONRenderer()
            ],
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
            logger_factory=structlog.stdlib.LoggerFactory(),
        )
        self.logger = structlog.get_logger()
        
        # 멀티레벨 캐싱 시스템
        self._setup_caching()
        
        # 커넥션 풀 설정
        self._setup_connection_pool()
        
        # 성능 모니터링
        self.metrics = PerformanceMetrics()
        self._start_performance_monitoring()
        
        # 메모리 관리
        self._setup_memory_management()
        
        # 실시간 모니터링 설정
        self._setup_realtime_monitoring()
        
        # WebSocket 설정
        self._setup_websocket()
    
    def _setup_caching(self):
        """멀티레벨 캐싱 시스템 설정"""
        # L1: 메모리 캐시 (LRU)
        self.memory_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # L2: 디스크 캐시
        self.cache_dir = Path('cache/kis_realtime')
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
        self.memory_monitor = MemoryMonitor(self.config.memory_limit_gb)
        self.gc_threshold = 0.8  # 80% 메모리 사용 시 GC
    
    def _setup_realtime_monitoring(self):
        """실시간 모니터링 설정"""
        self.is_monitoring = False
        self.realtime_data_history = deque(maxlen=10000)  # 최근 10000개 데이터
        self.detected_events = deque(maxlen=1000)  # 최근 1000개 이벤트
        self.last_realtime_data = {}
        self.emergency_mode = False
        
        # 시장 시간 설정
        self.market_start = datetime.strptime(self.config.market_hours_start, "%H:%M").time()
        self.market_end = datetime.strptime(self.config.market_hours_end, "%H:%M").time()
    
    def _setup_websocket(self):
        """WebSocket 설정"""
        self.websocket = None
        self.websocket_connected = False
        self.websocket_reconnect_count = 0
        self.websocket_last_heartbeat = None
    
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
            self.logger.error(f"세션 에러: {e}")
            if self.session:
                await self.session.close()
                self.session = None
            raise
    
    @lru_cache(maxsize=1000)
    def _get_cache_key(self, data_type: str, symbol: str, timestamp: str) -> str:
        """캐시 키 생성"""
        return hashlib.md5(f"{data_type}_{symbol}_{timestamp}".encode()).hexdigest()
    
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
                self.logger.error(f"디스크 캐시 로드 실패: {e}")
        
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
            self.logger.error(f"디스크 캐시 저장 실패: {e}")
    
    async def start_realtime_collection(self):
        """실시간 데이터 수집 시작"""
        if self.is_monitoring:
            self.logger.info("이미 실시간 수집 중입니다.")
            return
        
        self.is_monitoring = True
        self.logger.info("실시간 데이터 수집 시작")
        
        # 수집 태스크들 시작
        collection_tasks = [
            asyncio.create_task(self._continuous_realtime_collection()),
            asyncio.create_task(self._websocket_connection()),
            asyncio.create_task(self._market_event_detection()),
            asyncio.create_task(self._performance_monitoring()),
            asyncio.create_task(self._emergency_monitoring())
        ]
        
        try:
            await asyncio.gather(*collection_tasks)
        except Exception as e:
            self.logger.error(f"실시간 수집 태스크 에러: {e}")
        finally:
            self.is_monitoring = False
    
    async def stop_realtime_collection(self):
        """실시간 데이터 수집 중지"""
        self.is_monitoring = False
        if self.websocket:
            await self.websocket.close()
        self.logger.info("실시간 데이터 수집 중지")
    
    async def _continuous_realtime_collection(self):
        """지속적 실시간 데이터 수집"""
        while self.is_monitoring:
            try:
                # 시장 시간 체크
                if not self._is_market_hours():
                    await asyncio.sleep(300)  # 5분 대기
                    continue
                
                # 실시간 데이터 수집
                start_time = time.time()
                data = await self.collect_realtime_data_parallel([
                    DataType.REALTIME_PRICE, 
                    DataType.REALTIME_VOLUME,
                    DataType.REALTIME_ORDERBOOK
                ])
                
                # 수집 시간 기록
                self.metrics.total_collections += 1
                self.metrics.last_collection_time = datetime.now().isoformat()
                self.metrics.next_collection_time = (
                    datetime.now() + timedelta(seconds=self.config.monitoring_interval_seconds)
                ).isoformat()
                
                # 실시간 데이터 히스토리에 저장
                await self._process_realtime_data(data)
                
                execution_time = time.time() - start_time
                self._update_performance_metrics(execution_time)
                
                self.logger.info(f"실시간 데이터 수집 완료 (소요시간: {execution_time:.3f}초)")
                
                # 다음 수집까지 대기
                await asyncio.sleep(self.config.monitoring_interval_seconds)
                
            except Exception as e:
                self.logger.error(f"지속적 실시간 데이터 수집 에러: {e}")
                await asyncio.sleep(1)  # 에러 시 1초 대기
    
    async def _websocket_connection(self):
        """WebSocket 연결 및 실시간 데이터 수신"""
        while self.is_monitoring:
            try:
                if not self._is_market_hours():
                    await asyncio.sleep(60)
                    continue
                
                # WebSocket 연결
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
                
                async with websockets.connect(
                    self.config.websocket_url,
                    ssl=ssl_context,
                    extra_headers={
                        'authorization': f'Bearer {await self._get_access_token()}',
                        'appkey': self.config.kis_app_key,
                        'appsecret': self.config.kis_app_secret
                    }
                ) as websocket:
                    self.websocket = websocket
                    self.websocket_connected = True
                    self.metrics.websocket_connected = True
                    self.logger.info("WebSocket 연결 성공")
                    
                    # 실시간 데이터 구독
                    await self._subscribe_realtime_data(websocket)
                    
                    # 메시지 수신
                    async for message in websocket:
                        await self._process_websocket_message(message)
                        
            except Exception as e:
                self.logger.error(f"WebSocket 연결 에러: {e}")
                self.websocket_connected = False
                self.metrics.websocket_connected = False
                self.metrics.websocket_reconnects += 1
                await asyncio.sleep(self.config.websocket_reconnect_interval)
    
    async def _subscribe_realtime_data(self, websocket):
        """실시간 데이터 구독"""
        try:
            # 실시간 시세 구독
            subscribe_message = {
                "header": {
                    "approval_key": await self._get_access_token(),
                    "custtype": "P",
                    "tr_type": "1",
                    "content-type": "utf-8"
                },
                "body": {
                    "input": {
                        "tr_id": "H0_CNT0",
                        "tr_key": "005930"  # 삼성전자 예시
                    }
                }
            }
            
            await websocket.send(json.dumps(subscribe_message))
            self.logger.info("실시간 데이터 구독 요청 전송")
            
        except Exception as e:
            self.logger.error(f"실시간 데이터 구독 에러: {e}")
    
    async def _process_websocket_message(self, message: str):
        """WebSocket 메시지 처리"""
        try:
            data = json.loads(message)
            
            # 실시간 데이터 파싱
            if 'body' in data and 'output' in data['body']:
                output = data['body']['output']
                
                realtime_data = RealtimeData(
                    timestamp=datetime.now(),
                    symbol=output.get('hts_kor_isnm', ''),
                    data_type=DataType.REALTIME_PRICE,
                    price=float(output.get('stck_prpr', 0)),
                    volume=int(output.get('cntg_vol', 0)),
                    change_percent=float(output.get('prdy_vrss', 0)),
                    bid_price=float(output.get('bidp', 0)),
                    ask_price=float(output.get('askp', 0)),
                    bid_volume=int(output.get('bidp_rsqn', 0)),
                    ask_volume=int(output.get('askp_rsqn', 0))
                )
                
                self.realtime_data_history.append(realtime_data)
                self.last_realtime_data[realtime_data.symbol] = realtime_data
                
                # 이벤트 감지
                await self._detect_market_events(realtime_data)
                
        except Exception as e:
            self.logger.error(f"WebSocket 메시지 처리 에러: {e}")
    
    async def _detect_market_events(self, data: RealtimeData):
        """시장 이벤트 감지"""
        try:
            if len(self.realtime_data_history) < 2:
                return
            
            # 최근 데이터와 이전 데이터 비교
            current_data = self.realtime_data_history[-1]
            previous_data = self.realtime_data_history[-2]
            
            # 가격 변동 감지
            if current_data.price and previous_data.price:
                price_change = abs(current_data.price - previous_data.price) / previous_data.price
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
            if current_data.volume and previous_data.volume:
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
            
            # 대량 거래 감지
            if current_data.trade_volume and current_data.trade_volume > 10000:  # 1만주 이상
                event = MarketEvent(
                    event_type=MarketEventType.LARGE_TRADE,
                    symbol=current_data.symbol,
                    timestamp=current_data.timestamp,
                    description=f"대량 거래: {current_data.trade_volume:,}주",
                    severity=Priority.HIGH,
                    data={'trade_volume': current_data.trade_volume, 'trade_price': current_data.trade_price}
                )
                await self._handle_market_event(event)
                
        except Exception as e:
            self.logger.error(f"시장 이벤트 감지 에러: {e}")
    
    async def _handle_market_event(self, event: MarketEvent):
        """시장 이벤트 처리"""
        self.detected_events.append(event)
        self.metrics.market_events_detected += 1
        
        self.logger.info(f"시장 이벤트 감지: {event.description}")
        
        # 긴급 상황 시 더 자주 수집
        if event.severity == Priority.HIGH:
            self.emergency_mode = True
            asyncio.create_task(self._emergency_data_collection())
        
        # 이벤트 저장
        await self._save_market_event(event)
    
    async def _emergency_data_collection(self):
        """긴급 데이터 수집"""
        self.logger.info("긴급 데이터 수집 시작")
        
        for _ in range(100):  # 100회 긴급 수집
            try:
                data = await self.collect_realtime_data_parallel([DataType.REALTIME_PRICE])
                await self._process_realtime_data(data)
                await asyncio.sleep(self.config.emergency_collection_interval)
            except Exception as e:
                self.logger.error(f"긴급 데이터 수집 에러: {e}")
        
        self.emergency_mode = False
        self.logger.info("긴급 데이터 수집 완료")
    
    async def collect_realtime_data_parallel(self, data_types: List[DataType]) -> Dict[str, Any]:
        """병렬 실시간 데이터 수집 - 최신 Python 표준 활용"""
        start_time = time.time()
        
        try:
            # 비동기 병렬 처리
            tasks = [self._collect_single_realtime_data_type(data_type) for data_type in data_types]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 결과 통합
            combined_results = {}
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(f"실시간 데이터 수집 실패 ({data_types[i].name}): {result}")
                else:
                    combined_results[data_types[i].name] = result
            
            execution_time = time.time() - start_time
            self._update_performance_metrics(execution_time)
            
            return combined_results
            
        except Exception as e:
            self.logger.error(f"병렬 실시간 데이터 수집 실패: {e}")
            return {}
    
    async def _collect_single_realtime_data_type(self, data_type: DataType) -> Dict[str, Any]:
        """단일 실시간 데이터 타입 수집"""
        cache_key = self._get_cache_key(data_type.name, 'KIS', datetime.now().strftime('%Y%m%d%H%M%S'))
        
        # 캐시에서 먼저 확인
        cached_data = await self._multi_level_cache_get(cache_key)
        if cached_data:
            return cached_data
        
        try:
            async with self.get_session() as session:
                params = self._get_realtime_request_params(data_type)
                
                async with session.get(
                    'https://openapi.koreainvestment.com:9443/uapi/domestic-stock/v1/quotations/inquire-price',
                    params=params,
                    headers=self._get_realtime_headers(),
                    timeout=self.config.timeout
                ) as response:
                    
                    if response.status == 200:
                        data = await response.json()
                        
                        # 캐시에 저장
                        await self._multi_level_cache_set(cache_key, data)
                        
                        self.metrics.successful_requests += 1
                        return data
                    else:
                        raise Exception(f"HTTP {response.status}")
                        
        except Exception as e:
            self.metrics.failed_requests += 1
            self.logger.error(f"실시간 데이터 수집 실패 ({data_type.name}): {e}")
            raise
    
    def _get_realtime_request_params(self, data_type: DataType) -> Dict[str, str]:
        """실시간 요청 파라미터 생성"""
        base_params = {
            'FID_COND_MRKT_DIV_CODE': 'J',
            'FID_INPUT_ISCD': '005930'  # 삼성전자 예시
        }
        
        if data_type == DataType.REALTIME_PRICE:
            base_params['tr_id'] = 'FHKST01010100'
        elif data_type == DataType.REALTIME_VOLUME:
            base_params['tr_id'] = 'FHKST01010200'
        elif data_type == DataType.REALTIME_ORDERBOOK:
            base_params['tr_id'] = 'FHKST01010300'
        
        return base_params
    
    def _get_realtime_headers(self) -> Dict[str, str]:
        """실시간 요청 헤더 생성"""
        return {
            'authorization': f'Bearer {asyncio.run(self._get_access_token())}',
            'appkey': self.config.kis_app_key,
            'appsecret': self.config.kis_app_secret,
            'tr_id': 'FHKST01010100'
        }
    
    async def _get_access_token(self) -> str:
        """액세스 토큰 발급"""
        try:
            url = "https://openapi.koreainvestment.com:9443/oauth2/tokenP"
            headers = {"content-type": "application/json"}
            payload = {
                "grant_type": "client_credentials",
                "appkey": self.config.kis_app_key,
                "appsecret": self.config.kis_app_secret
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers) as response:
                    if response.status == 200:
                        result = await response.json()
                        token = result.get("access_token", "")
                        self.logger.info("KIS API 토큰 발급 성공")
                        return token
                    else:
                        raise Exception(f"토큰 발급 실패: {response.status}")
                        
        except Exception as e:
            self.logger.error(f"토큰 발급 에러: {e}")
            return ""
    
    async def _process_realtime_data(self, data: Dict[str, Any]):
        """실시간 데이터 처리"""
        try:
            # 데이터를 RealtimeData 객체로 변환
            for item in data.get('output', []):
                realtime_data = RealtimeData(
                    timestamp=datetime.now(),
                    symbol=item.get('hts_kor_isnm', ''),
                    data_type=DataType.REALTIME_PRICE,
                    price=float(item.get('stck_prpr', 0)),
                    volume=int(item.get('cntg_vol', 0)),
                    change_percent=float(item.get('prdy_vrss', 0)),
                    bid_price=float(item.get('bidp', 0)),
                    ask_price=float(item.get('askp', 0)),
                    bid_volume=int(item.get('bidp_rsqn', 0)),
                    ask_volume=int(item.get('askp_rsqn', 0))
                )
                
                self.realtime_data_history.append(realtime_data)
                self.last_realtime_data[realtime_data.symbol] = realtime_data
                
        except Exception as e:
            self.logger.error(f"실시간 데이터 처리 에러: {e}")
    
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
            
            events_file = Path('data/market_events_realtime.jsonl')
            events_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(events_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(event_data, ensure_ascii=False) + '\n')
                
        except Exception as e:
            self.logger.error(f"시장 이벤트 저장 실패: {e}")
    
    def _is_market_hours(self) -> bool:
        """시장 시간 체크"""
        now = datetime.now()
        current_time = now.time()
        
        # 주말 체크
        if not self.config.weekend_monitoring and now.weekday() >= 5:
            return False
        
        # 시장 시간 체크
        return self.market_start <= current_time <= self.market_end
    
    def _update_performance_metrics(self, execution_time: float):
        """성능 메트릭 업데이트"""
        self.metrics.total_requests += 1
        self.metrics.avg_response_time = (
            (self.metrics.avg_response_time * (self.metrics.total_requests - 1) + execution_time) 
            / self.metrics.total_requests
        )
        
        # 메모리 사용량 업데이트
        process = psutil.Process()
        self.metrics.memory_usage_mb = process.memory_info().rss / (1024 * 1024)
        self.metrics.cpu_usage_percent = process.cpu_percent()
    
    def _start_performance_monitoring(self):
        """성능 모니터링 시작 (실제 구현 필요시 확장)"""
        pass
    
    async def _performance_monitoring(self):
        """성능 모니터링"""
        while self.is_monitoring:
            try:
                # 메모리 사용량 체크
                if self.metrics.memory_usage_mb > self.config.memory_limit_gb * 1024 * 0.8:
                    self.logger.warning(f"메모리 사용량 높음: {self.metrics.memory_usage_mb:.2f}MB")
                    gc.collect()
                
                # 성능 리포트 저장
                await self._save_performance_report()
                
                await asyncio.sleep(60)  # 1분마다 체크
                
            except Exception as e:
                self.logger.error(f"성능 모니터링 에러: {e}")
                await asyncio.sleep(10)
    
    async def _emergency_monitoring(self):
        """긴급 상황 모니터링"""
        while self.is_monitoring:
            try:
                # 메모리 사용량 체크
                if self.metrics.memory_usage_mb > self.config.memory_limit_gb * 1024 * 0.9:
                    self.logger.critical("메모리 사용량 위험 수준")
                    gc.collect()
                
                # 에러율 체크
                error_rate = self.metrics.failed_requests / max(self.metrics.total_requests, 1)
                if error_rate > 0.1:  # 10% 이상 에러
                    self.logger.warning(f"높은 에러율: {error_rate:.1%}")
                
                await asyncio.sleep(30)  # 30초마다 체크
                
            except Exception as e:
                self.logger.error(f"긴급 모니터링 에러: {e}")
                await asyncio.sleep(10)
    
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
                    'realtime_data_count': len(self.realtime_data_history),
                    'detected_events_count': len(self.detected_events),
                    'is_market_hours': self._is_market_hours(),
                    'websocket_connected': self.websocket_connected
                }
            }
            
            report_file = Path('reports/realtime_performance_report.json')
            report_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            self.logger.error(f"성능 리포트 저장 실패: {e}")
    
    async def _market_event_detection(self):
        """실시간 이벤트 감지 (임시 구현)"""
        while self.is_monitoring:
            await asyncio.sleep(1)
    
    def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 반환"""
        return {
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
                'realtime_data_count': len(self.realtime_data_history),
                'detected_events_count': len(self.detected_events),
                'next_collection': self.metrics.next_collection_time,
                'websocket_connected': self.websocket_connected,
                'websocket_reconnects': self.metrics.websocket_reconnects
            }
        }

class MemoryMonitor:
    """메모리 모니터링 클래스"""
    
    def __init__(self, limit_gb: float):
        self.limit_bytes = limit_gb * 1024 * 1024 * 1024
        self.warning_threshold = 0.8
    
    def check_memory_usage(self) -> bool:
        """메모리 사용량 체크"""
        current_usage = psutil.virtual_memory().used
        usage_ratio = current_usage / self.limit_bytes
        
        if usage_ratio > self.warning_threshold:
            return False
        return True

async def main():
    """메인 실행 함수 - 실시간 데이터 수집"""
    # 시스템 초기화
    config = SystemConfig()
    collector = KISRealtimeCollector(config)
    
    try:
        # 실시간 데이터 수집 시작
        await collector.start_realtime_collection()
        
    except KeyboardInterrupt:
        print("\n사용자에 의해 중지되었습니다.")
        await collector.stop_realtime_collection()
    except Exception as e:
        print(f"시스템 에러: {e}")
        await collector.stop_realtime_collection()

if __name__ == "__main__":
    asyncio.run(main()) 