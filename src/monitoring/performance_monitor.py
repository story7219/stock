from prometheus_client import Counter
import Gauge
import Histogram, Summary
import psutil
from contextlib import asynccontextmanager
from core.logger import get_logger
from dataclasses import dataclass
import field
from datetime import datetime
import timedelta
from typing import Any
import Dict
import List, Optional
import asyncio
import json
import time

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: performance_monitor.py
모듈: 실시간 성능 모니터링 시스템
목적: 시스템 성능 지표 수집 및 분석

Author: Trading AI System
Created: 2025-01-27
Version: 1.0.0

Dependencies:
    - Python 3.11+
    - psutil==5.9.0
    - asyncio
    - aiohttp
    - prometheus_client

Performance:
    - 메트릭 수집 간격: 1초
    - 메모리 사용량: < 50MB
    - CPU 오버헤드: < 5%

License: MIT
"""


try:
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False



@dataclass
class SystemMetrics:
    """시스템 메트릭"""
    timestamp: datetime = field(default_factory=datetime.now)
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0
    network_io: Dict[str, int] = field(default_factory=dict)
    process_count: int = 0
    uptime: float = 0.0


@dataclass
class TradingMetrics:
    """트레이딩 메트릭"""
    timestamp: datetime = field(default_factory=datetime.now)
    signals_generated: int = 0
    trades_executed: int = 0
    successful_trades: int = 0
    failed_trades: int = 0
    total_pnl: float = 0.0
    win_rate: float = 0.0
    avg_trade_duration: float = 0.0
    active_positions: int = 0


@dataclass
class APIMetrics:
    """API 메트릭"""
    timestamp: datetime = field(default_factory=datetime.now)
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    active_connections: int = 0
    requests_per_minute: float = 0.0


class PerformanceMonitor:
    """성능 모니터링 시스템"""

    def __init__(self):
        self.logger = get_logger(__name__)
        self.running = False
        self.metrics_history: List[Dict[str, Any]] = []
        self.max_history_size = 1000
        self.system_metrics = None
        self.trading_metrics = None
        self.api_metrics = None
        self.stats = {
            'start_time': None,
            'total_metrics_collected': 0,
            'last_collection': None,
            'errors': []
        }
        if PROMETHEUS_AVAILABLE:
            self._setup_prometheus_metrics()

    def _setup_prometheus_metrics(self):
        self.cpu_gauge = Gauge('system_cpu_usage', 'CPU 사용률 (%)')
        self.memory_gauge = Gauge('system_memory_usage', '메모리 사용률 (%)')
        self.disk_gauge = Gauge('system_disk_usage', '디스크 사용률 (%)')
        self.signals_counter = Counter('trading_signals_total', '생성된 신호 수')
        self.trades_counter = Counter('trading_trades_total', '실행된 거래 수')
        self.api_requests_counter = Counter('api_requests_total', 'API 요청 수')
        self.response_time_histogram = Histogram('api_response_time_seconds', 'API 응답 시간')

    async def start(self):
        self.logger.info("🚀 성능 모니터링 시스템 시작")
        self.running = True
        self.stats['start_time'] = datetime.now()
        asyncio.create_task(self._collect_metrics_loop())
        self.logger.info("✅ 성능 모니터링 시스템 시작 완료")

    async def stop(self):
        self.logger.info("🛑 성능 모니터링 시스템 중지")
        self.running = False

    async def _collect_metrics_loop(self):
        while self.running:
            try:
                await self._collect_system_metrics()
                await self._collect_trading_metrics()
                await self._collect_api_metrics()
                await self._save_metrics()
                self.stats['total_metrics_collected'] += 1
                self.stats['last_collection'] = datetime.now()
                await asyncio.sleep(1)
            except Exception as e:
                self.logger.error(f"메트릭 수집 오류: {e}")
                self.stats['errors'].append({'timestamp': datetime.now(), 'error': str(e)})
                await asyncio.sleep(5)

    # ... (나머지 함수들)
