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

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from contextlib import asynccontextmanager

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    from prometheus_client import Counter, Gauge, Histogram, Summary
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

from core.logger import get_logger


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
        
        # 메트릭 저장소
        self.system_metrics: Optional[SystemMetrics] = None
        self.trading_metrics: Optional[TradingMetrics] = None
        self.api_metrics: Optional[APIMetrics] = None
        
        # 통계
        self.stats = {
            'start_time': None,
            'total_metrics_collected': 0,
            'last_collection': None,
            'errors': []
        }
        
        # Prometheus 메트릭 (선택적)
        if PROMETHEUS_AVAILABLE:
            self._setup_prometheus_metrics()

    def _setup_prometheus_metrics(self):
        """Prometheus 메트릭 설정"""
        self.cpu_gauge = Gauge('system_cpu_usage', 'CPU 사용률 (%)')
        self.memory_gauge = Gauge('system_memory_usage', '메모리 사용률 (%)')
        self.disk_gauge = Gauge('system_disk_usage', '디스크 사용률 (%)')
        self.signals_counter = Counter('trading_signals_total', '생성된 신호 수')
        self.trades_counter = Counter('trading_trades_total', '실행된 거래 수')
        self.api_requests_counter = Counter('api_requests_total', 'API 요청 수')
        self.response_time_histogram = Histogram('api_response_time_seconds', 'API 응답 시간')

    async def start(self) -> None:
        """모니터링 시작"""
        try:
            self.logger.info("🚀 성능 모니터링 시스템 시작")
            self.running = True
            self.stats['start_time'] = datetime.now()
            
            # 메트릭 수집 태스크 시작
            asyncio.create_task(self._collect_metrics_loop())
            
            self.logger.info("✅ 성능 모니터링 시스템 시작 완료")
            
        except Exception as e:
            self.logger.error(f"❌ 성능 모니터링 시스템 시작 실패: {e}")
            raise

    async def stop(self) -> None:
        """모니터링 중지"""
        self.logger.info("🛑 성능 모니터링 시스템 중지")
        self.running = False

    async def _collect_metrics_loop(self) -> None:
        """메트릭 수집 루프"""
        while self.running:
            try:
                # 시스템 메트릭 수집
                await self._collect_system_metrics()
                
                # 트레이딩 메트릭 수집
                await self._collect_trading_metrics()
                
                # API 메트릭 수집
                await self._collect_api_metrics()
                
                # 메트릭 저장
                await self._save_metrics()
                
                # 통계 업데이트
                self.stats['total_metrics_collected'] += 1
                self.stats['last_collection'] = datetime.now()
                
                # 1초 대기
                await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.error(f"메트릭 수집 오류: {e}")
                self.stats['errors'].append({
                    'timestamp': datetime.now(),
                    'error': str(e)
                })
                await asyncio.sleep(5)

    async def _collect_system_metrics(self) -> None:
        """시스템 메트릭 수집"""
        if not PSUTIL_AVAILABLE:
            return
        
        try:
            # CPU 사용률
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # 메모리 사용률
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # 디스크 사용률
            disk = psutil.disk_usage('/')
            disk_usage = (disk.used / disk.total) * 100
            
            # 네트워크 I/O
            network = psutil.net_io_counters()
            network_io = {
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv,
                'packets_sent': network.packets_sent,
                'packets_recv': network.packets_recv
            }
            
            # 프로세스 수
            process_count = len(psutil.pids())
            
            # 업타임
            uptime = time.time() - psutil.boot_time()
            
            self.system_metrics = SystemMetrics(
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                disk_usage=disk_usage,
                network_io=network_io,
                process_count=process_count,
                uptime=uptime
            )
            
            # Prometheus 메트릭 업데이트
            if PROMETHEUS_AVAILABLE:
                self.cpu_gauge.set(cpu_usage)
                self.memory_gauge.set(memory_usage)
                self.disk_gauge.set(disk_usage)
                
        except Exception as e:
            self.logger.error(f"시스템 메트릭 수집 오류: {e}")

    async def _collect_trading_metrics(self) -> None:
        """트레이딩 메트릭 수집"""
        try:
            # 실제 구현에서는 트레이딩 시스템에서 메트릭을 가져옴
            # 여기서는 예시 데이터 사용
            self.trading_metrics = TradingMetrics(
                signals_generated=10,
                trades_executed=5,
                successful_trades=4,
                failed_trades=1,
                total_pnl=150000,
                win_rate=0.8,
                avg_trade_duration=30.5,
                active_positions=3
            )
            
            # Prometheus 메트릭 업데이트
            if PROMETHEUS_AVAILABLE:
                self.signals_counter.inc(self.trading_metrics.signals_generated)
                self.trades_counter.inc(self.trading_metrics.trades_executed)
                
        except Exception as e:
            self.logger.error(f"트레이딩 메트릭 수집 오류: {e}")

    async def _collect_api_metrics(self) -> None:
        """API 메트릭 수집"""
        try:
            # 실제 구현에서는 API 서버에서 메트릭을 가져옴
            # 여기서는 예시 데이터 사용
            self.api_metrics = APIMetrics(
                total_requests=1000,
                successful_requests=950,
                failed_requests=50,
                avg_response_time=120.5,
                active_connections=25,
                requests_per_minute=60.0
            )
            
            # Prometheus 메트릭 업데이트
            if PROMETHEUS_AVAILABLE:
                self.api_requests_counter.inc(self.api_metrics.total_requests)
                self.response_time_histogram.observe(self.api_metrics.avg_response_time / 1000)
                
        except Exception as e:
            self.logger.error(f"API 메트릭 수집 오류: {e}")

    async def _save_metrics(self) -> None:
        """메트릭 저장"""
        try:
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'system': self.system_metrics.__dict__ if self.system_metrics else None,
                'trading': self.trading_metrics.__dict__ if self.trading_metrics else None,
                'api': self.api_metrics.__dict__ if self.api_metrics else None
            }
            
            self.metrics_history.append(metrics)
            
            # 히스토리 크기 제한
            if len(self.metrics_history) > self.max_history_size:
                self.metrics_history = self.metrics_history[-self.max_history_size:]
                
        except Exception as e:
            self.logger.error(f"메트릭 저장 오류: {e}")

    def get_current_metrics(self) -> Dict[str, Any]:
        """현재 메트릭 조회"""
        return {
            'system': self.system_metrics.__dict__ if self.system_metrics else None,
            'trading': self.trading_metrics.__dict__ if self.trading_metrics else None,
            'api': self.api_metrics.__dict__ if self.api_metrics else None,
            'stats': self.stats
        }

    def get_metrics_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """메트릭 히스토리 조회"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            metrics for metrics in self.metrics_history
            if datetime.fromisoformat(metrics['timestamp']) > cutoff_time
        ]

    def get_performance_summary(self) -> Dict[str, Any]:
        """성능 요약 조회"""
        if not self.metrics_history:
            return {}
        
        recent_metrics = self.get_metrics_history(hours=1)
        
        if not recent_metrics:
            return {}
        
        # 시스템 성능 평균
        system_metrics = [m['system'] for m in recent_metrics if m['system']]
        if system_metrics:
            avg_cpu = sum(m['cpu_usage'] for m in system_metrics) / len(system_metrics)
            avg_memory = sum(m['memory_usage'] for m in system_metrics) / len(system_metrics)
        else:
            avg_cpu = avg_memory = 0.0
        
        # 트레이딩 성능
        trading_metrics = [m['trading'] for m in recent_metrics if m['trading']]
        if trading_metrics:
            total_signals = sum(m['signals_generated'] for m in trading_metrics)
            total_trades = sum(m['trades_executed'] for m in trading_metrics)
            avg_win_rate = sum(m['win_rate'] for m in trading_metrics) / len(trading_metrics)
        else:
            total_signals = total_trades = avg_win_rate = 0.0
        
        # API 성능
        api_metrics = [m['api'] for m in recent_metrics if m['api']]
        if api_metrics:
            total_requests = sum(m['total_requests'] for m in api_metrics)
            avg_response_time = sum(m['avg_response_time'] for m in api_metrics) / len(api_metrics)
        else:
            total_requests = avg_response_time = 0.0
        
        return {
            'summary': {
                'avg_cpu_usage': round(avg_cpu, 2),
                'avg_memory_usage': round(avg_memory, 2),
                'total_signals_generated': total_signals,
                'total_trades_executed': total_trades,
                'avg_win_rate': round(avg_win_rate, 3),
                'total_api_requests': total_requests,
                'avg_api_response_time': round(avg_response_time, 2)
            },
            'alerts': self._generate_alerts()
        }

    def _generate_alerts(self) -> List[Dict[str, Any]]:
        """알림 생성"""
        alerts = []
        
        if self.system_metrics:
            # CPU 사용률 알림
            if self.system_metrics.cpu_usage > 80:
                alerts.append({
                    'level': 'warning',
                    'message': f"CPU 사용률이 높습니다: {self.system_metrics.cpu_usage}%",
                    'timestamp': datetime.now().isoformat()
                })
            
            # 메모리 사용률 알림
            if self.system_metrics.memory_usage > 85:
                alerts.append({
                    'level': 'critical',
                    'message': f"메모리 사용률이 매우 높습니다: {self.system_metrics.memory_usage}%",
                    'timestamp': datetime.now().isoformat()
                })
        
        if self.api_metrics:
            # API 응답 시간 알림
            if self.api_metrics.avg_response_time > 500:
                alerts.append({
                    'level': 'warning',
                    'message': f"API 응답 시간이 느립니다: {self.api_metrics.avg_response_time}ms",
                    'timestamp': datetime.now().isoformat()
                })
        
        return alerts

    @asynccontextmanager
    async def measure_response_time(self, endpoint: str):
        """응답 시간 측정 컨텍스트 매니저"""
        start_time = time.time()
        try:
            yield
        finally:
            response_time = (time.time() - start_time) * 1000  # ms
            if PROMETHEUS_AVAILABLE:
                self.response_time_histogram.observe(response_time / 1000)
            self.logger.debug(f"API 응답 시간 ({endpoint}): {response_time:.2f}ms")


# 전역 인스턴스
performance_monitor = PerformanceMonitor() 