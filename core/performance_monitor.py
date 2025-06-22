#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Ultra 성능 모니터 v5.0 - 고성능 실시간 성능 모니터링 시스템
- 실시간 메트릭 수집 & 분석
- 자동 성능 최적화 & 알림
- 멀티레벨 캐싱 & 백그라운드 처리
- 메모리 최적화 & 리소스 관리
"""

import asyncio
import time
import psutil
import threading
import weakref
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import functools
import inspect
import gc
import sys
import traceback
import json
from pathlib import Path
import structlog
import numpy as np
from collections import defaultdict, deque
import queue
import signal
import os

from core.cache_manager import get_cache_manager
from config.settings import settings

logger = structlog.get_logger(__name__)


class MetricType(Enum):
    """메트릭 유형"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    RATE = "rate"


class AlertLevel(Enum):
    """알림 레벨"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class OptimizationAction(Enum):
    """최적화 액션"""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    CACHE_CLEAR = "cache_clear"
    GC_COLLECT = "gc_collect"
    THREAD_ADJUST = "thread_adjust"
    MEMORY_OPTIMIZE = "memory_optimize"


@dataclass
class MetricData:
    """메트릭 데이터"""
    name: str
    value: Union[int, float]
    metric_type: MetricType
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceSnapshot:
    """성능 스냅샷"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    network_sent_mb: float
    network_recv_mb: float
    active_threads: int
    open_files: int
    connections: int
    gc_collections: Dict[int, int] = field(default_factory=dict)
    custom_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class FunctionMetrics:
    """함수 성능 메트릭"""
    name: str
    call_count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    avg_time: float = 0.0
    error_count: int = 0
    last_called: Optional[datetime] = None
    memory_usage: float = 0.0
    
    def update(self, execution_time: float, memory_delta: float = 0.0, error: bool = False):
        """메트릭 업데이트"""
        self.call_count += 1
        self.total_time += execution_time
        self.min_time = min(self.min_time, execution_time)
        self.max_time = max(self.max_time, execution_time)
        self.avg_time = self.total_time / self.call_count
        self.memory_usage += memory_delta
        self.last_called = datetime.now()
        
        if error:
            self.error_count += 1


@dataclass
class Alert:
    """성능 알림"""
    level: AlertLevel
    message: str
    metric_name: str
    current_value: float
    threshold: float
    timestamp: datetime
    resolved: bool = False
    action_taken: Optional[OptimizationAction] = None


@dataclass
class OptimizationRule:
    """최적화 규칙"""
    name: str
    condition: Callable[[PerformanceSnapshot], bool]
    action: OptimizationAction
    cooldown_seconds: int = 300
    last_executed: Optional[datetime] = None
    execution_count: int = 0
    
    def can_execute(self) -> bool:
        """실행 가능 여부 확인"""
        if self.last_executed is None:
            return True
        return (datetime.now() - self.last_executed).total_seconds() >= self.cooldown_seconds


class UltraPerformanceMonitor:
    """🚀 Ultra 성능 모니터 - 고성능 실시간 성능 모니터링"""
    
    def __init__(self):
        # 메트릭 저장소
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self._function_metrics: Dict[str, FunctionMetrics] = {}
        self._snapshots: deque = deque(maxlen=1000)
        
        # 알림 및 최적화
        self._alerts: deque = deque(maxlen=1000)
        self._optimization_rules: List[OptimizationRule] = []
        self._thresholds: Dict[str, Dict[str, float]] = {}
        
        # 비동기 처리
        self._metric_queue: asyncio.Queue = asyncio.Queue(maxsize=50000)
        self._workers: List[asyncio.Task] = []
        self._is_running = False
        
        # 스레드 풀 및 동기화
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="perf_monitor")
        self._lock = threading.RLock()
        
        # 캐시 매니저
        self._cache_manager = None
        
        # 성능 통계
        self._stats = {
            'metrics_processed': 0,
            'alerts_generated': 0,
            'optimizations_executed': 0,
            'errors_occurred': 0,
            'uptime_start': datetime.now()
        }
        
        # 시스템 정보 캐시
        self._system_info_cache = {}
        self._last_system_update = 0
        
        # 약한 참조로 모니터링 대상 관리
        self._monitored_objects: weakref.WeakSet = weakref.WeakSet()
        
        # 기본 최적화 규칙 설정
        self._setup_default_optimization_rules()
        
        # 기본 임계값 설정
        self._setup_default_thresholds()
        
        logger.info("Ultra 성능 모니터 초기화 완료")
    
    async def initialize(self) -> None:
        """성능 모니터 초기화"""
        try:
            # 캐시 매니저 연결
            self._cache_manager = get_cache_manager()
            
            # 백그라운드 워커 시작
            await self._start_workers()
            
            # 시스템 정보 수집 시작
            await self._start_system_monitoring()
            
            self._is_running = True
            logger.info("Ultra 성능 모니터 초기화 완료")
            
        except Exception as e:
            logger.error(f"성능 모니터 초기화 실패: {e}")
            raise
    
    async def _start_workers(self) -> None:
        """백그라운드 워커 시작"""
        try:
            # 메트릭 처리 워커
            for i in range(3):
                worker = asyncio.create_task(
                    self._metric_processor_worker(f"metric_worker_{i}")
                )
                self._workers.append(worker)
            
            # 시스템 모니터링 워커
            system_worker = asyncio.create_task(self._system_monitor_worker())
            self._workers.append(system_worker)
            
            # 알림 처리 워커
            alert_worker = asyncio.create_task(self._alert_processor_worker())
            self._workers.append(alert_worker)
            
            # 최적화 엔진 워커
            optimization_worker = asyncio.create_task(self._optimization_engine_worker())
            self._workers.append(optimization_worker)
            
            # 정리 워커
            cleanup_worker = asyncio.create_task(self._cleanup_worker())
            self._workers.append(cleanup_worker)
            
            logger.info(f"성능 모니터 워커 시작: {len(self._workers)}개")
            
        except Exception as e:
            logger.error(f"워커 시작 실패: {e}")
            raise
    
    async def _start_system_monitoring(self) -> None:
        """시스템 모니터링 시작"""
        try:
            # 초기 시스템 정보 수집
            await self._collect_system_info()
            
            # 주기적 수집 스케줄링
            asyncio.create_task(self._periodic_system_collection())
            
            logger.debug("시스템 모니터링 시작")
            
        except Exception as e:
            logger.error(f"시스템 모니터링 시작 실패: {e}")
    
    def _setup_default_optimization_rules(self) -> None:
        """기본 최적화 규칙 설정"""
        try:
            # CPU 사용률 기반 최적화
            self._optimization_rules.append(
                OptimizationRule(
                    name="high_cpu_gc",
                    condition=lambda s: s.cpu_percent > 80,
                    action=OptimizationAction.GC_COLLECT,
                    cooldown_seconds=60
                )
            )
            
            # 메모리 사용률 기반 최적화
            self._optimization_rules.append(
                OptimizationRule(
                    name="high_memory_cache_clear",
                    condition=lambda s: s.memory_percent > 85,
                    action=OptimizationAction.CACHE_CLEAR,
                    cooldown_seconds=120
                )
            )
            
            # 극심한 메모리 사용률
            self._optimization_rules.append(
                OptimizationRule(
                    name="critical_memory_optimize",
                    condition=lambda s: s.memory_percent > 95,
                    action=OptimizationAction.MEMORY_OPTIMIZE,
                    cooldown_seconds=30
                )
            )
            
            logger.debug(f"기본 최적화 규칙 설정: {len(self._optimization_rules)}개")
            
        except Exception as e:
            logger.error(f"최적화 규칙 설정 실패: {e}")
    
    def _setup_default_thresholds(self) -> None:
        """기본 임계값 설정"""
        self._thresholds = {
            'cpu_percent': {'warning': 70.0, 'critical': 90.0},
            'memory_percent': {'warning': 80.0, 'critical': 95.0},
            'disk_usage_percent': {'warning': 85.0, 'critical': 95.0},
            'response_time': {'warning': 1.0, 'critical': 5.0},
            'error_rate': {'warning': 0.05, 'critical': 0.10},
            'active_threads': {'warning': 100, 'critical': 200}
        }
    
    # 메트릭 수집
    async def record_metric(self, 
                          name: str, 
                          value: Union[int, float],
                          metric_type: MetricType = MetricType.GAUGE,
                          tags: Optional[Dict[str, str]] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> None:
        """메트릭 기록"""
        try:
            metric = MetricData(
                name=name,
                value=value,
                metric_type=metric_type,
                timestamp=datetime.now(),
                tags=tags or {},
                metadata=metadata or {}
            )
            
            # 큐에 추가 (논블로킹)
            if not self._metric_queue.full():
                await self._metric_queue.put(metric)
            else:
                logger.warning(f"메트릭 큐 가득참: {name}")
            
        except Exception as e:
            logger.error(f"메트릭 기록 실패 {name}: {e}")
    
    def record_metric_sync(self, 
                          name: str, 
                          value: Union[int, float],
                          metric_type: MetricType = MetricType.GAUGE) -> None:
        """동기 메트릭 기록"""
        try:
            with self._lock:
                self._metrics[name].append({
                    'value': value,
                    'timestamp': datetime.now(),
                    'type': metric_type.value
                })
                self._stats['metrics_processed'] += 1
                
        except Exception as e:
            logger.error(f"동기 메트릭 기록 실패 {name}: {e}")
    
    async def _collect_system_info(self) -> PerformanceSnapshot:
        """시스템 정보 수집"""
        try:
            # 캐시된 정보 확인 (1초 캐시)
            now = time.time()
            if now - self._last_system_update < 1.0 and self._system_info_cache:
                return self._system_info_cache.get('snapshot')
            
            # CPU 정보
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # 메모리 정보
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_mb = memory.used / (1024 * 1024)
            memory_available_mb = memory.available / (1024 * 1024)
            
            # 디스크 정보
            disk = psutil.disk_usage('/')
            disk_usage_percent = disk.percent
            
            # 네트워크 정보
            network = psutil.net_io_counters()
            network_sent_mb = network.bytes_sent / (1024 * 1024)
            network_recv_mb = network.bytes_recv / (1024 * 1024)
            
            # 프로세스 정보
            process = psutil.Process()
            active_threads = process.num_threads()
            open_files = len(process.open_files())
            connections = len(process.connections())
            
            # GC 정보
            gc_collections = {i: gc.get_count()[i] for i in range(3)}
            
            snapshot = PerformanceSnapshot(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_used_mb=memory_used_mb,
                memory_available_mb=memory_available_mb,
                disk_usage_percent=disk_usage_percent,
                network_sent_mb=network_sent_mb,
                network_recv_mb=network_recv_mb,
                active_threads=active_threads,
                open_files=open_files,
                connections=connections,
                gc_collections=gc_collections
            )
            
            # 캐시 업데이트
            self._system_info_cache['snapshot'] = snapshot
            self._last_system_update = now
            
            return snapshot
            
        except Exception as e:
            logger.error(f"시스템 정보 수집 실패: {e}")
            return PerformanceSnapshot(
                timestamp=datetime.now(),
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_used_mb=0.0,
                memory_available_mb=0.0,
                disk_usage_percent=0.0,
                network_sent_mb=0.0,
                network_recv_mb=0.0,
                active_threads=0,
                open_files=0,
                connections=0
            )
    
    # 워커들
    async def _metric_processor_worker(self, worker_name: str) -> None:
        """메트릭 처리 워커"""
        while self._is_running:
            try:
                metric = await asyncio.wait_for(self._metric_queue.get(), timeout=1.0)
                
                # 메트릭 저장
                with self._lock:
                    self._metrics[metric.name].append({
                        'value': metric.value,
                        'timestamp': metric.timestamp,
                        'type': metric.metric_type.value,
                        'tags': metric.tags,
                        'metadata': metric.metadata
                    })
                    self._stats['metrics_processed'] += 1
                
                # 알림 확인
                await self._check_alerts(metric)
                
                self._metric_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"{worker_name} 오류: {e}")
                self._stats['errors_occurred'] += 1
                await asyncio.sleep(0.1)
    
    async def _system_monitor_worker(self) -> None:
        """시스템 모니터링 워커"""
        while self._is_running:
            try:
                # 시스템 스냅샷 수집
                snapshot = await self._collect_system_info()
                
                # 스냅샷 저장
                with self._lock:
                    self._snapshots.append(snapshot)
                
                # 시스템 메트릭 기록
                await self._record_system_metrics(snapshot)
                
                await asyncio.sleep(5)  # 5초마다 수집
                
            except Exception as e:
                logger.error(f"시스템 모니터링 워커 오류: {e}")
                await asyncio.sleep(1)
    
    async def _alert_processor_worker(self) -> None:
        """알림 처리 워커"""
        while self._is_running:
            try:
                await asyncio.sleep(10)  # 10초마다 체크
                
                # 최근 스냅샷 확인
                if self._snapshots:
                    latest_snapshot = self._snapshots[-1]
                    await self._process_system_alerts(latest_snapshot)
                
            except Exception as e:
                logger.error(f"알림 처리 워커 오류: {e}")
                await asyncio.sleep(1)
    
    async def _optimization_engine_worker(self) -> None:
        """최적화 엔진 워커"""
        while self._is_running:
            try:
                await asyncio.sleep(30)  # 30초마다 최적화 체크
                
                if self._snapshots:
                    latest_snapshot = self._snapshots[-1]
                    await self._execute_optimizations(latest_snapshot)
                
            except Exception as e:
                logger.error(f"최적화 엔진 워커 오류: {e}")
                await asyncio.sleep(5)
    
    async def _cleanup_worker(self) -> None:
        """정리 워커"""
        while self._is_running:
            try:
                await asyncio.sleep(300)  # 5분마다 정리
                
                # 오래된 메트릭 정리
                cutoff_time = datetime.now() - timedelta(hours=1)
                
                with self._lock:
                    for name, metric_deque in self._metrics.items():
                        # 오래된 메트릭 제거
                        while metric_deque and metric_deque[0]['timestamp'] < cutoff_time:
                            metric_deque.popleft()
                
                # 해결된 알림 정리
                resolved_alerts = [alert for alert in self._alerts if alert.resolved]
                if len(resolved_alerts) > 100:
                    # 최근 100개만 유지
                    for alert in resolved_alerts[:-100]:
                        self._alerts.remove(alert)
                
                # 가비지 컬렉션
                gc.collect()
                
                logger.debug("정리 워커 실행 완료")
                
            except Exception as e:
                logger.error(f"정리 워커 오류: {e}")
                await asyncio.sleep(10)
    
    async def _record_system_metrics(self, snapshot: PerformanceSnapshot) -> None:
        """시스템 메트릭 기록"""
        try:
            # 각 시스템 메트릭을 개별적으로 기록
            await self.record_metric("cpu_percent", snapshot.cpu_percent)
            await self.record_metric("memory_percent", snapshot.memory_percent)
            await self.record_metric("memory_used_mb", snapshot.memory_used_mb)
            await self.record_metric("disk_usage_percent", snapshot.disk_usage_percent)
            await self.record_metric("active_threads", snapshot.active_threads)
            await self.record_metric("open_files", snapshot.open_files)
            await self.record_metric("connections", snapshot.connections)
            
        except Exception as e:
            logger.error(f"시스템 메트릭 기록 실패: {e}")
    
    async def _check_alerts(self, metric: MetricData) -> None:
        """알림 확인"""
        try:
            metric_name = metric.name
            value = metric.value
            
            # 임계값 확인
            if metric_name in self._thresholds:
                thresholds = self._thresholds[metric_name]
                
                # 경고 레벨
                if 'warning' in thresholds and value >= thresholds['warning']:
                    alert = Alert(
                        level=AlertLevel.WARNING,
                        message=f"{metric_name} 경고 임계값 초과: {value}",
                        metric_name=metric_name,
                        current_value=value,
                        threshold=thresholds['warning'],
                        timestamp=datetime.now()
                    )
                    self._alerts.append(alert)
                    self._stats['alerts_generated'] += 1
                
                # 위험 레벨
                if 'critical' in thresholds and value >= thresholds['critical']:
                    alert = Alert(
                        level=AlertLevel.CRITICAL,
                        message=f"{metric_name} 위험 임계값 초과: {value}",
                        metric_name=metric_name,
                        current_value=value,
                        threshold=thresholds['critical'],
                        timestamp=datetime.now()
                    )
                    self._alerts.append(alert)
                    self._stats['alerts_generated'] += 1
            
        except Exception as e:
            logger.error(f"알림 확인 실패: {e}")
    
    async def _process_system_alerts(self, snapshot: PerformanceSnapshot) -> None:
        """시스템 알림 처리"""
        try:
            # CPU 알림
            if snapshot.cpu_percent >= self._thresholds['cpu_percent']['critical']:
                alert = Alert(
                    level=AlertLevel.CRITICAL,
                    message=f"CPU 사용률 위험: {snapshot.cpu_percent:.1f}%",
                    metric_name="cpu_percent",
                    current_value=snapshot.cpu_percent,
                    threshold=self._thresholds['cpu_percent']['critical'],
                    timestamp=datetime.now()
                )
                self._alerts.append(alert)
            
            # 메모리 알림
            if snapshot.memory_percent >= self._thresholds['memory_percent']['critical']:
                alert = Alert(
                    level=AlertLevel.CRITICAL,
                    message=f"메모리 사용률 위험: {snapshot.memory_percent:.1f}%",
                    metric_name="memory_percent",
                    current_value=snapshot.memory_percent,
                    threshold=self._thresholds['memory_percent']['critical'],
                    timestamp=datetime.now()
                )
                self._alerts.append(alert)
            
        except Exception as e:
            logger.error(f"시스템 알림 처리 실패: {e}")
    
    async def _execute_optimizations(self, snapshot: PerformanceSnapshot) -> None:
        """최적화 실행"""
        try:
            for rule in self._optimization_rules:
                if rule.can_execute() and rule.condition(snapshot):
                    await self._execute_optimization_action(rule.action, snapshot)
                    rule.last_executed = datetime.now()
                    rule.execution_count += 1
                    self._stats['optimizations_executed'] += 1
                    
                    logger.info(f"최적화 실행: {rule.name} - {rule.action.value}")
            
        except Exception as e:
            logger.error(f"최적화 실행 실패: {e}")
    
    async def _execute_optimization_action(self, 
                                         action: OptimizationAction, 
                                         snapshot: PerformanceSnapshot) -> None:
        """최적화 액션 실행"""
        try:
            if action == OptimizationAction.GC_COLLECT:
                # 가비지 컬렉션 실행
                collected = gc.collect()
                logger.info(f"가비지 컬렉션 실행: {collected}개 객체 정리")
            
            elif action == OptimizationAction.CACHE_CLEAR:
                # 캐시 정리
                if self._cache_manager:
                    await self._cache_manager.clear_expired()
                    logger.info("만료된 캐시 정리 완료")
            
            elif action == OptimizationAction.MEMORY_OPTIMIZE:
                # 메모리 최적화
                gc.collect()
                # 추가적인 메모리 최적화 로직
                logger.info("메모리 최적화 실행")
            
        except Exception as e:
            logger.error(f"최적화 액션 실행 실패 {action}: {e}")
    
    async def _periodic_system_collection(self) -> None:
        """주기적 시스템 정보 수집"""
        while self._is_running:
            try:
                await asyncio.sleep(60)  # 1분마다
                
                # 확장 시스템 정보 수집
                await self._collect_extended_system_info()
                
            except Exception as e:
                logger.error(f"주기적 시스템 수집 실패: {e}")
    
    async def _collect_extended_system_info(self) -> None:
        """확장 시스템 정보 수집"""
        try:
            # 프로세스별 메모리 사용량
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'memory_percent']):
                try:
                    if proc.info['memory_percent'] > 1.0:  # 1% 이상 사용하는 프로세스만
                        processes.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # 상위 10개 프로세스
            top_processes = sorted(processes, key=lambda x: x['memory_percent'], reverse=True)[:10]
            
            # 메트릭으로 기록
            await self.record_metric(
                "top_memory_processes",
                len(top_processes),
                metadata={"processes": top_processes}
            )
            
        except Exception as e:
            logger.error(f"확장 시스템 정보 수집 실패: {e}")
    
    # 데코레이터
    def monitor_performance(self, 
                          name: Optional[str] = None,
                          track_memory: bool = True,
                          track_errors: bool = True) -> Callable:
        """성능 모니터링 데코레이터"""
        def decorator(func: Callable) -> Callable:
            func_name = name or f"{func.__module__}.{func.__qualname__}"
            
            if asyncio.iscoroutinefunction(func):
                @functools.wraps(func)
                async def async_wrapper(*args, **kwargs):
                    return await self._monitor_async_function(
                        func, func_name, track_memory, track_errors, *args, **kwargs
                    )
                return async_wrapper
            else:
                @functools.wraps(func)
                def sync_wrapper(*args, **kwargs):
                    return self._monitor_sync_function(
                        func, func_name, track_memory, track_errors, *args, **kwargs
                    )
                return sync_wrapper
        
        return decorator
    
    async def _monitor_async_function(self, 
                                    func: Callable,
                                    func_name: str,
                                    track_memory: bool,
                                    track_errors: bool,
                                    *args, **kwargs) -> Any:
        """비동기 함수 모니터링"""
        start_time = time.time()
        start_memory = 0
        error_occurred = False
        
        try:
            if track_memory:
                start_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            
            result = await func(*args, **kwargs)
            
            return result
            
        except Exception as e:
            error_occurred = True
            if track_errors:
                await self.record_metric(f"{func_name}.errors", 1, MetricType.COUNTER)
            raise
        
        finally:
            end_time = time.time()
            execution_time = end_time - start_time
            
            # 메모리 사용량 계산
            memory_delta = 0
            if track_memory:
                end_memory = psutil.Process().memory_info().rss / (1024 * 1024)
                memory_delta = end_memory - start_memory
            
            # 함수 메트릭 업데이트
            if func_name not in self._function_metrics:
                self._function_metrics[func_name] = FunctionMetrics(func_name)
            
            self._function_metrics[func_name].update(execution_time, memory_delta, error_occurred)
            
            # 메트릭 기록
            await self.record_metric(f"{func_name}.execution_time", execution_time, MetricType.TIMER)
            await self.record_metric(f"{func_name}.calls", 1, MetricType.COUNTER)
            
            if track_memory and memory_delta != 0:
                await self.record_metric(f"{func_name}.memory_delta", memory_delta, MetricType.GAUGE)
    
    def _monitor_sync_function(self, 
                             func: Callable,
                             func_name: str,
                             track_memory: bool,
                             track_errors: bool,
                             *args, **kwargs) -> Any:
        """동기 함수 모니터링"""
        start_time = time.time()
        start_memory = 0
        error_occurred = False
        
        try:
            if track_memory:
                start_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            
            result = func(*args, **kwargs)
            
            return result
            
        except Exception as e:
            error_occurred = True
            if track_errors:
                self.record_metric_sync(f"{func_name}.errors", 1, MetricType.COUNTER)
            raise
        
        finally:
            end_time = time.time()
            execution_time = end_time - start_time
            
            # 메모리 사용량 계산
            memory_delta = 0
            if track_memory:
                end_memory = psutil.Process().memory_info().rss / (1024 * 1024)
                memory_delta = end_memory - start_memory
            
            # 함수 메트릭 업데이트
            if func_name not in self._function_metrics:
                self._function_metrics[func_name] = FunctionMetrics(func_name)
            
            self._function_metrics[func_name].update(execution_time, memory_delta, error_occurred)
            
            # 메트릭 기록 (동기)
            self.record_metric_sync(f"{func_name}.execution_time", execution_time, MetricType.TIMER)
            self.record_metric_sync(f"{func_name}.calls", 1, MetricType.COUNTER)
            
            if track_memory and memory_delta != 0:
                self.record_metric_sync(f"{func_name}.memory_delta", memory_delta, MetricType.GAUGE)
    
    # 조회 API
    def get_metrics(self, name: str, limit: int = 100) -> List[Dict[str, Any]]:
        """메트릭 조회"""
        try:
            with self._lock:
                if name in self._metrics:
                    return list(self._metrics[name])[-limit:]
                return []
        except Exception as e:
            logger.error(f"메트릭 조회 실패 {name}: {e}")
            return []
    
    def get_function_metrics(self, name: Optional[str] = None) -> Union[FunctionMetrics, Dict[str, FunctionMetrics]]:
        """함수 메트릭 조회"""
        try:
            if name:
                return self._function_metrics.get(name)
            return dict(self._function_metrics)
        except Exception as e:
            logger.error(f"함수 메트릭 조회 실패: {e}")
            return {} if name is None else None
    
    def get_latest_snapshot(self) -> Optional[PerformanceSnapshot]:
        """최신 성능 스냅샷 조회"""
        try:
            return self._snapshots[-1] if self._snapshots else None
        except Exception as e:
            logger.error(f"스냅샷 조회 실패: {e}")
            return None
    
    def get_alerts(self, level: Optional[AlertLevel] = None, limit: int = 100) -> List[Alert]:
        """알림 조회"""
        try:
            alerts = list(self._alerts)
            
            if level:
                alerts = [alert for alert in alerts if alert.level == level]
            
            return alerts[-limit:]
        except Exception as e:
            logger.error(f"알림 조회 실패: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """통계 정보 조회"""
        try:
            uptime = datetime.now() - self._stats['uptime_start']
            
            return {
                **self._stats,
                'uptime_seconds': uptime.total_seconds(),
                'uptime_formatted': str(uptime),
                'active_workers': len(self._workers),
                'queue_size': self._metric_queue.qsize(),
                'total_metrics': sum(len(deque) for deque in self._metrics.values()),
                'total_function_metrics': len(self._function_metrics),
                'total_snapshots': len(self._snapshots),
                'total_alerts': len(self._alerts),
                'optimization_rules': len(self._optimization_rules)
            }
        except Exception as e:
            logger.error(f"통계 조회 실패: {e}")
            return {}
    
    def get_system_summary(self) -> Dict[str, Any]:
        """시스템 요약 정보"""
        try:
            latest_snapshot = self.get_latest_snapshot()
            if not latest_snapshot:
                return {}
            
            return {
                'cpu_percent': latest_snapshot.cpu_percent,
                'memory_percent': latest_snapshot.memory_percent,
                'memory_used_gb': latest_snapshot.memory_used_mb / 1024,
                'disk_usage_percent': latest_snapshot.disk_usage_percent,
                'active_threads': latest_snapshot.active_threads,
                'open_files': latest_snapshot.open_files,
                'connections': latest_snapshot.connections,
                'timestamp': latest_snapshot.timestamp.isoformat()
            }
        except Exception as e:
            logger.error(f"시스템 요약 조회 실패: {e}")
            return {}
    
    # 설정 관리
    def set_threshold(self, metric_name: str, warning: float, critical: float) -> None:
        """임계값 설정"""
        self._thresholds[metric_name] = {
            'warning': warning,
            'critical': critical
        }
    
    def add_optimization_rule(self, rule: OptimizationRule) -> None:
        """최적화 규칙 추가"""
        self._optimization_rules.append(rule)
    
    def remove_optimization_rule(self, rule_name: str) -> bool:
        """최적화 규칙 제거"""
        for i, rule in enumerate(self._optimization_rules):
            if rule.name == rule_name:
                del self._optimization_rules[i]
                return True
        return False
    
    # 정리
    async def cleanup(self) -> None:
        """성능 모니터 정리"""
        try:
            self._is_running = False
            
            # 워커 종료
            for worker in self._workers:
                worker.cancel()
            
            # 큐 정리
            while not self._metric_queue.empty():
                try:
                    self._metric_queue.get_nowait()
                    self._metric_queue.task_done()
                except asyncio.QueueEmpty:
                    break
            
            # 스레드 풀 종료
            self._executor.shutdown(wait=False)
            
            logger.info("Ultra 성능 모니터 정리 완료")
            
        except Exception as e:
            logger.error(f"성능 모니터 정리 중 오류: {e}")


# 전역 성능 모니터 인스턴스
_performance_monitor: Optional[UltraPerformanceMonitor] = None


def get_performance_monitor() -> UltraPerformanceMonitor:
    """전역 성능 모니터 반환"""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = UltraPerformanceMonitor()
    return _performance_monitor


async def initialize_performance_monitor() -> None:
    """성능 모니터 초기화"""
    monitor = get_performance_monitor()
    await monitor.initialize()


async def cleanup_performance_monitor() -> None:
    """성능 모니터 정리"""
    global _performance_monitor
    if _performance_monitor:
        await _performance_monitor.cleanup()
        _performance_monitor = None


# 편의 함수들
def monitor_performance(name: Optional[str] = None, 
                       track_memory: bool = True, 
                       track_errors: bool = True) -> Callable:
    """성능 모니터링 데코레이터 (편의 함수)"""
    monitor = get_performance_monitor()
    return monitor.monitor_performance(name, track_memory, track_errors)


async def record_metric(name: str, 
                       value: Union[int, float],
                       metric_type: MetricType = MetricType.GAUGE) -> None:
    """메트릭 기록 (편의 함수)"""
    monitor = get_performance_monitor()
    await monitor.record_metric(name, value, metric_type)


def record_metric_sync(name: str, 
                      value: Union[int, float],
                      metric_type: MetricType = MetricType.GAUGE) -> None:
    """동기 메트릭 기록 (편의 함수)"""
    monitor = get_performance_monitor()
    monitor.record_metric_sync(name, value, metric_type) 