"""
성능 모니터링 및 메모리 최적화 시스템
"""
import asyncio
import time
import psutil
import gc
import threading
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from functools import wraps
import weakref
from concurrent.futures import ThreadPoolExecutor
import memory_profiler
import structlog

from config.settings import settings

logger = structlog.get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """성능 메트릭 데이터 클래스"""
    timestamp: datetime = field(default_factory=datetime.now)
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_mb: float = 0.0
    memory_available_mb: float = 0.0
    disk_io_read_mb: float = 0.0
    disk_io_write_mb: float = 0.0
    network_sent_mb: float = 0.0
    network_recv_mb: float = 0.0
    active_threads: int = 0
    active_connections: int = 0
    cache_hit_rate: float = 0.0
    response_time_ms: float = 0.0


@dataclass
class FunctionMetrics:
    """함수 실행 메트릭"""
    name: str
    call_count: int = 0
    total_time: float = 0.0
    avg_time: float = 0.0
    max_time: float = 0.0
    min_time: float = float('inf')
    error_count: int = 0
    last_called: Optional[datetime] = None
    
    def update(self, execution_time: float, error: bool = False):
        """메트릭 업데이트"""
        self.call_count += 1
        self.total_time += execution_time
        self.avg_time = self.total_time / self.call_count
        self.max_time = max(self.max_time, execution_time)
        self.min_time = min(self.min_time, execution_time)
        self.last_called = datetime.now()
        
        if error:
            self.error_count += 1


class MemoryOptimizer:
    """메모리 최적화 관리자"""
    
    def __init__(self):
        self.weak_refs: weakref.WeakSet = weakref.WeakSet()
        self.memory_threshold_mb = settings.performance.memory_limit_mb
        self._cleanup_interval = 60  # 60초마다 정리
        self._last_cleanup = time.time()
    
    def register_object(self, obj: Any) -> None:
        """메모리 관리 대상 객체 등록"""
        try:
            self.weak_refs.add(obj)
        except TypeError:
            # WeakSet에 추가할 수 없는 객체는 무시
            pass
    
    def get_memory_usage(self) -> Dict[str, float]:
        """현재 메모리 사용량 조회"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,  # 물리 메모리
            "vms_mb": memory_info.vms / 1024 / 1024,  # 가상 메모리
            "percent": process.memory_percent(),
            "available_mb": psutil.virtual_memory().available / 1024 / 1024
        }
    
    def force_gc(self) -> Dict[str, int]:
        """강제 가비지 컬렉션 실행"""
        before = len(gc.get_objects())
        
        # 모든 세대의 가비지 컬렉션 실행
        collected = []
        for generation in range(3):
            collected.append(gc.collect(generation))
        
        after = len(gc.get_objects())
        
        return {
            "objects_before": before,
            "objects_after": after,
            "objects_freed": before - after,
            "collected_by_generation": collected
        }
    
    def optimize_memory(self) -> Dict[str, Any]:
        """메모리 최적화 실행"""
        current_memory = self.get_memory_usage()
        
        # 메모리 임계값 초과 시 최적화 실행
        if current_memory["rss_mb"] > self.memory_threshold_mb:
            logger.warning(f"메모리 사용량 임계값 초과: {current_memory['rss_mb']:.2f}MB")
            
            # 가비지 컬렉션 실행
            gc_result = self.force_gc()
            
            # 약한 참조 정리
            alive_refs = len(self.weak_refs)
            
            # 최적화 후 메모리 사용량
            optimized_memory = self.get_memory_usage()
            
            return {
                "before": current_memory,
                "after": optimized_memory,
                "freed_mb": current_memory["rss_mb"] - optimized_memory["rss_mb"],
                "gc_result": gc_result,
                "alive_weak_refs": alive_refs
            }
        
        return {"status": "no_optimization_needed", "current": current_memory}
    
    def should_cleanup(self) -> bool:
        """정리가 필요한지 확인"""
        return time.time() - self._last_cleanup > self._cleanup_interval
    
    async def periodic_cleanup(self) -> None:
        """주기적 메모리 정리"""
        if self.should_cleanup():
            self.optimize_memory()
            self._last_cleanup = time.time()


class PerformanceMonitor:
    """성능 모니터링 시스템"""
    
    def __init__(self):
        self.metrics_history: List[PerformanceMetrics] = []
        self.function_metrics: Dict[str, FunctionMetrics] = {}
        self.memory_optimizer = MemoryOptimizer()
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._lock = threading.RLock()
        self.max_history_size = 1000
        
        # 초기 시스템 정보
        self._initial_disk_io = psutil.disk_io_counters()
        self._initial_network_io = psutil.net_io_counters()
    
    async def start_monitoring(self, interval: float = 30.0) -> None:
        """성능 모니터링 시작"""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_loop(interval))
        logger.info(f"성능 모니터링 시작 (간격: {interval}초)")
    
    async def stop_monitoring(self) -> None:
        """성능 모니터링 중지"""
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("성능 모니터링 중지")
    
    async def _monitor_loop(self, interval: float) -> None:
        """모니터링 루프"""
        while self._monitoring:
            try:
                metrics = await self._collect_metrics()
                
                with self._lock:
                    self.metrics_history.append(metrics)
                    
                    # 히스토리 크기 제한
                    if len(self.metrics_history) > self.max_history_size:
                        self.metrics_history = self.metrics_history[-self.max_history_size:]
                
                # 메모리 최적화 확인
                await self.memory_optimizer.periodic_cleanup()
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"성능 모니터링 오류: {e}")
                await asyncio.sleep(interval)
    
    async def _collect_metrics(self) -> PerformanceMetrics:
        """시스템 메트릭 수집"""
        # CPU 사용률
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # 메모리 정보
        memory = psutil.virtual_memory()
        
        # 디스크 I/O
        disk_io = psutil.disk_io_counters()
        disk_read_mb = (disk_io.read_bytes - self._initial_disk_io.read_bytes) / 1024 / 1024
        disk_write_mb = (disk_io.write_bytes - self._initial_disk_io.write_bytes) / 1024 / 1024
        
        # 네트워크 I/O
        network_io = psutil.net_io_counters()
        network_sent_mb = (network_io.bytes_sent - self._initial_network_io.bytes_sent) / 1024 / 1024
        network_recv_mb = (network_io.bytes_recv - self._initial_network_io.bytes_recv) / 1024 / 1024
        
        # 스레드 수
        active_threads = threading.active_count()
        
        return PerformanceMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_mb=memory.used / 1024 / 1024,
            memory_available_mb=memory.available / 1024 / 1024,
            disk_io_read_mb=disk_read_mb,
            disk_io_write_mb=disk_write_mb,
            network_sent_mb=network_sent_mb,
            network_recv_mb=network_recv_mb,
            active_threads=active_threads
        )
    
    async def get_current_metrics(self) -> PerformanceMetrics:
        """현재 메트릭 조회 (비동기)"""
        with self._lock:
            if self.metrics_history:
                return self.metrics_history[-1]
        
        # 히스토리가 없으면 즉시 메트릭 수집
        return await self._collect_metrics()
    
    def get_metrics_summary(self, minutes: int = 60) -> Dict[str, Any]:
        """지정된 시간 범위의 메트릭 요약"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        with self._lock:
            recent_metrics = [
                m for m in self.metrics_history 
                if m.timestamp >= cutoff_time
            ]
        
        if not recent_metrics:
            return {}
        
        return {
            "period_minutes": minutes,
            "sample_count": len(recent_metrics),
            "cpu_percent": {
                "avg": sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics),
                "max": max(m.cpu_percent for m in recent_metrics),
                "min": min(m.cpu_percent for m in recent_metrics)
            },
            "memory_percent": {
                "avg": sum(m.memory_percent for m in recent_metrics) / len(recent_metrics),
                "max": max(m.memory_percent for m in recent_metrics),
                "min": min(m.memory_percent for m in recent_metrics)
            },
            "memory_used_mb": {
                "avg": sum(m.memory_used_mb for m in recent_metrics) / len(recent_metrics),
                "max": max(m.memory_used_mb for m in recent_metrics),
                "min": min(m.memory_used_mb for m in recent_metrics)
            }
        }
    
    def record_function_call(self, func_name: str, execution_time: float, error: bool = False) -> None:
        """함수 호출 메트릭 기록"""
        with self._lock:
            if func_name not in self.function_metrics:
                self.function_metrics[func_name] = FunctionMetrics(name=func_name)
            
            self.function_metrics[func_name].update(execution_time, error)
    
    def get_function_metrics(self) -> Dict[str, Dict[str, Any]]:
        """함수 메트릭 조회"""
        with self._lock:
            return {
                name: {
                    "call_count": metrics.call_count,
                    "avg_time_ms": metrics.avg_time * 1000,
                    "max_time_ms": metrics.max_time * 1000,
                    "min_time_ms": metrics.min_time * 1000 if metrics.min_time != float('inf') else 0,
                    "error_rate": metrics.error_count / max(metrics.call_count, 1),
                    "last_called": metrics.last_called.isoformat() if metrics.last_called else None
                }
                for name, metrics in self.function_metrics.items()
            }
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """메모리 통계 조회"""
        return self.memory_optimizer.get_memory_usage()
    
    def optimize_memory(self) -> Dict[str, Any]:
        """메모리 최적화 실행"""
        return self.memory_optimizer.optimize_memory()


# 전역 성능 모니터 인스턴스
performance_monitor = PerformanceMonitor()


def monitor_performance(func_name: Optional[str] = None):
    """성능 모니터링 데코레이터"""
    def decorator(func: Callable) -> Callable:
        name = func_name or f"{func.__module__}.{func.__name__}"
        
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                error = False
                
                try:
                    # 메모리 관리 대상으로 등록
                    performance_monitor.memory_optimizer.register_object(args)
                    performance_monitor.memory_optimizer.register_object(kwargs)
                    
                    result = await func(*args, **kwargs)
                    return result
                    
                except Exception as e:
                    error = True
                    raise
                finally:
                    execution_time = time.time() - start_time
                    performance_monitor.record_function_call(name, execution_time, error)
            
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                error = False
                
                try:
                    # 메모리 관리 대상으로 등록
                    performance_monitor.memory_optimizer.register_object(args)
                    performance_monitor.memory_optimizer.register_object(kwargs)
                    
                    result = func(*args, **kwargs)
                    return result
                    
                except Exception as e:
                    error = True
                    raise
                finally:
                    execution_time = time.time() - start_time
                    performance_monitor.record_function_call(name, execution_time, error)
            
            return sync_wrapper
    
    return decorator


async def initialize_performance_monitoring() -> None:
    """성능 모니터링 초기화"""
    await performance_monitor.start_monitoring()


async def cleanup_performance_monitoring() -> None:
    """성능 모니터링 정리"""
    await performance_monitor.stop_monitoring() 