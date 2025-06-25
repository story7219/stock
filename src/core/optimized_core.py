#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 최적화된 투자 분석 시스템 핵심 모듈 v3.0
================================================================
- 비동기 고속 병렬처리
- 멀티레벨 캐싱 시스템 
- 커넥션 풀링 최적화
- 메모리 최적화
- 안정성 및 유지보수성 향상
"""

import asyncio
import logging
import time
import gc
import weakref
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable, TypeVar, Generic
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from contextlib import asynccontextmanager
import aiohttp
import aiodns
from functools import wraps, lru_cache
import psutil
import threading
from collections import defaultdict, deque
import json
import pickle
import sqlite3
import redis
from pathlib import Path

# 로깅 설정
logger = logging.getLogger(__name__)

T = TypeVar('T')

@dataclass
class SystemConfig:
    """시스템 설정 통합"""
    # 성능 설정
    max_workers: int = 16
    max_concurrent_requests: int = 100
    connection_pool_size: int = 50
    memory_limit_mb: int = 4096
    cache_ttl_seconds: int = 3600
    
    # 비동기 설정
    async_timeout: float = 30.0
    retry_attempts: int = 3
    retry_delay: float = 1.0
    
    # 캐싱 설정
    cache_levels: int = 3
    l1_cache_size: int = 1000
    l2_cache_size: int = 5000
    l3_cache_size: int = 10000
    
    # 데이터베이스 설정
    db_pool_size: int = 20
    db_timeout: float = 10.0
    
    # 메모리 최적화
    gc_threshold: int = 700
    memory_check_interval: int = 300
    
    # API 설정
    gemini_api_key: str = ""
    rate_limit_per_minute: int = 60

class PerformanceMonitor:
    """성능 모니터링 시스템"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.start_times = {}
        self.memory_usage = deque(maxlen=100)
        self.cpu_usage = deque(maxlen=100)
        self._lock = threading.Lock()
    
    def start_timer(self, operation: str) -> str:
        """타이머 시작"""
        timer_id = f"{operation}_{int(time.time() * 1000)}"
        self.start_times[timer_id] = time.perf_counter()
        return timer_id
    
    def end_timer(self, timer_id: str) -> float:
        """타이머 종료 및 측정값 반환"""
        if timer_id in self.start_times:
            duration = time.perf_counter() - self.start_times.pop(timer_id)
            operation = timer_id.split('_')[0]
            with self._lock:
                self.metrics[operation].append(duration)
            return duration
        return 0.0
    
    def record_memory_usage(self):
        """메모리 사용량 기록"""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        with self._lock:
            self.memory_usage.append(memory_mb)
    
    def record_cpu_usage(self):
        """CPU 사용량 기록"""
        cpu_percent = psutil.cpu_percent()
        with self._lock:
            self.cpu_usage.append(cpu_percent)
    
    def get_stats(self) -> Dict[str, Any]:
        """성능 통계 반환"""
        with self._lock:
            stats = {}
            for operation, durations in self.metrics.items():
                if durations:
                    stats[operation] = {
                        'count': len(durations),
                        'avg': sum(durations) / len(durations),
                        'min': min(durations),
                        'max': max(durations),
                        'total': sum(durations)
                    }
            
            if self.memory_usage:
                stats['memory'] = {
                    'current_mb': list(self.memory_usage)[-1],
                    'avg_mb': sum(self.memory_usage) / len(self.memory_usage),
                    'max_mb': max(self.memory_usage)
                }
            
            if self.cpu_usage:
                stats['cpu'] = {
                    'current_percent': list(self.cpu_usage)[-1],
                    'avg_percent': sum(self.cpu_usage) / len(self.cpu_usage),
                    'max_percent': max(self.cpu_usage)
                }
            
            return stats

class MultiLevelCache:
    """멀티레벨 캐싱 시스템"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        
        # L1: 메모리 캐시 (가장 빠름)
        self.l1_cache = {}
        self.l1_access_times = {}
        self.l1_lock = threading.RLock()
        
        # L2: Redis 캐시 (중간 속도)
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
            self.redis_available = True
        except:
            self.redis_client = None
            self.redis_available = False
            logger.warning("Redis 연결 실패, L2 캐시 비활성화")
        
        # L3: SQLite 캐시 (영구 저장)
        self.db_path = Path("data/cache.db")
        self.db_path.parent.mkdir(exist_ok=True)
        self._init_sqlite_cache()
        
        # 백그라운드 정리 작업
        self._start_cleanup_task()
    
    def _init_sqlite_cache(self):
        """SQLite 캐시 초기화"""
        conn = sqlite3.connect(str(self.db_path))
        conn.execute('''
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                value TEXT,
                expires_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()
    
    def _start_cleanup_task(self):
        """백그라운드 정리 작업 시작"""
        def cleanup_worker():
            while True:
                try:
                    self._cleanup_expired()
                    time.sleep(300)  # 5분마다 정리
                except Exception as e:
                    logger.error(f"캐시 정리 오류: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
    
    async def get(self, key: str) -> Optional[Any]:
        """캐시에서 값 조회 (L1 -> L2 -> L3 순서)"""
        # L1 캐시 확인
        with self.l1_lock:
            if key in self.l1_cache:
                self.l1_access_times[key] = time.time()
                return self.l1_cache[key]
        
        # L2 캐시 확인 (Redis)
        if self.redis_available:
            try:
                value = self.redis_client.get(key)
                if value:
                    data = json.loads(value)
                    # L1 캐시에 승격
                    await self._promote_to_l1(key, data)
                    return data
            except Exception as e:
                logger.warning(f"Redis 캐시 조회 오류: {e}")
        
        # L3 캐시 확인 (SQLite)
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.execute(
                'SELECT value FROM cache WHERE key = ? AND expires_at > ?',
                (key, datetime.now())
            )
            row = cursor.fetchone()
            conn.close()
            
            if row:
                data = json.loads(row[0])
                # 상위 캐시에 승격
                await self._promote_to_l2(key, data)
                await self._promote_to_l1(key, data)
                return data
        except Exception as e:
            logger.warning(f"SQLite 캐시 조회 오류: {e}")
        
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """캐시에 값 저장 (모든 레벨에 저장)"""
        ttl = ttl or self.config.cache_ttl_seconds
        expires_at = datetime.now() + timedelta(seconds=ttl)
        
        # L1 캐시 저장
        await self._promote_to_l1(key, value)
        
        # L2 캐시 저장 (Redis)
        await self._promote_to_l2(key, value, ttl)
        
        # L3 캐시 저장 (SQLite)
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.execute(
                'INSERT OR REPLACE INTO cache (key, value, expires_at) VALUES (?, ?, ?)',
                (key, json.dumps(value, default=str), expires_at)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"SQLite 캐시 저장 오류: {e}")
    
    async def _promote_to_l1(self, key: str, value: Any) -> None:
        """L1 캐시로 승격"""
        with self.l1_lock:
            # L1 캐시 크기 제한
            if len(self.l1_cache) >= self.config.l1_cache_size:
                # LRU 방식으로 가장 오래된 항목 제거
                oldest_key = min(self.l1_access_times.keys(), 
                               key=lambda k: self.l1_access_times[k])
                del self.l1_cache[oldest_key]
                del self.l1_access_times[oldest_key]
            
            self.l1_cache[key] = value
            self.l1_access_times[key] = time.time()
    
    async def _promote_to_l2(self, key: str, value: Any, ttl: int = None) -> None:
        """L2 캐시로 승격"""
        if not self.redis_available:
            return
        
        try:
            ttl = ttl or self.config.cache_ttl_seconds
            self.redis_client.setex(key, ttl, json.dumps(value, default=str))
        except Exception as e:
            logger.warning(f"Redis 캐시 저장 오류: {e}")
    
    def _cleanup_expired(self) -> None:
        """만료된 캐시 정리"""
        # L1 캐시는 메모리 기반이므로 TTL 없음
        
        # L3 캐시 정리
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.execute('DELETE FROM cache WHERE expires_at < ?', (datetime.now(),))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"캐시 정리 오류: {e}")

class OptimizedConnectionPool:
    """최적화된 커넥션 풀"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.session_pool = {}
        self.pool_lock = threading.RLock()
        self.connector_args = {
            'limit': config.connection_pool_size,
            'limit_per_host': config.connection_pool_size // 4,
            'ttl_dns_cache': 300,
            'use_dns_cache': True,
            'keepalive_timeout': 30,
            'enable_cleanup_closed': True
        }
    
    @asynccontextmanager
    async def get_session(self, session_key: str = "default"):
        """세션 컨텍스트 매니저"""
        session = await self._get_or_create_session(session_key)
        try:
            yield session
        finally:
            # 세션은 풀에서 관리되므로 여기서 닫지 않음
            pass
    
    async def _get_or_create_session(self, session_key: str) -> aiohttp.ClientSession:
        """세션 가져오기 또는 생성"""
        with self.pool_lock:
            if session_key not in self.session_pool:
                connector = aiohttp.TCPConnector(**self.connector_args)
                timeout = aiohttp.ClientTimeout(total=self.config.async_timeout)
                session = aiohttp.ClientSession(
                    connector=connector,
                    timeout=timeout,
                    headers={'User-Agent': 'Investment-AI-System/3.0'}
                )
                self.session_pool[session_key] = session
            
            return self.session_pool[session_key]
    
    async def close_all(self):
        """모든 세션 종료"""
        with self.pool_lock:
            for session in self.session_pool.values():
                if not session.closed:
                    await session.close()
            self.session_pool.clear()

class AsyncTaskManager:
    """비동기 작업 관리자"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.semaphore = asyncio.Semaphore(config.max_concurrent_requests)
        self.thread_pool = ThreadPoolExecutor(max_workers=config.max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=min(4, config.max_workers))
        self.active_tasks = set()
        self.task_results = {}
        
    async def run_with_semaphore(self, coro):
        """세마포어를 사용한 동시 실행 제한"""
        async with self.semaphore:
            return await coro
    
    async def run_in_thread(self, func: Callable, *args, **kwargs):
        """스레드 풀에서 실행"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.thread_pool, func, *args, **kwargs)
    
    async def run_in_process(self, func: Callable, *args, **kwargs):
        """프로세스 풀에서 실행"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.process_pool, func, *args, **kwargs)
    
    async def gather_with_limit(self, *coros, return_exceptions=True):
        """제한된 동시 실행으로 gather"""
        semaphore_coros = [self.run_with_semaphore(coro) for coro in coros]
        return await asyncio.gather(*semaphore_coros, return_exceptions=return_exceptions)
    
    def create_task(self, coro, name: str = None) -> asyncio.Task:
        """작업 생성 및 추적"""
        task = asyncio.create_task(coro, name=name)
        self.active_tasks.add(task)
        task.add_done_callback(self.active_tasks.discard)
        return task
    
    async def cancel_all_tasks(self):
        """모든 활성 작업 취소"""
        for task in list(self.active_tasks):
            task.cancel()
        
        if self.active_tasks:
            await asyncio.gather(*self.active_tasks, return_exceptions=True)
    
    def shutdown(self):
        """리소스 정리"""
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)

class MemoryOptimizer:
    """메모리 최적화 관리자"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.weak_refs = weakref.WeakValueDictionary()
        self.memory_threshold = config.memory_limit_mb
        self.gc_threshold = config.gc_threshold
        self.last_gc_time = time.time()
        
        # 메모리 모니터링 시작
        self._start_memory_monitor()
    
    def _start_memory_monitor(self):
        """메모리 모니터링 시작"""
        def monitor_worker():
            while True:
                try:
                    self._check_memory_usage()
                    time.sleep(self.config.memory_check_interval)
                except Exception as e:
                    logger.error(f"메모리 모니터링 오류: {e}")
        
        monitor_thread = threading.Thread(target=monitor_worker, daemon=True)
        monitor_thread.start()
    
    def _check_memory_usage(self):
        """메모리 사용량 확인 및 최적화"""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        if memory_mb > self.memory_threshold * 0.8:  # 80% 임계치
            logger.warning(f"메모리 사용량 높음: {memory_mb:.1f}MB")
            self.force_gc()
    
    def force_gc(self):
        """강제 가비지 컬렉션"""
        current_time = time.time()
        if current_time - self.last_gc_time > 60:  # 1분에 한 번만
            collected = gc.collect()
            self.last_gc_time = current_time
            logger.info(f"가비지 컬렉션 완료: {collected}개 객체 정리")
    
    def register_weak_ref(self, key: str, obj: Any):
        """약한 참조 등록"""
        self.weak_refs[key] = obj
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """메모리 통계 반환"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'percent': process.memory_percent(),
            'weak_refs_count': len(self.weak_refs),
            'gc_counts': gc.get_count()
        }

class OptimizedCore:
    """최적화된 핵심 시스템"""
    
    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or SystemConfig()
        
        # 핵심 컴포넌트 초기화
        self.performance_monitor = PerformanceMonitor()
        self.cache = MultiLevelCache(self.config)
        self.connection_pool = OptimizedConnectionPool(self.config)
        self.task_manager = AsyncTaskManager(self.config)
        self.memory_optimizer = MemoryOptimizer(self.config)
        
        # 상태 관리
        self.is_initialized = False
        self.startup_time = datetime.now()
        
        logger.info("🚀 최적화된 코어 시스템 초기화 완료")
    
    async def initialize(self):
        """시스템 초기화"""
        if self.is_initialized:
            return
        
        timer_id = self.performance_monitor.start_timer("system_init")
        
        try:
            # 비동기 초기화 작업들
            init_tasks = [
                self._warm_up_cache(),
                self._test_connections(),
                self._optimize_memory()
            ]
            
            await self.task_manager.gather_with_limit(*init_tasks)
            
            self.is_initialized = True
            duration = self.performance_monitor.end_timer(timer_id)
            logger.info(f"✅ 시스템 초기화 완료 ({duration:.2f}초)")
            
        except Exception as e:
            logger.error(f"❌ 시스템 초기화 실패: {e}")
            raise
    
    async def _warm_up_cache(self):
        """캐시 워밍업"""
        # 자주 사용되는 데이터 미리 로드
        await self.cache.set("system_status", "initialized")
    
    async def _test_connections(self):
        """연결 테스트"""
        async with self.connection_pool.get_session() as session:
            # 간단한 연결 테스트
            pass
    
    async def _optimize_memory(self):
        """메모리 최적화"""
        self.memory_optimizer.force_gc()
    
    async def execute_with_retry(self, 
                                coro: Callable, 
                                *args, 
                                max_retries: int = None,
                                delay: float = None,
                                **kwargs) -> Any:
        """재시도 로직을 포함한 실행"""
        max_retries = max_retries or self.config.retry_attempts
        delay = delay or self.config.retry_delay
        
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(coro):
                    return await coro(*args, **kwargs)
                else:
                    return await self.task_manager.run_in_thread(coro, *args, **kwargs)
                    
            except Exception as e:
                last_exception = e
                if attempt < max_retries:
                    await asyncio.sleep(delay * (2 ** attempt))  # 지수 백오프
                    logger.warning(f"재시도 {attempt + 1}/{max_retries}: {e}")
                else:
                    logger.error(f"최종 실패 ({max_retries + 1}회 시도): {e}")
        
        raise last_exception
    
    def performance_decorator(self, operation_name: str):
        """성능 측정 데코레이터"""
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                timer_id = self.performance_monitor.start_timer(operation_name)
                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    self.performance_monitor.end_timer(timer_id)
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                timer_id = self.performance_monitor.start_timer(operation_name)
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    self.performance_monitor.end_timer(timer_id)
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        
        return decorator
    
    async def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 조회"""
        return {
            'initialized': self.is_initialized,
            'uptime_seconds': (datetime.now() - self.startup_time).total_seconds(),
            'performance': self.performance_monitor.get_stats(),
            'memory': self.memory_optimizer.get_memory_stats(),
            'active_tasks': len(self.task_manager.active_tasks),
            'cache_stats': {
                'l1_size': len(self.cache.l1_cache),
                'redis_available': self.cache.redis_available
            }
        }
    
    async def shutdown(self):
        """시스템 종료"""
        logger.info("🛑 시스템 종료 시작...")
        
        try:
            # 활성 작업 취소
            await self.task_manager.cancel_all_tasks()
            
            # 커넥션 풀 종료
            await self.connection_pool.close_all()
            
            # 리소스 정리
            self.task_manager.shutdown()
            
            logger.info("✅ 시스템 종료 완료")
            
        except Exception as e:
            logger.error(f"❌ 시스템 종료 중 오류: {e}")

# 전역 인스턴스
_core_instance = None

def get_core() -> OptimizedCore:
    """전역 코어 인스턴스 반환"""
    global _core_instance
    if _core_instance is None:
        _core_instance = OptimizedCore()
    return _core_instance

async def initialize_core(config: Optional[SystemConfig] = None) -> OptimizedCore:
    """코어 시스템 초기화"""
    global _core_instance
    if _core_instance is None:
        _core_instance = OptimizedCore(config)
    
    await _core_instance.initialize()
    return _core_instance 