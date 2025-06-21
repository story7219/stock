#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 고성능 비동기 병렬처리 핵심 모듈
- 멀티레벨 캐싱 시스템
- 커넥션 풀링
- 메모리 최적화
- 안정성 우선 설계
"""

import asyncio
import aiohttp
import logging
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import threading
import weakref
import gc
from functools import wraps
import json
import hashlib

# 로깅 설정
logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """캐시 엔트리 데이터 클래스"""
    data: Any
    timestamp: datetime
    ttl: int = 300  # 5분 기본 TTL
    access_count: int = 0
    
    def is_valid(self) -> bool:
        """캐시 유효성 검사"""
        return (datetime.now() - self.timestamp).seconds < self.ttl
    
    def touch(self):
        """캐시 접근 시 호출"""
        self.access_count += 1

@dataclass
class PerformanceMetrics:
    """성능 메트릭 데이터 클래스"""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_time: float = 0.0
    avg_response_time: float = 0.0
    error_count: int = 0
    
    @property
    def cache_hit_rate(self) -> float:
        """캐시 히트율 계산"""
        total = self.cache_hits + self.cache_misses
        return (self.cache_hits / total * 100) if total > 0 else 0.0

class MultiLevelCache:
    """🔥 멀티레벨 캐싱 시스템"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._metrics = PerformanceMetrics()
        
        # 메모리 관리를 위한 약한 참조
        self._weak_refs = weakref.WeakValueDictionary()
        
        logger.info(f"✅ 멀티레벨 캐시 초기화 (최대 크기: {max_size})")
    
    def _generate_key(self, *args, **kwargs) -> str:
        """캐시 키 생성"""
        key_data = f"{args}_{sorted(kwargs.items())}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """캐시에서 데이터 조회"""
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                if entry.is_valid():
                    entry.touch()
                    self._metrics.cache_hits += 1
                    logger.debug(f"🎯 캐시 히트: {key[:8]}...")
                    return entry.data
                else:
                    # 만료된 캐시 제거
                    del self._cache[key]
            
            self._metrics.cache_misses += 1
            logger.debug(f"❌ 캐시 미스: {key[:8]}...")
            return None
    
    def set(self, key: str, data: Any, ttl: int = 300):
        """캐시에 데이터 저장"""
        with self._lock:
            # 캐시 크기 제한 확인
            if len(self._cache) >= self.max_size:
                self._evict_lru()
            
            self._cache[key] = CacheEntry(
                data=data,
                timestamp=datetime.now(),
                ttl=ttl
            )
            logger.debug(f"💾 캐시 저장: {key[:8]}... (TTL: {ttl}s)")
    
    def _evict_lru(self):
        """LRU 방식으로 캐시 제거"""
        if not self._cache:
            return
        
        # 가장 오래된 항목 찾기
        oldest_key = min(
            self._cache.keys(),
            key=lambda k: (self._cache[k].timestamp, self._cache[k].access_count)
        )
        
        del self._cache[oldest_key]
        logger.debug(f"🗑️ LRU 캐시 제거: {oldest_key[:8]}...")
    
    def clear_expired(self):
        """만료된 캐시 정리"""
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if not entry.is_valid()
            ]
            
            for key in expired_keys:
                del self._cache[key]
            
            if expired_keys:
                logger.info(f"🧹 만료된 캐시 {len(expired_keys)}개 정리")
    
    def get_stats(self) -> Dict[str, Any]:
        """캐시 통계 반환"""
        with self._lock:
            return {
                "cache_size": len(self._cache),
                "max_size": self.max_size,
                "hit_rate": self._metrics.cache_hit_rate,
                "total_hits": self._metrics.cache_hits,
                "total_misses": self._metrics.cache_misses
            }

class ConnectionPool:
    """🔗 고성능 커넥션 풀"""
    
    def __init__(self, max_connections: int = 100, timeout: int = 30):
        self.max_connections = max_connections
        self.timeout = timeout
        self._session: Optional[aiohttp.ClientSession] = None
        self._connector: Optional[aiohttp.TCPConnector] = None
        self._lock = asyncio.Lock()
        
        logger.info(f"✅ 커넥션 풀 초기화 (최대 연결: {max_connections})")
    
    async def get_session(self) -> aiohttp.ClientSession:
        """세션 반환 (지연 초기화)"""
        if self._session is None or self._session.closed:
            async with self._lock:
                if self._session is None or self._session.closed:
                    await self._create_session()
        
        return self._session
    
    async def _create_session(self):
        """새로운 세션 생성"""
        # 기존 세션 정리
        if self._session and not self._session.closed:
            await self._session.close()
        
        # TCP 커넥터 설정
        self._connector = aiohttp.TCPConnector(
            limit=self.max_connections,
            limit_per_host=20,
            ttl_dns_cache=300,
            use_dns_cache=True,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        
        # 세션 생성
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        self._session = aiohttp.ClientSession(
            connector=self._connector,
            timeout=timeout,
            headers={
                'User-Agent': 'US-Stock-Analyzer/1.0',
                'Accept': 'application/json',
                'Connection': 'keep-alive'
            }
        )
        
        logger.info("🔗 새로운 HTTP 세션 생성 완료")
    
    async def close(self):
        """커넥션 풀 정리"""
        if self._session and not self._session.closed:
            await self._session.close()
            logger.info("🔗 커넥션 풀 정리 완료")

class AsyncTaskManager:
    """⚡ 비동기 작업 관리자"""
    
    def __init__(self, max_concurrent: int = 50, batch_size: int = 10):
        self.max_concurrent = max_concurrent
        self.batch_size = batch_size
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._active_tasks = set()
        
        logger.info(f"✅ 비동기 작업 관리자 초기화 (동시 작업: {max_concurrent}, 배치 크기: {batch_size})")
    
    async def run_batch(self, tasks: List[Callable], *args, **kwargs) -> List[Any]:
        """배치 단위로 작업 실행"""
        results = []
        
        # 배치로 나누어 처리
        for i in range(0, len(tasks), self.batch_size):
            batch = tasks[i:i + self.batch_size]
            batch_results = await self._run_concurrent_batch(batch, *args, **kwargs)
            results.extend(batch_results)
            
            # 배치 간 짧은 대기 (API 제한 회피)
            if i + self.batch_size < len(tasks):
                await asyncio.sleep(0.1)
        
        return results
    
    async def _run_concurrent_batch(self, batch: List[Callable], *args, **kwargs) -> List[Any]:
        """동시 실행 배치"""
        async def run_with_semaphore(task):
            async with self._semaphore:
                try:
                    if asyncio.iscoroutinefunction(task):
                        return await task(*args, **kwargs)
                    else:
                        return task(*args, **kwargs)
                except Exception as e:
                    logger.error(f"❌ 작업 실행 실패: {e}")
                    return None
        
        # 모든 작업을 동시에 시작
        batch_tasks = [run_with_semaphore(task) for task in batch]
        results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        
        # 예외 처리
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"❌ 배치 실행 중 예외: {result}")
                processed_results.append(None)
            else:
                processed_results.append(result)
        
        return processed_results

class MemoryOptimizer:
    """🧠 메모리 최적화 관리자"""
    
    def __init__(self, gc_threshold: int = 1000):
        self.gc_threshold = gc_threshold
        self._allocation_count = 0
        
        logger.info(f"✅ 메모리 최적화 관리자 초기화 (GC 임계값: {gc_threshold})")
    
    def track_allocation(self):
        """메모리 할당 추적"""
        self._allocation_count += 1
        
        if self._allocation_count >= self.gc_threshold:
            self.force_cleanup()
            self._allocation_count = 0
    
    def force_cleanup(self):
        """강제 메모리 정리"""
        collected = gc.collect()
        if collected > 0:
            logger.debug(f"🧹 가비지 컬렉션: {collected}개 객체 정리")
    
    @staticmethod
    def optimize_dict(data: Dict) -> Dict:
        """딕셔너리 메모리 최적화"""
        if not data:
            return {}
        
        # None 값 제거
        optimized = {k: v for k, v in data.items() if v is not None}
        
        # 문자열 최적화
        for key, value in optimized.items():
            if isinstance(value, str) and len(value) > 1000:
                optimized[key] = value[:1000] + "..."
        
        return optimized

def performance_monitor(func):
    """성능 모니터링 데코레이터"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(f"⏱️ {func.__name__} 실행 시간: {execution_time:.2f}초")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"❌ {func.__name__} 실행 실패 ({execution_time:.2f}초): {e}")
            raise
    
    return wrapper

class HighPerformanceCore:
    """🚀 고성능 핵심 시스템"""
    
    def __init__(self):
        self.cache = MultiLevelCache(max_size=20000)
        self.connection_pool = ConnectionPool(max_connections=100)
        self.task_manager = AsyncTaskManager(max_concurrent=50, batch_size=15)
        self.memory_optimizer = MemoryOptimizer(gc_threshold=1000)
        
        # 성능 통계
        self.start_time = datetime.now()
        self.total_requests = 0
        self.successful_requests = 0
        
        logger.info("🚀 고성능 핵심 시스템 초기화 완료")
    
    async def initialize(self):
        """시스템 초기화"""
        try:
            # 커넥션 풀 준비
            await self.connection_pool.get_session()
            
            # 캐시 정리 스케줄러 시작
            asyncio.create_task(self._cache_cleanup_scheduler())
            
            logger.info("✅ 고성능 시스템 초기화 완료")
            
        except Exception as e:
            logger.error(f"❌ 시스템 초기화 실패: {e}")
            raise
    
    async def _cache_cleanup_scheduler(self):
        """캐시 정리 스케줄러"""
        while True:
            try:
                await asyncio.sleep(300)  # 5분마다
                self.cache.clear_expired()
                self.memory_optimizer.force_cleanup()
                
                # 통계 출력
                stats = self.get_performance_stats()
                logger.info(f"📊 성능 통계: {stats}")
                
            except Exception as e:
                logger.error(f"❌ 캐시 정리 스케줄러 오류: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계 반환"""
        uptime = (datetime.now() - self.start_time).total_seconds()
        cache_stats = self.cache.get_stats()
        
        return {
            "uptime_seconds": uptime,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "success_rate": (self.successful_requests / self.total_requests * 100) if self.total_requests > 0 else 0,
            "cache_stats": cache_stats,
            "requests_per_second": self.total_requests / uptime if uptime > 0 else 0
        }
    
    async def cleanup(self):
        """시스템 정리"""
        try:
            await self.connection_pool.close()
            self.memory_optimizer.force_cleanup()
            logger.info("✅ 고성능 시스템 정리 완료")
        except Exception as e:
            logger.error(f"❌ 시스템 정리 중 오류: {e}")

# 전역 인스턴스 (싱글톤 패턴)
_performance_core = None

async def get_performance_core() -> HighPerformanceCore:
    """고성능 핵심 시스템 인스턴스 반환"""
    global _performance_core
    
    if _performance_core is None:
        _performance_core = HighPerformanceCore()
        await _performance_core.initialize()
    
    return _performance_core

if __name__ == "__main__":
    async def test_performance_core():
        """성능 핵심 시스템 테스트"""
        print("🧪 고성능 핵심 시스템 테스트 시작...")
        
        core = await get_performance_core()
        
        # 캐시 테스트
        core.cache.set("test_key", {"data": "test_value"}, ttl=60)
        cached_data = core.cache.get("test_key")
        print(f"📋 캐시 테스트: {cached_data}")
        
        # 성능 통계
        stats = core.get_performance_stats()
        print(f"📊 성능 통계: {stats}")
        
        await core.cleanup()
        print("✅ 테스트 완료!")
    
    asyncio.run(test_performance_core()) 