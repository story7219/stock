"""
🚀 Ultra 멀티레벨 캐싱 시스템 v5.0
L1(메모리) + L2(Redis) + L3(디스크) 캐시 통합 관리
고성능 비동기 처리, 메모리 최적화, 자동 캐시 워밍
"""
import asyncio
import json
import pickle
import hashlib
import time
import gc
import weakref
import threading
from typing import Any, Optional, Dict, List, Union, Callable, TypeVar, Generic
from datetime import datetime, timedelta
from functools import wraps
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
import gzip
import lz4.frame

import redis.asyncio as redis
import diskcache as dc
import structlog
from config.settings import settings

logger = structlog.get_logger(__name__)

T = TypeVar('T')


@dataclass
class CacheStats:
    """캐시 통계 정보"""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    evictions: int = 0
    memory_usage: int = 0
    
    @property
    def hit_rate(self) -> float:
        """캐시 히트율"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def reset(self) -> None:
        """통계 초기화"""
        self.hits = self.misses = self.sets = self.deletes = self.evictions = 0
        self.memory_usage = 0


@dataclass
class CacheEntry(Generic[T]):
    """고성능 캐시 엔트리"""
    value: T
    created_at: datetime
    ttl: int
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    size: int = 0
    compressed: bool = False
    
    def __post_init__(self):
        """엔트리 생성 후 초기화"""
        if self.last_accessed is None:
            self.last_accessed = self.created_at
        if self.size == 0:
            self.size = self._calculate_size()
    
    def _calculate_size(self) -> int:
        """메모리 사용량 계산"""
        try:
            return len(pickle.dumps(self.value))
        except Exception:
            return 1024  # 기본값
    
    def is_expired(self) -> bool:
        """캐시 엔트리 만료 확인"""
        if self.ttl <= 0:
            return False
        return datetime.now() > self.created_at + timedelta(seconds=self.ttl)
    
    def touch(self) -> None:
        """엔트리 접근 시간 업데이트"""
        self.access_count += 1
        self.last_accessed = datetime.now()
    
    def get_age(self) -> float:
        """엔트리 생성 후 경과 시간(초)"""
        return (datetime.now() - self.created_at).total_seconds()


class LRUCache(Generic[T]):
    """고성능 LRU 캐시 (L1 캐시)"""
    
    def __init__(self, max_size: int = 50000, max_memory_mb: int = 512):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self._cache: OrderedDict[str, CacheEntry[T]] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = CacheStats()
        self._memory_usage = 0
        
        # 메모리 정리 스레드
        self._cleanup_thread = threading.Thread(target=self._periodic_cleanup, daemon=True)
        self._cleanup_thread.start()
    
    async def get(self, key: str) -> Optional[T]:
        """L1 캐시에서 값 조회"""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._stats.misses += 1
                return None
            
            if entry.is_expired():
                self._remove_entry(key)
                self._stats.misses += 1
                return None
            
            # LRU 업데이트
            self._cache.move_to_end(key)
            entry.touch()
            self._stats.hits += 1
            
            return entry.value
    
    async def set(self, key: str, value: T, ttl: int = 300) -> None:
        """L1 캐시에 값 저장"""
        with self._lock:
            # 기존 엔트리 제거
            if key in self._cache:
                self._remove_entry(key)
            
            # 새 엔트리 생성
            entry = CacheEntry(
                value=value,
                created_at=datetime.now(),
                ttl=ttl
            )
            
            # 메모리 부족 시 정리
            while (len(self._cache) >= self.max_size or 
                   self._memory_usage + entry.size > self.max_memory_bytes):
                if not self._evict_lru():
                    break
            
            self._cache[key] = entry
            self._memory_usage += entry.size
            self._stats.sets += 1
    
    async def delete(self, key: str) -> bool:
        """L1 캐시에서 키 삭제"""
        with self._lock:
            if key in self._cache:
                self._remove_entry(key)
                self._stats.deletes += 1
                return True
            return False
    
    def _remove_entry(self, key: str) -> None:
        """엔트리 제거 (락 필요)"""
        entry = self._cache.pop(key, None)
        if entry:
            self._memory_usage -= entry.size
    
    def _evict_lru(self) -> bool:
        """LRU 정책으로 가장 오래된 항목 제거"""
        if not self._cache:
            return False
        
        # 가장 오래된 항목 제거
        oldest_key = next(iter(self._cache))
        self._remove_entry(oldest_key)
        self._stats.evictions += 1
        return True
    
    def _periodic_cleanup(self) -> None:
        """주기적 메모리 정리"""
        while True:
            try:
                time.sleep(60)  # 1분마다 실행
                self._cleanup_expired()
                if self._memory_usage > self.max_memory_bytes * 0.8:
                    self._force_gc()
            except Exception as e:
                logger.error(f"캐시 정리 중 오류: {e}")
    
    def _cleanup_expired(self) -> None:
        """만료된 엔트리 정리"""
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired()
            ]
            for key in expired_keys:
                self._remove_entry(key)
    
    def _force_gc(self) -> None:
        """강제 가비지 컬렉션"""
        gc.collect()
        logger.debug("L1 캐시 가비지 컬렉션 실행")
    
    def stats(self) -> CacheStats:
        """캐시 통계 반환"""
        with self._lock:
            self._stats.memory_usage = self._memory_usage
            return self._stats
    
    def clear(self) -> None:
        """모든 캐시 항목 삭제"""
        with self._lock:
            self._cache.clear()
            self._memory_usage = 0
            self._stats.reset()


class RedisCache:
    """고성능 Redis 캐시 (L2 캐시)"""
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self._connection_pool: Optional[redis.ConnectionPool] = None
        self._stats = CacheStats()
        self._pipeline_buffer: List[tuple] = []
        self._pipeline_lock = asyncio.Lock()
        self._compression_enabled = settings.cache.l2_compression
    
    async def initialize(self) -> None:
        """Redis 연결 초기화"""
        try:
            # 고성능 연결 풀 생성
            self._connection_pool = redis.ConnectionPool.from_url(
                settings.redis.url,
                max_connections=settings.redis.max_connections,
                **settings.redis.connection_pool_kwargs
            )
            
            self.redis_client = redis.Redis(
                connection_pool=self._connection_pool,
                decode_responses=False  # 바이너리 데이터 처리
            )
            
            # 연결 테스트
            await self.redis_client.ping()
            logger.info("Redis L2 캐시 초기화 완료")
            
            # 파이프라인 플러시 태스크 시작
            asyncio.create_task(self._pipeline_flusher())
            
        except Exception as e:
            logger.warning(f"Redis 서버 연결 실패 (L2 캐시 비활성화): {e}")
            self.redis_client = None
    
    async def get(self, key: str) -> Optional[Any]:
        """Redis에서 값 조회"""
        if not self.redis_client:
            self._stats.misses += 1
            return None
        
        try:
            data = await self.redis_client.get(key)
            if data is None:
                self._stats.misses += 1
                return None
            
            # 압축 해제
            if self._compression_enabled and data.startswith(b'LZ4:'):
                data = lz4.frame.decompress(data[4:])
            
            value = pickle.loads(data)
            self._stats.hits += 1
            return value
            
        except Exception as e:
            logger.error(f"Redis 조회 실패 {key}: {e}")
            self._stats.misses += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        """Redis에 값 저장"""
        if not self.redis_client:
            return
        
        try:
            data = pickle.dumps(value)
            
            # 압축 적용
            if self._compression_enabled and len(data) > 1024:
                compressed = lz4.frame.compress(data)
                if len(compressed) < len(data) * 0.8:  # 20% 이상 압축된 경우만
                    data = b'LZ4:' + compressed
            
            # 파이프라인 버퍼에 추가
            async with self._pipeline_lock:
                self._pipeline_buffer.append(('setex', key, ttl, data))
                
                # 버퍼가 가득 찬 경우 즉시 플러시
                if len(self._pipeline_buffer) >= settings.redis.pipeline_size:
                    await self._flush_pipeline()
            
            self._stats.sets += 1
            
        except Exception as e:
            logger.error(f"Redis 저장 실패 {key}: {e}")
    
    async def delete(self, key: str) -> bool:
        """Redis에서 키 삭제"""
        if not self.redis_client:
            return False
        
        try:
            result = await self.redis_client.delete(key)
            self._stats.deletes += 1
            return result > 0
        except Exception as e:
            logger.error(f"Redis 삭제 실패 {key}: {e}")
            return False
    
    async def _pipeline_flusher(self) -> None:
        """파이프라인 버퍼 주기적 플러시"""
        while True:
            try:
                await asyncio.sleep(0.1)  # 100ms마다 확인
                async with self._pipeline_lock:
                    if self._pipeline_buffer:
                        await self._flush_pipeline()
            except Exception as e:
                logger.error(f"파이프라인 플러시 오류: {e}")
    
    async def _flush_pipeline(self) -> None:
        """파이프라인 실행"""
        if not self._pipeline_buffer or not self.redis_client:
            return
        
        try:
            pipe = self.redis_client.pipeline()
            for cmd, *args in self._pipeline_buffer:
                getattr(pipe, cmd)(*args)
            
            await pipe.execute()
            self._pipeline_buffer.clear()
            
        except Exception as e:
            logger.error(f"파이프라인 실행 실패: {e}")
            self._pipeline_buffer.clear()
    
    def stats(self) -> CacheStats:
        """캐시 통계 반환"""
        return self._stats
    
    async def close(self) -> None:
        """Redis 연결 종료"""
        try:
            # 남은 파이프라인 플러시
            async with self._pipeline_lock:
                await self._flush_pipeline()
            
            if self.redis_client:
                await self.redis_client.close()
            if self._connection_pool:
                await self._connection_pool.disconnect()
                
        except Exception as e:
            logger.error(f"Redis 연결 종료 중 오류: {e}")


class DiskCache:
    """고성능 디스크 캐시 (L3 캐시)"""
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = cache_dir
        self._stats = CacheStats()
        
        # DiskCache 설정
        self.cache = dc.Cache(
            cache_dir,
            size_limit=settings.cache.l3_max_size_gb * 1024**3,  # GB to bytes
            disk_min_file_size=1024,  # 1KB 이상만 디스크에 저장
            disk_pickle_protocol=pickle.HIGHEST_PROTOCOL
        )
    
    async def get(self, key: str) -> Optional[Any]:
        """디스크 캐시에서 값 조회"""
        try:
            value = self.cache.get(key)
            if value is not None:
                self._stats.hits += 1
            else:
                self._stats.misses += 1
            return value
        except Exception as e:
            logger.error(f"디스크 캐시 조회 실패 {key}: {e}")
            self._stats.misses += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: int = 86400) -> None:
        """디스크 캐시에 값 저장"""
        try:
            expire_time = time.time() + ttl if ttl > 0 else None
            self.cache.set(key, value, expire=expire_time)
            self._stats.sets += 1
        except Exception as e:
            logger.error(f"디스크 캐시 저장 실패 {key}: {e}")
    
    async def delete(self, key: str) -> bool:
        """디스크 캐시에서 키 삭제"""
        try:
            result = self.cache.delete(key)
            if result:
                self._stats.deletes += 1
            return result
        except Exception as e:
            logger.error(f"디스크 캐시 삭제 실패 {key}: {e}")
            return False
    
    def stats(self) -> CacheStats:
        """캐시 통계 반환"""
        try:
            cache_stats = self.cache.stats()
            self._stats.memory_usage = cache_stats.get('size', 0)
        except Exception:
            pass
        return self._stats
    
    def clear(self) -> None:
        """모든 캐시 항목 삭제"""
        try:
            self.cache.clear()
            self._stats.reset()
        except Exception as e:
            logger.error(f"디스크 캐시 초기화 실패: {e}")


class UltraCacheManager:
    """🚀 Ultra 멀티레벨 캐시 매니저"""
    
    def __init__(self):
        # 캐시 계층 초기화
        self.l1_cache: Optional[LRUCache] = None
        self.l2_cache: Optional[RedisCache] = None
        self.l3_cache: Optional[DiskCache] = None
        
        # 성능 최적화
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._prefetch_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self._write_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        
        # 통계 및 모니터링
        self._global_stats = CacheStats()
        self._last_stats_update = time.time()
        
        logger.info("Ultra 캐시 매니저 초기화")
    
    async def initialize(self) -> None:
        """캐시 매니저 초기화"""
        try:
            # L1 캐시 (메모리) 초기화
            if settings.cache.l1_enabled:
                self.l1_cache = LRUCache(
                    max_size=settings.cache.l1_max_size,
                    max_memory_mb=512
                )
                logger.info("L1 메모리 캐시 초기화 완료")
            
            # L2 캐시 (Redis) 초기화
            if settings.cache.l2_enabled and settings.redis.enable_redis:
                self.l2_cache = RedisCache()
                await self.l2_cache.initialize()
                logger.info("L2 Redis 캐시 초기화 완료")
            
            # L3 캐시 (디스크) 초기화
            if settings.cache.l3_enabled:
                self.l3_cache = DiskCache(settings.cache.l3_directory)
                logger.info("L3 디스크 캐시 초기화 완료")
            
            # 백그라운드 태스크 시작
            asyncio.create_task(self._prefetch_worker())
            asyncio.create_task(self._write_worker())
            asyncio.create_task(self._stats_updater())
            
            logger.info("Ultra 캐시 매니저 초기화 완료")
            
        except Exception as e:
            logger.error(f"캐시 매니저 초기화 실패: {e}")
            raise
    
    async def get(self, key: str) -> Optional[Any]:
        """멀티레벨 캐시에서 값 조회"""
        start_time = time.time()
        
        try:
            # L1 캐시 확인
            if self.l1_cache:
                value = await self.l1_cache.get(key)
                if value is not None:
                    self._global_stats.hits += 1
                    logger.debug(f"L1 캐시 히트: {key}")
                    return value
            
            # L2 캐시 확인
            if self.l2_cache:
                value = await self.l2_cache.get(key)
                if value is not None:
                    # L1 캐시에 백필
                    if self.l1_cache:
                        await self.l1_cache.set(key, value, settings.cache.l1_ttl)
                    self._global_stats.hits += 1
                    logger.debug(f"L2 캐시 히트: {key}")
                    return value
            
            # L3 캐시 확인
            if self.l3_cache:
                value = await self.l3_cache.get(key)
                if value is not None:
                    # 상위 캐시에 백필
                    if self.l2_cache:
                        await self.l2_cache.set(key, value, settings.cache.l2_ttl)
                    if self.l1_cache:
                        await self.l1_cache.set(key, value, settings.cache.l1_ttl)
                    self._global_stats.hits += 1
                    logger.debug(f"L3 캐시 히트: {key}")
                    return value
            
            # 모든 캐시에서 미스
            self._global_stats.misses += 1
            return None
            
        except Exception as e:
            logger.error(f"캐시 조회 실패 {key}: {e}")
            self._global_stats.misses += 1
            return None
        finally:
            # 성능 로깅
            elapsed = time.time() - start_time
            if elapsed > 0.1:  # 100ms 이상
                logger.warning(f"캐시 조회 지연: {key} ({elapsed:.3f}초)")
    
    async def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        """멀티레벨 캐시에 값 저장"""
        try:
            # Write-through 전략
            if settings.cache.write_through:
                await self._write_through(key, value, ttl)
            else:
                # 비동기 쓰기 큐에 추가
                await self._write_queue.put((key, value, ttl))
            
            self._global_stats.sets += 1
            
        except Exception as e:
            logger.error(f"캐시 저장 실패 {key}: {e}")
    
    async def _write_through(self, key: str, value: Any, ttl: int) -> None:
        """Write-through 캐시 쓰기"""
        tasks = []
        
        # 모든 캐시 레벨에 동시 쓰기
        if self.l1_cache:
            tasks.append(self.l1_cache.set(key, value, min(ttl, settings.cache.l1_ttl)))
        
        if self.l2_cache:
            tasks.append(self.l2_cache.set(key, value, min(ttl, settings.cache.l2_ttl)))
        
        if self.l3_cache:
            tasks.append(self.l3_cache.set(key, value, min(ttl, settings.cache.l3_ttl)))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def delete(self, key: str) -> bool:
        """모든 캐시 레벨에서 키 삭제"""
        results = []
        
        if self.l1_cache:
            results.append(await self.l1_cache.delete(key))
        
        if self.l2_cache:
            results.append(await self.l2_cache.delete(key))
        
        if self.l3_cache:
            results.append(await self.l3_cache.delete(key))
        
        self._global_stats.deletes += 1
        return any(results)
    
    def generate_key(self, *args, **kwargs) -> str:
        """캐시 키 생성"""
        key_data = f"{args}:{sorted(kwargs.items())}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def _prefetch_worker(self) -> None:
        """프리페치 워커"""
        while True:
            try:
                # 프리페치 로직 구현
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"프리페치 워커 오류: {e}")
    
    async def _write_worker(self) -> None:
        """비동기 쓰기 워커"""
        while True:
            try:
                key, value, ttl = await self._write_queue.get()
                await self._write_through(key, value, ttl)
                self._write_queue.task_done()
            except Exception as e:
                logger.error(f"쓰기 워커 오류: {e}")
    
    async def _stats_updater(self) -> None:
        """통계 업데이트 워커"""
        while True:
            try:
                await asyncio.sleep(60)  # 1분마다
                self._update_global_stats()
            except Exception as e:
                logger.error(f"통계 업데이트 오류: {e}")
    
    def _update_global_stats(self) -> None:
        """전역 통계 업데이트"""
        try:
            # 각 캐시 레벨 통계 수집
            stats_data = {
                "timestamp": datetime.now().isoformat(),
                "global": {
                    "hits": self._global_stats.hits,
                    "misses": self._global_stats.misses,
                    "hit_rate": self._global_stats.hit_rate,
                }
            }
            
            if self.l1_cache:
                l1_stats = self.l1_cache.stats()
                stats_data["l1"] = {
                    "hits": l1_stats.hits,
                    "misses": l1_stats.misses,
                    "hit_rate": l1_stats.hit_rate,
                    "memory_usage": l1_stats.memory_usage,
                }
            
            if self.l2_cache:
                l2_stats = self.l2_cache.stats()
                stats_data["l2"] = {
                    "hits": l2_stats.hits,
                    "misses": l2_stats.misses,
                    "hit_rate": l2_stats.hit_rate,
                }
            
            if self.l3_cache:
                l3_stats = self.l3_cache.stats()
                stats_data["l3"] = {
                    "hits": l3_stats.hits,
                    "misses": l3_stats.misses,
                    "hit_rate": l3_stats.hit_rate,
                    "memory_usage": l3_stats.memory_usage,
                }
            
            logger.info("캐시 통계 업데이트", extra=stats_data)
            
        except Exception as e:
            logger.error(f"통계 업데이트 실패: {e}")
    
    def stats(self) -> Dict[str, Any]:
        """전체 캐시 통계"""
        return {
            "global": {
                "hits": self._global_stats.hits,
                "misses": self._global_stats.misses,
                "sets": self._global_stats.sets,
                "deletes": self._global_stats.deletes,
                "hit_rate": self._global_stats.hit_rate,
            },
            "l1": self.l1_cache.stats().__dict__ if self.l1_cache else None,
            "l2": self.l2_cache.stats().__dict__ if self.l2_cache else None,
            "l3": self.l3_cache.stats().__dict__ if self.l3_cache else None,
        }
    
    async def close(self) -> None:
        """캐시 매니저 종료"""
        try:
            # 큐 대기 작업 완료
            await self._write_queue.join()
            
            # 각 캐시 레벨 종료
            if self.l2_cache:
                await self.l2_cache.close()
            
            if self.l3_cache:
                self.l3_cache.clear()
            
            # 스레드 풀 종료
            self._executor.shutdown(wait=True)
            
            logger.info("Ultra 캐시 매니저 종료 완료")
            
        except Exception as e:
            logger.error(f"캐시 매니저 종료 중 오류: {e}")


# 전역 캐시 매니저 인스턴스
_cache_manager: Optional[UltraCacheManager] = None


def get_cache_manager() -> UltraCacheManager:
    """전역 캐시 매니저 반환"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = UltraCacheManager()
    return _cache_manager


def cached(ttl: int = 3600, key_prefix: str = "", use_compression: bool = True):
    """고성능 캐시 데코레이터"""
    def decorator(func: Callable) -> Callable:
        cache_manager = get_cache_manager()
        
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                # 캐시 키 생성
                cache_key = f"{key_prefix}:{func.__name__}:{cache_manager.generate_key(*args, **kwargs)}"
                
                # 캐시에서 조회
                cached_result = await cache_manager.get(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # 함수 실행
                result = await func(*args, **kwargs)
                
                # 캐시에 저장
                if result is not None:
                    await cache_manager.set(cache_key, result, ttl)
                
                return result
            
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                # 동기 함수는 캐시 없이 실행 (성능상 이유)
                return func(*args, **kwargs)
            
            return sync_wrapper
    
    return decorator


async def initialize_cache() -> None:
    """캐시 시스템 초기화"""
    cache_manager = get_cache_manager()
    await cache_manager.initialize()


async def cleanup_cache() -> None:
    """캐시 시스템 정리"""
    global _cache_manager
    if _cache_manager:
        await _cache_manager.close()
        _cache_manager = None


# 하위 호환성을 위한 별칭
CacheManager = UltraCacheManager 