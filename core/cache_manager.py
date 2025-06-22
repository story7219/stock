"""
멀티레벨 캐싱 시스템 - 메모리, Redis, 디스크 캐시 통합 관리
"""
import asyncio
import json
import pickle
import hashlib
import time
from typing import Any, Optional, Dict, List, Union, Callable
from datetime import datetime, timedelta
from functools import wraps
from dataclasses import dataclass
import weakref
import threading
from concurrent.futures import ThreadPoolExecutor

import redis.asyncio as redis
import diskcache as dc
import structlog
from config.settings import settings

logger = structlog.get_logger(__name__)


@dataclass
class CacheEntry:
    """캐시 엔트리 데이터 클래스"""
    value: Any
    created_at: datetime
    ttl: int
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    
    def is_expired(self) -> bool:
        """캐시 엔트리가 만료되었는지 확인"""
        if self.ttl <= 0:
            return False
        return datetime.now() > self.created_at + timedelta(seconds=self.ttl)
    
    def touch(self) -> None:
        """캐시 엔트리 접근 시간 업데이트"""
        self.access_count += 1
        self.last_accessed = datetime.now()


class MemoryCache:
    """메모리 캐시 (L1 캐시) - 가장 빠른 접근"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._weak_refs: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
    
    async def get(self, key: str) -> Optional[Any]:
        """메모리 캐시에서 값 조회"""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            
            if entry.is_expired():
                del self._cache[key]
                return None
            
            entry.touch()
            return entry.value
    
    async def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        """메모리 캐시에 값 저장"""
        with self._lock:
            # LRU 정책으로 캐시 크기 관리
            if len(self._cache) >= self.max_size:
                self._evict_lru()
            
            self._cache[key] = CacheEntry(
                value=value,
                created_at=datetime.now(),
                ttl=ttl
            )
    
    async def delete(self, key: str) -> bool:
        """메모리 캐시에서 키 삭제"""
        with self._lock:
            return self._cache.pop(key, None) is not None
    
    def _evict_lru(self) -> None:
        """LRU 정책으로 가장 오래된 항목 제거"""
        if not self._cache:
            return
        
        oldest_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k].last_accessed or self._cache[k].created_at
        )
        del self._cache[oldest_key]
    
    def clear(self) -> None:
        """모든 캐시 항목 삭제"""
        with self._lock:
            self._cache.clear()
    
    def stats(self) -> Dict[str, Any]:
        """캐시 통계 정보"""
        with self._lock:
            total_access = sum(entry.access_count for entry in self._cache.values())
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "total_access": total_access,
                "hit_rate": total_access / max(len(self._cache), 1)
            }


class RedisCache:
    """Redis 캐시 (L2 캐시) - 분산 캐시"""
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self._connection_pool: Optional[redis.ConnectionPool] = None
    
    async def initialize(self) -> None:
        """Redis 연결 초기화"""
        try:
            self._connection_pool = redis.ConnectionPool.from_url(
                settings.redis.url,
                max_connections=settings.redis.max_connections,
                retry_on_timeout=True,
                socket_keepalive=True,
                socket_keepalive_options={}
            )
            self.redis_client = redis.Redis(connection_pool=self._connection_pool)
            await self.redis_client.ping()
            logger.info("Redis 캐시 초기화 완료")
        except Exception as e:
            logger.warning(f"Redis 서버에 연결할 수 없습니다 (메모리 및 디스크 캐시로 동작): {e}")
            logger.info("Redis 없이도 시스템이 정상 동작합니다. 성능 향상을 위해 Redis 설치를 권장합니다.")
            self.redis_client = None
    
    async def get(self, key: str) -> Optional[Any]:
        """Redis에서 값 조회"""
        if not self.redis_client:
            return None
        
        try:
            data = await self.redis_client.get(key)
            if data is None:
                return None
            
            return pickle.loads(data)
        except Exception as e:
            logger.error(f"Redis 조회 실패 {key}: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        """Redis에 값 저장"""
        if not self.redis_client:
            return
        
        try:
            data = pickle.dumps(value)
            await self.redis_client.setex(key, ttl, data)
        except Exception as e:
            logger.error(f"Redis 저장 실패 {key}: {e}")
    
    async def delete(self, key: str) -> bool:
        """Redis에서 키 삭제"""
        if not self.redis_client:
            return False
        
        try:
            result = await self.redis_client.delete(key)
            return result > 0
        except Exception as e:
            logger.error(f"Redis 삭제 실패 {key}: {e}")
            return False
    
    async def close(self) -> None:
        """Redis 연결 종료"""
        if self.redis_client:
            await self.redis_client.close()
        if self._connection_pool:
            await self._connection_pool.disconnect()


class DiskCache:
    """디스크 캐시 (L3 캐시) - 영구 저장"""
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache = dc.Cache(cache_dir, size_limit=1024**3)  # 1GB 제한
    
    async def get(self, key: str) -> Optional[Any]:
        """디스크 캐시에서 값 조회"""
        try:
            return self.cache.get(key)
        except Exception as e:
            logger.error(f"디스크 캐시 조회 실패 {key}: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        """디스크 캐시에 값 저장"""
        try:
            self.cache.set(key, value, expire=ttl)
        except Exception as e:
            logger.error(f"디스크 캐시 저장 실패 {key}: {e}")
    
    async def delete(self, key: str) -> bool:
        """디스크 캐시에서 키 삭제"""
        try:
            return self.cache.delete(key)
        except Exception as e:
            logger.error(f"디스크 캐시 삭제 실패 {key}: {e}")
            return False
    
    def clear(self) -> None:
        """모든 디스크 캐시 삭제"""
        self.cache.clear()
    
    def stats(self) -> Dict[str, Any]:
        """디스크 캐시 통계"""
        return {
            "size": len(self.cache),
            "volume": self.cache.volume(),
            "stats": self.cache.stats()
        }


class CacheManager:
    """멀티레벨 캐시 매니저 - L1(메모리), L2(Redis), L3(디스크) 캐시 통합 관리"""
    
    def __init__(self):
        self.memory_cache = MemoryCache(max_size=settings.cache.memory_max_size)
        
        # Redis 활성화 여부에 따라 Redis 캐시 초기화
        if settings.redis.enable_redis:
            self.redis_cache = RedisCache()
        else:
            self.redis_cache = None
            logger.info("Redis가 비활성화되어 메모리+디스크 캐시만 사용합니다")
            
        self.disk_cache = DiskCache(settings.cache.disk_cache_dir)
        self._stats = {
            "hits": {"L1": 0, "L2": 0, "L3": 0},
            "misses": 0,
            "sets": 0
        }
    
    async def initialize(self) -> None:
        """캐시 시스템 초기화"""
        if self.redis_cache:
            await self.redis_cache.initialize()
            # Redis 실제 연결 상태 확인
            redis_status = "연결됨" if (self.redis_cache and hasattr(self.redis_cache, 'redis_client') and self.redis_cache.redis_client) else "연결 실패 (메모리+디스크 캐시 사용)"
        else:
            redis_status = "비활성화 (메모리+디스크 캐시 사용)"
            
        logger.info(f"멀티레벨 캐시 시스템 초기화 완료 - Redis: {redis_status}")
    
    async def get(self, key: str) -> Optional[Any]:
        """멀티레벨 캐시에서 값 조회 (L1 -> L2 -> L3 순서)"""
        # L1 캐시 확인
        value = await self.memory_cache.get(key)
        if value is not None:
            self._stats["hits"]["L1"] += 1
            return value
        
        # L2 캐시 확인
        value = await self.redis_cache.get(key) if self.redis_cache else None
        if value is not None:
            self._stats["hits"]["L2"] += 1
            # L1 캐시에 역순 저장
            await self.memory_cache.set(key, value)
            return value
        
        # L3 캐시 확인
        value = await self.disk_cache.get(key)
        if value is not None:
            self._stats["hits"]["L3"] += 1
            # L2, L1 캐시에 역순 저장
            await self.redis_cache.set(key, value) if self.redis_cache else None
            await self.memory_cache.set(key, value)
            return value
        
        # 캐시 미스
        self._stats["misses"] += 1
        return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        """모든 레벨의 캐시에 값 저장"""
        self._stats["sets"] += 1
        
        # None이 아닌 태스크만 수집
        tasks = [
            self.memory_cache.set(key, value, ttl),
            self.disk_cache.set(key, value, ttl)
        ]
        
        if self.redis_cache:
            tasks.append(self.redis_cache.set(key, value, ttl))
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def delete(self, key: str) -> bool:
        """모든 레벨의 캐시에서 키 삭제"""
        # None이 아닌 태스크만 수집
        tasks = [
            self.memory_cache.delete(key),
            self.disk_cache.delete(key)
        ]
        
        if self.redis_cache:
            tasks.append(self.redis_cache.delete(key))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return any(result for result in results if not isinstance(result, Exception))
    
    def generate_key(self, *args, **kwargs) -> str:
        """캐시 키 생성"""
        key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, default=str)
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def stats(self) -> Dict[str, Any]:
        """캐시 통계 정보"""
        total_hits = sum(self._stats["hits"].values())
        total_requests = total_hits + self._stats["misses"]
        hit_rate = total_hits / total_requests if total_requests > 0 else 0
        
        return {
            **self._stats,
            "total_requests": total_requests,
            "hit_rate": hit_rate,
            "memory_stats": self.memory_cache.stats(),
            "disk_stats": self.disk_cache.stats()
        }
    
    async def close(self) -> None:
        """캐시 시스템 종료"""
        if self.redis_cache:
            await self.redis_cache.close()
        self.memory_cache.clear()
        self.disk_cache.clear()


# 전역 캐시 인스턴스
cache_manager = CacheManager()


def cached(ttl: int = 3600, key_prefix: str = ""):
    """캐시 데코레이터"""
    def decorator(func: Callable) -> Callable:
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                # 캐시 키 생성
                cache_key = f"{key_prefix}:{func.__name__}:" + cache_manager.generate_key(*args, **kwargs)
                
                # 캐시에서 조회
                cached_result = await cache_manager.get(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # 함수 실행
                result = await func(*args, **kwargs)
                
                # 캐시에 저장
                await cache_manager.set(cache_key, result, ttl)
                return result
            
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                # 동기 함수는 캐시 없이 실행
                return func(*args, **kwargs)
            
            return sync_wrapper
    return decorator


async def initialize_cache() -> None:
    """캐시 시스템 초기화"""
    await cache_manager.initialize()


async def cleanup_cache() -> None:
    """캐시 시스템 정리"""
    await cache_manager.close() 