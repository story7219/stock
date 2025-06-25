"""
Multi-level cache manager for the investment system.
"""

import asyncio
import json
import logging
import pickle
import time
from typing import Any, Dict, List, Optional, Union
from collections import OrderedDict
from dataclasses import dataclass
from functools import wraps
import hashlib
import threading
import weakref

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class CacheItem:
    """Cache item with metadata."""
    value: Any
    timestamp: float
    ttl: int
    access_count: int = 0
    last_access: float = 0.0
    
    def is_expired(self) -> bool:
        """Check if cache item is expired."""
        return time.time() - self.timestamp > self.ttl
    
    def touch(self) -> None:
        """Update access metadata."""
        self.access_count += 1
        self.last_access = time.time()


class MemoryCache:
    """Thread-safe in-memory cache with LRU eviction."""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheItem] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'expired': 0
        }
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key not in self._cache:
                self._stats['misses'] += 1
                return None
            
            item = self._cache[key]
            
            if item.is_expired():
                del self._cache[key]
                self._stats['expired'] += 1
                self._stats['misses'] += 1
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            item.touch()
            self._stats['hits'] += 1
            return item.value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache."""
        with self._lock:
            ttl = ttl or self.default_ttl
            item = CacheItem(
                value=value,
                timestamp=time.time(),
                ttl=ttl
            )
            
            if key in self._cache:
                self._cache[key] = item
                self._cache.move_to_end(key)
            else:
                self._cache[key] = item
                
                # Evict if necessary
                while len(self._cache) > self.max_size:
                    oldest_key = next(iter(self._cache))
                    del self._cache[oldest_key]
                    self._stats['evictions'] += 1
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
    
    def cleanup_expired(self) -> int:
        """Remove expired entries and return count."""
        with self._lock:
            expired_keys = [
                key for key, item in self._cache.items()
                if item.is_expired()
            ]
            
            for key in expired_keys:
                del self._cache[key]
            
            self._stats['expired'] += len(expired_keys)
            return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._stats['hits'] + self._stats['misses']
            hit_rate = self._stats['hits'] / total_requests if total_requests > 0 else 0
            
            return {
                **self._stats,
                'size': len(self._cache),
                'max_size': self.max_size,
                'hit_rate': hit_rate
            }


class RedisCache:
    """Redis-based cache for distributed caching."""
    
    def __init__(self, redis_url: str, default_ttl: int = 3600):
        self.redis_url = redis_url
        self.default_ttl = default_ttl
        self._redis: Optional[redis.Redis] = None
        self._connected = False
    
    async def connect(self) -> bool:
        """Connect to Redis."""
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available, skipping Redis cache")
            return False
        
        try:
            self._redis = redis.from_url(self.redis_url)
            await self._redis.ping()
            self._connected = True
            logger.info("Connected to Redis cache")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self._connected = False
            return False
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache."""
        if not self._connected or not self._redis:
            return None
        
        try:
            data = await self._redis.get(key)
            if data is None:
                return None
            
            return pickle.loads(data)
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in Redis cache."""
        if not self._connected or not self._redis:
            return False
        
        try:
            ttl = ttl or self.default_ttl
            data = pickle.dumps(value)
            await self._redis.setex(key, ttl, data)
            return True
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from Redis cache."""
        if not self._connected or not self._redis:
            return False
        
        try:
            result = await self._redis.delete(key)
            return result > 0
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False
    
    async def clear(self) -> bool:
        """Clear all Redis cache entries."""
        if not self._connected or not self._redis:
            return False
        
        try:
            await self._redis.flushdb()
            return True
        except Exception as e:
            logger.error(f"Redis clear error: {e}")
            return False
    
    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            self._connected = False


class CacheManager:
    """Multi-level cache manager."""
    
    def __init__(self, 
                 memory_cache_size: int = 1000,
                 default_ttl: int = 3600,
                 redis_url: Optional[str] = None,
                 enable_redis: bool = True):
        self.default_ttl = default_ttl
        self.memory_cache = MemoryCache(memory_cache_size, default_ttl)
        self.redis_cache = RedisCache(redis_url, default_ttl) if redis_url and enable_redis else None
        self._cleanup_interval = 300  # 5 minutes
        self._cleanup_task: Optional[asyncio.Task] = None
    
    async def initialize(self) -> None:
        """Initialize cache manager."""
        if self.redis_cache:
            await self.redis_cache.connect()
        
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Cache manager initialized")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache (memory first, then Redis)."""
        # Try memory cache first
        value = self.memory_cache.get(key)
        if value is not None:
            return value
        
        # Try Redis cache
        if self.redis_cache:
            value = await self.redis_cache.get(key)
            if value is not None:
                # Store in memory cache for faster access
                self.memory_cache.set(key, value)
                return value
        
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache (both memory and Redis)."""
        ttl = ttl or self.default_ttl
        
        # Set in memory cache
        self.memory_cache.set(key, value, ttl)
        
        # Set in Redis cache
        if self.redis_cache:
            await self.redis_cache.set(key, value, ttl)
    
    async def delete(self, key: str) -> None:
        """Delete key from cache (both memory and Redis)."""
        self.memory_cache.delete(key)
        if self.redis_cache:
            await self.redis_cache.delete(key)
    
    async def clear(self) -> None:
        """Clear all cache entries."""
        self.memory_cache.clear()
        if self.redis_cache:
            await self.redis_cache.clear()
    
    def generate_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_parts = [prefix]
        
        # Add positional arguments
        for arg in args:
            if isinstance(arg, (str, int, float, bool)):
                key_parts.append(str(arg))
            else:
                key_parts.append(hashlib.md5(str(arg).encode()).hexdigest()[:8])
        
        # Add keyword arguments
        for k, v in sorted(kwargs.items()):
            if isinstance(v, (str, int, float, bool)):
                key_parts.append(f"{k}={v}")
            else:
                key_parts.append(f"{k}={hashlib.md5(str(v).encode()).hexdigest()[:8]}")
        
        return ":".join(key_parts)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {
            'memory': self.memory_cache.get_stats(),
            'redis_enabled': self.redis_cache is not None,
            'redis_connected': self.redis_cache._connected if self.redis_cache else False
        }
        return stats
    
    async def _cleanup_loop(self) -> None:
        """Periodic cleanup of expired entries."""
        while True:
            try:
                await asyncio.sleep(self._cleanup_interval)
                expired_count = self.memory_cache.cleanup_expired()
                if expired_count > 0:
                    logger.debug(f"Cleaned up {expired_count} expired cache entries")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
    
    async def close(self) -> None:
        """Close cache manager."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        if self.redis_cache:
            await self.redis_cache.close()


def cached(ttl: int = 3600, key_prefix: str = "cache"):
    """Decorator for caching function results."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            cache_manager = getattr(async_wrapper, '_cache_manager', None)
            if not cache_manager:
                return await func(*args, **kwargs)
            
            # Generate cache key
            key = cache_manager.generate_key(f"{key_prefix}:{func.__name__}", *args, **kwargs)
            
            # Try to get from cache
            result = await cache_manager.get(key)
            if result is not None:
                return result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            await cache_manager.set(key, result, ttl)
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            cache_manager = getattr(sync_wrapper, '_cache_manager', None)
            if not cache_manager:
                return func(*args, **kwargs)
            
            # Generate cache key
            key = cache_manager.generate_key(f"{key_prefix}:{func.__name__}", *args, **kwargs)
            
            # Try to get from cache (sync access to memory cache only)
            result = cache_manager.memory_cache.get(key)
            if result is not None:
                return result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_manager.memory_cache.set(key, result, ttl)
            return result
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Global cache manager instance
cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """Get global cache manager instance."""
    global cache_manager
    if cache_manager is None:
        from .config import config
        cache_manager = CacheManager(
            memory_cache_size=config.cache.memory_cache_size,
            default_ttl=config.cache.cache_ttl,
            redis_url=config.get_redis_url() if config.cache.multi_level_enabled else None,
            enable_redis=config.cache.multi_level_enabled
        )
    return cache_manager 