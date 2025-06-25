"""
Memory optimizer for efficient memory usage and garbage collection.
"""

import gc
import logging
import psutil
import threading
import time
import weakref
from typing import Any, Dict, List, Optional, Set, Callable
from dataclasses import dataclass
from collections import defaultdict
import sys
import tracemalloc
from functools import wraps

logger = logging.getLogger(__name__)


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    total_memory: int = 0
    available_memory: int = 0
    used_memory: int = 0
    memory_percent: float = 0.0
    gc_collections: Dict[int, int] = None
    object_counts: Dict[str, int] = None
    peak_memory: int = 0
    last_gc_time: float = 0.0
    
    def __post_init__(self):
        if self.gc_collections is None:
            self.gc_collections = {}
        if self.object_counts is None:
            self.object_counts = {}


class ObjectTracker:
    """Track object creation and destruction."""
    
    def __init__(self):
        self._tracked_objects: Dict[int, weakref.ReferenceType] = {}
        self._object_types: Dict[str, Set[int]] = defaultdict(set)
        self._creation_times: Dict[int, float] = {}
        self._lock = threading.Lock()
    
    def track_object(self, obj: Any, obj_type: Optional[str] = None) -> int:
        """Track an object."""
        obj_id = id(obj)
        obj_type = obj_type or type(obj).__name__
        
        with self._lock:
            # Create weak reference with cleanup callback
            def cleanup_callback(ref):
                self._cleanup_object(obj_id, obj_type)
            
            weak_ref = weakref.ref(obj, cleanup_callback)
            self._tracked_objects[obj_id] = weak_ref
            self._object_types[obj_type].add(obj_id)
            self._creation_times[obj_id] = time.time()
        
        return obj_id
    
    def _cleanup_object(self, obj_id: int, obj_type: str) -> None:
        """Clean up tracked object."""
        with self._lock:
            self._tracked_objects.pop(obj_id, None)
            self._object_types[obj_type].discard(obj_id)
            self._creation_times.pop(obj_id, None)
    
    def get_object_count(self, obj_type: Optional[str] = None) -> int:
        """Get count of tracked objects."""
        with self._lock:
            if obj_type:
                return len(self._object_types[obj_type])
            return len(self._tracked_objects)
    
    def get_object_types(self) -> Dict[str, int]:
        """Get count of objects by type."""
        with self._lock:
            return {obj_type: len(obj_ids) for obj_type, obj_ids in self._object_types.items()}
    
    def get_old_objects(self, max_age: float = 3600) -> List[int]:
        """Get objects older than max_age seconds."""
        current_time = time.time()
        old_objects = []
        
        with self._lock:
            for obj_id, creation_time in self._creation_times.items():
                if (current_time - creation_time) > max_age:
                    old_objects.append(obj_id)
        
        return old_objects
    
    def cleanup_old_objects(self, max_age: float = 3600) -> int:
        """Clean up objects older than max_age seconds."""
        old_objects = self.get_old_objects(max_age)
        
        # Objects will be cleaned up automatically by weak references
        # when they're actually garbage collected
        return len(old_objects)


class MemoryPool:
    """Memory pool for object reuse."""
    
    def __init__(self, max_pool_size: int = 1000):
        self.max_pool_size = max_pool_size
        self._pools: Dict[str, List[Any]] = defaultdict(list)
        self._pool_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {'created': 0, 'reused': 0, 'returned': 0})
        self._lock = threading.Lock()
    
    def get_object(self, obj_type: str, factory: Callable[[], Any]) -> Any:
        """Get object from pool or create new one."""
        with self._lock:
            pool = self._pools[obj_type]
            
            if pool:
                obj = pool.pop()
                self._pool_stats[obj_type]['reused'] += 1
                return obj
            else:
                obj = factory()
                self._pool_stats[obj_type]['created'] += 1
                return obj
    
    def return_object(self, obj: Any, obj_type: str, reset_func: Optional[Callable[[Any], None]] = None) -> None:
        """Return object to pool."""
        with self._lock:
            pool = self._pools[obj_type]
            
            if len(pool) < self.max_pool_size:
                # Reset object if reset function provided
                if reset_func:
                    try:
                        reset_func(obj)
                    except Exception as e:
                        logger.warning(f"Failed to reset object {obj_type}: {e}")
                        return
                
                pool.append(obj)
                self._pool_stats[obj_type]['returned'] += 1
    
    def get_pool_stats(self) -> Dict[str, Dict[str, int]]:
        """Get pool statistics."""
        with self._lock:
            return dict(self._pool_stats)
    
    def clear_pool(self, obj_type: Optional[str] = None) -> int:
        """Clear pool(s)."""
        cleared = 0
        
        with self._lock:
            if obj_type:
                cleared = len(self._pools[obj_type])
                self._pools[obj_type].clear()
            else:
                for pool in self._pools.values():
                    cleared += len(pool)
                    pool.clear()
                self._pools.clear()
                self._pool_stats.clear()
        
        return cleared


class MemoryMonitor:
    """Monitor memory usage."""
    
    def __init__(self, check_interval: float = 60.0):
        self.check_interval = check_interval
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._callbacks: List[Callable[[MemoryStats], None]] = []
        self._lock = threading.Lock()
        self._peak_memory = 0
        self._last_gc_time = 0.0
    
    def start(self) -> None:
        """Start memory monitoring."""
        if self._running:
            return
        
        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info(f"Memory monitor started with {self.check_interval}s interval")
    
    def stop(self) -> None:
        """Stop memory monitoring."""
        if not self._running:
            return
        
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join()
        logger.info("Memory monitor stopped")
    
    def add_callback(self, callback: Callable[[MemoryStats], None]) -> None:
        """Add memory stats callback."""
        with self._lock:
            self._callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[MemoryStats], None]) -> None:
        """Remove memory stats callback."""
        with self._lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)
    
    def _monitor_loop(self) -> None:
        """Memory monitoring loop."""
        while self._running:
            try:
                stats = self.get_memory_stats()
                
                # Update peak memory
                if stats.used_memory > self._peak_memory:
                    self._peak_memory = stats.used_memory
                    stats.peak_memory = self._peak_memory
                else:
                    stats.peak_memory = self._peak_memory
                
                # Call callbacks
                with self._lock:
                    for callback in self._callbacks:
                        try:
                            callback(stats)
                        except Exception as e:
                            logger.error(f"Memory monitor callback error: {e}")
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Memory monitor error: {e}")
                time.sleep(self.check_interval)
    
    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics."""
        # System memory
        memory = psutil.virtual_memory()
        
        # GC statistics
        gc_stats = {}
        for i in range(3):
            gc_stats[i] = gc.get_count()[i]
        
        # Object counts
        object_counts = {}
        if hasattr(sys, 'gettotalrefcount'):
            object_counts['total_refs'] = sys.gettotalrefcount()
        
        # Get object counts by type
        from collections import Counter
        type_counts = Counter(type(obj).__name__ for obj in gc.get_objects())
        object_counts.update(dict(type_counts.most_common(10)))
        
        return MemoryStats(
            total_memory=memory.total,
            available_memory=memory.available,
            used_memory=memory.used,
            memory_percent=memory.percent,
            gc_collections=gc_stats,
            object_counts=object_counts,
            peak_memory=self._peak_memory,
            last_gc_time=self._last_gc_time
        )


class MemoryOptimizer:
    """Main memory optimizer class."""
    
    def __init__(self,
                 gc_threshold: float = 80.0,
                 auto_gc_enabled: bool = True,
                 memory_pool_enabled: bool = True,
                 object_tracking_enabled: bool = True,
                 monitoring_enabled: bool = True,
                 monitor_interval: float = 60.0):
        self.gc_threshold = gc_threshold
        self.auto_gc_enabled = auto_gc_enabled
        self.memory_pool_enabled = memory_pool_enabled
        self.object_tracking_enabled = object_tracking_enabled
        self.monitoring_enabled = monitoring_enabled
        
        # Initialize components
        self.object_tracker = ObjectTracker() if object_tracking_enabled else None
        self.memory_pool = MemoryPool() if memory_pool_enabled else None
        self.memory_monitor = MemoryMonitor(monitor_interval) if monitoring_enabled else None
        
        # Memory optimization settings
        self._optimization_callbacks: List[Callable[[], None]] = []
        self._lock = threading.Lock()
        
        # Setup automatic GC
        if auto_gc_enabled and self.memory_monitor:
            self.memory_monitor.add_callback(self._auto_gc_callback)
        
        # Enable tracemalloc for memory profiling
        if not tracemalloc.is_tracing():
            tracemalloc.start()
    
    def initialize(self) -> None:
        """Initialize memory optimizer."""
        if self.memory_monitor:
            self.memory_monitor.start()
        
        # Set GC thresholds for better performance
        gc.set_threshold(700, 10, 10)
        
        logger.info("Memory optimizer initialized")
    
    def shutdown(self) -> None:
        """Shutdown memory optimizer."""
        if self.memory_monitor:
            self.memory_monitor.stop()
        
        # Final cleanup
        self.force_gc()
        
        logger.info("Memory optimizer shutdown")
    
    def _auto_gc_callback(self, stats: MemoryStats) -> None:
        """Automatic garbage collection callback."""
        if stats.memory_percent > self.gc_threshold:
            logger.warning(f"Memory usage {stats.memory_percent:.1f}% exceeds threshold {self.gc_threshold}%")
            self.force_gc()
            
            # Run optimization callbacks
            with self._lock:
                for callback in self._optimization_callbacks:
                    try:
                        callback()
                    except Exception as e:
                        logger.error(f"Memory optimization callback error: {e}")
    
    def force_gc(self) -> Dict[str, int]:
        """Force garbage collection."""
        start_time = time.time()
        
        # Collect all generations
        collected = {}
        for generation in range(3):
            collected[f'gen_{generation}'] = gc.collect(generation)
        
        # Update last GC time
        if self.memory_monitor:
            self.memory_monitor._last_gc_time = time.time()
        
        gc_time = time.time() - start_time
        logger.debug(f"Garbage collection completed in {gc_time:.3f}s: {collected}")
        
        return collected
    
    def add_optimization_callback(self, callback: Callable[[], None]) -> None:
        """Add memory optimization callback."""
        with self._lock:
            self._optimization_callbacks.append(callback)
    
    def remove_optimization_callback(self, callback: Callable[[], None]) -> None:
        """Remove memory optimization callback."""
        with self._lock:
            if callback in self._optimization_callbacks:
                self._optimization_callbacks.remove(callback)
    
    def track_object(self, obj: Any, obj_type: Optional[str] = None) -> Optional[int]:
        """Track object for memory monitoring."""
        if self.object_tracker:
            return self.object_tracker.track_object(obj, obj_type)
        return None
    
    def get_from_pool(self, obj_type: str, factory: Callable[[], Any]) -> Any:
        """Get object from memory pool."""
        if self.memory_pool:
            return self.memory_pool.get_object(obj_type, factory)
        return factory()
    
    def return_to_pool(self, obj: Any, obj_type: str, reset_func: Optional[Callable[[Any], None]] = None) -> None:
        """Return object to memory pool."""
        if self.memory_pool:
            self.memory_pool.return_object(obj, obj_type, reset_func)
    
    def get_memory_stats(self) -> Optional[MemoryStats]:
        """Get current memory statistics."""
        if self.memory_monitor:
            return self.memory_monitor.get_memory_stats()
        return None
    
    def get_memory_profile(self) -> Dict[str, Any]:
        """Get detailed memory profile."""
        profile = {}
        
        # Basic memory stats
        if self.memory_monitor:
            stats = self.memory_monitor.get_memory_stats()
            profile['memory_stats'] = stats.__dict__
        
        # Object tracking stats
        if self.object_tracker:
            profile['object_types'] = self.object_tracker.get_object_types()
            profile['total_tracked_objects'] = self.object_tracker.get_object_count()
        
        # Memory pool stats
        if self.memory_pool:
            profile['pool_stats'] = self.memory_pool.get_pool_stats()
        
        # Tracemalloc stats
        if tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            profile['tracemalloc'] = {
                'current': current,
                'peak': peak
            }
            
            # Top memory consumers
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')[:10]
            profile['top_memory_consumers'] = [
                {
                    'filename': stat.traceback.format()[0],
                    'size': stat.size,
                    'count': stat.count
                }
                for stat in top_stats
            ]
        
        return profile
    
    async def optimize_memory(self):
        """메모리 최적화 실행"""
        try:
            logger.info("메모리 최적화 시작")
            
            # 가비지 컬렉션 실행
            gc.collect()
            
            # 메모리 사용량 체크
            memory_usage = self.get_memory_usage()
            
            # 대용량 객체 정리
            self._cleanup_large_objects()
            
            # 캐시 정리
            self._cleanup_cache()
            
            logger.info(f"메모리 최적화 완료 - 사용량: {memory_usage:.1f}MB")
            
        except Exception as e:
            logger.error(f"메모리 최적화 중 오류: {e}")
            raise

    def _cleanup_large_objects(self):
        """대용량 객체 정리"""
        # 임시 데이터 정리
        for obj_name in list(globals().keys()):
            if obj_name.startswith('temp_') or obj_name.startswith('cache_'):
                try:
                    del globals()[obj_name]
                except:
                    pass

    def _cleanup_cache(self):
        """캐시 정리"""
        # 내부 캐시 정리
        if hasattr(self, '_cache'):
            self._cache.clear()


# Decorator for memory optimization
def memory_optimized(auto_cleanup: bool = True,
                    track_objects: bool = True,
                    use_pool: bool = False,
                    pool_type: Optional[str] = None):
    """Decorator for memory-optimized functions."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            optimizer = get_memory_optimizer()
            
            # Track function execution
            if track_objects and optimizer.object_tracker:
                func_id = optimizer.track_object(func, 'function')
            
            try:
                result = func(*args, **kwargs)
                
                # Auto cleanup after function execution
                if auto_cleanup:
                    if hasattr(optimizer, '_last_cleanup'):
                        if time.time() - optimizer._last_cleanup > 300:  # 5 minutes
                            optimizer.force_gc()
                            optimizer._last_cleanup = time.time()
                    else:
                        optimizer._last_cleanup = time.time()
                
                return result
                
            except Exception:
                # Force cleanup on exception
                if auto_cleanup:
                    optimizer.force_gc()
                raise
        
        return wrapper
    return decorator


# Global memory optimizer instance
_memory_optimizer: Optional[MemoryOptimizer] = None


def get_memory_optimizer() -> MemoryOptimizer:
    """Get global memory optimizer instance."""
    global _memory_optimizer
    if _memory_optimizer is None:
        _memory_optimizer = MemoryOptimizer()
        _memory_optimizer.initialize()
    return _memory_optimizer


def cleanup_memory() -> Dict[str, Any]:
    """Convenience function for memory cleanup."""
    optimizer = get_memory_optimizer()
    return optimizer.optimize_memory() 