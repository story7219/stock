"""
Async executor for high-performance parallel processing.
"""

import asyncio
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Union, Coroutine
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading
from functools import wraps
import weakref
import gc

logger = logging.getLogger(__name__)


@dataclass
class TaskResult:
    """Task execution result."""
    task_id: str
    result: Any = None
    error: Optional[Exception] = None
    execution_time: float = 0.0
    start_time: float = 0.0
    end_time: float = 0.0
    
    @property
    def is_success(self) -> bool:
        """Check if task execution was successful."""
        return self.error is None
    
    @property
    def is_failed(self) -> bool:
        """Check if task execution failed."""
        return self.error is not None


@dataclass
class ExecutorStats:
    """Executor statistics."""
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    active_tasks: int = 0
    avg_execution_time: float = 0.0
    total_execution_time: float = 0.0
    peak_concurrent_tasks: int = 0
    last_reset: float = 0.0


class TaskQueue:
    """High-performance task queue with priority support."""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._queue = asyncio.PriorityQueue(maxsize=max_size)
        self._task_counter = 0
        self._lock = asyncio.Lock()
    
    async def put(self, 
                  coro: Coroutine,
                  priority: int = 0,
                  task_id: Optional[str] = None) -> str:
        """Add task to queue."""
        if task_id is None:
            async with self._lock:
                self._task_counter += 1
                task_id = f"task_{self._task_counter}"
        
        # Lower number = higher priority
        await self._queue.put((priority, task_id, coro))
        return task_id
    
    async def get(self) -> tuple[int, str, Coroutine]:
        """Get task from queue."""
        return await self._queue.get()
    
    def task_done(self) -> None:
        """Mark task as done."""
        self._queue.task_done()
    
    async def join(self) -> None:
        """Wait for all tasks to complete."""
        await self._queue.join()
    
    def qsize(self) -> int:
        """Get queue size."""
        return self._queue.qsize()
    
    def empty(self) -> bool:
        """Check if queue is empty."""
        return self._queue.empty()
    
    def full(self) -> bool:
        """Check if queue is full."""
        return self._queue.full()


class WorkerPool:
    """Worker pool for executing async tasks."""
    
    def __init__(self, 
                 num_workers: int = 10,
                 task_queue: Optional[TaskQueue] = None):
        self.num_workers = num_workers
        self.task_queue = task_queue or TaskQueue()
        self._workers: List[asyncio.Task] = []
        self._results: Dict[str, TaskResult] = {}
        self._running = False
        self._stats = ExecutorStats()
        self._stats.last_reset = time.time()
        self._result_callbacks: Dict[str, List[Callable]] = {}
        self._lock = asyncio.Lock()
    
    async def start(self) -> None:
        """Start worker pool."""
        if self._running:
            return
        
        self._running = True
        self._workers = []
        
        for i in range(self.num_workers):
            worker = asyncio.create_task(self._worker(f"worker_{i}"))
            self._workers.append(worker)
        
        logger.info(f"Worker pool started with {self.num_workers} workers")
    
    async def stop(self) -> None:
        """Stop worker pool."""
        if not self._running:
            return
        
        self._running = False
        
        # Cancel all workers
        for worker in self._workers:
            worker.cancel()
        
        # Wait for workers to finish
        await asyncio.gather(*self._workers, return_exceptions=True)
        
        self._workers.clear()
        logger.info("Worker pool stopped")
    
    async def _worker(self, worker_name: str) -> None:
        """Worker coroutine."""
        logger.debug(f"Worker {worker_name} started")
        
        while self._running:
            try:
                # Get task from queue with timeout
                try:
                    priority, task_id, coro = await asyncio.wait_for(
                        self.task_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Execute task
                start_time = time.time()
                result = TaskResult(
                    task_id=task_id,
                    start_time=start_time
                )
                
                async with self._lock:
                    self._stats.active_tasks += 1
                    if self._stats.active_tasks > self._stats.peak_concurrent_tasks:
                        self._stats.peak_concurrent_tasks = self._stats.active_tasks
                
                try:
                    result.result = await coro
                    logger.debug(f"Task {task_id} completed successfully")
                    
                except Exception as e:
                    result.error = e
                    logger.error(f"Task {task_id} failed: {e}")
                
                finally:
                    end_time = time.time()
                    result.end_time = end_time
                    result.execution_time = end_time - start_time
                    
                    # Update stats
                    async with self._lock:
                        self._stats.active_tasks -= 1
                        self._stats.total_tasks += 1
                        
                        if result.is_success:
                            self._stats.completed_tasks += 1
                        else:
                            self._stats.failed_tasks += 1
                        
                        # Update average execution time
                        self._stats.total_execution_time += result.execution_time
                        self._stats.avg_execution_time = (
                            self._stats.total_execution_time / self._stats.total_tasks
                        )
                    
                    # Store result
                    self._results[task_id] = result
                    
                    # Execute callbacks
                    if task_id in self._result_callbacks:
                        for callback in self._result_callbacks[task_id]:
                            try:
                                if asyncio.iscoroutinefunction(callback):
                                    await callback(result)
                                else:
                                    callback(result)
                            except Exception as e:
                                logger.error(f"Callback error for task {task_id}: {e}")
                        
                        # Clean up callbacks
                        del self._result_callbacks[task_id]
                    
                    # Mark task as done
                    self.task_queue.task_done()
            
            except asyncio.CancelledError:
                logger.debug(f"Worker {worker_name} cancelled")
                break
            except Exception as e:
                logger.error(f"Worker {worker_name} error: {e}")
        
        logger.debug(f"Worker {worker_name} stopped")
    
    async def submit(self, 
                    coro: Coroutine,
                    priority: int = 0,
                    task_id: Optional[str] = None,
                    callback: Optional[Callable] = None) -> str:
        """Submit task to worker pool."""
        if not self._running:
            await self.start()
        
        task_id = await self.task_queue.put(coro, priority, task_id)
        
        if callback:
            if task_id not in self._result_callbacks:
                self._result_callbacks[task_id] = []
            self._result_callbacks[task_id].append(callback)
        
        return task_id
    
    async def get_result(self, task_id: str, timeout: Optional[float] = None) -> TaskResult:
        """Get task result."""
        start_time = time.time()
        
        while task_id not in self._results:
            if timeout and (time.time() - start_time) > timeout:
                raise asyncio.TimeoutError(f"Task {task_id} timeout")
            await asyncio.sleep(0.01)
        
        return self._results[task_id]
    
    def get_result_nowait(self, task_id: str) -> Optional[TaskResult]:
        """Get task result without waiting."""
        return self._results.get(task_id)
    
    async def wait_for_results(self, task_ids: List[str], timeout: Optional[float] = None) -> List[TaskResult]:
        """Wait for multiple task results."""
        results = []
        
        for task_id in task_ids:
            result = await self.get_result(task_id, timeout)
            results.append(result)
        
        return results
    
    def get_stats(self) -> ExecutorStats:
        """Get executor statistics."""
        return self._stats
    
    def reset_stats(self) -> None:
        """Reset executor statistics."""
        self._stats = ExecutorStats()
        self._stats.last_reset = time.time()
    
    def cleanup_results(self, max_age: float = 3600) -> int:
        """Clean up old results."""
        current_time = time.time()
        cleaned = 0
        
        to_remove = []
        for task_id, result in self._results.items():
            if (current_time - result.end_time) > max_age:
                to_remove.append(task_id)
        
        for task_id in to_remove:
            del self._results[task_id]
            cleaned += 1
        
        return cleaned


class AsyncExecutor:
    """High-performance async executor with advanced features."""
    
    def __init__(self,
                 max_workers: int = 50,
                 max_queue_size: int = 10000,
                 thread_pool_size: int = 10):
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size
        
        # Create task queue and worker pool
        self.task_queue = TaskQueue(max_size=max_queue_size)
        self.worker_pool = WorkerPool(num_workers=max_workers, task_queue=self.task_queue)
        
        # Thread pool for CPU-bound tasks
        self.thread_pool = ThreadPoolExecutor(max_workers=thread_pool_size)
        
        # Batch processing
        self._batch_queues: Dict[str, List[Coroutine]] = {}
        self._batch_timers: Dict[str, asyncio.Task] = {}
        self._batch_lock = asyncio.Lock()
    
    async def initialize(self) -> None:
        """Initialize the executor."""
        await self.worker_pool.start()
        logger.info(f"AsyncExecutor initialized with {self.max_workers} workers")
    
    async def shutdown(self) -> None:
        """Shutdown the executor."""
        await self.worker_pool.stop()
        self.thread_pool.shutdown(wait=True)
        
        # Cancel batch timers
        for timer in self._batch_timers.values():
            timer.cancel()
        
        logger.info("AsyncExecutor shutdown completed")
    
    async def submit(self,
                    coro: Coroutine,
                    priority: int = 0,
                    task_id: Optional[str] = None,
                    callback: Optional[Callable] = None) -> str:
        """Submit coroutine for execution."""
        return await self.worker_pool.submit(coro, priority, task_id, callback)
    
    async def submit_batch(self,
                          coros: List[Coroutine],
                          priority: int = 0,
                          batch_id: Optional[str] = None) -> List[str]:
        """Submit batch of coroutines."""
        task_ids = []
        
        for i, coro in enumerate(coros):
            task_id = f"{batch_id}_{i}" if batch_id else None
            task_id = await self.submit(coro, priority, task_id)
            task_ids.append(task_id)
        
        return task_ids
    
    async def submit_delayed_batch(self,
                                  coros: List[Coroutine],
                                  batch_key: str,
                                  delay: float = 0.1,
                                  max_batch_size: int = 100) -> None:
        """Submit coroutines with batching and delay."""
        async with self._batch_lock:
            if batch_key not in self._batch_queues:
                self._batch_queues[batch_key] = []
            
            self._batch_queues[batch_key].extend(coros)
            
            # Cancel existing timer
            if batch_key in self._batch_timers:
                self._batch_timers[batch_key].cancel()
            
            # Schedule batch processing
            self._batch_timers[batch_key] = asyncio.create_task(
                self._process_delayed_batch(batch_key, delay, max_batch_size)
            )
    
    async def _process_delayed_batch(self,
                                   batch_key: str,
                                   delay: float,
                                   max_batch_size: int) -> None:
        """Process delayed batch."""
        await asyncio.sleep(delay)
        
        async with self._batch_lock:
            if batch_key not in self._batch_queues:
                return
            
            coros = self._batch_queues[batch_key][:max_batch_size]
            self._batch_queues[batch_key] = self._batch_queues[batch_key][max_batch_size:]
            
            if not self._batch_queues[batch_key]:
                del self._batch_queues[batch_key]
            
            if batch_key in self._batch_timers:
                del self._batch_timers[batch_key]
        
        if coros:
            await self.submit_batch(coros, batch_id=batch_key)
            logger.debug(f"Processed delayed batch {batch_key} with {len(coros)} tasks")
    
    def submit_cpu_bound(self, func: Callable, *args, **kwargs) -> asyncio.Future:
        """Submit CPU-bound function to thread pool."""
        loop = asyncio.get_event_loop()
        return loop.run_in_executor(self.thread_pool, func, *args, **kwargs)
    
    async def gather_results(self,
                           task_ids: List[str],
                           timeout: Optional[float] = None,
                           return_exceptions: bool = False) -> List[Any]:
        """Gather results from multiple tasks."""
        results = await self.worker_pool.wait_for_results(task_ids, timeout)
        
        if return_exceptions:
            return [r.result if r.is_success else r.error for r in results]
        else:
            # Raise first exception found
            for r in results:
                if r.is_failed:
                    raise r.error
            return [r.result for r in results]
    
    async def map_async(self,
                       func: Callable,
                       items: List[Any],
                       max_concurrency: int = 10,
                       timeout: Optional[float] = None) -> List[Any]:
        """Map function over items asynchronously."""
        semaphore = asyncio.Semaphore(max_concurrency)
        
        async def bounded_func(item):
            async with semaphore:
                if asyncio.iscoroutinefunction(func):
                    return await func(item)
                else:
                    return func(item)
        
        coros = [bounded_func(item) for item in items]
        task_ids = await self.submit_batch(coros)
        
        return await self.gather_results(task_ids, timeout)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get executor statistics."""
        return {
            'worker_pool': self.worker_pool.get_stats().__dict__,
            'task_queue_size': self.task_queue.qsize(),
            'batch_queues': {k: len(v) for k, v in self._batch_queues.items()},
            'active_batch_timers': len(self._batch_timers)
        }
    
    def cleanup(self, max_result_age: float = 3600) -> Dict[str, int]:
        """Cleanup old data."""
        cleaned_results = self.worker_pool.cleanup_results(max_result_age)
        
        # Force garbage collection
        gc.collect()
        
        return {
            'cleaned_results': cleaned_results
        }

    async def execute_single(self, coro):
        """단일 코루틴 실행"""
        try:
            return await coro
        except Exception as e:
            logger.error(f"코루틴 실행 중 오류: {e}")
            raise


# Decorator for async retry
def async_retry(max_attempts: int = 3,
               delay: float = 1.0,
               backoff: float = 2.0,
               exceptions: tuple = (Exception,)):
    """Decorator for async function retry logic."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        wait_time = delay * (backoff ** attempt)
                        logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"All {max_attempts} attempts failed: {e}")
            
            raise last_exception
        
        return wrapper
    return decorator


# Global executor instance
executor: Optional[AsyncExecutor] = None


def get_executor() -> AsyncExecutor:
    """Get global executor instance."""
    global executor
    if executor is None:
        executor = AsyncExecutor()
    return executor 