import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Any, Optional
from tqdm import tqdm
import time

class AsyncBatchExecutor:
    def __init__(self, max_workers: int = 32, chunk_size: int = 50, retry: int = 2, timeout: int = 30):
        self.max_workers = max_workers
        self.chunk_size = chunk_size
        self.retry = retry
        self.timeout = timeout

    def run(self, items: List[Any], func: Callable, desc: Optional[str] = None, *args, **kwargs) -> List[Any]:
        results = []
        chunks = [items[i:i+self.chunk_size] for i in range(0, len(items), self.chunk_size)]
        for chunk in tqdm(chunks, desc=desc or "Batch Processing"):
            results.extend(self._run_chunk(chunk, func, *args, **kwargs))
        return results

    def _run_chunk(self, chunk, func, *args, **kwargs):
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._retry_func, func, item, *args, **kwargs) for item in chunk]
            results = []
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=self.timeout)
                    if result is not None:
                        results.append(result)
                except Exception:
                    continue
            return results

    def _retry_func(self, func, item, *args, **kwargs):
        for _ in range(self.retry):
            try:
                return func(item, *args, **kwargs)
            except Exception:
                time.sleep(0.5)
        return None

    async def run_async(self, items: List[Any], func: Callable, desc: Optional[str] = None, *args, **kwargs) -> List[Any]:
        results = []
        chunks = [items[i:i+self.chunk_size] for i in range(0, len(items), self.chunk_size)]
        for chunk in tqdm(chunks, desc=desc or "Async Batch Processing"):
            results.extend(await self._run_chunk_async(chunk, func, *args, **kwargs))
        return results

    async def _run_chunk_async(self, chunk, func, *args, **kwargs):
        tasks = [self._retry_func_async(func, item, *args, **kwargs) for item in chunk]
        done, _ = await asyncio.wait(tasks, timeout=self.timeout)
        return [task.result() for task in done if task.result() is not None]

    async def _retry_func_async(self, func, item, *args, **kwargs):
        for _ in range(self.retry):
            try:
                return await func(item, *args, **kwargs)
            except Exception:
                await asyncio.sleep(0.5)
        return None 