import asyncio
import aiohttp
import logging
import time
import json
import pickle
import hashlib
from typing import Dict, List, Any, Optional, Callable, Union
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
import redis
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from tqdm import tqdm
import traceback


class AsyncBatchExecutor:
    """비동기 배치 처리 실행기"""
    
    def __init__(self, max_workers: int = 10, max_concurrent: int = 50):
        self.max_workers = max_workers
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.session = None
        
    async def __aenter__(self):
        connector = aiohttp.TCPConnector(limit=self.max_concurrent, limit_per_host=10)
        self.session = aiohttp.ClientSession(connector=connector)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            
    async def execute_batch(self, tasks: List, desc: str = "Processing") -> List[Any]:
        """배치 작업 실행"""
        async def execute_with_semaphore(task):
            async with self.semaphore:
                return await task
                
        results = []
        with tqdm(total=len(tasks), desc=desc) as pbar:
            for batch in self._chunk_list(tasks, self.max_workers):
                batch_results = await asyncio.gather(*[execute_with_semaphore(task) for task in batch], return_exceptions=True)
                results.extend(batch_results)
                pbar.update(len(batch))
                
        return results
        
    def _chunk_list(self, lst: List, chunk_size: int) -> List[List]:
        """리스트를 청크로 분할"""
        return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


class CacheManager:
    """멀티레벨 캐시 관리자"""
    
    def __init__(self, redis_url: str = None, memory_cache_size: int = 1000):
        self.memory_cache = {}
        self.memory_cache_size = memory_cache_size
        self.redis_client = None
        if redis_url:
            try:
                self.redis_client = redis.from_url(redis_url)
                self.redis_client.ping()
            except Exception as e:
                logging.warning(f"Redis 연결 실패: {e}")
                
    def _generate_key(self, *args, **kwargs) -> str:
        """캐시 키 생성"""
        key_data = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_data.encode()).hexdigest()
        
    def get(self, key: str, default: Any = None) -> Any:
        """캐시에서 데이터 조회"""
        # 메모리 캐시 확인
        if key in self.memory_cache:
            data, expiry = self.memory_cache[key]
            if expiry is None or time.time() < expiry:
                return data
            else:
                del self.memory_cache[key]
                
        # Redis 캐시 확인
        if self.redis_client:
            try:
                data = self.redis_client.get(key)
                if data:
                    return pickle.loads(data)
            except Exception as e:
                logging.warning(f"Redis 조회 실패: {e}")
                
        return default
        
    def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        """캐시에 데이터 저장"""
        expiry = time.time() + ttl if ttl > 0 else None
        
        # 메모리 캐시 저장
        if len(self.memory_cache) >= self.memory_cache_size:
            # LRU 방식으로 오래된 항목 제거
            oldest_key = min(self.memory_cache.keys(), 
                           key=lambda k: self.memory_cache[k][1] or 0)
            del self.memory_cache[oldest_key]
            
        self.memory_cache[key] = (value, expiry)
        
        # Redis 캐시 저장
        if self.redis_client:
            try:
                serialized_data = pickle.dumps(value)
                self.redis_client.setex(key, ttl, serialized_data)
            except Exception as e:
                logging.warning(f"Redis 저장 실패: {e}")
                
    def clear(self) -> None:
        """캐시 클리어"""
        self.memory_cache.clear()
        if self.redis_client:
            try:
                self.redis_client.flushdb()
            except Exception as e:
                logging.warning(f"Redis 클리어 실패: {e}")


class ConnectionPool:
    """커넥션 풀 관리자"""
    
    def __init__(self, max_connections: int = 100, max_per_host: int = 10):
        self.max_connections = max_connections
        self.max_per_host = max_per_host
        self.connector = None
        self.session = None
        
    async def __aenter__(self):
        self.connector = aiohttp.TCPConnector(
            limit=self.max_connections,
            limit_per_host=self.max_per_host,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        self.session = aiohttp.ClientSession(
            connector=self.connector,
            timeout=timeout
        )
        return self.session
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()


class RetryHandler:
    """재시도 핸들러"""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        
    async def execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """재시도 로직과 함께 함수 실행"""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries:
                    delay = self.base_delay * (2 ** attempt)
                    logging.warning(f"재시도 {attempt + 1}/{self.max_retries + 1}: {e}")
                    await asyncio.sleep(delay)
                else:
                    logging.error(f"최대 재시도 횟수 초과: {e}")
                    raise last_exception


class PerformanceMonitor:
    """성능 모니터링"""
    
    def __init__(self):
        self.metrics = {}
        
    def start_timer(self, name: str) -> None:
        """타이머 시작"""
        self.metrics[name] = {'start': time.time()}
        
    def end_timer(self, name: str) -> float:
        """타이머 종료 및 실행 시간 반환"""
        if name in self.metrics:
            duration = time.time() - self.metrics[name]['start']
            self.metrics[name]['duration'] = duration
            self.metrics[name]['end'] = time.time()
            return duration
        return 0.0
        
    def get_metrics(self) -> Dict[str, Any]:
        """메트릭 반환"""
        return self.metrics.copy()


class DataProcessor:
    """데이터 처리 유틸리티"""
    
    @staticmethod
    def chunk_dataframe(df: pd.DataFrame, chunk_size: int = 1000) -> List[pd.DataFrame]:
        """DataFrame을 청크로 분할"""
        return [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
        
    @staticmethod
    def safe_convert_numeric(series: pd.Series) -> pd.Series:
        """안전한 숫자 변환"""
        try:
            return pd.to_numeric(series, errors='coerce')
        except Exception:
            return series
            
    @staticmethod
    def remove_outliers(df: pd.DataFrame, columns: List[str], 
                       method: str = 'iqr') -> pd.DataFrame:
        """이상치 제거"""
        df_clean = df.copy()
        
        for col in columns:
            if col in df_clean.columns:
                if method == 'iqr':
                    Q1 = df_clean[col].quantile(0.25)
                    Q3 = df_clean[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    df_clean = df_clean[
                        (df_clean[col] >= lower_bound) & 
                        (df_clean[col] <= upper_bound)
                    ]
                    
        return df_clean


class AsyncLogger:
    """비동기 로거"""
    
    def __init__(self, name: str, level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            
    async def info(self, message: str) -> None:
        """정보 로그"""
        self.logger.info(message)
        
    async def warning(self, message: str) -> None:
        """경고 로그"""
        self.logger.warning(message)
        
    async def error(self, message: str) -> None:
        """에러 로그"""
        self.logger.error(message)
        
    async def debug(self, message: str) -> None:
        """디버그 로그"""
        self.logger.debug(message)


# 데코레이터들
def async_retry(max_retries: int = 3, base_delay: float = 1.0):
    """비동기 재시도 데코레이터"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            retry_handler = RetryHandler(max_retries, base_delay)
            return await retry_handler.execute_with_retry(func, *args, **kwargs)
        return wrapper
    return decorator


def cache_result(ttl: int = 3600):
    """결과 캐싱 데코레이터"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache_manager = CacheManager()
            cache_key = f"{func.__name__}_{hash(str(args) + str(kwargs))}"
            
            # 캐시에서 조회
            cached_result = cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result
                
            # 함수 실행
            result = await func(*args, **kwargs)
            
            # 결과 캐싱
            cache_manager.set(cache_key, result, ttl)
            return result
        return wrapper
    return decorator


def performance_monitor(name: str = None):
    """성능 모니터링 데코레이터"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            monitor_name = name or func.__name__
            monitor = PerformanceMonitor()
            monitor.start_timer(monitor_name)
            
            try:
                result = await func(*args, **kwargs)
                duration = monitor.end_timer(monitor_name)
                logging.info(f"{monitor_name} 실행 시간: {duration:.2f}초")
                return result
            except Exception as e:
                duration = monitor.end_timer(monitor_name)
                logging.error(f"{monitor_name} 실패 (시간: {duration:.2f}초): {e}")
                raise
        return wrapper
    return decorator


# 전역 인스턴스들
cache_manager = CacheManager()
retry_handler = RetryHandler()
performance_monitor = PerformanceMonitor()
async_logger = AsyncLogger("common_utils") 