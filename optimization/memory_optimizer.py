#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: memory_optimizer.py
모듈: 메모리 최적화 전문 시스템
목적: 메모리 사용량 극한 최적화, 메모리 누수 방지, 가비지 컬렉션 최적화

Author: World-Class AI Trading System
Created: 2025-01-27
Version: 1.0.0

최적화 기법:
🧠 메모리 관리:
- 메모리 매핑 파일 I/O
- 청크 단위 스트리밍 처리
- 메모리 풀 재사용
- 약한 참조 활용

⚡ 가비지 컬렉션:
- 세대별 GC 최적화
- 수동 GC 트리거
- 순환 참조 방지
- 메모리 임계값 관리

📊 메모리 모니터링:
- 실시간 메모리 사용량 추적
- 메모리 누수 탐지
- 객체 수명 관리
- 메모리 프로파일링

🔧 최적화 전략:
- 데이터 구조 최적화
- 메모리 정렬 최적화
- 캐시 친화적 알고리즘
- 메모리 압축 기법

License: MIT
"""

from __future__ import annotations
import gc
import mmap
import os
import sys
import time
import threading
import weakref
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union, Iterator
import logging
import psutil
import resource
from collections import defaultdict, deque
from functools import wraps
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from memory_profiler import profile
import pympler
from pympler import tracker, muppy, summary
import tracemalloc
import linecache
import pickle
import joblib
import lz4.frame
import zstandard as zstd
from concurrent.futures import ThreadPoolExecutor
import asyncio

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 메모리 설정
MEMORY_CONFIG = {
    'max_memory_gb': 16,
    'gc_threshold_ratio': 0.8,
    'chunk_size': 1024 * 1024,  # 1MB 청크
    'mmap_threshold': 100 * 1024 * 1024,  # 100MB 이상 시 mmap 사용
    'weak_ref_cache_size': 10000,
    'memory_pool_size': 1000,
    'compression_threshold': 50 * 1024 * 1024,  # 50MB 이상 시 압축
}

@dataclass
class MemoryStats:
    """메모리 통계 클래스"""
    total_memory: int = 0
    available_memory: int = 0
    used_memory: int = 0
    memory_percent: float = 0.0
    process_memory: int = 0
    gc_collections: int = 0
    gc_collected: int = 0
    gc_uncollectable: int = 0
    timestamp: float = field(default_factory=time.time)

class MemoryPool:
    """메모리 풀 관리자"""

    def __init__(self, pool_size: int = 1000):
        self.pool_size = pool_size
        self.byte_pools = defaultdict(deque)
        self.array_pools = defaultdict(deque)
        self.dataframe_pools = deque()
        self.lock = threading.Lock()
        self.stats = {
            'allocations': 0,
            'deallocations': 0,
            'pool_hits': 0,
            'pool_misses': 0,
        }

    def get_bytes(self, size: int) -> bytearray:
        """바이트 배열 획득"""
        with self.lock:
            pool = self.byte_pools[size]
            if pool:
                self.stats['pool_hits'] += 1
                return pool.popleft()
            else:
                self.stats['pool_misses'] += 1
                self.stats['allocations'] += 1
                return bytearray(size)

    def return_bytes(self, data: bytearray):
        """바이트 배열 반환"""
        with self.lock:
            size = len(data)
            pool = self.byte_pools[size]
            if len(pool) < self.pool_size:
                # 데이터 초기화
                data[:] = b'\x00' * size
                pool.append(data)
                self.stats['deallocations'] += 1

    def get_array(self, shape: tuple, dtype: np.dtype) -> np.ndarray:
        """NumPy 배열 획득"""
        with self.lock:
            key = (shape, dtype)
            pool = self.array_pools[key]
            if pool:
                self.stats['pool_hits'] += 1
                return pool.popleft()
            else:
                self.stats['pool_misses'] += 1
                self.stats['allocations'] += 1
                return np.zeros(shape, dtype=dtype)

    def return_array(self, array: np.ndarray):
        """NumPy 배열 반환"""
        with self.lock:
            key = (array.shape, array.dtype)
            pool = self.array_pools[key]
            if len(pool) < self.pool_size:
                # 배열 초기화
                array.fill(0)
                pool.append(array)
                self.stats['deallocations'] += 1

    def get_dataframe(self, rows: int, cols: int) -> pd.DataFrame:
        """DataFrame 획득"""
        with self.lock:
            if self.dataframe_pools:
                df = self.dataframe_pools.popleft()
                # 크기 조정
                if len(df) >= rows and len(df.columns) >= cols:
                    self.stats['pool_hits'] += 1
                    return df.iloc[:rows, :cols].copy()

            self.stats['pool_misses'] += 1
            self.stats['allocations'] += 1
            return pd.DataFrame(np.zeros((rows, cols)))

    def return_dataframe(self, df: pd.DataFrame):
        """DataFrame 반환"""
        with self.lock:
            if len(self.dataframe_pools) < self.pool_size:
                # 데이터 초기화
                df.iloc[:, :] = 0
                self.dataframe_pools.append(df)
                self.stats['deallocations'] += 1

    def get_stats(self) -> Dict[str, int]:
        """풀 통계 반환"""
        with self.lock:
            return self.stats.copy()

class WeakRefCache:
    """약한 참조 캐시"""

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.cache = {}
        self.access_order = deque()
        self.lock = threading.Lock()
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
        }

    def get(self, key: str) -> Optional[Any]:
        """캐시에서 객체 조회"""
        with self.lock:
            if key in self.cache:
                weak_ref = self.cache[key]
                obj = weak_ref()
                if obj is not None:
                    # 액세스 순서 업데이트
                    self.access_order.remove(key)
                    self.access_order.append(key)
                    self.stats['hits'] += 1
                    return obj
                else:
                    # 객체가 가비지 컬렉션됨
                    del self.cache[key]

            self.stats['misses'] += 1
            return None

    def set(self, key: str, obj: Any):
        """캐시에 객체 저장"""
        with self.lock:
            # 크기 제한 확인
            if len(self.cache) >= self.max_size:
                self._evict_oldest()

            # 약한 참조로 저장
            def cleanup_callback(weak_ref):
                with self.lock:
                    if key in self.cache and self.cache[key] is weak_ref:
                        del self.cache[key]
                        if key in self.access_order:
                            self.access_order.remove(key)

            self.cache[key] = weakref.ref(obj, cleanup_callback)
            self.access_order.append(key)

    def _evict_oldest(self):
        """가장 오래된 항목 제거"""
        if self.access_order:
            oldest_key = self.access_order.popleft()
            if oldest_key in self.cache:
                del self.cache[oldest_key]
                self.stats['evictions'] += 1

    def get_stats(self) -> Dict[str, int]:
        """캐시 통계 반환"""
        with self.lock:
            return self.stats.copy()

class MemoryMappedFile:
    """메모리 매핑 파일 관리자"""

    def __init__(self, file_path: Union[str, Path], mode: str = 'r+b'):
        self.file_path = Path(file_path)
        self.mode = mode
        self.file_handle = None
        self.mmap_handle = None
        self.size = 0

    def __enter__(self):
        """컨텍스트 매니저 진입"""
        self.file_handle = open(self.file_path, self.mode)
        self.size = os.path.getsize(self.file_path)

        if self.size > 0:
            self.mmap_handle = mmap.mmap(
                self.file_handle.fileno(),
                self.size,
                access=mmap.ACCESS_READ if 'r' in self.mode else mmap.ACCESS_WRITE
            )

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """컨텍스트 매니저 종료"""
        if self.mmap_handle:
            self.mmap_handle.close()
        if self.file_handle:
            self.file_handle.close()

    def read_chunk(self, offset: int, size: int) -> bytes:
        """청크 단위 읽기"""
        if self.mmap_handle:
            return self.mmap_handle[offset:offset + size]
        return b''

    def write_chunk(self, offset: int, data: bytes):
        """청크 단위 쓰기"""
        if self.mmap_handle and 'w' in self.mode:
            self.mmap_handle[offset:offset + len(data)] = data
            self.mmap_handle.flush()

class ChunkedDataProcessor:
    """청크 단위 데이터 처리기"""

    def __init__(self, chunk_size: int = 1024 * 1024):
        self.chunk_size = chunk_size
        self.memory_pool = MemoryPool()
        self.compressor = zstd.ZstdCompressor(level=3)
        self.decompressor = zstd.ZstdDecompressor()

    def process_large_file(self, file_path: Path, processor: Callable) -> Iterator[Any]:
        """대용량 파일 청크 단위 처리"""
        file_size = file_path.stat().st_size

        if file_size > MEMORY_CONFIG['mmap_threshold']:
            # 메모리 매핑 사용
            with MemoryMappedFile(file_path) as mmap_file:
                offset = 0
                while offset < file_size:
                    chunk_size = min(self.chunk_size, file_size - offset)
                    chunk_data = mmap_file.read_chunk(offset, chunk_size)

                    result = processor(chunk_data)
                    if result is not None:
                        yield result

                    offset += chunk_size
        else:
            # 일반 파일 읽기
            with open(file_path, 'rb') as f:
                while True:
                    chunk_data = f.read(self.chunk_size)
                    if not chunk_data:
                        break

                    result = processor(chunk_data)
                    if result is not None:
                        yield result

    def process_dataframe_chunks(self, df: pd.DataFrame, processor: Callable) -> Iterator[pd.DataFrame]:
        """DataFrame 청크 단위 처리"""
        total_rows = len(df)
        rows_per_chunk = self.chunk_size // (df.memory_usage(deep=True).sum() // total_rows)

        for start_idx in range(0, total_rows, rows_per_chunk):
            end_idx = min(start_idx + rows_per_chunk, total_rows)
            chunk_df = df.iloc[start_idx:end_idx].copy()

            result = processor(chunk_df)
            if result is not None:
                yield result

            # 메모리 정리
            del chunk_df
            gc.collect()

    def compress_data(self, data: bytes) -> bytes:
        """데이터 압축"""
        if len(data) > MEMORY_CONFIG['compression_threshold']:
            return self.compressor.compress(data)
        return data

    def decompress_data(self, compressed_data: bytes) -> bytes:
        """데이터 압축 해제"""
        try:
            return self.decompressor.decompress(compressed_data)
        except:
            return compressed_data

class GarbageCollectionOptimizer:
    """가비지 컬렉션 최적화"""

    def __init__(self):
        self.gc_stats = {
            'collections': [0, 0, 0],
            'collected': [0, 0, 0],
            'uncollectable': [0, 0, 0],
        }
        self.last_gc_time = time.time()
        self.gc_threshold = MEMORY_CONFIG['gc_threshold_ratio']

    def optimize_gc_settings(self):
        """GC 설정 최적화"""
        # GC 임계값 조정
        gc.set_threshold(700, 10, 10)  # 더 적극적인 GC

        # 디버그 플래그 설정
        gc.set_debug(gc.DEBUG_STATS)

        logger.info("가비지 컬렉션 설정 최적화 완료")

    def should_collect(self) -> bool:
        """GC 수행 여부 결정"""
        memory_percent = psutil.virtual_memory().percent / 100
        time_since_last_gc = time.time() - self.last_gc_time

        return (memory_percent > self.gc_threshold or
                time_since_last_gc > 60)  # 1분마다 또는 메모리 부족 시

    def collect_garbage(self) -> Dict[str, int]:
        """가비지 컬렉션 수행"""
        start_time = time.time()

        # 전체 세대 GC 수행
        collected = gc.collect()

        # 통계 업데이트
        self.last_gc_time = time.time()
        gc_time = self.last_gc_time - start_time

        # GC 통계 수집
        stats = gc.get_stats()
        for i, stat in enumerate(stats):
            self.gc_stats['collections'][i] += stat.get('collections', 0)
            self.gc_stats['collected'][i] += stat.get('collected', 0)
            self.gc_stats['uncollectable'][i] += stat.get('uncollectable', 0)

        result = {
            'collected_objects': collected,
            'gc_time_seconds': gc_time,
            'generation_0_collections': self.gc_stats['collections'][0],
            'generation_1_collections': self.gc_stats['collections'][1],
            'generation_2_collections': self.gc_stats['collections'][2],
        }

        logger.info(f"가비지 컬렉션 완료: {result}")
        return result

    def find_memory_leaks(self) -> List[Dict[str, Any]]:
        """메모리 누수 탐지"""
        # 참조되지 않는 객체 찾기
        unreachable = gc.garbage

        leaks = []
        for obj in unreachable:
            leak_info = {
                'type': type(obj).__name__,
                'id': id(obj),
                'size': sys.getsizeof(obj),
                'referrers': len(gc.get_referrers(obj)),
                'referents': len(gc.get_referents(obj)),
            }
            leaks.append(leak_info)

        return leaks

class MemoryProfiler:
    """메모리 프로파일러"""

    def __init__(self):
        self.tracker = tracker.SummaryTracker()
        self.snapshots = []
        self.start_time = time.time()

    def start_profiling(self):
        """프로파일링 시작"""
        tracemalloc.start()
        self.start_time = time.time()
        logger.info("메모리 프로파일링 시작")

    def take_snapshot(self, name: str = None):
        """메모리 스냅샷 생성"""
        if not tracemalloc.is_tracing():
            return

        snapshot = tracemalloc.take_snapshot()
        self.snapshots.append({
            'name': name or f"snapshot_{len(self.snapshots)}",
            'timestamp': time.time(),
            'snapshot': snapshot,
        })

        logger.info(f"메모리 스냅샷 생성: {name}")

    def get_top_memory_usage(self, limit: int = 10) -> List[Dict[str, Any]]:
        """메모리 사용량 상위 항목 반환"""
        if not self.snapshots:
            return []

        latest_snapshot = self.snapshots[-1]['snapshot']
        top_stats = latest_snapshot.statistics('lineno')

        memory_usage = []
        for stat in top_stats[:limit]:
            memory_usage.append({
                'filename': stat.traceback.format()[0],
                'size_mb': stat.size / 1024 / 1024,
                'count': stat.count,
                'average_size': stat.size / stat.count if stat.count > 0 else 0,
            })

        return memory_usage

    def compare_snapshots(self, name1: str, name2: str) -> List[Dict[str, Any]]:
        """스냅샷 비교"""
        snapshot1 = next((s for s in self.snapshots if s['name'] == name1), None)
        snapshot2 = next((s for s in self.snapshots if s['name'] == name2), None)

        if not snapshot1 or not snapshot2:
            return []

        top_stats = snapshot2['snapshot'].compare_to(snapshot1['snapshot'], 'lineno')

        differences = []
        for stat in top_stats[:10]:
            differences.append({
                'filename': stat.traceback.format()[0],
                'size_diff_mb': stat.size_diff / 1024 / 1024,
                'count_diff': stat.count_diff,
                'size_mb': stat.size / 1024 / 1024,
            })

        return differences

    def stop_profiling(self):
        """프로파일링 종료"""
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        logger.info("메모리 프로파일링 종료")

class MemoryOptimizer:
    """메모리 최적화 메인 클래스"""

    def __init__(self):
        self.memory_pool = MemoryPool()
        self.weak_cache = WeakRefCache()
        self.chunked_processor = ChunkedDataProcessor()
        self.gc_optimizer = GarbageCollectionOptimizer()
        self.profiler = MemoryProfiler()

        # 모니터링 스레드
        self.monitoring_thread = None
        self.is_monitoring = False

        # 통계
        self.stats = {
            'optimizations_performed': 0,
            'memory_saved_mb': 0,
            'gc_collections': 0,
            'start_time': time.time(),
        }

    def __enter__(self):
        """컨텍스트 매니저 진입"""
        self.start_optimization()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """컨텍스트 매니저 종료"""
        self.stop_optimization()

    def start_optimization(self):
        """최적화 시작"""
        logger.info("메모리 최적화 시작")

        # GC 설정 최적화
        self.gc_optimizer.optimize_gc_settings()

        # 프로파일링 시작
        self.profiler.start_profiling()

        # 모니터링 시작
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitor_memory)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()

    def stop_optimization(self):
        """최적화 종료"""
        logger.info("메모리 최적화 종료")

        # 모니터링 중지
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)

        # 프로파일링 종료
        self.profiler.stop_profiling()

        # 최종 GC 수행
        self.gc_optimizer.collect_garbage()

    def _monitor_memory(self):
        """메모리 모니터링 스레드"""
        while self.is_monitoring:
            try:
                # 메모리 사용량 확인
                memory_stats = self.get_memory_stats()

                # 임계값 초과 시 최적화 수행
                if memory_stats.memory_percent > MEMORY_CONFIG['gc_threshold_ratio']:
                    self.perform_optimization()

                time.sleep(10)  # 10초마다 확인

            except Exception as e:
                logger.error(f"메모리 모니터링 오류: {e}")
                time.sleep(5)

    def get_memory_stats(self) -> MemoryStats:
        """메모리 통계 조회"""
        virtual_memory = psutil.virtual_memory()
        process = psutil.Process()

        return MemoryStats(
            total_memory=virtual_memory.total,
            available_memory=virtual_memory.available,
            used_memory=virtual_memory.used,
            memory_percent=virtual_memory.percent,
            process_memory=process.memory_info().rss,
            gc_collections=sum(self.gc_optimizer.gc_stats['collections']),
            gc_collected=sum(self.gc_optimizer.gc_stats['collected']),
            gc_uncollectable=sum(self.gc_optimizer.gc_stats['uncollectable']),
        )

    def perform_optimization(self):
        """최적화 수행"""
        logger.info("메모리 최적화 수행")

        start_memory = psutil.Process().memory_info().rss

        # 가비지 컬렉션 수행
        if self.gc_optimizer.should_collect():
            gc_result = self.gc_optimizer.collect_garbage()
            self.stats['gc_collections'] += 1

        # 메모리 누수 탐지
        leaks = self.gc_optimizer.find_memory_leaks()
        if leaks:
            logger.warning(f"메모리 누수 탐지: {len(leaks)}개")

        # 최적화 후 메모리 사용량
        end_memory = psutil.Process().memory_info().rss
        memory_saved = (start_memory - end_memory) / 1024 / 1024  # MB

        self.stats['optimizations_performed'] += 1
        self.stats['memory_saved_mb'] += memory_saved

        logger.info(f"최적화 완료: {memory_saved:.2f}MB 절약")

    @contextmanager
    def optimized_processing(self, name: str = "processing"):
        """최적화된 처리 컨텍스트"""
        # 시작 스냅샷
        self.profiler.take_snapshot(f"{name}_start")
        start_memory = psutil.Process().memory_info().rss

        try:
            yield
        finally:
            # 종료 스냅샷
            self.profiler.take_snapshot(f"{name}_end")
            end_memory = psutil.Process().memory_info().rss

            # 메모리 사용량 분석
            memory_diff = (end_memory - start_memory) / 1024 / 1024
            logger.info(f"{name} 메모리 사용량: {memory_diff:.2f}MB")

            # 필요시 최적화 수행
            if memory_diff > 100:  # 100MB 이상 증가 시
                self.perform_optimization()

    def optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """DataFrame 메모리 최적화"""
        logger.info(f"DataFrame 최적화 시작: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f}MB")

        # 데이터 타입 최적화
        optimized_df = df.copy()

        for col in optimized_df.columns:
            col_type = optimized_df[col].dtype

            if col_type == 'object':
                # 문자열 카테고리화
                unique_ratio = optimized_df[col].nunique() / len(optimized_df)
                if unique_ratio < 0.5:  # 고유값이 50% 미만이면 카테고리화
                    optimized_df[col] = optimized_df[col].astype('category')

            elif col_type in ['int64', 'int32']:
                # 정수 타입 최적화
                col_min = optimized_df[col].min()
                col_max = optimized_df[col].max()

                if col_min >= 0:
                    if col_max < 255:
                        optimized_df[col] = optimized_df[col].astype('uint8')
                    elif col_max < 65535:
                        optimized_df[col] = optimized_df[col].astype('uint16')
                    elif col_max < 4294967295:
                        optimized_df[col] = optimized_df[col].astype('uint32')
                else:
                    if col_min > -128 and col_max < 127:
                        optimized_df[col] = optimized_df[col].astype('int8')
                    elif col_min > -32768 and col_max < 32767:
                        optimized_df[col] = optimized_df[col].astype('int16')
                    elif col_min > -2147483648 and col_max < 2147483647:
                        optimized_df[col] = optimized_df[col].astype('int32')

            elif col_type == 'float64':
                # 실수 타입 최적화
                optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='float')

        optimized_size = optimized_df.memory_usage(deep=True).sum() / 1024 / 1024
        original_size = df.memory_usage(deep=True).sum() / 1024 / 1024
        saved_memory = original_size - optimized_size

        logger.info(f"DataFrame 최적화 완료: {saved_memory:.2f}MB 절약 ({optimized_size:.2f}MB)")

        return optimized_df

    def get_optimization_stats(self) -> Dict[str, Any]:
        """최적화 통계 반환"""
        runtime = time.time() - self.stats['start_time']
        memory_stats = self.get_memory_stats()

        return {
            'runtime_seconds': runtime,
            'optimizations_performed': self.stats['optimizations_performed'],
            'memory_saved_mb': self.stats['memory_saved_mb'],
            'gc_collections': self.stats['gc_collections'],
            'current_memory_usage_mb': memory_stats.process_memory / 1024 / 1024,
            'memory_usage_percent': memory_stats.memory_percent,
            'pool_stats': self.memory_pool.get_stats(),
            'cache_stats': self.weak_cache.get_stats(),
            'top_memory_usage': self.profiler.get_top_memory_usage(),
        }

# 메모리 최적화 데코레이터
def memory_optimized(func):
    """메모리 최적화 데코레이터"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        with MemoryOptimizer() as optimizer:
            with optimizer.optimized_processing(func.__name__):
                return func(*args, **kwargs)
    return wrapper

# 사용 예시
async def main():
    """메모리 최적화 테스트"""
    try:
        with MemoryOptimizer() as optimizer:

            # 대용량 데이터 생성
            logger.info("대용량 데이터 생성")
            large_df = pd.DataFrame({
                'id': range(1000000),
                'value': np.random.randn(1000000),
                'category': np.random.choice(['A', 'B', 'C'], 1000000),
                'timestamp': pd.date_range('2020-01-01', periods=1000000, freq='1min')
            })

            # 메모리 최적화
            logger.info("DataFrame 메모리 최적화")
            optimized_df = optimizer.optimize_dataframe(large_df)

            # 청크 단위 처리
            logger.info("청크 단위 처리")
            chunk_results = []
            for chunk in optimizer.chunked_processor.process_dataframe_chunks(:
                optimized_df, :
                lambda x: x.groupby('category').agg({'value': 'mean'})
            ):
                chunk_results.append(chunk)

            # 통계 출력
            stats = optimizer.get_optimization_stats()
            logger.info(f"최적화 통계: {stats}")

    except Exception as e:
        logger.error(f"메모리 최적화 테스트 실패: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
