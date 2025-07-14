#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: async_batch_processor.py
모듈: 비동기 배치 처리 최적화 시스템
목적: 대용량 데이터 배치 처리, 동적 로드 밸런싱, 백프레셔 제어

Author: World-Class AI Trading System
Created: 2025-01-27
Version: 1.0.0

고급 배치 처리 기법:
⚡ 비동기 배치 처리:
- 동적 배치 크기 조정
- 적응적 처리량 제어
- 백프레셔 메커니즘
- 우선순위 큐 관리

🔄 로드 밸런싱:
- 작업 부하 분산
- 동적 워커 스케일링
- 처리 시간 기반 스케줄링
- 리소스 사용량 모니터링

📊 성능 최적화:
- 처리량 극대화
- 지연시간 최소화
- 메모리 효율성
- CPU 활용률 최적화

🛡️ 안정성 보장:
- 오류 복구 메커니즘
- 재시도 정책
- 데드락 방지
- 타임아웃 관리

License: MIT
"""

from __future__ import annotations
import asyncio
import time
import logging
from collections import deque, defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import (
    Dict, List, Optional, Any, Callable, Coroutine,
    AsyncIterator, Union, Tuple, TypeVar, Generic
)
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import wraps
import threading
import multiprocessing as mp
import psutil
import weakref
import heapq
from enum import Enum
import uuid
import pickle
import json
import numpy as np
import pandas as pd

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')

class TaskPriority(Enum):
    """작업 우선순위"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

class TaskStatus(Enum):
    """작업 상태"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class BatchTask:
    """배치 작업 클래스"""
    id: str
    data: Any
    processor: Callable
    priority: TaskPriority = TaskPriority.NORMAL
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    error: Optional[Exception] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout: float = 300.0

    def __lt__(self, other):
        """우선순위 비교"""
        return self.priority.value > other.priority.value

@dataclass
class BatchConfig:
    """배치 처리 설정"""
    # 기본 설정
    batch_size: int = 100
    max_batch_size: int = 1000
    min_batch_size: int = 10
    max_workers: int = mp.cpu_count()

    # 타이밍 설정
    batch_timeout: float = 5.0
    processing_timeout: float = 300.0
    worker_timeout: float = 60.0

    # 백프레셔 설정
    max_queue_size: int = 10000
    backpressure_threshold: float = 0.8

    # 동적 조정 설정
    enable_dynamic_sizing: bool = True
    performance_window: int = 100
    adjustment_factor: float = 1.2

    # 재시도 설정
    max_retries: int = 3
    retry_delay: float = 1.0
    exponential_backoff: bool = True

class PerformanceMetrics:
    """성능 메트릭 수집기"""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.processing_times = deque(maxlen=window_size)
        self.throughput_samples = deque(maxlen=window_size)
        self.error_rates = deque(maxlen=window_size)
        self.queue_sizes = deque(maxlen=window_size)
        self.memory_usage = deque(maxlen=window_size)
        self.cpu_usage = deque(maxlen=window_size)

        self.total_processed = 0
        self.total_errors = 0
        self.start_time = time.time()

    def record_processing_time(self, duration: float):
        """처리 시간 기록"""
        self.processing_times.append(duration)

    def record_throughput(self, items_per_second: float):
        """처리량 기록"""
        self.throughput_samples.append(items_per_second)

    def record_error_rate(self, error_rate: float):
        """오류율 기록"""
        self.error_rates.append(error_rate)

    def record_queue_size(self, size: int):
        """큐 크기 기록"""
        self.queue_sizes.append(size)

    def record_system_metrics(self):
        """시스템 메트릭 기록"""
        self.memory_usage.append(psutil.virtual_memory().percent)
        self.cpu_usage.append(psutil.cpu_percent())

    def get_avg_processing_time(self) -> float:
        """평균 처리 시간"""
        return np.mean(self.processing_times) if self.processing_times else 0.0

    def get_avg_throughput(self) -> float:
        """평균 처리량"""
        return np.mean(self.throughput_samples) if self.throughput_samples else 0.0

    def get_avg_error_rate(self) -> float:
        """평균 오류율"""
        return np.mean(self.error_rates) if self.error_rates else 0.0

    def get_p95_processing_time(self) -> float:
        """95% 처리 시간"""
        return np.percentile(self.processing_times, 95) if self.processing_times else 0.0

    def get_system_load(self) -> Dict[str, float]:
        """시스템 부하 정보"""
        return {
            'cpu_percent': np.mean(self.cpu_usage) if self.cpu_usage else 0.0,
            'memory_percent': np.mean(self.memory_usage) if self.memory_usage else 0.0,
            'avg_queue_size': np.mean(self.queue_sizes) if self.queue_sizes else 0.0,
        }

class DynamicBatchSizer:
    """동적 배치 크기 조정기"""

    def __init__(self, config: BatchConfig, metrics: PerformanceMetrics):
        self.config = config
        self.metrics = metrics
        self.current_batch_size = config.batch_size
        self.last_adjustment = time.time()
        self.adjustment_interval = 30.0  # 30초마다 조정

    def should_adjust(self) -> bool:
        """조정 필요 여부 확인"""
        return (time.time() - self.last_adjustment > self.adjustment_interval and
                len(self.metrics.processing_times) >= self.config.performance_window)

    def adjust_batch_size(self) -> int:
        """배치 크기 조정"""
        if not self.should_adjust():
            return self.current_batch_size

        # 성능 지표 분석
        avg_processing_time = self.metrics.get_avg_processing_time()
        avg_throughput = self.metrics.get_avg_throughput()
        system_load = self.metrics.get_system_load()

        # 조정 결정
        adjustment_factor = 1.0

        # 처리 시간 기반 조정
        if avg_processing_time > 10.0:  # 10초 이상이면 배치 크기 감소
            adjustment_factor *= 0.8
        elif avg_processing_time < 2.0:  # 2초 미만이면 배치 크기 증가
            adjustment_factor *= 1.2

        # 시스템 부하 기반 조정
        if system_load['cpu_percent'] > 80:
            adjustment_factor *= 0.9
        elif system_load['cpu_percent'] < 50:
            adjustment_factor *= 1.1

        if system_load['memory_percent'] > 85:
            adjustment_factor *= 0.8

        # 새 배치 크기 계산
        new_batch_size = int(self.current_batch_size * adjustment_factor)
        new_batch_size = max(self.config.min_batch_size,
                           min(self.config.max_batch_size, new_batch_size))

        if new_batch_size != self.current_batch_size:
            logger.info(f"배치 크기 조정: {self.current_batch_size} → {new_batch_size}")
            self.current_batch_size = new_batch_size
            self.last_adjustment = time.time()

        return self.current_batch_size

class BackpressureController:
    """백프레셔 제어기"""

    def __init__(self, config: BatchConfig):
        self.config = config
        self.queue_size_history = deque(maxlen=50)
        self.processing_rate_history = deque(maxlen=50)
        self.backpressure_active = False
        self.last_check = time.time()

    def check_backpressure(self, queue_size: int, processing_rate: float) -> bool:
        """백프레셔 상태 확인"""
        self.queue_size_history.append(queue_size)
        self.processing_rate_history.append(processing_rate)

        # 큐 크기 기반 백프레셔
        queue_ratio = queue_size / self.config.max_queue_size
        if queue_ratio > self.config.backpressure_threshold:
            if not self.backpressure_active:
                logger.warning(f"백프레셔 활성화: 큐 크기 {queue_size}/{self.config.max_queue_size}")
                self.backpressure_active = True
            return True

        # 처리율 기반 백프레셔
        if len(self.processing_rate_history) >= 10:
            avg_rate = np.mean(list(self.processing_rate_history)[-10:])
            if avg_rate < 1.0:  # 초당 1개 미만
                if not self.backpressure_active:
                    logger.warning(f"백프레셔 활성화: 낮은 처리율 {avg_rate:.2f}/s")
                    self.backpressure_active = True
                return True

        # 백프레셔 해제
        if self.backpressure_active and queue_ratio < 0.5:
            logger.info("백프레셔 해제")
            self.backpressure_active = False

        return False

    def get_backpressure_delay(self) -> float:
        """백프레셔 지연 시간 계산"""
        if not self.backpressure_active:
            return 0.0

        # 큐 크기에 따른 지연 시간
        if self.queue_size_history:
            latest_queue_size = self.queue_size_history[-1]
            queue_ratio = latest_queue_size / self.config.max_queue_size
            return min(5.0, queue_ratio * 2.0)  # 최대 5초

        return 1.0

class WorkerPool:
    """워커 풀 관리자"""

    def __init__(self, config: BatchConfig):
        self.config = config
        self.workers = []
        self.active_workers = 0
        self.worker_stats = defaultdict(lambda: {'processed': 0, 'errors': 0, 'avg_time': 0.0})
        self.lock = asyncio.Lock()

    async def start_workers(self, worker_func: Callable):
        """워커 시작"""
        async with self.lock:
            for i in range(self.config.max_workers):
                worker = asyncio.create_task(self._worker_loop(f"worker-{i}", worker_func))
                self.workers.append(worker)

            logger.info(f"워커 풀 시작: {len(self.workers)}개 워커")

    async def stop_workers(self):
        """워커 종료"""
        async with self.lock:
            for worker in self.workers:
                worker.cancel()

            await asyncio.gather(*self.workers, return_exceptions=True)
            self.workers.clear()

            logger.info("워커 풀 종료")

    async def _worker_loop(self, worker_id: str, worker_func: Callable):
        """워커 루프"""
        logger.info(f"워커 {worker_id} 시작")

        try:
            while True:
                try:
                    # 작업 처리
                    await worker_func(worker_id)

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"워커 {worker_id} 오류: {e}")
                    self.worker_stats[worker_id]['errors'] += 1
                    await asyncio.sleep(1)

        finally:
            logger.info(f"워커 {worker_id} 종료")

    def get_worker_stats(self) -> Dict[str, Dict[str, Any]]:
        """워커 통계 반환"""
        return dict(self.worker_stats)

class AsyncBatchProcessor(Generic[T, R]):
    """비동기 배치 처리기"""

    def __init__(self, config: BatchConfig):
        self.config = config
        self.metrics = PerformanceMetrics(config.performance_window)
        self.batch_sizer = DynamicBatchSizer(config, self.metrics)
        self.backpressure = BackpressureController(config)
        self.worker_pool = WorkerPool(config)

        # 큐 관리
        self.task_queue = asyncio.PriorityQueue(maxsize=config.max_queue_size)
        self.result_queue = asyncio.Queue()
        self.batch_queue = asyncio.Queue()

        # 상태 관리
        self.is_running = False
        self.tasks_in_progress = weakref.WeakSet()
        self.completed_tasks = 0
        self.failed_tasks = 0

        # 배치 관리
        self.current_batch = []
        self.batch_timer = None
        self.batch_lock = asyncio.Lock()

    async def __aenter__(self):
        """비동기 컨텍스트 매니저 진입"""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """비동기 컨텍스트 매니저 종료"""
        await self.stop()

    async def start(self):
        """배치 처리기 시작"""
        logger.info("비동기 배치 처리기 시작")

        self.is_running = True

        # 워커 시작
        await self.worker_pool.start_workers(self._worker_process)

        # 배치 관리자 시작
        asyncio.create_task(self._batch_manager())

        # 메트릭 수집기 시작
        asyncio.create_task(self._metrics_collector())

    async def stop(self):
        """배치 처리기 종료"""
        logger.info("비동기 배치 처리기 종료")

        self.is_running = False

        # 남은 작업 처리
        await self._process_remaining_tasks()

        # 워커 종료
        await self.worker_pool.stop_workers()

        # 통계 출력
        self._print_final_stats()

    async def submit_task(self, data: T, processor: Callable[[T], R],
                         priority: TaskPriority = TaskPriority.NORMAL,
                         timeout: float = None) -> str:
        """작업 제출"""
        # 백프레셔 확인
        queue_size = self.task_queue.qsize()
        processing_rate = self.metrics.get_avg_throughput()

        if self.backpressure.check_backpressure(queue_size, processing_rate):
            delay = self.backpressure.get_backpressure_delay()
            if delay > 0:
                await asyncio.sleep(delay)

        # 작업 생성
        task = BatchTask(
            id=str(uuid.uuid4()),
            data=data,
            processor=processor,
            priority=priority,
            timeout=timeout or self.config.processing_timeout
        )

        # 큐에 추가
        try:
            await self.task_queue.put(task)
            return task.id
        except asyncio.QueueFull:
            raise Exception("작업 큐가 가득 참")

    async def submit_batch(self, batch_data: List[T], processor: Callable[[List[T]], List[R]],
                          priority: TaskPriority = TaskPriority.NORMAL) -> List[str]:
        """배치 작업 제출"""
        task_ids = []

        for data in batch_data:
            task_id = await self.submit_task(data, processor, priority)
            task_ids.append(task_id)

        return task_ids

    async def get_result(self, task_id: str, timeout: float = None) -> R:
        """작업 결과 조회"""
        timeout = timeout or self.config.processing_timeout

        try:
            # 결과 큐에서 대기
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    result = await asyncio.wait_for(
                        self.result_queue.get(),
                        timeout=min(1.0, timeout - (time.time() - start_time))
                    )

                    if result['task_id'] == task_id:
                        if result['status'] == TaskStatus.COMPLETED:
                            return result['result']
                        else:
                            raise Exception(f"작업 실패: {result['error']}")
                    else:
                        # 다른 작업의 결과면 다시 큐에 넣기
                        await self.result_queue.put(result)

                except asyncio.TimeoutError:
                    continue

            raise asyncio.TimeoutError(f"작업 {task_id} 타임아웃")

        except Exception as e:
            logger.error(f"결과 조회 실패: {e}")
            raise

    async def _batch_manager(self):
        """배치 관리자"""
        while self.is_running:
            try:
                # 배치 크기 조정
                target_batch_size = self.batch_sizer.adjust_batch_size()

                # 배치 수집
                batch = await self._collect_batch(target_batch_size)

                if batch:
                    await self.batch_queue.put(batch)

                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"배치 관리자 오류: {e}")
                await asyncio.sleep(1)

    async def _collect_batch(self, target_size: int) -> List[BatchTask]:
        """배치 수집"""
        batch = []
        batch_timeout = self.config.batch_timeout
        start_time = time.time()

        while (len(batch) < target_size and:
               time.time() - start_time < batch_timeout and:
               self.is_running):

            try:
                task = await asyncio.wait_for(
                    self.task_queue.get(),
                    timeout=min(0.1, batch_timeout - (time.time() - start_time))
                )
                batch.append(task)

            except asyncio.TimeoutError:
                break

        return batch

    async def _worker_process(self, worker_id: str):
        """워커 처리 함수"""
        try:
            # 배치 가져오기
            batch = await asyncio.wait_for(self.batch_queue.get(), timeout=1.0)

            if batch:
                await self._process_batch(batch, worker_id)

        except asyncio.TimeoutError:
            pass
        except Exception as e:
            logger.error(f"워커 {worker_id} 처리 오류: {e}")

    async def _process_batch(self, batch: List[BatchTask], worker_id: str):
        """배치 처리"""
        start_time = time.time()

        try:
            # 배치별 처리
            for task in batch:
                task.status = TaskStatus.RUNNING
                task.started_at = time.time()

                try:
                    # 작업 처리
                    if asyncio.iscoroutinefunction(task.processor):
                        result = await task.processor(task.data)
                    else:
                        result = task.processor(task.data)

                    # 결과 저장
                    task.result = result
                    task.status = TaskStatus.COMPLETED
                    task.completed_at = time.time()

                    # 결과 큐에 추가
                    await self.result_queue.put({
                        'task_id': task.id,
                        'status': task.status,
                        'result': result,
                        'error': None
                    })

                    self.completed_tasks += 1

                except Exception as e:
                    # 오류 처리
                    task.error = e
                    task.status = TaskStatus.FAILED
                    task.completed_at = time.time()

                    # 재시도 로직
                    if task.retry_count < task.max_retries:
                        task.retry_count += 1
                        task.status = TaskStatus.PENDING
                        await self.task_queue.put(task)
                        logger.warning(f"작업 재시도: {task.id} ({task.retry_count}/{task.max_retries})")
                    else:
                        await self.result_queue.put({
                            'task_id': task.id,
                            'status': task.status,
                            'result': None,
                            'error': str(e)
                        })
                        self.failed_tasks += 1
                        logger.error(f"작업 실패: {task.id} - {e}")

            # 성능 메트릭 기록
            processing_time = time.time() - start_time
            self.metrics.record_processing_time(processing_time)

            # 워커 통계 업데이트
            self.worker_pool.worker_stats[worker_id]['processed'] += len(batch)
            self.worker_pool.worker_stats[worker_id]['avg_time'] = processing_time / len(batch)

        except Exception as e:
            logger.error(f"배치 처리 실패: {e}")
            for task in batch:
                if task.status == TaskStatus.RUNNING:
                    task.status = TaskStatus.FAILED
                    task.error = e
                    self.failed_tasks += 1

    async def _metrics_collector(self):
        """메트릭 수집기"""
        while self.is_running:
            try:
                # 시스템 메트릭 수집
                self.metrics.record_system_metrics()
                self.metrics.record_queue_size(self.task_queue.qsize())

                # 처리량 계산
                current_time = time.time()
                if hasattr(self, '_last_metrics_time'):
                    time_diff = current_time - self._last_metrics_time
                    if time_diff > 0:
                        completed_diff = self.completed_tasks - getattr(self, '_last_completed', 0)
                        throughput = completed_diff / time_diff
                        self.metrics.record_throughput(throughput)

                self._last_metrics_time = current_time
                self._last_completed = self.completed_tasks

                await asyncio.sleep(5)  # 5초마다 수집

            except Exception as e:
                logger.error(f"메트릭 수집 오류: {e}")
                await asyncio.sleep(1)

    async def _process_remaining_tasks(self):
        """남은 작업 처리"""
        logger.info("남은 작업 처리 중...")

        timeout = 30.0  # 30초 타임아웃
        start_time = time.time()

        while (not self.task_queue.empty() and :
               time.time() - start_time < timeout):
            await asyncio.sleep(0.1)

        if not self.task_queue.empty():
            logger.warning(f"타임아웃으로 인해 {self.task_queue.qsize()}개 작업 미처리")

    def _print_final_stats(self):
        """최종 통계 출력"""
        total_tasks = self.completed_tasks + self.failed_tasks
        success_rate = self.completed_tasks / total_tasks if total_tasks > 0 else 0

        logger.info(f"배치 처리 완료 통계:")
        logger.info(f"  총 작업: {total_tasks}")
        logger.info(f"  성공: {self.completed_tasks}")
        logger.info(f"  실패: {self.failed_tasks}")
        logger.info(f"  성공률: {success_rate:.2%}")
        logger.info(f"  평균 처리 시간: {self.metrics.get_avg_processing_time():.3f}초")
        logger.info(f"  평균 처리량: {self.metrics.get_avg_throughput():.2f}개/초")

        # 워커 통계
        worker_stats = self.worker_pool.get_worker_stats()
        for worker_id, stats in worker_stats.items():
            logger.info(f"  {worker_id}: {stats['processed']}개 처리, 평균 {stats['avg_time']:.3f}초")

    def get_status(self) -> Dict[str, Any]:
        """현재 상태 반환"""
        return {
            'is_running': self.is_running,
            'queue_size': self.task_queue.qsize(),
            'batch_queue_size': self.batch_queue.qsize(),
            'completed_tasks': self.completed_tasks,
            'failed_tasks': self.failed_tasks,
            'current_batch_size': self.batch_sizer.current_batch_size,
            'backpressure_active': self.backpressure.backpressure_active,
            'avg_processing_time': self.metrics.get_avg_processing_time(),
            'avg_throughput': self.metrics.get_avg_throughput(),
            'system_load': self.metrics.get_system_load(),
            'worker_stats': self.worker_pool.get_worker_stats(),
        }

# 사용 예시
async def main():
    """배치 처리기 테스트"""
    try:
        # 설정
        config = BatchConfig(
            batch_size=50,
            max_batch_size=200,
            max_workers=8,
            enable_dynamic_sizing=True,
        )

        # 테스트 처리 함수
        def process_data(data: Dict[str, Any]) -> Dict[str, Any]:
            """테스트 데이터 처리"""
            time.sleep(0.1)  # 처리 시간 시뮬레이션
            return {
                'id': data['id'],
                'result': data['value'] * 2,
                'processed_at': time.time()
            }

        # 배치 처리기 시작
        async with AsyncBatchProcessor(config) as processor:

            # 테스트 데이터 생성
            test_data = [
                {'id': i, 'value': i * 10}
                for i in range(1000):
            ]:
            :
            logger.info(f"테스트 데이터 제출: {len(test_data)}개")

            # 작업 제출
            task_ids = []
            for data in test_data:
                task_id = await processor.submit_task(
                    data,
                    process_data,
                    priority=TaskPriority.NORMAL
                )
                task_ids.append(task_id)

            # 결과 수집
            results = []
            for task_id in task_ids:
                try:
                    result = await processor.get_result(task_id, timeout=30.0)
                    results.append(result)
                except Exception as e:
                    logger.error(f"결과 조회 실패: {e}")

            logger.info(f"처리 완료: {len(results)}개 결과")

            # 상태 확인
            status = processor.get_status()
            logger.info(f"최종 상태: {status}")

    except Exception as e:
        logger.error(f"배치 처리 테스트 실패: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
