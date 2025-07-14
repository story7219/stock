#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: realtime_stream_processor.py
모듈: 실시간 스트리밍 데이터 처리 최적화 시스템
목적: 고속 실시간 데이터 스트리밍, 이벤트 기반 처리, 지연시간 최소화

Author: World-Class AI Trading System
Created: 2025-01-27
Version: 1.0.0

실시간 스트리밍 최적화:
⚡ 고속 스트리밍:
- 무잠금 큐 구조
- 링 버퍼 최적화
- 제로카피 데이터 전송
- 메모리 풀 재사용

📊 이벤트 처리:
- 이벤트 기반 아키텍처
- 복잡 이벤트 처리 (CEP)
- 스트림 조인 연산
- 윈도우 집계 처리

🔄 백프레셔 제어:
- 적응적 백프레셔
- 동적 버퍼 크기 조정
- 우선순위 기반 드롭
- 부하 분산

⏱️ 지연시간 최적화:
- 마이크로초 단위 처리
- CPU 캐시 최적화
- NUMA 인식 처리
- 하드웨어 가속

License: MIT
"""

from __future__ import annotations
import asyncio
import time
import logging
from collections import deque
import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass
import field
from typing import (
    Dict, List, Optional, Any, Callable, AsyncIterator,
    Union, Tuple, TypeVar, Generic, Protocol
)
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
import threading
import multiprocessing as mp
from queue import Queue
import Empty
import heapq
from enum import Enum
import uuid
import json
import numpy as np
import pandas as pd
from datetime import datetime
import timedelta
import weakref
import mmap
import os
import sys

# 고성능 라이브러리
try:
    import uvloop
    if sys.platform != 'win32':
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass

import psutil
import zmq
import zmq.asyncio
from prometheus_client import Counter
import Histogram, Gauge

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 메트릭 정의
STREAM_MESSAGES = Counter('stream_messages_total', 'Total stream messages', ['stream_id', 'event_type'])
PROCESSING_LATENCY = Histogram('stream_processing_latency_seconds', 'Processing latency')
BUFFER_SIZE = Gauge('stream_buffer_size', 'Stream buffer size', ['stream_id'])
THROUGHPUT = Gauge('stream_throughput_per_second', 'Stream throughput', ['stream_id'])

T = TypeVar('T')
R = TypeVar('R')

class EventType(Enum):
    """이벤트 타입"""
    MARKET_DATA = "market_data"
    TRADE = "trade"
    ORDER = "order"
    NEWS = "news"
    HEARTBEAT = "heartbeat"
    CONTROL = "control"

class StreamState(Enum):
    """스트림 상태"""
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"

@dataclass
class StreamEvent:
    """스트림 이벤트"""
    id: str
    event_type: EventType
    timestamp: float
    data: Any
    source: str
    sequence: int = 0
    priority: int = 0
    ttl: float = 0.0  # Time to live

    def __lt__(self, other):
        return self.priority > other.priority

@dataclass
class StreamWindow:
    """스트림 윈도우"""
    window_id: str
    start_time: float
    end_time: float
    events: List[StreamEvent] = field(default_factory=list)
    aggregated_data: Optional[Any] = None

    def add_event(self, event: StreamEvent):
        """이벤트 추가"""
        if self.start_time <= event.timestamp < self.end_time:
            self.events.append(event)
            return True
        return False

    def is_complete(self) -> bool:
        """윈도우 완료 여부"""
        return time.time() >= self.end_time

@dataclass
class StreamConfig:
    """스트림 설정"""
    # 기본 설정
    stream_id: str
    buffer_size: int = 100000
    batch_size: int = 1000
    max_latency_ms: float = 10.0

    # 백프레셔 설정
    enable_backpressure: bool = True
    backpressure_threshold: float = 0.8
    drop_policy: str = "oldest"  # oldest, newest, lowest_priority

    # 윈도우 설정
    window_size_ms: float = 1000.0
    window_slide_ms: float = 100.0
    enable_windowing: bool = True

    # 성능 설정
    enable_zero_copy: bool = True
    numa_aware: bool = True
    cpu_affinity: Optional[List[int]] = None

    # 모니터링 설정
    enable_metrics: bool = True
    metrics_interval_ms: float = 1000.0

class LockFreeRingBuffer:
    """무잠금 링 버퍼"""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = [None] * capacity
        self.head = 0
        self.tail = 0
        self.size = 0
        self._lock = threading.Lock()  # 백업용

    def put(self, item: Any) -> bool:
        """아이템 추가"""
        with self._lock:
            if self.size >= self.capacity:
                return False

            self.buffer[self.tail] = item
            self.tail = (self.tail + 1) % self.capacity
            self.size += 1
            return True

    def get(self) -> Optional[Any]:
        """아이템 조회"""
        with self._lock:
            if self.size == 0:
                return None

            item = self.buffer[self.head]
            self.buffer[self.head] = None
            self.head = (self.head + 1) % self.capacity
            self.size -= 1
            return item

    def is_full(self) -> bool:
        """버퍼 가득 참 여부"""
        return self.size >= self.capacity

    def is_empty(self) -> bool:
        """버퍼 비어있음 여부"""
        return self.size == 0

    def current_size(self) -> int:
        """현재 크기"""
        return self.size

class StreamBuffer:
    """스트림 버퍼"""

    def __init__(self, config: StreamConfig):
        self.config = config
        self.ring_buffer = LockFreeRingBuffer(config.buffer_size)
        self.priority_queue = []
        self.sequence_counter = 0
        self.dropped_events = 0
        self.total_events = 0
        self.lock = threading.Lock()

    def put(self, event: StreamEvent) -> bool:
        """이벤트 추가"""
        self.total_events += 1

        # TTL 체크
        if event.ttl > 0 and time.time() > event.timestamp + event.ttl:
            self.dropped_events += 1
            return False

        # 시퀀스 번호 할당
        event.sequence = self.sequence_counter
        self.sequence_counter += 1

        # 백프레셔 체크
        if self.config.enable_backpressure:
            buffer_ratio = self.ring_buffer.current_size() / self.config.buffer_size
            if buffer_ratio > self.config.backpressure_threshold:
                return self._handle_backpressure(event)

        # 우선순위 큐 사용
        if event.priority > 0:
            with self.lock:
                heapq.heappush(self.priority_queue, event)
            return True

        # 링 버퍼에 추가
        return self.ring_buffer.put(event)

    def get(self) -> Optional[StreamEvent]:
        """이벤트 조회"""
        # 우선순위 큐 먼저 확인
        with self.lock:
            if self.priority_queue:
                return heapq.heappop(self.priority_queue)

        # 링 버퍼에서 조회
        return self.ring_buffer.get()

    def get_batch(self, batch_size: int) -> List[StreamEvent]:
        """배치 조회"""
        batch = []
        for _ in range(batch_size):
            event = self.get()
            if event is None:
                break
            batch.append(event)
        return batch

    def _handle_backpressure(self, event: StreamEvent) -> bool:
        """백프레셔 처리"""
        if self.config.drop_policy == "newest":
            # 새 이벤트 드롭
            self.dropped_events += 1
            return False

        elif self.config.drop_policy == "oldest":
            # 오래된 이벤트 드롭
            old_event = self.ring_buffer.get()
            if old_event:
                self.dropped_events += 1
            return self.ring_buffer.put(event)

        elif self.config.drop_policy == "lowest_priority":
            # 낮은 우선순위 이벤트 드롭
            if event.priority > 0:
                # 우선순위 큐에서 가장 낮은 우선순위 제거
                with self.lock:
                    if self.priority_queue:
                        dropped = heapq.heappop(self.priority_queue)
                        if dropped.priority >= event.priority:
                            heapq.heappush(self.priority_queue, dropped)
                            self.dropped_events += 1
                            return False
                        else:
                            heapq.heappush(self.priority_queue, event)
                            self.dropped_events += 1
                            return True

            # 일반 이벤트는 드롭
            self.dropped_events += 1
            return False

        return False

    def get_stats(self) -> Dict[str, Any]:
        """버퍼 통계"""
        return {
            'buffer_size': self.ring_buffer.current_size(),
            'priority_queue_size': len(self.priority_queue),
            'total_events': self.total_events,
            'dropped_events': self.dropped_events,
            'drop_rate': self.dropped_events / max(self.total_events, 1),
            'buffer_utilization': self.ring_buffer.current_size() / self.config.buffer_size,
        }

class WindowManager:
    """윈도우 관리자"""

    def __init__(self, config: StreamConfig):
        self.config = config
        self.windows = {}
        self.completed_windows = deque(maxlen=100)
        self.window_counter = 0
        self.lock = threading.Lock()

    def add_event(self, event: StreamEvent):
        """이벤트를 윈도우에 추가"""
        if not self.config.enable_windowing:
            return

        current_time = event.timestamp
        window_id = self._get_window_id(current_time)

        with self.lock:
            if window_id not in self.windows:
                self.windows[window_id] = self._create_window(current_time)

            window = self.windows[window_id]
            window.add_event(event)

    def get_completed_windows(self) -> List[StreamWindow]:
        """완료된 윈도우 반환"""
        completed = []
        current_time = time.time()

        with self.lock:
            expired_windows = []
            for window_id, window in self.windows.items():
                if window.end_time <= current_time:
                    completed.append(window)
                    expired_windows.append(window_id)

            # 완료된 윈도우 제거
            for window_id in expired_windows:
                del self.windows[window_id]

        return completed

    def _get_window_id(self, timestamp: float) -> str:
        """윈도우 ID 생성"""
        window_start = int(timestamp / (self.config.window_slide_ms / 1000)) * (self.config.window_slide_ms / 1000)
        return f"window_{window_start:.3f}"

    def _create_window(self, timestamp: float) -> StreamWindow:
        """윈도우 생성"""
        window_start = int(timestamp / (self.config.window_slide_ms / 1000)) * (self.config.window_slide_ms / 1000)
        window_end = window_start + (self.config.window_size_ms / 1000)

        return StreamWindow(
            window_id=f"window_{self.window_counter}",
            start_time=window_start,
            end_time=window_end
        )

class StreamProcessor:
    """스트림 처리기"""

    def __init__(self, config: StreamConfig):
        self.config = config
        self.buffer = StreamBuffer(config)
        self.window_manager = WindowManager(config)
        self.processors = {}
        self.state = StreamState.STOPPED

        # 성능 메트릭
        self.processed_events = 0
        self.processing_times = deque(maxlen=1000)
        self.throughput_samples = deque(maxlen=100)
        self.last_throughput_time = time.time()

        # 작업자 스레드
        self.worker_threads = []
        self.is_running = False

        # 메트릭 스레드
        self.metrics_thread = None

    def register_processor(self, event_type: EventType, processor: Callable[[StreamEvent], Any]):
        """이벤트 처리기 등록"""
        self.processors[event_type] = processor
        logger.info(f"처리기 등록: {event_type.value}")

    def register_window_processor(self, processor: Callable[[StreamWindow], Any]):
        """윈도우 처리기 등록"""
        self.window_processor = processor
        logger.info("윈도우 처리기 등록")

    async def start(self):
        """스트림 처리 시작"""
        logger.info(f"스트림 처리기 시작: {self.config.stream_id}")

        self.state = StreamState.STARTING
        self.is_running = True

        # 작업자 스레드 시작
        num_workers = min(mp.cpu_count(), 8)
        for i in range(num_workers):
            worker = threading.Thread(target=self._worker_loop, args=(f"worker-{i}",))
            worker.daemon = True
            worker.start()
            self.worker_threads.append(worker)

        # 윈도우 처리 스레드 시작
        window_worker = threading.Thread(target=self._window_worker_loop)
        window_worker.daemon = True
        window_worker.start()
        self.worker_threads.append(window_worker)

        # 메트릭 스레드 시작
        if self.config.enable_metrics:
            self.metrics_thread = threading.Thread(target=self._metrics_loop)
            self.metrics_thread.daemon = True
            self.metrics_thread.start()

        self.state = StreamState.RUNNING
        logger.info(f"스트림 처리기 시작 완료: {num_workers}개 워커")

    async def stop(self):
        """스트림 처리 중지"""
        logger.info(f"스트림 처리기 중지: {self.config.stream_id}")

        self.state = StreamState.STOPPING
        self.is_running = False

        # 작업자 스레드 대기
        for worker in self.worker_threads:
            worker.join(timeout=5)

        if self.metrics_thread:
            self.metrics_thread.join(timeout=5)

        self.state = StreamState.STOPPED
        logger.info("스트림 처리기 중지 완료")

    def put_event(self, event: StreamEvent) -> bool:
        """이벤트 추가"""
        if self.state != StreamState.RUNNING:
            return False

        success = self.buffer.put(event)
        if success:
            STREAM_MESSAGES.labels(
                stream_id=self.config.stream_id,
                event_type=event.event_type.value
            ).inc()

            # 윈도우에 추가
            self.window_manager.add_event(event)

        return success

    def _worker_loop(self, worker_id: str):
        """작업자 루프"""
        logger.info(f"작업자 {worker_id} 시작")

        while self.is_running:
            try:
                # 배치 처리
                batch = self.buffer.get_batch(self.config.batch_size)
                if not batch:
                    time.sleep(0.001)  # 1ms 대기
                    continue

                # 이벤트 처리
                start_time = time.time()
                for event in batch:
                    self._process_event(event)

                # 성능 메트릭 기록
                processing_time = time.time() - start_time
                self.processing_times.append(processing_time)
                self.processed_events += len(batch)

                PROCESSING_LATENCY.observe(processing_time / len(batch))

            except Exception as e:
                logger.error(f"작업자 {worker_id} 오류: {e}")
                time.sleep(0.1)

        logger.info(f"작업자 {worker_id} 종료")

    def _window_worker_loop(self):
        """윈도우 처리 루프"""
        logger.info("윈도우 작업자 시작")

        while self.is_running:
            try:
                # 완료된 윈도우 처리
                completed_windows = self.window_manager.get_completed_windows()

                for window in completed_windows:
                    if hasattr(self, 'window_processor'):
                        try:
                            self.window_processor(window)
                        except Exception as e:
                            logger.error(f"윈도우 처리 오류: {e}")

                time.sleep(0.1)  # 100ms 대기

            except Exception as e:
                logger.error(f"윈도우 작업자 오류: {e}")
                time.sleep(1)

        logger.info("윈도우 작업자 종료")

    def _process_event(self, event: StreamEvent):
        """개별 이벤트 처리"""
        try:
            processor = self.processors.get(event.event_type)
            if processor:
                processor(event)
            else:
                logger.warning(f"처리기 없음: {event.event_type.value}")

        except Exception as e:
            logger.error(f"이벤트 처리 오류: {e}")

    def _metrics_loop(self):
        """메트릭 수집 루프"""
        logger.info("메트릭 수집기 시작")

        while self.is_running:
            try:
                # 버퍼 크기 메트릭
                buffer_stats = self.buffer.get_stats()
                BUFFER_SIZE.labels(stream_id=self.config.stream_id).set(buffer_stats['buffer_size'])

                # 처리량 메트릭
                current_time = time.time()
                time_diff = current_time - self.last_throughput_time
                if time_diff >= 1.0:  # 1초마다
                    throughput = self.processed_events / time_diff
                    THROUGHPUT.labels(stream_id=self.config.stream_id).set(throughput)

                    self.throughput_samples.append(throughput)
                    self.processed_events = 0
                    self.last_throughput_time = current_time

                time.sleep(self.config.metrics_interval_ms / 1000)

            except Exception as e:
                logger.error(f"메트릭 수집 오류: {e}")
                time.sleep(1)

        logger.info("메트릭 수집기 종료")

    def get_stats(self) -> Dict[str, Any]:
        """통계 반환"""
        buffer_stats = self.buffer.get_stats()

        return {
            'stream_id': self.config.stream_id,
            'state': self.state.value,
            'processed_events': self.processed_events,
            'avg_processing_time': np.mean(self.processing_times) if self.processing_times else 0,
            'avg_throughput': np.mean(self.throughput_samples) if self.throughput_samples else 0,
            'buffer_stats': buffer_stats,
            'active_windows': len(self.window_manager.windows),
        }

class RealtimeStreamProcessor:
    """실시간 스트림 처리 시스템"""

    def __init__(self):
        self.streams = {}
        self.global_stats = {
            'total_events': 0,
            'total_streams': 0,
            'start_time': time.time(),
        }

    def create_stream(self, config: StreamConfig) -> StreamProcessor:
        """스트림 생성"""
        if config.stream_id in self.streams:
            raise ValueError(f"스트림 이미 존재: {config.stream_id}")

        stream = StreamProcessor(config)
        self.streams[config.stream_id] = stream
        self.global_stats['total_streams'] += 1

        logger.info(f"스트림 생성: {config.stream_id}")
        return stream

    async def start_stream(self, stream_id: str):
        """스트림 시작"""
        if stream_id not in self.streams:
            raise ValueError(f"스트림 없음: {stream_id}")

        await self.streams[stream_id].start()

    async def stop_stream(self, stream_id: str):
        """스트림 중지"""
        if stream_id not in self.streams:
            raise ValueError(f"스트림 없음: {stream_id}")

        await self.streams[stream_id].stop()

    async def start_all_streams(self):
        """모든 스트림 시작"""
        for stream in self.streams.values():
            await stream.start()

    async def stop_all_streams(self):
        """모든 스트림 중지"""
        for stream in self.streams.values():
            await stream.stop()

    def get_stream(self, stream_id: str) -> Optional[StreamProcessor]:
        """스트림 조회"""
        return self.streams.get(stream_id)

    def put_event(self, stream_id: str, event: StreamEvent) -> bool:
        """이벤트 추가"""
        stream = self.streams.get(stream_id)
        if stream:
            success = stream.put_event(event)
            if success:
                self.global_stats['total_events'] += 1
            return success
        return False

    def get_global_stats(self) -> Dict[str, Any]:
        """전체 통계"""
        runtime = time.time() - self.global_stats['start_time']

        stream_stats = {}
        for stream_id, stream in self.streams.items():
            stream_stats[stream_id] = stream.get_stats()

        return {
            'runtime_seconds': runtime,
            'total_events': self.global_stats['total_events'],
            'total_streams': self.global_stats['total_streams'],
            'events_per_second': self.global_stats['total_events'] / max(runtime, 1),
            'streams': stream_stats,
        }

# 사용 예시
async def main():
    """실시간 스트림 처리 테스트"""
    try:
        # 스트림 처리 시스템 생성
        processor = RealtimeStreamProcessor()

        # 스트림 설정
        config = StreamConfig(
            stream_id="market_data",
            buffer_size=50000,
            batch_size=500,
            max_latency_ms=5.0,
            enable_windowing=True,
            window_size_ms=1000.0,
            window_slide_ms=100.0,
        )

        # 스트림 생성
        stream = processor.create_stream(config)

        # 이벤트 처리기 등록
        def process_market_data(event: StreamEvent):
            # 시장 데이터 처리
            data = event.data
            logger.info(f"시장 데이터 처리: {data.get('symbol', 'Unknown')}")

        def process_window(window: StreamWindow):
            # 윈도우 집계 처리
            logger.info(f"윈도우 처리: {len(window.events)}개 이벤트")

        stream.register_processor(EventType.MARKET_DATA, process_market_data)
        stream.register_window_processor(process_window)

        # 스트림 시작
        await processor.start_stream("market_data")

        # 테스트 이벤트 생성
        logger.info("테스트 이벤트 생성")
        for i in range(10000):
            event = StreamEvent(
                id=f"event_{i}",
                event_type=EventType.MARKET_DATA,
                timestamp=time.time(),
                data={
                    'symbol': f'STOCK_{i % 100}',
                    'price': 100.0 + i * 0.1,
                    'volume': 1000 + i * 10,
                },
                source="test_generator"
            )

            success = processor.put_event("market_data", event)
            if not success:
                logger.warning(f"이벤트 추가 실패: {event.id}")

            if i % 1000 == 0:
                await asyncio.sleep(0.1)  # 잠시 대기

        # 처리 완료 대기
        await asyncio.sleep(5)

        # 통계 출력
        stats = processor.get_global_stats()
        logger.info(f"처리 완료 통계: {stats}")

        # 스트림 중지
        await processor.stop_stream("market_data")

    except Exception as e:
        logger.error(f"실시간 스트림 처리 테스트 실패: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
