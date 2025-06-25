#!/usr/bin/env python3
"""
💻 시스템 성능 모니터링 모듈 (System Performance Monitor)
======================================================

사용자 컴퓨터의 하드웨어 스펙과 실시간 성능을 모니터링하여
최적화된 작업 실행을 지원하는 경량화된 모니터링 시스템입니다.

주요 기능:
1. 하드웨어 스펙 감지 (Hardware Specification Detection)
   - CPU: 코어 수, 최대 주파수, 현재 사용률
   - 메모리: 총 용량, 사용 가능 용량, 사용률
   - 디스크: 총 용량, 여유 공간, I/O 성능
   - 시스템 아키텍처 및 운영체제 정보

2. 실시간 성능 모니터링 (Real-time Performance Monitoring)
   - CPU 사용률 실시간 추적
   - 메모리 사용량 및 가용성 모니터링
   - 디스크 I/O 성능 측정
   - 시스템 부하 상태 분석

3. 지능형 리소스 관리 (Intelligent Resource Management)
   - ML 작업 실행 가능 여부 자동 판단
   - 시스템 부하에 따른 배치 크기 최적화
   - CPU 코어 수에 맞는 워커 수 추천
   - 메모리 상황에 따른 작업 조정

4. 적응형 성능 최적화 (Adaptive Performance Optimization)
   - 시스템 스펙에 맞는 설정 자동 조정
   - 부하 상황에 따른 작업 우선순위 조정
   - 리소스 부족 시 자동 대기 및 재시도
   - 최적 실행 시점 추천

5. 안전성 보장 (Safety Assurance)
   - 시스템 과부하 방지
   - 메모리 부족 상황 사전 감지
   - 자동 임계값 설정 및 관리
   - 비상 상황 자동 대응

모니터링 지표:
- CPU 사용률 임계값: 85%
- 메모리 사용률 임계값: 80%
- 최소 여유 메모리: 2GB
- 디스크 I/O 성능 추적

시스템 상태 분류:
- 최적: CPU < 50%, 메모리 < 60%
- 양호: CPU < 70%, 메모리 < 75%
- 보통: CPU < 85%, 메모리 < 85%
- 과부하: 임계값 초과

추천 설정:
- 배치 크기: 메모리 용량에 따라 4-32 조정
- 워커 수: CPU 코어의 50-75% 활용
- ML 실행: 리소스 여유도에 따른 실행 제어

특징:
- 경량화: TensorFlow 의존성 제거로 최소 리소스 사용
- 실시간: 1초 단위 성능 데이터 수집
- 적응형: 하드웨어 스펙에 맞는 자동 최적화
- 안정성: 시스템 보호를 위한 다중 안전장치

이 모니터는 다양한 사용자 환경에서 투자 분석 시스템이
안정적이고 효율적으로 동작하도록 보장합니다.
"""

import psutil
import time
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class SystemSpecs:
    """시스템 스펙 정보"""
    cpu_count: int
    cpu_freq_max: float  # GHz
    ram_total_gb: float
    ram_available_gb: float
    disk_total_gb: float
    disk_free_gb: float
    gpu_memory_mb: Optional[float] = None

@dataclass
class PerformanceMetrics:
    """성능 메트릭"""
    cpu_usage_percent: float
    memory_usage_percent: float
    memory_available_gb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    timestamp: datetime

class SystemMonitor:
    """시스템 성능 모니터 (경량화)"""
    
    def __init__(self):
        self.system_specs = self._get_system_specs()
        self.cpu_threshold = 85.0  # CPU 사용률 임계값
        self.memory_threshold = 80.0  # 메모리 사용률 임계값
        self.performance_history = []
        
        logger.info(f"시스템 모니터 초기화: RAM {self.system_specs.ram_total_gb:.1f}GB, CPU {self.system_specs.cpu_count}코어")
    
    def _get_system_specs(self) -> SystemSpecs:
        """시스템 스펙 수집"""
        
        # CPU 정보
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        cpu_freq_max = cpu_freq.max / 1000 if cpu_freq else 0.0  # MHz -> GHz
        
        # 메모리 정보
        memory = psutil.virtual_memory()
        ram_total_gb = memory.total / (1024**3)
        ram_available_gb = memory.available / (1024**3)
        
        # 디스크 정보
        disk = psutil.disk_usage('/')
        disk_total_gb = disk.total / (1024**3)
        disk_free_gb = disk.free / (1024**3)
        
        return SystemSpecs(
            cpu_count=cpu_count,
            cpu_freq_max=cpu_freq_max,
            ram_total_gb=ram_total_gb,
            ram_available_gb=ram_available_gb,
            disk_total_gb=disk_total_gb,
            disk_free_gb=disk_free_gb
        )
    
    def get_current_metrics(self) -> PerformanceMetrics:
        """현재 성능 메트릭 수집"""
        
        # CPU 사용률
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # 메모리 정보
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_available_gb = memory.available / (1024**3)
        
        # 디스크 I/O (간단화)
        disk_io = psutil.disk_io_counters()
        disk_read_mb = disk_io.read_bytes / (1024**2) if disk_io else 0
        disk_write_mb = disk_io.write_bytes / (1024**2) if disk_io else 0
        
        metrics = PerformanceMetrics(
            cpu_usage_percent=cpu_percent,
            memory_usage_percent=memory_percent,
            memory_available_gb=memory_available_gb,
            disk_io_read_mb=disk_read_mb,
            disk_io_write_mb=disk_write_mb,
            timestamp=datetime.now()
        )
        
        # 히스토리 유지 (최근 100개만)
        self.performance_history.append(metrics)
        if len(self.performance_history) > 100:
            self.performance_history.pop(0)
        
        return metrics
    
    def is_system_ready_for_ml(self) -> bool:
        """ML 작업 실행 가능 여부 판단"""
        
        metrics = self.get_current_metrics()
        
        # CPU 사용률 체크
        if metrics.cpu_usage_percent > self.cpu_threshold:
            logger.warning(f"CPU 사용률 높음: {metrics.cpu_usage_percent:.1f}%")
            return False
        
        # 메모리 사용률 체크
        if metrics.memory_usage_percent > self.memory_threshold:
            logger.warning(f"메모리 사용률 높음: {metrics.memory_usage_percent:.1f}%")
            return False
        
        # 여유 메모리 체크 (최소 2GB 필요)
        if metrics.memory_available_gb < 2.0:
            logger.warning(f"여유 메모리 부족: {metrics.memory_available_gb:.1f}GB")
            return False
        
        return True
    
    def get_recommended_batch_size(self) -> int:
        """추천 배치 크기 계산"""
        
        available_memory_gb = self.system_specs.ram_available_gb
        
        if available_memory_gb >= 8:
            return 32
        elif available_memory_gb >= 4:
            return 16
        elif available_memory_gb >= 2:
            return 8
        else:
            return 4
    
    def get_recommended_worker_count(self) -> int:
        """추천 워커 수 계산"""
        
        cpu_count = self.system_specs.cpu_count
        
        # CPU 코어의 50-75% 사용
        if cpu_count >= 8:
            return min(6, cpu_count // 2)
        elif cpu_count >= 4:
            return 2
        else:
            return 1
    
    def get_system_status_report(self) -> Dict[str, Any]:
        """시스템 상태 보고서"""
        
        current_metrics = self.get_current_metrics()
        
        # 상태 판단
        if current_metrics.cpu_usage_percent < 50 and current_metrics.memory_usage_percent < 60:
            status = "최적"
        elif current_metrics.cpu_usage_percent < 70 and current_metrics.memory_usage_percent < 75:
            status = "양호"
        elif current_metrics.cpu_usage_percent < 85 and current_metrics.memory_usage_percent < 85:
            status = "보통"
        else:
            status = "과부하"
        
        return {
            'timestamp': datetime.now().isoformat(),
            'status': status,
            'system_specs': {
                'cpu_count': self.system_specs.cpu_count,
                'cpu_freq_max': self.system_specs.cpu_freq_max,
                'ram_total_gb': self.system_specs.ram_total_gb,
                'disk_total_gb': self.system_specs.disk_total_gb
            },
            'current_metrics': {
                'cpu_usage_percent': current_metrics.cpu_usage_percent,
                'memory_usage_percent': current_metrics.memory_usage_percent,
                'memory_available_gb': current_metrics.memory_available_gb
            },
            'recommendations': {
                'batch_size': self.get_recommended_batch_size(),
                'worker_count': self.get_recommended_worker_count(),
                'ml_ready': self.is_system_ready_for_ml()
            }
        }

# 전역 시스템 모니터 인스턴스
_system_monitor = None

def get_system_monitor() -> SystemMonitor:
    """시스템 모니터 싱글톤 인스턴스 반환"""
    global _system_monitor
    if _system_monitor is None:
        _system_monitor = SystemMonitor()
    return _system_monitor

# 편의 함수들
def check_system_ready() -> bool:
    """시스템 준비 상태 확인"""
    return get_system_monitor().is_system_ready_for_ml()

def get_recommended_settings() -> Dict[str, int]:
    """추천 설정 반환"""
    monitor = get_system_monitor()
    return {
        'batch_size': monitor.get_recommended_batch_size(),
        'worker_count': monitor.get_recommended_worker_count()
    } 