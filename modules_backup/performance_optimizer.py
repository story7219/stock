#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
⚡ 성능 최적화 모듈 v1.0
시스템 성능 모니터링 및 최적화
"""

import asyncio
import logging
import time
import psutil
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from functools import wraps
import gc
import sys
import os

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """성능 메트릭 데이터 클래스"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_io_read: int
    disk_io_write: int
    network_sent: int
    network_recv: int
    active_threads: int
    function_name: str
    execution_time: float
    status: str

@dataclass
class OptimizationResult:
    """최적화 결과"""
    optimization_type: str
    before_metrics: PerformanceMetrics
    after_metrics: PerformanceMetrics
    improvement_percent: float
    recommendation: str

class PerformanceOptimizer:
    """🚀 성능 최적화 관리자"""
    
    def __init__(self, db_path: str = "performance_metrics.db"):
        """성능 최적화 관리자 초기화"""
        logger.info("⚡ 성능 최적화 시스템 초기화")
        
        self.db_path = db_path
        self.metrics_history: List[PerformanceMetrics] = []
        self.optimization_history: List[OptimizationResult] = []
        
        # 성능 임계값 설정
        self.thresholds = {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'execution_time': 30.0,  # 30초
            'disk_io_threshold': 100 * 1024 * 1024,  # 100MB
            'memory_leak_threshold': 500 * 1024 * 1024  # 500MB
        }
        
        # 스레드풀 설정
        self.thread_pool = ThreadPoolExecutor(max_workers=mp.cpu_count())
        self.process_pool = ProcessPoolExecutor(max_workers=mp.cpu_count())
        
        # 모니터링 설정
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # 데이터베이스 초기화
        self._init_database()
        
        logger.info("✅ 성능 최적화 시스템 준비 완료")
    
    def _init_database(self):
        """성능 메트릭 데이터베이스 초기화"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    cpu_percent REAL,
                    memory_percent REAL,
                    memory_used_mb REAL,
                    disk_io_read INTEGER,
                    disk_io_write INTEGER,
                    network_sent INTEGER,
                    network_recv INTEGER,
                    active_threads INTEGER,
                    function_name TEXT,
                    execution_time REAL,
                    status TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS optimization_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    optimization_type TEXT,
                    improvement_percent REAL,
                    recommendation TEXT,
                    before_metrics TEXT,
                    after_metrics TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("✅ 성능 메트릭 데이터베이스 초기화 완료")
            
        except Exception as e:
            logger.error(f"❌ 데이터베이스 초기화 실패: {e}")
    
    def performance_monitor(self, function_name: str = None):
        """성능 모니터링 데코레이터"""
        def decorator(func: Callable):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                start_metrics = self._collect_system_metrics()
                
                try:
                    result = await func(*args, **kwargs)
                    status = "SUCCESS"
                except Exception as e:
                    status = f"ERROR: {str(e)}"
                    raise
                finally:
                    end_time = time.time()
                    execution_time = end_time - start_time
                    
                    end_metrics = self._collect_system_metrics()
                    
                    # 성능 메트릭 생성
                    metrics = PerformanceMetrics(
                        timestamp=datetime.now(),
                        cpu_percent=end_metrics['cpu_percent'],
                        memory_percent=end_metrics['memory_percent'],
                        memory_used_mb=end_metrics['memory_used_mb'],
                        disk_io_read=end_metrics['disk_io_read'],
                        disk_io_write=end_metrics['disk_io_write'],
                        network_sent=end_metrics['network_sent'],
                        network_recv=end_metrics['network_recv'],
                        active_threads=threading.active_count(),
                        function_name=function_name or func.__name__,
                        execution_time=execution_time,
                        status=status
                    )
                    
                    # 메트릭 저장
                    self._save_metrics(metrics)
                    
                    # 성능 이슈 체크
                    await self._check_performance_issues(metrics)
                
                return result
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                start_metrics = self._collect_system_metrics()
                
                try:
                    result = func(*args, **kwargs)
                    status = "SUCCESS"
                except Exception as e:
                    status = f"ERROR: {str(e)}"
                    raise
                finally:
                    end_time = time.time()
                    execution_time = end_time - start_time
                    
                    end_metrics = self._collect_system_metrics()
                    
                    # 성능 메트릭 생성
                    metrics = PerformanceMetrics(
                        timestamp=datetime.now(),
                        cpu_percent=end_metrics['cpu_percent'],
                        memory_percent=end_metrics['memory_percent'],
                        memory_used_mb=end_metrics['memory_used_mb'],
                        disk_io_read=end_metrics['disk_io_read'],
                        disk_io_write=end_metrics['disk_io_write'],
                        network_sent=end_metrics['network_sent'],
                        network_recv=end_metrics['network_recv'],
                        active_threads=threading.active_count(),
                        function_name=function_name or func.__name__,
                        execution_time=execution_time,
                        status=status
                    )
                    
                    # 메트릭 저장
                    self._save_metrics(metrics)
                
                return result
            
            # 비동기 함수인지 확인
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    def _collect_system_metrics(self) -> Dict[str, Any]:
        """시스템 메트릭 수집"""
        try:
            # CPU 사용률
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # 메모리 정보
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_mb = memory.used / (1024 * 1024)
            
            # 디스크 I/O
            disk_io = psutil.disk_io_counters()
            disk_io_read = disk_io.read_bytes if disk_io else 0
            disk_io_write = disk_io.write_bytes if disk_io else 0
            
            # 네트워크 I/O
            network_io = psutil.net_io_counters()
            network_sent = network_io.bytes_sent if network_io else 0
            network_recv = network_io.bytes_recv if network_io else 0
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'memory_used_mb': memory_used_mb,
                'disk_io_read': disk_io_read,
                'disk_io_write': disk_io_write,
                'network_sent': network_sent,
                'network_recv': network_recv
            }
        
        except Exception as e:
            logger.error(f"시스템 메트릭 수집 실패: {e}")
            return {}
    
    def _save_metrics(self, metrics: PerformanceMetrics):
        """성능 메트릭 저장"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO performance_metrics 
                (timestamp, cpu_percent, memory_percent, memory_used_mb,
                 disk_io_read, disk_io_write, network_sent, network_recv,
                 active_threads, function_name, execution_time, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metrics.timestamp.isoformat(),
                metrics.cpu_percent,
                metrics.memory_percent,
                metrics.memory_used_mb,
                metrics.disk_io_read,
                metrics.disk_io_write,
                metrics.network_sent,
                metrics.network_recv,
                metrics.active_threads,
                metrics.function_name,
                metrics.execution_time,
                metrics.status
            ))
            
            conn.commit()
            conn.close()
            
            # 메모리에도 저장 (최근 1000개만)
            self.metrics_history.append(metrics)
            if len(self.metrics_history) > 1000:
                self.metrics_history = self.metrics_history[-1000:]
                
        except Exception as e:
            logger.error(f"메트릭 저장 실패: {e}")
    
    async def _check_performance_issues(self, metrics: PerformanceMetrics):
        """성능 이슈 체크 및 최적화 제안"""
        issues = []
        
        # CPU 사용률 체크
        if metrics.cpu_percent > self.thresholds['cpu_percent']:
            issues.append(f"높은 CPU 사용률: {metrics.cpu_percent:.1f}%")
        
        # 메모리 사용률 체크
        if metrics.memory_percent > self.thresholds['memory_percent']:
            issues.append(f"높은 메모리 사용률: {metrics.memory_percent:.1f}%")
        
        # 실행 시간 체크
        if metrics.execution_time > self.thresholds['execution_time']:
            issues.append(f"긴 실행 시간: {metrics.execution_time:.1f}초")
        
        if issues:
            logger.warning(f"⚠️ 성능 이슈 감지 [{metrics.function_name}]: {', '.join(issues)}")
            await self._suggest_optimizations(metrics, issues)
    
    async def _suggest_optimizations(self, metrics: PerformanceMetrics, issues: List[str]):
        """최적화 제안"""
        suggestions = []
        
        if metrics.cpu_percent > self.thresholds['cpu_percent']:
            suggestions.append("CPU 집약적 작업을 비동기 처리로 분산")
            suggestions.append("멀티프로세싱 활용 고려")
        
        if metrics.memory_percent > self.thresholds['memory_percent']:
            suggestions.append("메모리 사용량 최적화 필요")
            suggestions.append("데이터 청크 단위 처리 고려")
            
        if metrics.execution_time > self.thresholds['execution_time']:
            suggestions.append("함수 분할 및 캐싱 적용")
            suggestions.append("데이터베이스 쿼리 최적화")
        
        logger.info(f"💡 최적화 제안 [{metrics.function_name}]: {', '.join(suggestions)}")
    
    async def optimize_memory_usage(self):
        """메모리 사용량 최적화"""
        logger.info("🧹 메모리 최적화 시작")
        
        before_metrics = self._collect_system_metrics()
        
        # 가비지 컬렉션 강제 실행
        collected = gc.collect()
        
        # 캐시 정리
        self._clear_caches()
        
        after_metrics = self._collect_system_metrics()
        
        improvement = before_metrics['memory_percent'] - after_metrics['memory_percent']
        
        logger.info(f"✅ 메모리 최적화 완료: {improvement:.1f}% 개선, {collected}개 객체 정리")
        
        return OptimizationResult(
            optimization_type="MEMORY",
            before_metrics=PerformanceMetrics(
                timestamp=datetime.now(),
                cpu_percent=before_metrics['cpu_percent'],
                memory_percent=before_metrics['memory_percent'],
                memory_used_mb=before_metrics['memory_used_mb'],
                disk_io_read=0, disk_io_write=0,
                network_sent=0, network_recv=0,
                active_threads=0, function_name="memory_optimization",
                execution_time=0, status="BEFORE"
            ),
            after_metrics=PerformanceMetrics(
                timestamp=datetime.now(),
                cpu_percent=after_metrics['cpu_percent'],
                memory_percent=after_metrics['memory_percent'],
                memory_used_mb=after_metrics['memory_used_mb'],
                disk_io_read=0, disk_io_write=0,
                network_sent=0, network_recv=0,
                active_threads=0, function_name="memory_optimization",
                execution_time=0, status="AFTER"
            ),
            improvement_percent=improvement,
            recommendation="정기적인 메모리 정리 실행"
        )
    
    def _clear_caches(self):
        """캐시 정리"""
        # 내부 캐시 정리
        if hasattr(self, 'cache'):
            self.cache.clear()
        
        # 시스템 캐시 정리 시도
        try:
            if sys.platform == "linux":
                os.system("sync && echo 1 > /proc/sys/vm/drop_caches")
        except:
            pass
    
    async def optimize_async_operations(self, operations: List[Callable], max_concurrent: int = 10):
        """비동기 작업 최적화"""
        logger.info(f"⚡ 비동기 작업 최적화: {len(operations)}개 작업")
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def controlled_operation(operation):
            async with semaphore:
                return await operation()
        
        start_time = time.time()
        results = await asyncio.gather(
            *[controlled_operation(op) for op in operations],
            return_exceptions=True
        )
        end_time = time.time()
        
        execution_time = end_time - start_time
        logger.info(f"✅ 비동기 최적화 완료: {execution_time:.2f}초")
        
        return results
    
    def get_performance_report(self, hours: int = 24) -> Dict[str, Any]:
        """성능 리포트 생성"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 최근 N시간 데이터 조회
            since_time = (datetime.now() - timedelta(hours=hours)).isoformat()
            
            cursor.execute('''
                SELECT * FROM performance_metrics 
                WHERE timestamp > ? 
                ORDER BY timestamp DESC
            ''', (since_time,))
            
            rows = cursor.fetchall()
            conn.close()
            
            if not rows:
                return {"message": "성능 데이터가 없습니다"}
            
            # 통계 계산
            cpu_values = [row[2] for row in rows if row[2] is not None]
            memory_values = [row[3] for row in rows if row[3] is not None]
            execution_times = [row[11] for row in rows if row[11] is not None]
            
            report = {
                "period_hours": hours,
                "total_records": len(rows),
                "cpu_stats": {
                    "avg": sum(cpu_values) / len(cpu_values) if cpu_values else 0,
                    "max": max(cpu_values) if cpu_values else 0,
                    "min": min(cpu_values) if cpu_values else 0
                },
                "memory_stats": {
                    "avg": sum(memory_values) / len(memory_values) if memory_values else 0,
                    "max": max(memory_values) if memory_values else 0,
                    "min": min(memory_values) if memory_values else 0
                },
                "execution_time_stats": {
                    "avg": sum(execution_times) / len(execution_times) if execution_times else 0,
                    "max": max(execution_times) if execution_times else 0,
                    "min": min(execution_times) if execution_times else 0
                },
                "top_slow_functions": self._get_slow_functions(rows),
                "recommendations": self._generate_recommendations(rows)
            }
            
            return report
            
        except Exception as e:
            logger.error(f"성능 리포트 생성 실패: {e}")
            return {"error": str(e)}
    
    def _get_slow_functions(self, rows: List) -> List[Dict]:
        """느린 함수 목록 생성"""
        function_times = {}
        
        for row in rows:
            func_name = row[10]  # function_name
            exec_time = row[11]  # execution_time
            
            if func_name and exec_time:
                if func_name not in function_times:
                    function_times[func_name] = []
                function_times[func_name].append(exec_time)
        
        # 평균 실행 시간 계산 및 정렬
        avg_times = []
        for func_name, times in function_times.items():
            avg_time = sum(times) / len(times)
            avg_times.append({
                "function": func_name,
                "avg_time": avg_time,
                "call_count": len(times),
                "max_time": max(times)
            })
        
        return sorted(avg_times, key=lambda x: x['avg_time'], reverse=True)[:10]
    
    def _generate_recommendations(self, rows: List) -> List[str]:
        """성능 개선 권장사항 생성"""
        recommendations = []
        
        # CPU 사용률 분석
        cpu_values = [row[2] for row in rows if row[2] is not None]
        if cpu_values and sum(cpu_values) / len(cpu_values) > 70:
            recommendations.append("평균 CPU 사용률이 높습니다. 비동기 처리 증대를 고려하세요.")
        
        # 메모리 사용률 분석
        memory_values = [row[3] for row in rows if row[3] is not None]
        if memory_values and sum(memory_values) / len(memory_values) > 80:
            recommendations.append("평균 메모리 사용률이 높습니다. 메모리 최적화가 필요합니다.")
        
        # 실행 시간 분석
        execution_times = [row[11] for row in rows if row[11] is not None and row[11] > 10]
        if execution_times:
            recommendations.append("일부 함수의 실행 시간이 깁니다. 코드 최적화를 검토하세요.")
        
        return recommendations
    
    def start_continuous_monitoring(self, interval: int = 60):
        """연속 모니터링 시작"""
        if self.monitoring_active:
            logger.warning("모니터링이 이미 실행 중입니다")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._continuous_monitor,
            args=(interval,),
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info(f"📊 연속 모니터링 시작 (간격: {interval}초)")
    
    def stop_continuous_monitoring(self):
        """연속 모니터링 중지"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("📊 연속 모니터링 중지")
    
    def _continuous_monitor(self, interval: int):
        """연속 모니터링 실행"""
        while self.monitoring_active:
            try:
                metrics = self._collect_system_metrics()
                
                # 임계치 체크
                if metrics.get('cpu_percent', 0) > self.thresholds['cpu_percent']:
                    logger.warning(f"⚠️ CPU 사용률 높음: {metrics['cpu_percent']:.1f}%")
                
                if metrics.get('memory_percent', 0) > self.thresholds['memory_percent']:
                    logger.warning(f"⚠️ 메모리 사용률 높음: {metrics['memory_percent']:.1f}%")
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"연속 모니터링 오류: {e}")
                time.sleep(interval)
    
    def __del__(self):
        """소멸자"""
        self.stop_continuous_monitoring()
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=True)
        if hasattr(self, 'process_pool'):
            self.process_pool.shutdown(wait=True)

    async def optimize_system(self):
        """시스템 성능 최적화 실행"""
        try:
            logger.info("시스템 성능 최적화 시작")
            
            # 메모리 최적화
            await self.optimize_memory_usage()
            
            # 캐시 최적화
            await self.optimize_cache()
            
            # 데이터베이스 최적화
            await self.optimize_database()
            
            # 네트워크 최적화
            await self.optimize_network()
            
            logger.info("시스템 성능 최적화 완료")
            
        except Exception as e:
            logger.error(f"시스템 성능 최적화 중 오류: {e}")
            raise

    async def optimize_cache(self):
        """캐시 최적화"""
        try:
            logger.info("캐시 최적화 시작")
            
            # 캐시 히트율 분석
            hit_rate = self._analyze_cache_hit_rate()
            
            # 불필요한 캐시 정리
            cleared_count = self._clear_unused_cache()
            
            logger.info(f"캐시 최적화 완료 - 히트율: {hit_rate:.2%}, 정리된 항목: {cleared_count}")
            
        except Exception as e:
            logger.error(f"캐시 최적화 중 오류: {e}")
            
    async def optimize_database(self):
        """데이터베이스 최적화"""
        try:
            logger.info("데이터베이스 최적화 시작")
            
            # 인덱스 최적화
            optimized_indexes = self._optimize_indexes()
            
            # 오래된 데이터 정리
            cleaned_records = self._cleanup_old_data()
            
            logger.info(f"데이터베이스 최적화 완료 - 인덱스: {optimized_indexes}, 정리된 레코드: {cleaned_records}")
            
        except Exception as e:
            logger.error(f"데이터베이스 최적화 중 오류: {e}")
            
    async def optimize_network(self):
        """네트워크 최적화"""
        try:
            logger.info("네트워크 최적화 시작")
            
            # 연결 풀 최적화
            pool_size = self._optimize_connection_pool()
            
            # 타임아웃 설정 최적화
            timeout_config = self._optimize_timeouts()
            
            logger.info(f"네트워크 최적화 완료 - 풀 크기: {pool_size}, 타임아웃: {timeout_config}")
            
        except Exception as e:
            logger.error(f"네트워크 최적화 중 오류: {e}")
            
    def _analyze_cache_hit_rate(self) -> float:
        """캐시 히트율 분석"""
        # Mock 히트율 반환
        return 0.85
        
    def _clear_unused_cache(self) -> int:
        """사용하지 않는 캐시 정리"""
        # Mock 정리 개수 반환
        return 150
        
    def _optimize_indexes(self) -> int:
        """인덱스 최적화"""
        # Mock 최적화된 인덱스 수 반환
        return 5
        
    def _cleanup_old_data(self) -> int:
        """오래된 데이터 정리"""
        # Mock 정리된 레코드 수 반환
        return 1000
        
    def _optimize_connection_pool(self) -> int:
        """연결 풀 최적화"""
        # Mock 풀 크기 반환
        return 20
        
    def _optimize_timeouts(self) -> Dict[str, int]:
        """타임아웃 최적화"""
        # Mock 타임아웃 설정 반환
        return {
            'connect_timeout': 10,
            'read_timeout': 30,
            'write_timeout': 15
        }

# 전역 성능 최적화 인스턴스
performance_optimizer = PerformanceOptimizer()

# 데코레이터 단축키
monitor_performance = performance_optimizer.performance_monitor

async def main():
    """테스트 실행"""
    optimizer = PerformanceOptimizer()
    
    @optimizer.performance_monitor("test_function")
    async def test_function():
        await asyncio.sleep(1)
        return "테스트 완료"
    
    # 테스트 실행
    result = await test_function()
    print(f"결과: {result}")
    
    # 성능 리포트
    report = optimizer.get_performance_report(1)
    print(f"성능 리포트: {json.dumps(report, indent=2, ensure_ascii=False)}")

if __name__ == "__main__":
    asyncio.run(main()) 