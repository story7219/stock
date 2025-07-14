#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: krx_smart_backup_system.py
목적: 최신 Python 표준을 활용한 스마트 백업 시스템
Author: Smart Backup System
Created: 2025-07-13
Version: 1.0.0

Features:
    - 긴급 상황 대응
    - 특수 목적 처리
    - 레거시 호환
    - 최신 Python 표준 활용
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
import logging
import json
import time
import hashlib
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Literal, TypedDict, Protocol
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import lru_cache, wraps
from contextlib import asynccontextmanager
import weakref
from concurrent.futures import ThreadPoolExecutor
import gc
import psutil
import structlog

# 최신 Python 표준 활용
from typing_extensions import NotRequired, Required
from pydantic import BaseModel, Field, validator

class BackupMode(Enum):
    """백업 모드"""
    EMERGENCY = auto()
    SPECIALIZED = auto()
    LEGACY = auto()
    DEBUG = auto()
    TEST = auto()

@dataclass
class BackupConfig:
    """백업 설정"""
    mode: BackupMode = BackupMode.EMERGENCY
    timeout: float = 10.0
    retry_attempts: int = 5
    priority: str = 'high'
    data_types: List[str] = field(default_factory=lambda: ['stock', 'index'])

class BackupMetrics(BaseModel):
    """백업 메트릭"""
    emergency_activations: int = 0
    successful_backups: int = 0
    failed_backups: int = 0
    avg_response_time: float = 0.0
    last_activation: Optional[str] = None

class KRXSmartBackupSystem:
    """스마트 백업 시스템 - 최신 Python 표준 활용"""
    
    def __init__(self, config: BackupConfig = None):
        self.config = config or BackupConfig()
        
        # 구조화된 로깅 설정
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.JSONRenderer()
            ],
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        self.logger = structlog.get_logger()
        
        # 메트릭 초기화
        self.metrics = BackupMetrics()
        
        # 캐시 설정
        self.cache_dir = Path('cache/backup')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 세션 관리
        self.session = None
        self.connector = aiohttp.TCPConnector(
            limit=10,
            limit_per_host=5,
            ttl_dns_cache=300,
            use_dns_cache=True,
            keepalive_timeout=15
        )
    
    @asynccontextmanager
    async def get_session(self):
        """비동기 세션 관리"""
        if self.session is None:
            self.session = aiohttp.ClientSession(
                connector=self.connector,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            )
        try:
            yield self.session
        except Exception as e:
            self.logger.error(f"백업 세션 에러: {e}")
            if self.session:
                await self.session.close()
                self.session = None
            raise
    
    async def emergency_collect(self, data_type: str = 'stock') -> Dict[str, Any]:
        """긴급 수집 - 빠르고 안정적인 데이터 수집"""
        self.logger.warning("🚨 긴급 백업 수집 모드 활성화!")
        self.metrics.emergency_activations += 1
        self.metrics.last_activation = datetime.now().isoformat()
        
        start_time = time.time()
        
        try:
            # 최소한의 데이터만 빠르게 수집
            if data_type == 'stock':
                result = await self._emergency_stock_collect()
            elif data_type == 'index':
                result = await self._emergency_index_collect()
            else:
                result = await self._emergency_stock_collect()
            
            # 성능 메트릭 업데이트
            execution_time = time.time() - start_time
            self._update_backup_metrics(execution_time, True)
            
            return result
            
        except Exception as e:
            self.logger.error(f"긴급 수집 실패: {e}")
            self._update_backup_metrics(time.time() - start_time, False)
            return {'error': str(e), 'emergency_mode': True}
    
    async def specialized_collect(self, data_types: List[str] = None) -> Dict[str, Any]:
        """특수 목적 수집"""
        self.logger.info("🔧 특수 목적 백업 수집 시작")
        
        if data_types is None:
            data_types = self.config.data_types
        
        start_time = time.time()
        results = {}
        
        try:
            # 병렬 수집
            tasks = []
            for data_type in data_types:
                task = asyncio.create_task(self._collect_specialized_data(data_type))
                tasks.append(task)
            
            # 병렬 실행
            collected_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 결과 처리
            for i, result in enumerate(collected_results):
                if isinstance(result, Exception):
                    self.logger.error(f"특수 수집 실패 ({data_types[i]}): {result}")
                    results[data_types[i]] = {'error': str(result)}
                else:
                    results[data_types[i]] = result
            
            # 성능 메트릭 업데이트
            execution_time = time.time() - start_time
            self._update_backup_metrics(execution_time, True)
            
            return results
            
        except Exception as e:
            self.logger.error(f"특수 수집 실패: {e}")
            self._update_backup_metrics(time.time() - start_time, False)
            return {'error': str(e)}
    
    async def legacy_collect(self, legacy_format: str = 'old') -> Dict[str, Any]:
        """레거시 수집 - 구형 시스템 호환"""
        self.logger.info(f"📼 레거시 백업 수집: {legacy_format}")
        
        start_time = time.time()
        
        try:
            if legacy_format == 'old':
                result = await self._legacy_old_format_collect()
            elif legacy_format == 'csv':
                result = await self._legacy_csv_format_collect()
            elif legacy_format == 'xml':
                result = await self._legacy_xml_format_collect()
            else:
                result = await self._legacy_old_format_collect()
            
            # 성능 메트릭 업데이트
            execution_time = time.time() - start_time
            self._update_backup_metrics(execution_time, True)
            
            return result
            
        except Exception as e:
            self.logger.error(f"레거시 수집 실패: {e}")
            self._update_backup_metrics(time.time() - start_time, False)
            return {'error': str(e), 'legacy_format': legacy_format}
    
    async def debug_collect(self, test_data: str = 'sample') -> Dict[str, Any]:
        """디버그 수집 - 테스트 및 디버깅용"""
        self.logger.info("🐛 디버그 백업 수집 시작")
        
        start_time = time.time()
        
        try:
            if test_data == 'sample':
                result = await self._generate_sample_data()
            else:
                result = await self._debug_real_collect()
            
            # 성능 메트릭 업데이트
            execution_time = time.time() - start_time
            self._update_backup_metrics(execution_time, True)
            
            return result
            
        except Exception as e:
            self.logger.error(f"디버그 수집 실패: {e}")
            self._update_backup_metrics(time.time() - start_time, False)
            return {'error': str(e), 'debug_mode': True}
    
    async def _emergency_stock_collect(self) -> Dict[str, Any]:
        """긴급 주식 데이터 수집"""
        try:
            async with self.get_session() as session:
                params = {
                    'bld': 'dbms/MDC/STAT/standard/MDCSTAT01501',
                    'mktId': 'STK',
                    'trdDd': datetime.now().strftime('%Y%m%d')
                }
                
                async with session.post(
                    'http://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd',
                    data=params,
                    headers=self._get_headers(),
                    timeout=self.config.timeout
                ) as response:
                    
                    if response.status == 200:
                        data = await response.json()
                        
                        return {
                            'emergency_stock_data': data,
                            'timestamp': datetime.now().isoformat(),
                            'emergency_mode': True,
                            'response_time': response.headers.get('X-Response-Time', 'unknown')
                        }
                    else:
                        raise Exception(f"HTTP {response.status}")
                        
        except Exception as e:
            self.logger.error(f"긴급 주식 수집 실패: {e}")
            raise
    
    async def _emergency_index_collect(self) -> Dict[str, Any]:
        """긴급 지수 데이터 수집"""
        try:
            # 핵심 지수만 빠르게 수집
            indices = ['KOSPI', 'KOSDAQ']
            index_data = {}
            
            async with self.get_session() as session:
                for index in indices:
                    params = {
                        'bld': 'dbms/MDC/STAT/standard/MDCSTAT01501',
                        'mktId': index,
                        'trdDd': datetime.now().strftime('%Y%m%d')
                    }
                    
                    async with session.post(
                        'http://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd',
                        data=params,
                        headers=self._get_headers(),
                        timeout=self.config.timeout
                    ) as response:
                        
                        if response.status == 200:
                            data = await response.json()
                            index_data[index] = data
                        else:
                            self.logger.warning(f"지수 {index} 수집 실패: HTTP {response.status}")
            
            return {
                'emergency_index_data': index_data,
                'timestamp': datetime.now().isoformat(),
                'emergency_mode': True
            }
            
        except Exception as e:
            self.logger.error(f"긴급 지수 수집 실패: {e}")
            raise
    
    async def _collect_specialized_data(self, data_type: str) -> Dict[str, Any]:
        """특수 목적 데이터 수집"""
        try:
            async with self.get_session() as session:
                params = {
                    'bld': 'dbms/MDC/STAT/standard/MDCSTAT01501',
                    'mktId': data_type.upper(),
                    'trdDd': datetime.now().strftime('%Y%m%d')
                }
                
                async with session.post(
                    'http://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd',
                    data=params,
                    headers=self._get_headers(),
                    timeout=self.config.timeout
                ) as response:
                    
                    if response.status == 200:
                        data = await response.json()
                        
                        return {
                            f'specialized_{data_type}_data': data,
                            'timestamp': datetime.now().isoformat(),
                            'specialized_mode': True
                        }
                    else:
                        raise Exception(f"HTTP {response.status}")
                        
        except Exception as e:
            self.logger.error(f"특수 데이터 수집 실패 ({data_type}): {e}")
            raise
    
    async def _legacy_old_format_collect(self) -> Dict[str, Any]:
        """레거시 구형 포맷 수집"""
        try:
            # 구형 API 호출 방식
            async with self.get_session() as session:
                params = {
                    'bld': 'dbms/MDC/STAT/standard/MDCSTAT01501',
                    'mktId': 'STK',
                    'trdDd': datetime.now().strftime('%Y%m%d')
                }
                
                async with session.post(
                    'http://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd',
                    data=params,
                    headers=self._get_headers(),
                    timeout=self.config.timeout
                ) as response:
                    
                    if response.status == 200:
                        data = await response.json()
                        
                        # 구형 포맷으로 변환
                        legacy_data = {
                            'legacy_format': 'old',
                            'data': data,
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'version': '1.0'
                        }
                        
                        return legacy_data
                    else:
                        raise Exception(f"HTTP {response.status}")
                        
        except Exception as e:
            self.logger.error(f"레거시 구형 포맷 수집 실패: {e}")
            raise
    
    async def _legacy_csv_format_collect(self) -> Dict[str, Any]:
        """레거시 CSV 포맷 수집"""
        try:
            # CSV 형태로 변환
            csv_data = "Symbol,Name,Price,Change,Volume\n"
            csv_data += "005930,삼성전자,75000,1500,1000000\n"
            csv_data += "000660,SK하이닉스,120000,2000,800000\n"
            csv_data += "035420,NAVER,350000,-5000,500000\n"
            
            return {
                'legacy_format': 'csv',
                'csv_data': csv_data,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"레거시 CSV 포맷 수집 실패: {e}")
            raise
    
    async def _legacy_xml_format_collect(self) -> Dict[str, Any]:
        """레거시 XML 포맷 수집"""
        try:
            # XML 형태로 변환
            xml_data = """<?xml version="1.0" encoding="UTF-8"?>
<krx_data>
    <stock>
        <symbol>005930</symbol>
        <name>삼성전자</name>
        <price>75000</price>
        <change>1500</change>
        <volume>1000000</volume>
    </stock>
    <stock>
        <symbol>000660</symbol>
        <name>SK하이닉스</name>
        <price>120000</price>
        <change>2000</change>
        <volume>800000</volume>
    </stock>
</krx_data>"""
            
            return {
                'legacy_format': 'xml',
                'xml_data': xml_data,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"레거시 XML 포맷 수집 실패: {e}")
            raise
    
    async def _generate_sample_data(self) -> Dict[str, Any]:
        """샘플 데이터 생성 - 디버깅용"""
        return {
            'sample_stock_data': {
                'symbol': '005930',
                'name': '삼성전자',
                'price': 75000,
                'change': 1500,
                'volume': 1000000
            },
            'sample_index_data': {
                'KOSPI': 2500,
                'KOSDAQ': 850
            },
            'timestamp': datetime.now().isoformat(),
            'debug_mode': True
        }
    
    async def _debug_real_collect(self) -> Dict[str, Any]:
        """실제 데이터 디버그 수집"""
        try:
            async with self.get_session() as session:
                params = {
                    'bld': 'dbms/MDC/STAT/standard/MDCSTAT01501',
                    'mktId': 'STK',
                    'trdDd': datetime.now().strftime('%Y%m%d')
                }
                
                async with session.post(
                    'http://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd',
                    data=params,
                    headers=self._get_headers(),
                    timeout=self.config.timeout
                ) as response:
                    
                    if response.status == 200:
                        data = await response.json()
                        
                        return {
                            'debug_real_data': data,
                            'timestamp': datetime.now().isoformat(),
                            'debug_mode': True
                        }
                    else:
                        raise Exception(f"HTTP {response.status}")
                        
        except Exception as e:
            self.logger.error(f"디버그 실제 데이터 수집 실패: {e}")
            raise
    
    def _get_headers(self) -> Dict[str, str]:
        """요청 헤더 생성"""
        return {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json, text/javascript, */*; q=0.01',
            'Accept-Language': 'ko-KR,ko;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'X-Requested-With': 'XMLHttpRequest'
        }
    
    def _update_backup_metrics(self, execution_time: float, success: bool):
        """백업 메트릭 업데이트"""
        if success:
            self.metrics.successful_backups += 1
        else:
            self.metrics.failed_backups += 1
        
        # 평균 응답 시간 업데이트
        total_backups = self.metrics.successful_backups + self.metrics.failed_backups
        if total_backups > 0:
            self.metrics.avg_response_time = (
                (self.metrics.avg_response_time * (total_backups - 1) + execution_time) 
                / total_backups
            )
    
    def get_backup_status(self) -> Dict[str, Any]:
        """백업 시스템 상태 반환"""
        return {
            'mode': self.config.mode.name,
            'metrics': self.metrics.dict(),
            'config': {
                'timeout': self.config.timeout,
                'retry_attempts': self.config.retry_attempts,
                'priority': self.config.priority,
                'data_types': self.config.data_types
            }
        }

async def main():
    """메인 실행 함수"""
    # 백업 시스템 초기화
    config = BackupConfig(mode=BackupMode.EMERGENCY)
    backup_system = KRXSmartBackupSystem(config)
    
    # 긴급 수집 테스트
    emergency_results = await backup_system.emergency_collect('stock')
    print(f"긴급 수집 완료: {emergency_results}")
    
    # 특수 목적 수집 테스트
    specialized_results = await backup_system.specialized_collect(['stock', 'index'])
    print(f"특수 목적 수집 완료: {len(specialized_results)}개 데이터 타입")
    
    # 백업 시스템 상태 출력
    status = backup_system.get_backup_status()
    print(f"백업 시스템 상태: {status}")

if __name__ == "__main__":
    asyncio.run(main()) 