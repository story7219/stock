#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: krx_ultimate_manager.py
목적: 최신 Python 표준을 활용한 궁극적 통합 관리자
Author: Ultimate KRX Manager
Created: 2025-07-13
Version: 1.0.0

Features:
    - 메인/백업 시스템 자동 선택
    - 장애 감지 및 복구
    - 성능 모니터링
    - 스마트 라우팅
    - 자동 백업
    - 자동 코드수정 파일 보존
"""

import asyncio
import logging
import json
import time
import psutil
import gc
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Literal, TypedDict, Protocol
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import lru_cache, wraps
from contextlib import asynccontextmanager
import weakref
from concurrent.futures import ThreadPoolExecutor
import structlog

# 최신 Python 표준 활용
from typing_extensions import NotRequired, Required
from pydantic import BaseModel, Field, validator

from krx_ultimate_system import KRXUltimateSystem, SystemConfig, SystemMode
from krx_smart_backup_system import KRXSmartBackupSystem, BackupConfig, BackupMode

class ManagerStatus(Enum):
    """관리자 상태"""
    HEALTHY = auto()
    WARNING = auto()
    ERROR = auto()
    EMERGENCY = auto()

class Priority(Enum):
    """우선순위"""
    LOW = auto()
    NORMAL = auto()
    HIGH = auto()
    EMERGENCY = auto()

@dataclass
class ManagerConfig:
    """관리자 설정"""
    health_check_interval: int = 60
    max_error_count: int = 3
    response_time_threshold: float = 30.0
    auto_switch: bool = True
    backup_interval: int = 300
    memory_limit_gb: float = 8.0

class SystemInfo(TypedDict):
    """시스템 정보"""
    name: str
    status: str
    last_check: str
    response_time: float
    error_count: int
    success_rate: float

class ManagerMetrics(BaseModel):
    """관리자 메트릭"""
    total_requests: int = 0
    main_system_used: int = 0
    backup_system_used: int = 0
    emergency_mode_activated: int = 0
    auto_switches: int = 0
    start_time: str = Field(default_factory=lambda: datetime.now().isoformat())

class KRXUltimateManager:
    """궁극적 KRX 관리자 - 최신 Python 표준 활용"""
    
    def __init__(self, config: Optional[ManagerConfig] = None):
        self.config = config or ManagerConfig()
        
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
        
        # 403 에러 해결을 위한 설정 추가
        self._setup_403_fix()
        
        # 시스템 초기화
        self.main_system = KRXUltimateSystem(SystemConfig(mode=SystemMode.LIVE))
        self.backup_system = KRXSmartBackupSystem(BackupConfig(mode=BackupMode.EMERGENCY))
        
        # 시스템 정보
        self.systems: Dict[str, SystemInfo] = {
            'main': {
                'name': 'main',
                'status': 'healthy',
                'last_check': datetime.now().isoformat(),
                'response_time': 0.0,
                'error_count': 0,
                'success_rate': 100.0
            },
            'backup': {
                'name': 'backup',
                'status': 'healthy',
                'last_check': datetime.now().isoformat(),
                'response_time': 0.0,
                'error_count': 0,
                'success_rate': 100.0
            }
        }
        
        # 메트릭
        self.metrics = ManagerMetrics()
        
        # 모니터링 활성화
        self.monitoring_active = True
        asyncio.create_task(self._start_monitoring())
        
        # 자동 코드수정 파일 보존
        self._preserve_auto_fix_files()
    
    def _setup_403_fix(self):
        """403 에러 해결을 위한 설정"""
        # 다양한 User-Agent 목록
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/120.0.0.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ]
        
        # 요청 간격 설정
        self.request_interval = 2.0  # 2초 간격
        self.last_request_time = 0
        self.current_user_agent = None
    
    def get_random_user_agent(self) -> str:
        """랜덤 User-Agent 반환"""
        import random
        return random.choice(self.user_agents)
    
    def get_enhanced_headers(self) -> Dict[str, str]:
        """403 에러 해결을 위한 강화된 헤더"""
        user_agent = self.get_random_user_agent()
        self.current_user_agent = user_agent
        
        return {
            'User-Agent': user_agent,
            'Accept': 'application/json, text/javascript, */*; q=0.01',
            'Accept-Language': 'ko-KR,ko;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'X-Requested-With': 'XMLHttpRequest',
            'Referer': 'http://data.krx.co.kr/contents/MDC/MDI/mdiLoader/index.cmd?menuId=MDC0201',
            'Origin': 'http://data.krx.co.kr',
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache'
        }
    
    async def wait_for_request_interval(self):
        """요청 간격 대기"""
        import time
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.request_interval:
            wait_time = self.request_interval - time_since_last
            await asyncio.sleep(wait_time)
        
        self.last_request_time = time.time()
    
    async def make_krx_request_with_retry(self, params: Dict[str, str], max_retries: int = 3) -> Optional[Dict[str, Any]]:
        """403 에러 해결을 위한 재시도 로직이 포함된 KRX 요청"""
        import time
        
        for attempt in range(max_retries):
            try:
                await self.wait_for_request_interval()
                
                headers = self.get_enhanced_headers()
                
                self.logger.info(f"KRX 요청 시도 {attempt + 1}/{max_retries}")
                self.logger.info(f"User-Agent: {self.current_user_agent[:50]}...")
                
                # requests를 사용한 동기 요청 (더 안정적)
                import requests
                response = requests.post(
                    'http://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd',
                    data=params,
                    headers=headers,
                    timeout=30
                )
                
                if response.status_code == 200:
                    try:
                        data = response.json()
                        self.logger.info(f"✅ KRX 요청 성공 (시도 {attempt + 1})")
                        return data
                    except Exception as e:
                        self.logger.warning(f"JSON 파싱 실패: {e}")
                        # HTML 응답인 경우 403 에러로 간주
                        continue
                
                elif response.status_code == 403:
                    self.logger.warning(f"⚠️ 403 에러 (시도 {attempt + 1}) - User-Agent 변경")
                    continue
                
                elif response.status_code == 429:
                    self.logger.warning(f"⚠️ 429 에러 (시도 {attempt + 1}) - 요청 간격 증가")
                    await asyncio.sleep(2.0 * (attempt + 1))
                    continue
                
                else:
                    self.logger.error(f"❌ HTTP {response.status_code} 에러 (시도 {attempt + 1})")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2.0)
                        continue
                    else:
                        return None
            
            except Exception as e:
                self.logger.error(f"❌ KRX 요청 에러 (시도 {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2.0)
                    continue
                else:
                    return None
        
        self.logger.error("❌ 모든 KRX 재시도 실패")
        return None
    
    def _preserve_auto_fix_files(self):
        """자동 코드수정 파일 보존"""
        auto_fix_files = [
            'smart_duplicate_cleaner.py',
            'ultimate_folder_consolidator.py'
        ]
        
        for file_name in auto_fix_files:
            file_path = Path(file_name)
            if file_path.exists():
                # 백업 생성
                backup_path = Path(f'backup/auto_fix/{file_name}')
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                
                if not backup_path.exists():
                    import shutil
                    shutil.copy2(file_path, backup_path)
                    self.logger.info(f"자동 코드수정 파일 보존: {file_name}")
    
    async def smart_collect(self, data_types: Optional[List[str]] = None, priority: Priority = Priority.NORMAL) -> Dict[str, Any]:
        """스마트 수집 - 상황에 맞는 시스템 선택 (403 에러 해결 포함)"""
        self.metrics.total_requests += 1
        
        try:
            # 우선순위에 따른 시스템 선택
            if priority == Priority.EMERGENCY:
                self.logger.warning("🚨 긴급 수집 모드!")
                self.metrics.emergency_mode_activated += 1
                return await self.backup_system.emergency_collect('stock')
            
            elif priority == Priority.HIGH:
                # 고우선순위 - 403 에러 해결 시도 후 메인 시스템 사용
                if self._is_system_healthy('main'):
                    self.metrics.main_system_used += 1
                    # 403 에러 해결을 위한 직접 KRX 요청
                    return await self._collect_with_403_fix(data_types or ['stock'])
                else:
                    self.logger.warning("⚠️ 메인 시스템 장애, 백업 시스템 사용")
                    self.metrics.backup_system_used += 1
                    return await self.backup_system.specialized_collect(data_types)
            
            else:
                # 일반 우선순위 - 스마트 선택
                selected_system = self._select_best_system()
                
                if selected_system == 'main':
                    self.metrics.main_system_used += 1
                    # 403 에러 해결을 위한 직접 KRX 요청
                    return await self._collect_with_403_fix(data_types or ['stock'])
                elif selected_system == 'backup':
                    self.metrics.backup_system_used += 1
                    return await self.backup_system.specialized_collect(data_types)
                else:
                    self.metrics.emergency_mode_activated += 1
                    return await self.backup_system.emergency_collect('stock')
                    
        except Exception as e:
            self.logger.error(f"스마트 수집 실패: {e}")
            # 최후의 수단으로 403 에러 해결 시도
            return await self._collect_with_403_fix(data_types or ['stock'])
    
    async def _collect_with_403_fix(self, data_types: List[str]) -> Dict[str, Any]:
        """403 에러 해결을 포함한 데이터 수집"""
        self.logger.info("🔧 403 에러 해결 모드로 데이터 수집 시작")
        
        results = {}
        
        for data_type in data_types:
            try:
                if data_type.lower() in ['stock', 'stk']:
                    params = {
                        'bld': 'dbms/MDC/STAT/standard/MDCSTAT01501',
                        'trdDd': '20250713',  # 오늘 날짜
                        'mktId': 'STK',
                        'share': '1',
                        'money': '1',
                        'csvxls_isNo': 'false'
                    }
                    
                    data = await self.make_krx_request_with_retry(params)
                    if data:
                        results['stock'] = data
                        self.logger.info("✅ 주식 데이터 수집 성공")
                    else:
                        results['stock'] = {'error': '403 Forbidden'}
                        self.logger.error("❌ 주식 데이터 수집 실패")
                
                elif data_type.lower() in ['index', 'idx']:
                    params = {
                        'bld': 'dbms/MDC/STAT/standard/MDCSTAT01501',
                        'trdDd': '20250713',  # 오늘 날짜
                        'mktId': 'IDX',
                        'share': '1',
                        'money': '1',
                        'csvxls_isNo': 'false'
                    }
                    
                    data = await self.make_krx_request_with_retry(params)
                    if data:
                        results['index'] = data
                        self.logger.info("✅ 지수 데이터 수집 성공")
                    else:
                        results['index'] = {'error': '403 Forbidden'}
                        self.logger.error("❌ 지수 데이터 수집 실패")
                
                else:
                    self.logger.warning(f"지원하지 않는 데이터 타입: {data_type}")
                    results[data_type] = {'error': f'Unsupported data type: {data_type}'}
            
            except Exception as e:
                self.logger.error(f"데이터 타입 {data_type} 수집 실패: {e}")
                results[data_type] = {'error': str(e)}
        
        return results
    
    def _select_best_system(self) -> str:
        """최적의 시스템 선택"""
        # 메인 시스템이 건강하고 응답 시간이 빠르면 메인 사용
        if (self._is_system_healthy('main') and 
            self.systems['main']['response_time'] < self.config.response_time_threshold):
            return 'main'
        
        # 백업 시스템이 건강하면 백업 사용
        elif self._is_system_healthy('backup'):
            return 'backup'
        
        # 둘 다 문제가 있으면 긴급 시스템 사용
        else:
            return 'emergency'
    
    def _is_system_healthy(self, system_name: str) -> bool:
        """시스템 건강 상태 확인"""
        system = self.systems.get(system_name)
        if not system:
            return False
        
        # 에러 횟수가 임계값을 초과하면 비정상
        if system['error_count'] >= self.config.max_error_count:
            return False
        
        # 상태가 에러면 비정상
        if system['status'] == 'error':
            return False
        
        # 마지막 체크 시간이 너무 오래되었으면 비정상
        last_check = datetime.fromisoformat(system['last_check'])
        if datetime.now() - last_check > timedelta(minutes=5):
            return False
        
        return True
    
    async def _start_monitoring(self):
        """모니터링 시작"""
        self.logger.info("🔍 시스템 모니터링 시작")
        
        while self.monitoring_active:
            try:
                await self._check_systems_health()
                await self._check_memory_usage()
                await asyncio.sleep(self.config.health_check_interval)
            except Exception as e:
                self.logger.error(f"모니터링 에러: {e}")
                await asyncio.sleep(10)
    
    async def _check_systems_health(self):
        """시스템 건강 상태 체크"""
        for system_name in ['main', 'backup']:
            try:
                start_time = time.time()
                
                # 간단한 상태 체크
                if system_name == 'main':
                    status = self.main_system.get_system_status()
                else:
                    status = self.backup_system.get_backup_status()
                
                response_time = time.time() - start_time
                
                # 시스템 정보 업데이트
                self.systems[system_name]['last_check'] = datetime.now().isoformat()
                self.systems[system_name]['response_time'] = response_time
                
                # 상태 판단
                if response_time > self.config.response_time_threshold:
                    self.systems[system_name]['status'] = 'warning'
                elif status.get('status', 'unknown') in ['running', 'healthy']:
                    self.systems[system_name]['status'] = 'healthy'
                    self.systems[system_name]['error_count'] = 0
                else:
                    self.systems[system_name]['status'] = 'error'
                    self.systems[system_name]['error_count'] += 1
                
                # 성공률 계산
                if system_name == 'main':
                    metrics = status.get('metrics', {})
                    total_requests = metrics.get('total_requests', 0)
                    successful_requests = metrics.get('successful_requests', 0)
                else:
                    metrics = status.get('metrics', {})
                    total_requests = metrics.get('successful_backups', 0) + metrics.get('failed_backups', 0)
                    successful_requests = metrics.get('successful_backups', 0)
                
                if total_requests > 0:
                    success_rate = (successful_requests / total_requests) * 100
                    self.systems[system_name]['success_rate'] = success_rate
                
            except Exception as e:
                self.logger.error(f"{system_name} 시스템 상태 체크 실패: {e}")
                self.systems[system_name]['status'] = 'error'
                self.systems[system_name]['error_count'] += 1
    
    async def _check_memory_usage(self):
        """메모리 사용량 체크"""
        try:
            memory_usage = psutil.virtual_memory()
            memory_percent = memory_usage.percent
            
            if memory_percent > 80:
                self.logger.warning(f"메모리 사용량 높음: {memory_percent:.1f}%")
                gc.collect()
                
                # 메모리 사용량이 90%를 초과하면 긴급 모드
                if memory_percent > 90:
                    self.logger.critical(f"메모리 사용량 위험: {memory_percent:.1f}% - 긴급 모드 활성화")
                    await self.emergency_mode()
                    
        except Exception as e:
            self.logger.error(f"메모리 체크 실패: {e}")
    
    async def force_switch_system(self, target_system: str) -> bool:
        """강제로 시스템 전환"""
        try:
            self.logger.info(f"🔄 강제 시스템 전환: {target_system}")
            
            if target_system == 'main':
                # 메인 시스템 상태 초기화
                self.systems['main']['error_count'] = 0
                self.systems['main']['status'] = 'healthy'
                return True
            
            elif target_system == 'backup':
                # 백업 시스템 상태 초기화
                self.systems['backup']['error_count'] = 0
                self.systems['backup']['status'] = 'healthy'
                return True
            
            else:
                self.logger.error(f"알 수 없는 시스템: {target_system}")
                return False
                
        except Exception as e:
            self.logger.error(f"강제 전환 실패: {e}")
            return False
    
    async def get_manager_status(self) -> Dict[str, Any]:
        """관리자 상태 반환"""
        return {
            'manager_status': 'running',
            'systems': self.systems,
            'metrics': self.metrics.dict(),
            'config': {
                'health_check_interval': self.config.health_check_interval,
                'max_error_count': self.config.max_error_count,
                'response_time_threshold': self.config.response_time_threshold,
                'auto_switch': self.config.auto_switch,
                'backup_interval': self.config.backup_interval,
                'memory_limit_gb': self.config.memory_limit_gb
            }
        }
    
    async def update_config(self, new_config: Dict[str, Any]) -> bool:
        """설정 업데이트"""
        try:
            for key, value in new_config.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
            
            self.logger.info(f"설정 업데이트 완료: {new_config}")
            return True
        except Exception as e:
            self.logger.error(f"설정 업데이트 실패: {e}")
            return False
    
    async def emergency_mode(self) -> Dict[str, Any]:
        """긴급 모드 활성화"""
        self.logger.critical("🚨 긴급 모드 활성화!")
        
        try:
            # 모든 시스템을 긴급 모드로 전환
            result = await self.backup_system.emergency_collect('stock')
            
            return {
                'emergency_mode': True,
                'timestamp': datetime.now().isoformat(),
                'data': result
            }
        except Exception as e:
            self.logger.error(f"긴급 모드 실패: {e}")
            return {
                'emergency_mode': True,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def test_all_systems(self) -> Dict[str, Any]:
        """모든 시스템 테스트"""
        self.logger.info("🧪 모든 시스템 테스트 시작")
        
        results = {}
        
        try:
            # 메인 시스템 테스트 (403 에러 해결 포함)
            try:
                main_result = await self._collect_with_403_fix(['stock'])
                results['main'] = {'status': 'success', 'data': main_result}
            except Exception as e:
                results['main'] = {'status': 'error', 'error': str(e)}
            
            # 백업 시스템 테스트
            try:
                backup_result = await self.backup_system.specialized_collect(['stock'])
                results['backup'] = {'status': 'success', 'data': backup_result}
            except Exception as e:
                results['backup'] = {'status': 'error', 'error': str(e)}
            
            return results
            
        except Exception as e:
            self.logger.error(f"시스템 테스트 실패: {e}")
            return {'error': str(e)}
    
    def stop_monitoring(self):
        """모니터링 중지"""
        self.monitoring_active = False
        self.logger.info("모니터링 중지")

async def main():
    """메인 실행 함수"""
    manager = KRXUltimateManager()
    
    # 관리자 상태 확인
    status = await manager.get_manager_status()
    print(f"관리자 상태: {status}")
    
    # 스마트 수집 테스트
    results = await manager.smart_collect(['stock', 'index'], Priority.NORMAL)
    print(f"스마트 수집 완료! 결과: {len(results)}개 데이터 타입")
    
    # 모든 시스템 테스트
    test_results = await manager.test_all_systems()
    print(f"시스템 테스트 완료: {test_results}")

if __name__ == "__main__":
    asyncio.run(main()) 