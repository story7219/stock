"""
API 호출 속도 및 횟수 제한을 관리하는 모듈
"""
import asyncio
import time
from collections import deque
from typing import Dict, Deque

class APILimiter:
    """API 종류별 호출 속도를 제어합니다."""

    def __init__(self, limits: Dict[str, Dict[str, int]]):
        """
        호출 제한 설정을 사용하여 초기화합니다.
        
        Args:
            limits (Dict): API 그룹별 초당 호출 제한.
                           예: {'market_data': {'per_sec': 5}, 'order': {'per_sec': 10}}
        """
        self.limits = limits
        self.timestamps: Dict[str, Deque[float]] = {
            group: deque() for group in limits
        }
        self.locks: Dict[str, asyncio.Lock] = {
            group: asyncio.Lock() for group in limits
        }

    async def wait_for_slot(self, api_group: str):
        """
        지정된 API 그룹의 다음 호출이 가능할 때까지 대기합니다.
        
        Args:
            api_group (str): 확인할 API 그룹 이름 (예: 'market_data')
        
        Raises:
            ValueError: 설정되지 않은 API 그룹일 경우
        """
        if api_group not in self.limits:
            raise ValueError(f"'{api_group}'에 대한 API 제한이 설정되지 않았습니다.")

        limit_per_sec = self.limits[api_group]['per_sec']
        
        async with self.locks[api_group]:
            while True:
                now = time.time()
                
                # 1초가 지난 타임스탬프 제거
                while self.timestamps[api_group] and self.timestamps[api_group][0] <= now - 1:
                    self.timestamps[api_group].popleft()
                
                # 현재 윈도우 내 호출 횟수 확인
                if len(self.timestamps[api_group]) < limit_per_sec:
                    self.timestamps[api_group].append(now)
                    break
                
                # 다음 슬롯까지 대기
                sleep_duration = (self.timestamps[api_group][0] + 1) - now
                await asyncio.sleep(sleep_duration) 