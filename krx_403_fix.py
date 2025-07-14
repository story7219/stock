#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: krx_403_fix.py
목적: KRX 403 에러 해결 도구 - IP 차단, User-Agent, 요청 패턴 개선
Author: KRX 403 Fix Tool
Created: 2025-07-13
Version: 1.0.0

Features:
    - 다양한 User-Agent 로테이션
    - 요청 간격 조절
    - 프록시 지원
    - 세션 관리
    - 재시도 로직
"""

import asyncio
import aiohttp
import requests
import time
import random
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

@dataclass
class KRXConfig:
    """KRX 설정"""
    base_url: str = "http://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd"
    timeout: int = 30
    max_retries: int = 5
    retry_delay: float = 2.0
    request_interval: float = 1.0  # 요청 간격 (초)
    use_proxy: bool = False
    proxy_list: List[str] = None

class KRX403Fixer:
    """KRX 403 에러 해결 도구"""
    
    def __init__(self, config: KRXConfig = None):
        self.config = config or KRXConfig()
        
        # 다양한 User-Agent 목록
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/120.0.0.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ]
        
        # 세션 관리
        self.session = None
        self.current_user_agent = None
        self.request_count = 0
        self.last_request_time = 0
        
        # 프록시 설정
        if self.config.proxy_list is None:
            self.config.proxy_list = []
    
    async def get_session(self) -> aiohttp.ClientSession:
        """세션 생성"""
        if self.session is None:
            connector = aiohttp.TCPConnector(
                limit=10,
                limit_per_host=5,
                ttl_dns_cache=300,
                use_dns_cache=True,
                keepalive_timeout=30
            )
            
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout
            )
        
        return self.session
    
    def get_random_user_agent(self) -> str:
        """랜덤 User-Agent 반환"""
        return random.choice(self.user_agents)
    
    def get_headers(self) -> Dict[str, str]:
        """헤더 생성"""
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
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.config.request_interval:
            wait_time = self.config.request_interval - time_since_last
            await asyncio.sleep(wait_time)
        
        self.last_request_time = time.time()
    
    async def make_request_with_retry(self, params: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """재시도 로직이 포함된 요청"""
        for attempt in range(self.config.max_retries):
            try:
                await self.wait_for_request_interval()
                
                session = await self.get_session()
                headers = self.get_headers()
                
                logger.info(f"요청 시도 {attempt + 1}/{self.config.max_retries}")
                logger.info(f"User-Agent: {self.current_user_agent[:50]}...")
                
                async with session.post(
                    self.config.base_url,
                    data=params,
                    headers=headers,
                    timeout=self.config.timeout
                ) as response:
                    
                    self.request_count += 1
                    
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"✅ 요청 성공 (시도 {attempt + 1})")
                        return data
                    
                    elif response.status == 403:
                        logger.warning(f"⚠️ 403 에러 (시도 {attempt + 1}) - User-Agent 변경")
                        # User-Agent 변경
                        continue
                    
                    elif response.status == 429:
                        logger.warning(f"⚠️ 429 에러 (시도 {attempt + 1}) - 요청 간격 증가")
                        await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                        continue
                    
                    else:
                        logger.error(f"❌ HTTP {response.status} 에러 (시도 {attempt + 1})")
                        if attempt < self.config.max_retries - 1:
                            await asyncio.sleep(self.config.retry_delay)
                            continue
                        else:
                            return None
            
            except asyncio.TimeoutError:
                logger.error(f"⏰ 타임아웃 에러 (시도 {attempt + 1})")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay)
                    continue
                else:
                    return None
            
            except Exception as e:
                logger.error(f"❌ 요청 에러 (시도 {attempt + 1}): {e}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay)
                    continue
                else:
                    return None
        
        logger.error("❌ 모든 재시도 실패")
        return None
    
    async def test_krx_connection(self) -> Dict[str, Any]:
        """KRX 연결 테스트"""
        logger.info("🔍 KRX 연결 테스트 시작")
        
        # 간단한 요청으로 테스트
        test_params = {
            'bld': 'dbms/MDC/STAT/standard/MDCSTAT01501',
            'trdDd': '20250713',
            'mktId': 'STK'
        }
        
        result = await self.make_request_with_retry(test_params)
        
        if result:
            logger.info("✅ KRX 연결 성공")
            return {
                'status': 'success',
                'data': result,
                'request_count': self.request_count,
                'user_agent': self.current_user_agent
            }
        else:
            logger.error("❌ KRX 연결 실패")
            return {
                'status': 'error',
                'error': '403 Forbidden',
                'request_count': self.request_count,
                'user_agent': self.current_user_agent
            }
    
    async def collect_stock_data(self, date: str = None) -> Dict[str, Any]:
        """주식 데이터 수집"""
        if date is None:
            date = time.strftime('%Y%m%d')
        
        logger.info(f"📊 주식 데이터 수집 시작: {date}")
        
        params = {
            'bld': 'dbms/MDC/STAT/standard/MDCSTAT01501',
            'trdDd': date,
            'mktId': 'STK',
            'share': '1',
            'money': '1',
            'csvxls_isNo': 'false'
        }
        
        result = await self.make_request_with_retry(params)
        
        if result:
            logger.info(f"✅ 주식 데이터 수집 성공: {len(result.get('OutBlock_1', []))}건")
            return {
                'status': 'success',
                'data': result,
                'count': len(result.get('OutBlock_1', []))
            }
        else:
            logger.error("❌ 주식 데이터 수집 실패")
            return {
                'status': 'error',
                'error': '403 Forbidden'
            }
    
    async def collect_index_data(self, date: str = None) -> Dict[str, Any]:
        """지수 데이터 수집"""
        if date is None:
            date = time.strftime('%Y%m%d')
        
        logger.info(f"📈 지수 데이터 수집 시작: {date}")
        
        params = {
            'bld': 'dbms/MDC/STAT/standard/MDCSTAT01501',
            'trdDd': date,
            'mktId': 'IDX',
            'share': '1',
            'money': '1',
            'csvxls_isNo': 'false'
        }
        
        result = await self.make_request_with_retry(params)
        
        if result:
            logger.info(f"✅ 지수 데이터 수집 성공: {len(result.get('OutBlock_1', []))}건")
            return {
                'status': 'success',
                'data': result,
                'count': len(result.get('OutBlock_1', []))
            }
        else:
            logger.error("❌ 지수 데이터 수집 실패")
            return {
                'status': 'error',
                'error': '403 Forbidden'
            }
    
    async def close_session(self):
        """세션 종료"""
        if self.session:
            await self.session.close()
            self.session = None
    
    def get_status(self) -> Dict[str, Any]:
        """상태 정보 반환"""
        return {
            'request_count': self.request_count,
            'current_user_agent': self.current_user_agent,
            'config': {
                'timeout': self.config.timeout,
                'max_retries': self.config.max_retries,
                'retry_delay': self.config.retry_delay,
                'request_interval': self.config.request_interval
            }
        }

async def main():
    """메인 실행 함수"""
    # 설정
    config = KRXConfig(
        timeout=30,
        max_retries=5,
        retry_delay=2.0,
        request_interval=2.0  # 2초 간격으로 요청
    )
    
    fixer = KRX403Fixer(config)
    
    try:
        # 연결 테스트
        test_result = await fixer.test_krx_connection()
        print(f"연결 테스트 결과: {test_result['status']}")
        
        if test_result['status'] == 'success':
            # 주식 데이터 수집
            stock_result = await fixer.collect_stock_data()
            print(f"주식 데이터 수집: {stock_result['status']}")
            
            # 지수 데이터 수집
            index_result = await fixer.collect_index_data()
            print(f"지수 데이터 수집: {index_result['status']}")
        
        # 상태 정보 출력
        status = fixer.get_status()
        print(f"상태 정보: {status}")
        
    except Exception as e:
        logger.error(f"메인 실행 에러: {e}")
    
    finally:
        await fixer.close_session()

if __name__ == "__main__":
    asyncio.run(main()) 