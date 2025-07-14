#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: find_krx_recent_data.py
목적: 최근 날짜부터 시작해서 KRX에서 실제로 데이터를 받을 수 있는 날짜 찾기
"""

import asyncio
import aiohttp
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import time

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KRXRecentDataFinder:
    """KRX 최근 데이터 날짜 찾기 클래스"""
    
    def __init__(self):
        self.base_url = "http://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd"
        self.session = None
        
    async def __aenter__(self):
        """비동기 컨텍스트 매니저 진입"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            connector=aiohttp.TCPConnector(limit=10, limit_per_host=5)
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """비동기 컨텍스트 매니저 종료"""
        if self.session:
            await self.session.close()
    
    async def test_date_for_data(self, date_str: str, data_type: str = "stock") -> Tuple[bool, Dict]:
        """특정 날짜에 데이터가 있는지 테스트"""
        try:
            if data_type == "stock":
                params = {
                    'bld': 'dbms/MDC/STAT/standard/MDCSTAT01501',
                    'mktId': 'STK',
                    'trdDd': date_str,
                    'share': '1',
                    'money': '1',
                    'csvxls_isNo': 'false'
                }
            elif data_type == "etf":
                params = {
                    'bld': 'dbms/MDC/STAT/standard/MDCSTAT01501',
                    'mktId': 'ETF',
                    'trdDd': date_str,
                    'share': '1',
                    'money': '1',
                    'csvxls_isNo': 'false'
                }
            elif data_type == "index":
                params = {
                    'bld': 'dbms/MDC/STAT/standard/MDCSTAT01501',
                    'mktId': 'IDX',
                    'trdDd': date_str,
                    'share': '1',
                    'money': '1',
                    'csvxls_isNo': 'false'
                }
            else:
                return False, {}
            
            async with self.session.post(self.base_url, data=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('OutBlock_1') and len(data['OutBlock_1']) > 0:
                        return True, data
                    else:
                        return False, data
                else:
                    return False, {}
                    
        except Exception as e:
            logger.error(f"날짜 {date_str} 테스트 중 오류: {e}")
            return False, {}
    
    async def find_recent_data(self, days_back: int = 30) -> Dict:
        """최근 N일부터 시작해서 데이터가 있는 날짜 찾기"""
        logger.info(f"최근 {days_back}일부터 데이터 검색 시작...")
        
        current_date = datetime.now()
        found_data = []
        
        for i in range(days_back):
            test_date = current_date - timedelta(days=i)
            date_str = test_date.strftime("%Y%m%d")
            
            logger.info(f"테스트 날짜: {test_date.strftime('%Y-%m-%d')}")
            
            for data_type in ["stock", "etf", "index"]:
                has_data, data = await self.test_date_for_data(date_str, data_type)
                
                if has_data:
                    logger.info(f"✅ {test_date.strftime('%Y-%m-%d')} {data_type} 데이터 발견!")
                    found_data.append({
                        'date': test_date,
                        'date_str': date_str,
                        'data_type': data_type,
                        'data_count': len(data.get('OutBlock_1', [])),
                        'sample_data': data
                    })
                    break
            
            await asyncio.sleep(0.5)  # API 제한 방지
        
        return found_data
    
    async def find_earliest_from_recent(self, start_date: datetime, days_to_search: int = 365) -> Dict:
        """특정 날짜부터 과거로 검색해서 가장 오래된 데이터 찾기"""
        logger.info(f"{start_date.strftime('%Y-%m-%d')}부터 {days_to_search}일 전까지 검색...")
        
        earliest_found = None
        
        for i in range(days_to_search):
            test_date = start_date - timedelta(days=i)
            date_str = test_date.strftime("%Y%m%d")
            
            if i % 10 == 0:  # 10일마다 로그 출력
                logger.info(f"검색 진행: {test_date.strftime('%Y-%m-%d')}")
            
            for data_type in ["stock", "etf", "index"]:
                has_data, data = await self.test_date_for_data(date_str, data_type)
                
                if has_data:
                    logger.info(f"✅ {test_date.strftime('%Y-%m-%d')} {data_type} 데이터 발견!")
                    if earliest_found is None or test_date < earliest_found['date']:
                        earliest_found = {
                            'date': test_date,
                            'date_str': date_str,
                            'data_type': data_type,
                            'data_count': len(data.get('OutBlock_1', [])),
                            'sample_data': data
                        }
                    break
            
            await asyncio.sleep(0.3)  # API 제한 방지
        
        return earliest_found

async def main():
    """메인 함수"""
    logger.info("KRX 최근 데이터 날짜 찾기 시작...")
    
    async with KRXRecentDataFinder() as finder:
        # 1단계: 최근 30일 데이터 검색
        logger.info("=== 1단계: 최근 30일 데이터 검색 ===")
        recent_data = await finder.find_recent_data(30)
        
        if recent_data:
            logger.info(f"최근 데이터 발견: {len(recent_data)}개")
            
            # 가장 오래된 데이터 찾기
            earliest_recent = min(recent_data, key=lambda x: x['date'])
            logger.info(f"최근 데이터 중 가장 오래된 것: {earliest_recent['date'].strftime('%Y-%m-%d')} ({earliest_recent['data_type']})")
            
            # 2단계: 해당 날짜부터 과거로 더 검색
            logger.info("=== 2단계: 과거 데이터 상세 검색 ===")
            earliest_data = await finder.find_earliest_from_recent(earliest_recent['date'], 365)
            
            if earliest_data:
                logger.info(f"🎉 최종 결과: {earliest_data['date'].strftime('%Y-%m-%d')} ({earliest_data['data_type']}) - {earliest_data['data_count']}개 데이터")
                
                # 결과 저장
                result_data = {
                    'earliest_date': earliest_data['date'].strftime('%Y-%m-%d'),
                    'earliest_date_str': earliest_data['date_str'],
                    'data_type': earliest_data['data_type'],
                    'data_count': earliest_data['data_count'],
                    'recent_data_found': len(recent_data),
                    'search_completed_at': datetime.now().isoformat()
                }
                
                with open('krx_earliest_data_result.json', 'w', encoding='utf-8') as f:
                    json.dump(result_data, f, ensure_ascii=False, indent=2)
                
                logger.info("결과가 krx_earliest_data_result.json에 저장되었습니다.")
                return earliest_data
            else:
                logger.warning("과거 데이터 검색에서 더 오래된 데이터를 찾을 수 없습니다.")
                return earliest_recent
        else:
            logger.warning("최근 30일 데이터를 찾을 수 없습니다.")
    
    return None

if __name__ == "__main__":
    asyncio.run(main()) 