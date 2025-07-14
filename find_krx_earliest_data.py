#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: find_krx_earliest_data.py
목적: KRX에서 데이터를 받을 수 있는 가장 오래된 날짜 찾기
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

class KRXDataFinder:
    """KRX 데이터 최초 날짜 찾기 클래스"""
    
    def __init__(self):
        self.base_url = "http://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd"
        self.session = None
        self.cache = {}
        
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
    
    async def binary_search_earliest_date(self, start_year: int = 1990, end_year: int = 2024) -> Dict:
        """이진 탐색으로 가장 오래된 데이터 날짜 찾기"""
        logger.info(f"{start_year}년부터 {end_year}년까지 이진 탐색 시작...")
        
        earliest_found = None
        data_types = ["stock", "etf", "index"]
        
        # 연도별로 테스트
        for year in range(start_year, end_year + 1):
            logger.info(f"{year}년 테스트 중...")
            
            # 각 연도의 1월 1일부터 테스트
            test_date = datetime(year, 1, 1)
            
            for data_type in data_types:
                date_str = test_date.strftime("%Y%m%d")
                has_data, data = await self.test_date_for_data(date_str, data_type)
                
                if has_data:
                    logger.info(f"✅ {year}년 {data_type} 데이터 발견: {date_str}")
                    if earliest_found is None or test_date < earliest_found['date']:
                        earliest_found = {
                            'date': test_date,
                            'date_str': date_str,
                            'data_type': data_type,
                            'sample_data': data
                        }
                    break
            
            # 1초 대기 (API 제한 방지)
            await asyncio.sleep(1)
        
        return earliest_found
    
    async def find_earliest_by_month(self, start_year: int = 1990) -> Dict:
        """월별로 더 정확한 최초 날짜 찾기"""
        logger.info("월별 상세 검색 시작...")
        
        earliest_found = None
        
        for year in range(start_year, 2025):
            for month in range(1, 13):
                # 각 월의 1일부터 테스트
                test_date = datetime(year, month, 1)
                date_str = test_date.strftime("%Y%m%d")
                
                for data_type in ["stock", "etf", "index"]:
                    has_data, data = await self.test_date_for_data(date_str, data_type)
                    
                    if has_data:
                        logger.info(f"✅ {year}년 {month}월 {data_type} 데이터 발견: {date_str}")
                        if earliest_found is None or test_date < earliest_found['date']:
                            earliest_found = {
                                'date': test_date,
                                'date_str': date_str,
                                'data_type': data_type,
                                'sample_data': data
                            }
                        break
                
                # 0.5초 대기
                await asyncio.sleep(0.5)
        
        return earliest_found
    
    async def find_earliest_by_day(self, start_date: datetime) -> Dict:
        """일별로 정확한 최초 날짜 찾기"""
        logger.info(f"{start_date.strftime('%Y-%m-%d')}부터 일별 상세 검색 시작...")
        
        current_date = start_date
        end_date = datetime(2024, 12, 31)
        
        while current_date <= end_date:
            date_str = current_date.strftime("%Y%m%d")
            
            for data_type in ["stock", "etf", "index"]:
                has_data, data = await self.test_date_for_data(date_str, data_type)
                
                if has_data:
                    logger.info(f"✅ 최초 데이터 발견: {current_date.strftime('%Y-%m-%d')} ({data_type})")
                    return {
                        'date': current_date,
                        'date_str': date_str,
                        'data_type': data_type,
                        'sample_data': data
                    }
            
            current_date += timedelta(days=1)
            await asyncio.sleep(0.2)  # API 제한 방지
        
        return None

async def main():
    """메인 함수"""
    logger.info("KRX 최초 데이터 날짜 찾기 시작...")
    
    async with KRXDataFinder() as finder:
        # 1단계: 연도별 이진 탐색
        logger.info("=== 1단계: 연도별 탐색 ===")
        year_result = await finder.binary_search_earliest_date(1990, 2024)
        
        if year_result:
            logger.info(f"연도별 탐색 결과: {year_result['date'].strftime('%Y-%m-%d')} ({year_result['data_type']})")
            
            # 2단계: 해당 연도의 월별 탐색
            logger.info("=== 2단계: 월별 상세 탐색 ===")
            month_result = await finder.find_earliest_by_month(year_result['date'].year)
            
            if month_result:
                logger.info(f"월별 탐색 결과: {month_result['date'].strftime('%Y-%m-%d')} ({month_result['data_type']})")
                
                # 3단계: 해당 월의 일별 탐색
                logger.info("=== 3단계: 일별 상세 탐색 ===")
                day_result = await finder.find_earliest_by_day(month_result['date'])
                
                if day_result:
                    logger.info(f"🎉 최종 결과: {day_result['date'].strftime('%Y-%m-%d')} ({day_result['data_type']})")
                    
                    # 결과 저장
                    result_data = {
                        'earliest_date': day_result['date'].strftime('%Y-%m-%d'),
                        'earliest_date_str': day_result['date_str'],
                        'data_type': day_result['data_type'],
                        'sample_data_count': len(day_result['sample_data'].get('OutBlock_1', [])),
                        'search_completed_at': datetime.now().isoformat()
                    }
                    
                    with open('krx_earliest_data_result.json', 'w', encoding='utf-8') as f:
                        json.dump(result_data, f, ensure_ascii=False, indent=2)
                    
                    logger.info("결과가 krx_earliest_data_result.json에 저장되었습니다.")
                    return day_result
                else:
                    logger.warning("일별 탐색에서 데이터를 찾을 수 없습니다.")
            else:
                logger.warning("월별 탐색에서 데이터를 찾을 수 없습니다.")
        else:
            logger.warning("연도별 탐색에서 데이터를 찾을 수 없습니다.")
    
    return None

if __name__ == "__main__":
    asyncio.run(main()) 