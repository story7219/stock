#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: test_krx_trading_days.py
목적: 실제 거래일을 찾기 위해 월요일부터 금요일까지의 최근 날짜들을 테스트
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

class KRXTradingDayFinder:
    """KRX 거래일 찾기 클래스"""
    
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
    
    def get_weekdays(self, days_back: int = 60) -> List[datetime]:
        """최근 N일 중 월요일부터 금요일까지의 날짜들 반환"""
        weekdays = []
        current_date = datetime.now()
        
        for i in range(days_back):
            test_date = current_date - timedelta(days=i)
            if test_date.weekday() < 5:  # 0=월요일, 4=금요일
                weekdays.append(test_date)
        
        return weekdays
    
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
    
    async def find_trading_days(self, days_back: int = 60) -> Dict:
        """거래일 찾기"""
        logger.info(f"최근 {days_back}일 중 거래일 검색 시작...")
        
        weekdays = self.get_weekdays(days_back)
        logger.info(f"검색할 거래일 수: {len(weekdays)}개")
        
        found_data = []
        
        for i, test_date in enumerate(weekdays):
            date_str = test_date.strftime("%Y%m%d")
            
            if i % 10 == 0:  # 10개마다 진행상황 출력
                logger.info(f"진행률: {i}/{len(weekdays)} - 테스트 날짜: {test_date.strftime('%Y-%m-%d')}")
            
            for data_type in ["stock", "etf", "index"]:
                has_data, data = await self.test_date_for_data(date_str, data_type)
                
                if has_data:
                    logger.info(f"✅ {test_date.strftime('%Y-%m-%d')} {data_type} 데이터 발견! ({len(data.get('OutBlock_1', []))}개)")
                    found_data.append({
                        'date': test_date,
                        'date_str': date_str,
                        'data_type': data_type,
                        'data_count': len(data.get('OutBlock_1', [])),
                        'sample_data': data
                    })
                    break
            
            await asyncio.sleep(0.3)  # API 제한 방지
        
        return found_data

async def main():
    """메인 함수"""
    logger.info("KRX 거래일 데이터 찾기 시작...")
    
    async with KRXTradingDayFinder() as finder:
        # 거래일 데이터 검색
        trading_data = await finder.find_trading_days(60)
        
        if trading_data:
            logger.info(f"거래일 데이터 발견: {len(trading_data)}개")
            
            # 가장 오래된 데이터 찾기
            earliest_data = min(trading_data, key=lambda x: x['date'])
            logger.info(f"가장 오래된 거래일 데이터: {earliest_data['date'].strftime('%Y-%m-%d')} ({earliest_data['data_type']}) - {earliest_data['data_count']}개")
            
            # 가장 최근 데이터 찾기
            latest_data = max(trading_data, key=lambda x: x['date'])
            logger.info(f"가장 최근 거래일 데이터: {latest_data['date'].strftime('%Y-%m-%d')} ({latest_data['data_type']}) - {latest_data['data_count']}개")
            
            # 결과 저장
            result_data = {
                'total_found': len(trading_data),
                'earliest_date': earliest_data['date'].strftime('%Y-%m-%d'),
                'earliest_date_str': earliest_data['date_str'],
                'earliest_data_type': earliest_data['data_type'],
                'earliest_data_count': earliest_data['data_count'],
                'latest_date': latest_data['date'].strftime('%Y-%m-%d'),
                'latest_date_str': latest_data['date_str'],
                'latest_data_type': latest_data['data_type'],
                'latest_data_count': latest_data['data_count'],
                'all_trading_days': [
                    {
                        'date': data['date'].strftime('%Y-%m-%d'),
                        'data_type': data['data_type'],
                        'data_count': data['data_count']
                    }
                    for data in trading_data
                ],
                'search_completed_at': datetime.now().isoformat()
            }
            
            with open('krx_trading_days_result.json', 'w', encoding='utf-8') as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)
            
            logger.info("결과가 krx_trading_days_result.json에 저장되었습니다.")
            return result_data
        else:
            logger.warning("거래일 데이터를 찾을 수 없습니다.")
    
    return None

if __name__ == "__main__":
    asyncio.run(main()) 