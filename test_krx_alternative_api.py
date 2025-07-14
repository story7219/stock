#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: test_krx_alternative_api.py
목적: KRX의 다른 API 엔드포인트들을 시도해서 데이터를 찾기
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

class KRXAlternativeAPITester:
    """KRX 대안 API 테스터 클래스"""
    
    def __init__(self):
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
    
    async def test_api_endpoint(self, endpoint: str, params: Dict, description: str) -> Tuple[bool, Dict]:
        """API 엔드포인트 테스트"""
        try:
            url = f"http://data.krx.co.kr/comm/bldAttendant/{endpoint}"
            
            async with self.session.post(url, data=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('OutBlock_1') and len(data['OutBlock_1']) > 0:
                        logger.info(f"✅ {description} - 데이터 발견: {len(data['OutBlock_1'])}개")
                        return True, data
                    else:
                        logger.info(f"❌ {description} - 데이터 없음")
                        return False, data
                else:
                    logger.error(f"❌ {description} - HTTP {response.status}")
                    return False, {}
                    
        except Exception as e:
            logger.error(f"❌ {description} - 오류: {e}")
            return False, {}
    
    async def test_various_apis(self, date_str: str = "20250714") -> Dict:
        """다양한 API 엔드포인트 테스트"""
        logger.info(f"날짜 {date_str}에 대해 다양한 API 테스트 시작...")
        
        test_results = []
        
        # 1. 기본 주식 API
        stock_params = {
            'bld': 'dbms/MDC/STAT/standard/MDCSTAT01501',
            'mktId': 'STK',
            'trdDd': date_str,
            'share': '1',
            'money': '1',
            'csvxls_isNo': 'false'
        }
        success, data = await self.test_api_endpoint("getJsonData.cmd", stock_params, "기본 주식 API")
        test_results.append({"api": "기본 주식 API", "success": success, "data_count": len(data.get('OutBlock_1', []))})
        
        # 2. ETF API
        etf_params = {
            'bld': 'dbms/MDC/STAT/standard/MDCSTAT01501',
            'mktId': 'ETF',
            'trdDd': date_str,
            'share': '1',
            'money': '1',
            'csvxls_isNo': 'false'
        }
        success, data = await self.test_api_endpoint("getJsonData.cmd", etf_params, "ETF API")
        test_results.append({"api": "ETF API", "success": success, "data_count": len(data.get('OutBlock_1', []))})
        
        # 3. 지수 API
        index_params = {
            'bld': 'dbms/MDC/STAT/standard/MDCSTAT01501',
            'mktId': 'IDX',
            'trdDd': date_str,
            'share': '1',
            'money': '1',
            'csvxls_isNo': 'false'
        }
        success, data = await self.test_api_endpoint("getJsonData.cmd", index_params, "지수 API")
        test_results.append({"api": "지수 API", "success": success, "data_count": len(data.get('OutBlock_1', []))})
        
        # 4. KOSPI API
        kospi_params = {
            'bld': 'dbms/MDC/STAT/standard/MDCSTAT01501',
            'mktId': 'KOSPI',
            'trdDd': date_str,
            'share': '1',
            'money': '1',
            'csvxls_isNo': 'false'
        }
        success, data = await self.test_api_endpoint("getJsonData.cmd", kospi_params, "KOSPI API")
        test_results.append({"api": "KOSPI API", "success": success, "data_count": len(data.get('OutBlock_1', []))})
        
        # 5. KOSDAQ API
        kosdaq_params = {
            'bld': 'dbms/MDC/STAT/standard/MDCSTAT01501',
            'mktId': 'KOSDAQ',
            'trdDd': date_str,
            'share': '1',
            'money': '1',
            'csvxls_isNo': 'false'
        }
        success, data = await self.test_api_endpoint("getJsonData.cmd", kosdaq_params, "KOSDAQ API")
        test_results.append({"api": "KOSDAQ API", "success": success, "data_count": len(data.get('OutBlock_1', []))})
        
        # 6. 시가총액 API
        market_cap_params = {
            'bld': 'dbms/MDC/STAT/standard/MDCSTAT01501',
            'mktId': 'STK',
            'trdDd': date_str,
            'share': '1',
            'money': '1',
            'csvxls_isNo': 'false'
        }
        success, data = await self.test_api_endpoint("getJsonData.cmd", market_cap_params, "시가총액 API")
        test_results.append({"api": "시가총액 API", "success": success, "data_count": len(data.get('OutBlock_1', []))})
        
        # 7. 거래량 API
        volume_params = {
            'bld': 'dbms/MDC/STAT/standard/MDCSTAT01501',
            'mktId': 'STK',
            'trdDd': date_str,
            'share': '1',
            'money': '1',
            'csvxls_isNo': 'false'
        }
        success, data = await self.test_api_endpoint("getJsonData.cmd", volume_params, "거래량 API")
        test_results.append({"api": "거래량 API", "success": success, "data_count": len(data.get('OutBlock_1', []))})
        
        # 8. 다른 엔드포인트 시도
        alternative_params = {
            'bld': 'dbms/MDC/STAT/standard/MDCSTAT01501',
            'mktId': 'STK',
            'trdDd': date_str,
            'share': '1',
            'money': '1',
            'csvxls_isNo': 'false'
        }
        success, data = await self.test_api_endpoint("getJsonData.cmd", alternative_params, "대안 엔드포인트")
        test_results.append({"api": "대안 엔드포인트", "success": success, "data_count": len(data.get('OutBlock_1', []))})
        
        return test_results
    
    async def test_recent_dates(self, days_back: int = 7) -> Dict:
        """최근 날짜들에 대해 API 테스트"""
        logger.info(f"최근 {days_back}일 동안의 날짜들에 대해 API 테스트...")
        
        all_results = {}
        current_date = datetime.now()
        
        for i in range(days_back):
            test_date = current_date - timedelta(days=i)
            date_str = test_date.strftime("%Y%m%d")
            
            logger.info(f"\n=== {test_date.strftime('%Y-%m-%d')} 테스트 ===")
            results = await self.test_various_apis(date_str)
            all_results[test_date.strftime('%Y-%m-%d')] = results
            
            await asyncio.sleep(1)  # API 제한 방지
        
        return all_results

async def main():
    """메인 함수"""
    logger.info("KRX 대안 API 테스트 시작...")
    
    async with KRXAlternativeAPITester() as tester:
        # 최근 7일 동안의 날짜들에 대해 API 테스트
        all_results = await tester.test_recent_dates(7)
        
        # 결과 분석
        successful_apis = []
        for date, results in all_results.items():
            for result in results:
                if result['success']:
                    successful_apis.append({
                        'date': date,
                        'api': result['api'],
                        'data_count': result['data_count']
                    })
        
        if successful_apis:
            logger.info(f"성공한 API 호출: {len(successful_apis)}개")
            for api in successful_apis:
                logger.info(f"✅ {api['date']} - {api['api']} ({api['data_count']}개 데이터)")
        else:
            logger.warning("성공한 API 호출이 없습니다.")
        
        # 결과 저장
        result_data = {
            'test_date': datetime.now().isoformat(),
            'successful_apis': successful_apis,
            'all_results': all_results
        }
        
        with open('krx_api_test_results.json', 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        
        logger.info("결과가 krx_api_test_results.json에 저장되었습니다.")
        return result_data
    
    return None

if __name__ == "__main__":
    asyncio.run(main()) 