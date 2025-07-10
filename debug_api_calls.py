#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: debug_api_calls.py
목적: API 호출 디버깅 및 문제 진단
"""

import asyncio
import logging
import os
import time
from datetime import datetime, timedelta
import aiohttp
import json

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug_api_calls.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class APIDebugger:
    """API 호출 디버깅 클래스"""
    
    def __init__(self):
        self.app_key = os.getenv('MOCK_KIS_APP_KEY')
        self.app_secret = os.getenv('MOCK_KIS_APP_SECRET')
        self.base_url = "https://openapivts.koreainvestment.com:29443"
        self.session = None
        self.access_token = None
        
    async def initialize(self):
        """초기화"""
        self.session = aiohttp.ClientSession()
        await self._get_access_token()
    
    async def close(self):
        """세션 종료"""
        if self.session:
            await self.session.close()
    
    async def _get_access_token(self):
        """토큰 발급"""
        try:
            url = f"{self.base_url}/oauth2/tokenP"
            
            headers = {
                "content-type": "application/json"
            }
            
            data = {
                "grant_type": "client_credentials",
                "appkey": self.app_key,
                "appsecret": self.app_secret
            }
            
            logger.info(f"토큰 발급 요청: {url}")
            logger.info(f"Headers: {headers}")
            logger.info(f"Data: {data}")
            
            async with self.session.post(url, headers=headers, json=data) as response:
                logger.info(f"토큰 발급 응답 상태: {response.status}")
                logger.info(f"응답 헤더: {dict(response.headers)}")
                
                response_text = await response.text()
                logger.info(f"응답 내용: {response_text}")
                
                if response.status == 200:
                    result = await response.json()
                    self.access_token = result["access_token"]
                    logger.info("토큰 발급 성공")
                else:
                    logger.error(f"토큰 발급 실패: {response.status}")
                    
        except Exception as e:
            logger.error(f"토큰 발급 중 오류: {e}")
    
    async def test_stock_price_api(self, symbol: str = "005930"):
        """주가 API 테스트"""
        try:
            if not self.access_token:
                logger.error("토큰이 없습니다")
                return
            
            url = f"{self.base_url}/uapi/domestic-stock/v1/quotations/inquire-daily-price"
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.access_token}",
                "appkey": self.app_key,
                "appsecret": self.app_secret,
                "tr_id": "FHKST01010400"
            }
            
            # 최근 10일 데이터만 테스트
            end_date = datetime.now().strftime("%Y%m%d")
            start_date = (datetime.now() - timedelta(days=10)).strftime("%Y%m%d")
            
            params = {
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_COND_SCR_DIV_CODE": "20171",
                "FID_INPUT_ISCD": symbol,
                "FID_INPUT_DATE_1": start_date,
                "FID_INPUT_DATE_2": end_date,
                "FID_VOL_CNT": ""
            }
            
            logger.info(f"주가 API 호출: {symbol}")
            logger.info(f"URL: {url}")
            logger.info(f"Headers: {headers}")
            logger.info(f"Params: {params}")
            
            # 1초 대기 후 호출
            await asyncio.sleep(1)
            
            async with self.session.get(url, headers=headers, params=params) as response:
                logger.info(f"주가 API 응답 상태: {response.status}")
                logger.info(f"응답 헤더: {dict(response.headers)}")
                
                response_text = await response.text()
                logger.info(f"응답 내용 길이: {len(response_text)}")
                logger.info(f"응답 내용: {response_text}")
                
                if response.status == 200:
                    try:
                        result = await response.json()
                        data = result.get("output", [])
                        logger.info(f"데이터 개수: {len(data)}")
                        if data:
                            logger.info(f"첫 번째 데이터: {data[0]}")
                    except Exception as e:
                        logger.error(f"JSON 파싱 실패: {e}")
                else:
                    logger.error(f"API 호출 실패: {response.status}")
                    
        except Exception as e:
            logger.error(f"주가 API 테스트 중 오류: {e}")
    
    async def test_rate_limiting(self):
        """Rate limiting 테스트"""
        logger.info("Rate limiting 테스트 시작")
        
        for i in range(5):
            logger.info(f"테스트 호출 {i+1}/5")
            await self.test_stock_price_api("005930")
            await asyncio.sleep(2)  # 2초 대기
        
        logger.info("Rate limiting 테스트 완료")

async def main():
    """메인 함수"""
    debugger = APIDebugger()
    
    try:
        await debugger.initialize()
        await debugger.test_stock_price_api()
        await debugger.test_rate_limiting()
    finally:
        await debugger.close()

if __name__ == "__main__":
    asyncio.run(main()) 