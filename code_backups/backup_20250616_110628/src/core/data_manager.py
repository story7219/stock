"""
📊 데이터 관리자 - 시장 데이터 수집 및 관리
"""

import asyncio
import logging
import aiohttp
from typing import Dict, Any, Optional
from datetime import datetime

class DataManager:
    """데이터 관리자"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.is_initialized = False
        self.session = None
    
    async def initialize(self):
        """데이터 관리자 초기화"""
        if self.is_initialized:
            return
            
        self.logger.info("📊 데이터 관리자 초기화")
        self.session = aiohttp.ClientSession()
        self.is_initialized = True
    
    async def cleanup(self):
        """정리"""
        if self.session:
            await self.session.close()
        self.logger.info("📊 데이터 관리자 정리 완료")
    
    async def get_current_price(self, symbol: str) -> float:
        """현재 가격 조회"""
        try:
            self.logger.info(f"💰 현재 가격 조회: {symbol}")
            
            # 실제로는 한국투자증권 API 호출
            # 여기서는 시뮬레이션
            await asyncio.sleep(0.1)
            
            # 예시 가격 (실제로는 API에서 가져옴)
            import random
            base_prices = {
                "005930": 70000,  # 삼성전자
                "000660": 120000,  # SK하이닉스
                "035420": 200000,  # NAVER
                "051910": 800000,  # LG화학
                "006400": 600000,  # 삼성SDI
            }
            
            base_price = base_prices.get(symbol, 50000)
            # ±5% 변동
            current_price = base_price * (1 + random.uniform(-0.05, 0.05))
            
            self.logger.info(f"💰 {symbol} 현재가: {current_price:,.0f}원")
            return current_price
            
        except Exception as e:
            self.logger.error(f"❌ 가격 조회 오류 ({symbol}): {e}")
            return 50000.0  # 기본값 