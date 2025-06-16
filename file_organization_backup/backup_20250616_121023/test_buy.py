"""
🧪 매수 테스트 스크립트
"""

import asyncio
import sys
import os

sys.path.append('src')

from core.order_executor import OrderExecutor
from config import config

async def test_buy():
    """매수 테스트"""
    executor = OrderExecutor()
    
    try:
        print("🔧 초기화 중...")
        await executor.initialize()
        
        print("💰 계좌 잔고 확인...")
        balance = await executor.get_account_balance()
        print(f"현금 잔고: {balance['cash']:,}원")
        
        print("🛒 삼성전자 1주 매수 테스트...")
        success = await executor.buy_market_order("005930", 1)
        
        if success:
            print("✅ 매수 성공!")
        else:
            print("❌ 매수 실패!")
            
    except Exception as e:
        print(f"❌ 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_buy()) 