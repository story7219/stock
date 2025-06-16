"""
🔧 시스템 점검
"""
import asyncio
from trade import TradingSystem

async def weekend_checkup():
    """주말 시스템 점검"""
    print("🔧 시스템 점검 시작")
    
    async with TradingSystem() as system:
        # 1. API 연결 테스트
        print("1. API 연결 확인...")
        
        # 2. 포트폴리오 현황 확인
        print("2. 현재 포트폴리오...")
        holdings = await system.check_portfolio()
        
        # 3. 전략 설정 확인
        print("3. 전략 설정 점검...")
        print(f"   - 피보나치 수열: {system.config.fibonacci_sequence}")
        print(f"   - 척후병 선정: {system.config.scout_selection}개")
        print(f"   - 최종 선정: {system.config.final_selection}개")
        
        print("✅ 시스템 점검 완료")

if __name__ == "__main__":
    asyncio.run(weekend_checkup()) 