"""
매일 실행하는 간단한 스크립트
"""

import asyncio
from personal_trading_system import PersonalTradingSystem, PersonalTradingConfig

async def run_daily_trading():
    """매일 실행할 트레이딩 루틴"""
    
    # 본인의 투자 규모에 맞게 설정
    config = PersonalTradingConfig(
        total_capital=5_000_000,   # 500만원 (본인 자본금으로 수정)
        max_stocks=5,              # 최대 5종목
        stop_loss=-0.10,           # 10% 손절 (보수적)
        take_profit=0.20           # 20% 익절 (현실적)
    )
    
    system = PersonalTradingSystem(config)
    
    # 분석 및 실행
    analysis = await system.daily_analysis()
    execution = system.execute_trades(analysis)
    report = system.generate_daily_report(analysis, execution)
    
    print(report)
    
    # 파일로 저장
    with open(f"trading_report_{datetime.now().strftime('%Y%m%d')}.txt", "w", encoding="utf-8") as f:
        f.write(report)

if __name__ == "__main__":
    from datetime import datetime
    asyncio.run(run_daily_trading()) 