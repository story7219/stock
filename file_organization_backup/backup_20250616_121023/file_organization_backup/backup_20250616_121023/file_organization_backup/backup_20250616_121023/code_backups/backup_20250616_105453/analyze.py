"""
📈 전략 성과 분석
"""
import asyncio
from trade import TradingSystem

async def analyze_strategy():
    """전략 분석"""
    print("📈 전략 성과 분석")
    
    strategies = {
        "척후병": "5개→4개→2개 선정 방식",
        "피보나치": "1,1,2,3,5,8,13 분할매수",
        "추세전환": "상승 추세 전환점 포착",
        "눌림목": "일시적 하락 후 매수",
        "전고점돌파": "저항선 돌파 시 매수"
    }
    
    for name, desc in strategies.items():
        print(f"🎯 {name}: {desc}")
    
    print("✅ 분석 완료")

if __name__ == "__main__":
    asyncio.run(analyze_strategy()) 