#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 새로운 투자 전략 테스트 (존 헨리 & 브루스 코브너)
===================================================
- 존 헨리: 알고리즘 트렌드 추종 전략
- 브루스 코브너: 매크로 헤지펀드 전략
- 한글 투자 전략 유형 테스트
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.base_interfaces import StrategyType
from src.modules.optimized_investment_strategies import StrategyFactory, JohnHenryStrategy, BruceKovnerStrategy

async def test_new_strategies():
    """새로운 전략 테스트"""
    print("🚀 새로운 투자 전략 테스트 시작")
    print("=" * 50)
    
    # 한글 전략 유형 확인
    print("📋 한글 전략 유형 목록:")
    for strategy_type in StrategyType:
        print(f"  - {strategy_type.name}: {strategy_type.value}")
    print()
    
    # 샘플 데이터
    sample_data = {
        "AAPL": {
            "symbol": "AAPL",
            "market": "nasdaq100",
            "indicators": {
                "current_price": 180.0,
                "sma_20": 175.0,
                "sma_50": 170.0,
                "sma_200": 165.0,
                "rsi": 55.0,
                "macd": 2.5,
                "macd_signal": 2.0,
                "bb_upper": 185.0,
                "bb_middle": 180.0,
                "bb_lower": 175.0,
                "adx": 65.0,
                "volume": 50000000,
                "volume_sma": 45000000
            },
            "signals": {
                "trend": "UPTREND",
                "bollinger": "NEUTRAL"
            }
        },
        "005930": {
            "symbol": "005930",
            "market": "kospi200",
            "indicators": {
                "current_price": 70000.0,
                "sma_20": 69000.0,
                "sma_50": 68000.0,
                "sma_200": 67000.0,
                "rsi": 45.0,
                "macd": -1.0,
                "macd_signal": -0.5,
                "bb_upper": 72000.0,
                "bb_middle": 70000.0,
                "bb_lower": 68000.0,
                "adx": 40.0,
                "volume": 10000000,
                "volume_sma": 12000000
            },
            "signals": {
                "trend": "SIDEWAYS",
                "bollinger": "NEUTRAL"
            }
        }
    }
    
    # 존 헨리 전략 테스트
    print("🤖 존 헨리 전략 (알고리즘 트렌드 추종) 테스트:")
    john_henry = JohnHenryStrategy()
    
    for symbol, data in sample_data.items():
        signal = await john_henry.analyze_stock(data)
        if signal:
            print(f"  📊 {symbol}:")
            print(f"     신호강도: {signal.signal_strength:.1f}/100")
            print(f"     신뢰도: {signal.confidence:.1f}/100")
            print(f"     목표가: {signal.target_price:.2f}")
            print(f"     손절가: {signal.stop_loss:.2f}")
            print(f"     리스크: {signal.risk_score:.1f}/100")
            print(f"     근거: {signal.reasoning}")
            print()
    
    # 브루스 코브너 전략 테스트
    print("🌍 브루스 코브너 전략 (매크로 헤지펀드) 테스트:")
    bruce_kovner = BruceKovnerStrategy()
    
    for symbol, data in sample_data.items():
        signal = await bruce_kovner.analyze_stock(data)
        if signal:
            print(f"  📊 {symbol}:")
            print(f"     신호강도: {signal.signal_strength:.1f}/100")
            print(f"     신뢰도: {signal.confidence:.1f}/100")
            print(f"     목표가: {signal.target_price:.2f}")
            print(f"     손절가: {signal.stop_loss:.2f}")
            print(f"     리스크: {signal.risk_score:.1f}/100")
            print(f"     근거: {signal.reasoning}")
            print()
    
    # 팩토리 테스트
    print("🏭 전략 팩토리 테스트:")
    try:
        john_henry_from_factory = StrategyFactory.create_strategy(StrategyType.JOHN_HENRY)
        bruce_kovner_from_factory = StrategyFactory.create_strategy(StrategyType.BRUCE_KOVNER)
        
        print(f"  ✅ 존 헨리 전략 생성: {john_henry_from_factory.__class__.__name__}")
        print(f"  ✅ 브루스 코브너 전략 생성: {bruce_kovner_from_factory.__class__.__name__}")
        
        all_strategies = StrategyFactory.get_all_strategies()
        print(f"  📈 총 전략 수: {len(all_strategies)}개")
        
        for strategy in all_strategies:
            print(f"     - {strategy.strategy_type.value} ({strategy.category.value})")
            
    except Exception as e:
        print(f"  ❌ 팩토리 테스트 실패: {e}")
    
    print("\n✅ 새로운 전략 테스트 완료!")

if __name__ == "__main__":
    asyncio.run(test_new_strategies()) 