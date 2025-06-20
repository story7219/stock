#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
모든 투자 전략 순차 테스트 스크립트
1번부터 6번까지 모든 전략을 자동으로 테스트합니다.
"""

import sys
import time
import asyncio
from main import StockAnalysisSystem

async def test_all_strategies():
    """모든 투자 전략을 순차적으로 테스트합니다."""
    
    strategies = [
        ("윌리엄 오닐", "윌리엄 오닐"),      # 1번 - 이미 성공 확인
        ("제시 리버모어", "제시 리버모어"),  # 2번
        ("워렌 버핏", "워렌 버핏"),       # 3번  
        ("피터 린치", "피터 린치"),          # 4번
        ("일목산인", "일목산인"),            # 5번
        ("벤저민 그레이엄", "벤저민 그레이엄")               # 6번
    ]
    
    print("🚀 모든 투자 전략 순차 테스트 시작!")
    print("=" * 80)
    
    # 시스템 초기화
    try:
        print("📊 시스템 초기화 중...")
        system = StockAnalysisSystem()
        
        # 중요: initialize() 메서드 호출
        if not await system.initialize():
            print("❌ 시스템 초기화 실패")
            return
            
        print("✅ 시스템 초기화 완료")
    except Exception as e:
        print(f"❌ 시스템 초기화 실패: {e}")
        return
    
    # 각 전략별 테스트
    for i, (strategy_name, kor_strategy_name) in enumerate(strategies, 1):
        print(f"\n📊 {i}번. {kor_strategy_name} 전략 테스트 시작...")
        print("-" * 60)
        
        try:
            start_time = time.time()
            
            # 전략 분석 실행 - 매개변수 순서 수정
            await system.analyze_strategy(strategy_name, kor_strategy_name)
            
            end_time = time.time()
            elapsed = end_time - start_time
            
            print(f"✅ {kor_strategy_name} 전략 테스트 완료! (소요시간: {elapsed:.1f}초)")
            
            # 잠시 대기 (API 호출 제한 고려)
            if i < len(strategies):
                print("⏳ 다음 테스트를 위해 5초 대기...")
                await asyncio.sleep(5)  # 비동기 sleep 사용
                
        except Exception as e:
            print(f"❌ {kor_strategy_name} 전략 테스트 실패: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "=" * 80)
    print("🎉 모든 전략 테스트 완료!")
    
    # 리소스 정리
    try:
        await system.cleanup()
        print("✅ 시스템 리소스 정리 완료")
    except Exception as e:
        print(f"⚠️ 리소스 정리 중 오류: {e}")

if __name__ == "__main__":
    asyncio.run(test_all_strategies()) 