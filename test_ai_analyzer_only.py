#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI 분석기 단독 테스트 스크립트
KIS API나 다른 외부 의존성 없이 AI 분석기만 테스트합니다.
"""

import asyncio
import os
from dotenv import load_dotenv
from personal_blackrock.ai_analyzer import AIAnalyzer
from personal_blackrock.stock_data_manager import DataManager

# 환경 변수 로드
load_dotenv()

async def test_ai_analyzer_only():
    """AI 분석기만 단독으로 테스트합니다."""
    print("🤖 AI 분석기 전체 전략 테스트 시작!")
    print("=" * 60)
    
    # Gemini API 키 확인
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        print("❌ GEMINI_API_KEY 환경 변수가 설정되지 않았습니다.")
        print("💡 .env 파일에 GEMINI_API_KEY를 추가해주세요.")
        return
    
    print(f"✅ Gemini API 키 확인됨: ...{gemini_api_key[-10:]}")
    
    try:
        # 데이터 매니저 초기화 (빠른 모드)
        print("\n📊 데이터 매니저 초기화 중...")
        data_manager = DataManager(preload_data=False)
        print("✅ 데이터 매니저 초기화 완료")
        
        # AI 분석기 초기화
        print("\n🤖 AI 분석기 초기화 중...")
        ai_analyzer = AIAnalyzer(data_manager=data_manager)
        print("✅ AI 분석기 초기화 완료")
        
        # 테스트할 종목과 전략들
        test_stock = "005930"  # 삼성전자
        strategies = [
            "윌리엄 오닐",
            "제시 리버모어", 
            "피터 린치",
            "워렌 버핏",
            "일목산인"
        ]
        
        print(f"\n🔍 {test_stock} (삼성전자) 전략별 분석 시작...")
        print("=" * 60)
        
        for i, strategy in enumerate(strategies, 1):
            print(f"\n📊 [{i}/5] {strategy} 전략 분석 중...")
            print("-" * 40)
            
            try:
                # AI 분석 수행
                result = await ai_analyzer.analyze_stock_with_strategy(test_stock, strategy)
                
                if result and 'error' not in result:
                    print(f"🎉 {strategy} 전략 분석 성공!")
                    print(f"📈 종목: {result.get('name', 'N/A')} ({test_stock})")
                    print(f"🏆 점수: {result.get('점수', 'N/A')}점")
                    print(f"📊 등급: {result.get('추천 등급', 'N/A')}")
                    print(f"💡 결론: {result.get('결론', 'N/A')}")
                    print(f"🎯 진입가격: {result.get('진입 가격', 'N/A')}")
                    print(f"📈 목표가격: {result.get('목표 가격', 'N/A')}")
                    print(f"🔒 신뢰도: {result.get('신뢰도', 'N/A')}")
                    
                    # 분석 내용 요약 (처음 300자만)
                    analysis = result.get('분석', '')
                    if len(analysis) > 300:
                        analysis = analysis[:300] + "..."
                    print(f"\n📝 분석 요약: {analysis}")
                    
                    # 추천 이유 요약 (처음 200자만)
                    reason = result.get('추천 이유', '')
                    if len(reason) > 200:
                        reason = reason[:200] + "..."
                    print(f"💬 추천 이유: {reason}")
                    
                else:
                    print(f"❌ {strategy} 전략 분석 실패: {result.get('error', '알 수 없는 오류')}")
                    
            except Exception as e:
                print(f"❌ {strategy} 전략 분석 중 오류: {e}")
            
            # 전략 간 간격
            if i < len(strategies):
                print("\n" + "="*40)
        
        print("\n" + "="*60)
        print("🎉 전체 전략 테스트 완료!")
        print("💡 각 전략별로 서로 다른 관점에서 분석이 수행되었습니다.")
        print("📊 윌리엄 오닐: CAN SLIM + 차트 패턴")
        print("🎯 제시 리버모어: 피버럴 포인트 + 추세 추종")
        print("🔍 피터 린치: 성장주 발굴")
        print("🏰 워렌 버핏: 해자(Moat) + 가치투자")
        print("☁️ 일목산인: 일목균형표 기술적 분석")
        
    except Exception as e:
        print(f"❌ 테스트 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_ai_analyzer_only()) 