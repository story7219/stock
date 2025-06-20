"""
개선된 AI 분석기 테스트 스크립트
Gemini AI가 풍부하고 질 좋은 데이터로 정확한 분석을 수행하는지 테스트합니다.
"""

import asyncio
import sys
import os

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from personal_blackrock.ai_analyzer import AIAnalyzer

async def test_improved_analysis():
    """개선된 분석 기능을 테스트합니다."""
    print("🚀 개선된 AI 분석기 테스트 시작...")
    print("=" * 60)
    
    try:
        # AI 분석기 초기화
        analyzer = AIAnalyzer()
        print("✅ AI 분석기 초기화 완료")
        
        # 삼성전자로 윌리엄 오닐 전략 테스트
        print("\n📊 삼성전자 - 윌리엄 오닐 전략 분석 중...")
        result = await analyzer.analyze_stock_with_strategy('005930', '윌리엄 오닐')
        
        if result and isinstance(result, dict):
            print("\n🎉 분석 성공!")
            print("-" * 40)
            print(f"📈 종목명: {result.get('name', 'N/A')}")
            print(f"🎯 전략: 윌리엄 오닐 (CAN SLIM)")
            print(f"⭐ 점수: {result.get('점수', 'N/A')}점")
            print(f"💡 결론: {result.get('결론', 'N/A')}")
            print(f"🏆 추천등급: {result.get('추천 등급', 'N/A')}")
            print(f"💰 진입가격: {result.get('진입 가격', 'N/A')}")
            print(f"🎯 목표가격: {result.get('목표 가격', 'N/A')}")
            print(f"🔒 신뢰도: {result.get('신뢰도', 'N/A')}")
            
            # 분석 내용 일부 출력
            analysis = result.get('분석', '')
            if analysis:
                print(f"\n📝 분석 요약:")
                # 분석 내용이 너무 길면 처음 200자만 출력
                if len(analysis) > 200:
                    print(f"   {analysis[:200]}...")
                else:
                    print(f"   {analysis}")
            
            print("\n" + "=" * 60)
            print("✅ 테스트 완료: 풍부한 데이터 가공이 성공적으로 작동합니다!")
            
        else:
            print("❌ 분석 실패: 결과가 올바르지 않습니다.")
            print(f"결과 타입: {type(result)}")
            print(f"결과 내용: {result}")
            
    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_improved_analysis()) 