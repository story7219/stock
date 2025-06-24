"""
GeminiAnalyzer 정밀 진단 스크립트
API 연결, 모델 초기화, 실제 분석 호출까지 단계별로 테스트합니다.
"""

import os
import sys
from pathlib import Path
import logging

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 로거 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def diagnose():
    print("="*50)
    print("🔬 GeminiAnalyzer 정밀 진단을 시작합니다...")
    print("="*50)

    # 1. 환경 변수 확인
    print("\n[1/4] 🔑 환경 변수(.env) 로드 확인...")
    try:
        from dotenv import load_dotenv
        load_dotenv()
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if gemini_api_key and "your" not in gemini_api_key:
            print("✅ GEMINI_API_KEY 로드 성공 (키의 일부: ...{})".format(gemini_api_key[-4:]))
        else:
            print("❌ GEMINI_API_KEY가 .env 파일에 없거나 유효하지 않습니다.")
            return
    except Exception as e:
        print(f"❌ .env 파일 로드 실패: {e}")
        return

    # 2. GeminiAnalyzer 초기화 및 헬스 체크
    print("\n[2/4] 🚀 GeminiAnalyzer 초기화 및 헬스 체크...")
    analyzer = None
    try:
        from src.gemini_analyzer import GeminiAnalyzer
        analyzer = GeminiAnalyzer()
        print("✅ GeminiAnalyzer 객체 생성 성공")
        
        health_check_result = analyzer.health_check()
        if health_check_result:
            print("✅ health_check() 통과: API 키 및 모델 설정이 정상입니다.")
        else:
            print("❌ health_check() 실패: API 키가 유효하지 않거나 모델 설정에 문제가 있습니다.")
            return
            
    except Exception as e:
        print(f"❌ GeminiAnalyzer 초기화 또는 헬스 체크 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. 간단한 데이터로 분석 함수 호출
    print("\n[3/4] 📊 더미 데이터로 실제 분석 API 호출 테스트...")
    try:
        from src.gemini_analyzer import StockData
        dummy_data = [
            StockData(
                symbol="005930.KS",
                name="삼성전자",
                market="KOSPI200",
                price=80000.0,
                volume=10000000,
                market_cap=477e12, # 477조
                rsi=60.0,
                macd=150.0,
                ma_50=78000.0
            )
        ]
        
        analysis_result = analyzer.analyze_by_all_strategies(dummy_data)

        if analysis_result and analysis_result.get('top5_stocks'):
            print("✅ AI 분석 API 호출 성공!")
            print("📊 분석 결과 (일부):")
            # print(analysis_result)
            for stock in analysis_result.get('top5_stocks', [])[:1]:
                print(f"  - Rank {stock.get('rank')}: {stock.get('name')}({stock.get('symbol')}), Score: {stock.get('score')}")
        else:
            print("❌ AI 분석 API 호출은 성공했으나, 유효한 분석 결과가 반환되지 않았습니다.")
            print("   - API 응답 형식 변경, 프롬프트 문제 또는 모델 정책 변경 가능성이 있습니다.")
            print(f"   - 실제 반환값: {analysis_result}")

    except Exception as e:
        print(f"❌ AI 분석 API 호출 중 치명적인 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return
        
    print("\n[4/4] ✅ 모든 진단 과정이 완료되었습니다.")
    print("="*50)

if __name__ == "__main__":
    diagnose() 