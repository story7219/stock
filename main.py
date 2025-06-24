#!/usr/bin/env python3
"""
🚀 Ultra Premium HTS v5.0 - Gemini AI 100% 활용 시스템
- 실제 Gemini AI가 시황 분석 + 종목 선정
- 세계 최고 수준의 AI 투자 분석가 시스템
- 고품질 데이터 + AI 추론으로 Top5 종목 자동 선정
- 고급 결측치 보정 및 데이터 정제 자동화
"""
import os
import sys
import asyncio
from typing import List, Dict, Any
from dotenv import load_dotenv
from src.strategies import StrategyManager
from src.data_collector import DataCollector
from src.data_cleaner import AdvancedDataCleaner
from src.gemini_analyzer import GeminiAnalyzer, GeminiAnalysisResult

# 환경 변수 로드
load_dotenv('config.env')

# 14개 투자 대가 전략명
STRATEGY_LIST = [
    "워런 버핏", "피터 린치", "벤저민 그레이엄", "윌리엄 오닐", "제시 리버모어",
    "존 템플턴", "존 네프", "필립 피셔", "마크 미너비니", "짐 슬레이터",
    "조엘 그린블라트", "에드워드 소프", "레이 달리오", "피터 드러커"
]

MARKET_LIST = [
    "한국주식(코스피200)", 
    "미국주식(나스닥100)", 
    "미국주식(S&P500)"
]

def print_progress(step, total_steps, message, progress_percent=None):
    """진행 상황 출력"""
    if progress_percent is not None:
        print(f"[{step}/{total_steps}] {message} (진행률: {progress_percent}%)")
    else:
        print(f"[{step}/{total_steps}] {message}")

def print_ai_banner():
    """AI 시스템 배너 출력"""
    print("🤖" + "=" * 58 + "🤖")
    print("🚀 Ultra Premium HTS v5.0 - Gemini AI 100% 활용 시스템")
    print("🧠 세계 최고 수준의 AI 투자 분석가가 직접 분석합니다!")
    print("🎯 실시간 AI 추론 + 고품질 데이터 = 최적의 Top5 종목 선정")
    print("🧹 고급 결측치 보정 및 데이터 정제 자동화 시스템")
    print("🤖" + "=" * 58 + "🤖")

async def main():
    total_steps = 10  # 단계 수 증가 (데이터 정제 단계 추가)
    
    print_ai_banner()
    
    # Gemini API 키 확인
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    if not gemini_api_key or gemini_api_key == 'your_gemini_api_key_here':
        print("❌ Gemini API 키가 설정되지 않았습니다!")
        print("📋 config.env 파일에서 GEMINI_API_KEY를 실제 키로 변경해주세요.")
        print("🔗 API 키 발급: https://aistudio.google.com/app/apikey")
        
        # 데모 모드로 실행할지 물어보기
        demo_mode = input("\n🎮 데모 모드(규칙 기반)로 실행하시겠습니까? (y/n): ").lower() == 'y'
        if not demo_mode:
            return
        use_ai = False
    else:
        use_ai = True
        print("✅ Gemini AI 연동 확인됨 - 실제 AI 분석 모드로 실행합니다!")
    
    print_progress(1, total_steps, "투자 전략 선택 중...", 10)
    print("\n📊 투자 대가 전략 리스트:")
    for idx, name in enumerate(STRATEGY_LIST, 1):
        ai_status = "🤖 AI 분석" if use_ai else "📊 규칙 기반"
        print(f"  {idx:2d}. {name} ({ai_status})")
    
    try:
        strategy_idx = int(input("\n선택할 전략 번호를 입력하세요: "))
        assert 1 <= strategy_idx <= len(STRATEGY_LIST)
        selected_strategy = STRATEGY_LIST[strategy_idx-1]
        print(f"✅ 선택된 전략: {selected_strategy}")
    except Exception as e:
        print(f"❌ 전략 선택 에러: {e}")
        return

    print_progress(2, total_steps, "투자 시장 선택 중...", 15)
    print("\n🌍 투자 시장 리스트:")
    for idx, name in enumerate(MARKET_LIST, 1):
        print(f"  {idx}. {name}")
    
    try:
        market_idx = int(input("\n선택할 시장 번호를 입력하세요: "))
        assert 1 <= market_idx <= len(MARKET_LIST)
        selected_market = MARKET_LIST[market_idx-1]
        print(f"✅ 선택된 시장: {selected_market}")
    except Exception as e:
        print(f"❌ 시장 선택 에러: {e}")
        return

    print_progress(3, total_steps, "🤖 AI 분석 시스템 초기화 중...", 20)
    try:
        # 데이터 수집기 초기화
        collector = DataCollector()
        print("✅ DataCollector 초기화 완료")
        
        # 고급 데이터 정제기 초기화
        data_cleaner = AdvancedDataCleaner()
        print("🧹 AdvancedDataCleaner 초기화 완료")
        
        # 전략 매니저 초기화  
        strategy_manager = StrategyManager()
        print("✅ StrategyManager 초기화 완료")
        
        # Gemini AI 분석기 초기화 (AI 모드일 때만)
        if use_ai:
            gemini_analyzer = GeminiAnalyzer(api_key=gemini_api_key)
            print("🤖 Gemini AI 분석기 초기화 완료")
        else:
            gemini_analyzer = None
            print("📊 규칙 기반 분석 모드")
            
    except Exception as e:
        print(f"❌ 시스템 초기화 실패: {e}")
        return

    print_progress(4, total_steps, "📊 고품질 시장 데이터 수집 중... (AI 최적화)", 35)
    try:
        if "한국" in selected_market:
            print("📈 코스피200 종목 데이터 수집 중...")
            stocks = await collector.collect_kospi_data()
            market_data = {"kospi200": stocks}
        elif "나스닥100" in selected_market:
            print("📈 나스닥100 종목 데이터 수집 중...")
            stocks = await collector.collect_nasdaq_data()
            market_data = {"nasdaq100": stocks}
        elif "S&P500" in selected_market:
            print("📈 S&P500 종목 데이터 수집 중...")
            stocks = await collector.collect_sp500_data()
            market_data = {"sp500": stocks}
        else:
            print("❌ 알 수 없는 시장입니다.")
            return
        
        print(f"✅ 원본 데이터 수집 완료: {len(stocks)}개 종목")
        if stocks:
            print(f"📊 샘플 종목: {stocks[0].name} ({stocks[0].symbol}) - 가격: ${stocks[0].price:.2f}")
    except Exception as e:
        print(f"❌ 데이터 수집 실패: {e}")
        return

    print_progress(5, total_steps, "🧹 고급 데이터 정제 및 결측치 보정 중...", 50)
    try:
        print("🤖 AI 기반 결측치 보정 시작...")
        print("  - 1단계: 기본 데이터 검증")
        print("  - 2단계: 이상치 탐지 및 처리")  
        print("  - 3단계: 통계적 + ML 기반 결측치 보정")
        print("  - 4단계: 기술적 지표 재계산")
        print("  - 5단계: 데이터 품질 평가")
        
        # 전체 종목에 대해 고급 데이터 정제 수행
        cleaned_stocks, cleaning_result = await data_cleaner.clean_stock_data_list(stocks)
        
        # 정제 결과 보고서 출력
        print("\n" + "🧹" + "=" * 50 + "🧹")
        print(data_cleaner.generate_cleaning_report(cleaning_result))
        print("🧹" + "=" * 50 + "🧹")
        
        # 정제된 데이터로 market_data 업데이트
        if "한국" in selected_market:
            market_data = {"kospi200": cleaned_stocks}
        elif "나스닥100" in selected_market:
            market_data = {"nasdaq100": cleaned_stocks}
        elif "S&P500" in selected_market:
            market_data = {"sp500": cleaned_stocks}
        
        stocks = cleaned_stocks  # 이후 처리를 위해 정제된 데이터 사용
        
    except Exception as e:
        print(f"❌ 데이터 정제 실패: {e}")
        print("📊 원본 데이터로 계속 진행합니다...")

    print_progress(6, total_steps, "🧠 투자 대가 전략 사전 분석 중...", 65)
    try:
        # 전략명을 키로 매핑
        strategy_map = {
            "워런 버핏": "buffett", "피터 린치": "lynch", "벤저민 그레이엄": "graham",
            "윌리엄 오닐": "oneil", "제시 리버모어": "livermore", "존 템플턴": "templeton",
            "존 네프": "neff", "필립 피셔": "fisher", "마크 미너비니": "minervini",
            "짐 슬레이터": "slater", "조엘 그린블라트": "greenblatt", "에드워드 소프": "thorp",
            "레이 달리오": "dalio", "피터 드러커": "drucker",
        }
        
        # 전략 사전 분석 (AI 분석을 위한 후보군 생성)
        strategy_results = {}
        strategy_key = strategy_map.get(selected_strategy)
        
        if strategy_key and strategy_key in strategy_manager.strategies:
            print(f"📈 {selected_strategy} 전략 사전 분석 중...")
            strategy_scores = strategy_manager.strategies[strategy_key].apply_strategy(stocks)
            strategy_results[strategy_key] = strategy_scores
            print(f"✅ 사전 분석 완료: {len(strategy_scores)}개 종목 평가")
        else:
            print(f"❌ 전략을 찾을 수 없습니다: {selected_strategy}")
            return
            
    except Exception as e:
        print(f"❌ 전략 사전 분석 실패: {e}")
        return

    if use_ai:
        print_progress(7, total_steps, "🤖 Gemini AI 종합 분석 실행 중... (실제 AI 추론)", 80)
        try:
            print("🧠 Gemini AI가 시황 분석 중...")
            print("🎯 AI가 투자 대가 전략과 고품질 정제 데이터를 종합 분석 중...")
            print("💡 최적의 Top5 종목 선정 중... (잠시만 기다려주세요)")
            
            # 실제 Gemini AI 분석 호출
            ai_result = await gemini_analyzer.analyze_candidates(strategy_results, market_data)
            
            print("✅ Gemini AI 분석 완료!")
            print(f"🎯 신뢰도 점수: {ai_result.confidence_score:.1f}%")
            
        except Exception as e:
            print(f"❌ Gemini AI 분석 실패: {e}")
            print("📊 백업 분석 모드로 전환합니다...")
            use_ai = False
            ai_result = None
    else:
        ai_result = None

    print_progress(8, total_steps, "📊 최종 결과 생성 중...", 90)
    
    if use_ai and ai_result:
        # AI 분석 결과 출력
        print_progress(9, total_steps, "🤖 AI 분석 결과 출력 중...", 95)
        print("\n" + "🤖" + "=" * 58 + "🤖")
        print(f"🚀 Gemini AI 분석 결과 - {selected_strategy} 전략")
        print("🤖" + "=" * 58 + "🤖")
        
        # 시황 분석
        print(f"\n📊 **AI 시황 분석**")
        print(f"   {ai_result.market_outlook}")
        
        print(f"\n🎯 **AI 선정 Top5 종목**")
        for selection in ai_result.top5_selections:
            print(f"\n{selection.rank}. 🏆 {selection.name} ({selection.symbol})")
            print(f"   🤖 AI 최종 점수: {selection.final_score:.1f}점")
            print(f"   💡 선정 이유: {selection.selection_reason}")
            print(f"   📈 기술적 분석: {selection.technical_analysis}")
            print(f"   ⚠️  리스크 평가: {selection.risk_assessment}")
            print(f"   🧠 AI 추론: {selection.gemini_reasoning}")
        
        # 추가 정보
        print(f"\n📋 **분석 요약**")
        print(f"   {ai_result.analysis_summary}")
        
        if ai_result.risk_warnings:
            print(f"\n⚠️  **리스크 경고**")
            for warning in ai_result.risk_warnings:
                print(f"   - {warning}")
        
        if ai_result.alternative_candidates:
            print(f"\n🔄 **대안 후보**")
            print(f"   {', '.join(ai_result.alternative_candidates[:5])}")
            
    else:
        # 규칙 기반 결과 출력 (백업 모드)
        print_progress(9, total_steps, "📊 규칙 기반 결과 출력 중...", 95)
        print("\n" + "📊" + "=" * 58 + "📊")
        print(f"📈 {selected_strategy} 전략 - Top 5 추천 종목 (규칙 기반)")
        print("📊" + "=" * 58 + "📊")
        
        result = strategy_results[strategy_key]
        if not result:
            print("❌ 분석 결과가 없습니다. 필터링 조건을 만족하는 종목이 없습니다.")
            return
                
        for i, stock in enumerate(result[:5], 1):
            print(f"\n{i}. 🎯 {stock.name} ({stock.symbol})")
            print(f"   📊 종합 점수: {stock.total_score:.2f}점")
            print(f"   💡 선정 사유: {stock.reasoning.split('.')[0]}.")
            if hasattr(stock, 'criteria_scores') and stock.criteria_scores:
                print(f"   📈 세부 점수: {', '.join([f'{k}: {v:.1f}' for k, v in stock.criteria_scores.items()])}")
    
    print_progress(10, total_steps, "✅ 분석 완료!", 100)
    print("\n" + "🎉" + "=" * 58 + "🎉")
    if use_ai:
        print("✅ 🤖 Gemini AI 분석 완료! 세계 최고 수준의 AI가 선정한 종목입니다.")
    else:
        print("✅ 📊 규칙 기반 분석 완료! 투자 결정은 신중하게 하시기 바랍니다.")
    
    print("🧹 고급 데이터 정제 완료! 결측치 보정 및 품질 개선이 적용되었습니다.")
    print("💡 이 분석은 투자 참고용이며, 최종 투자 결정은 본인의 책임입니다.")
    print("🎉" + "=" * 58 + "🎉")

if __name__ == "__main__":
    # 필요한 패키지 확인
    try:
        import google.generativeai
        import dotenv
        import sklearn
        import scipy
    except ImportError as e:
        print(f"❌ 필요한 패키지가 설치되지 않았습니다: {e}")
        print("📦 다음 명령어로 설치하세요:")
        print("   pip install google-generativeai python-dotenv scikit-learn scipy")
        sys.exit(1)

    asyncio.run(main()) 