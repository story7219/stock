#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Ultra Stock Analysis System - 데모 스크립트
GUI 없이 핵심 기능을 테스트할 수 있는 스크립트
"""

import asyncio
import logging
from datetime import datetime
from typing import List
import os
import sys

# 프로젝트 루트 경로 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_collector import DataCollector, StockData
from src.strategies import StrategyManager
from src.gemini_analyzer import Top5Selector
from src.technical_analyzer import ChartAnalyzer
from src.report_generator import ReportGenerator

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_demo_data() -> List[StockData]:
    """데모용 주식 데이터 생성"""
    demo_stocks = [
        StockData(
            symbol="AAPL",
            name="Apple Inc.",
            price=150.0,
            volume=50000000,
            market_cap=2500000000000,
            pe_ratio=25.5,
            pb_ratio=6.2,
            dividend_yield=0.005,
            moving_avg_20=148.5,
            moving_avg_60=145.2,
            rsi=65.3,
            bollinger_upper=155.0,
            bollinger_lower=145.0,
            macd=2.1,
            macd_signal=1.8
        ),
        StockData(
            symbol="MSFT",
            name="Microsoft Corporation",
            price=280.0,
            volume=30000000,
            market_cap=2100000000000,
            pe_ratio=28.2,
            pb_ratio=5.8,
            dividend_yield=0.008,
            moving_avg_20=275.3,
            moving_avg_60=270.1,
            rsi=58.7,
            bollinger_upper=290.0,
            bollinger_lower=265.0,
            macd=1.5,
            macd_signal=1.2
        ),
        StockData(
            symbol="GOOGL",
            name="Alphabet Inc.",
            price=2800.0,
            volume=25000000,
            market_cap=1800000000000,
            pe_ratio=22.1,
            pb_ratio=4.9,
            dividend_yield=0.0,
            moving_avg_20=2750.0,
            moving_avg_60=2700.0,
            rsi=72.1,
            bollinger_upper=2900.0,
            bollinger_lower=2650.0,
            macd=45.2,
            macd_signal=42.1
        ),
        StockData(
            symbol="TSLA",
            name="Tesla Inc.",
            price=800.0,
            volume=80000000,
            market_cap=800000000000,
            pe_ratio=45.6,
            pb_ratio=12.3,
            dividend_yield=0.0,
            moving_avg_20=780.0,
            moving_avg_60=750.0,
            rsi=68.9,
            bollinger_upper=850.0,
            bollinger_lower=720.0,
            macd=15.3,
            macd_signal=12.8
        ),
        StockData(
            symbol="NVDA",
            name="NVIDIA Corporation",
            price=450.0,
            volume=60000000,
            market_cap=1100000000000,
            pe_ratio=65.2,
            pb_ratio=15.8,
            dividend_yield=0.001,
            moving_avg_20=440.0,
            moving_avg_60=420.0,
            rsi=75.4,
            bollinger_upper=480.0,
            bollinger_lower=410.0,
            macd=8.7,
            macd_signal=7.2
        )
    ]
    
    return demo_stocks

async def demo_data_collection():
    """데이터 수집 데모"""
    print("📊 데이터 수집 데모")
    print("-" * 30)
    
    # 데모 데이터 사용
    stocks = create_demo_data()
    
    print(f"✅ {len(stocks)}개 종목 데이터 수집 완료")
    for stock in stocks:
        print(f"   - {stock.name} ({stock.symbol}): ${stock.price:.2f}")
    
    return stocks

def demo_strategy_analysis(stocks: List[StockData]):
    """전략 분석 데모"""
    print("\n🧠 투자 전략 분석 데모")
    print("-" * 30)
    
    strategy_manager = StrategyManager()
    results = strategy_manager.apply_all_strategies(stocks)
    
    for strategy_name, scores in results.items():
        print(f"\n📈 {strategy_name} 전략 결과:")
        for i, score in enumerate(scores[:3], 1):  # 상위 3개만 표시
            print(f"   {i}. {score.name} - 점수: {score.total_score:.1f}")
    
    return results

async def demo_gemini_analysis(strategy_results, market_data):
    """Gemini AI 분석 데모"""
    print("\n🤖 Gemini AI 분석 데모")
    print("-" * 30)
    
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("⚠️ GEMINI_API_KEY 환경 변수가 설정되지 않았습니다.")
        print("💡 실제 API 키를 설정하면 AI 분석을 사용할 수 있습니다.")
        return None
    
    try:
        gemini_selector = Top5Selector(api_key)
        result = await gemini_selector.select_top5_stocks(
            strategy_results, {"NASDAQ": market_data}
        )
        
        print("✅ AI Top5 종목 선정 완료:")
        for selection in result.top5_selections:
            print(f"   {selection.rank}. {selection.name} - 점수: {selection.final_score:.1f}")
            print(f"      이유: {selection.selection_reason[:100]}...")
        
        return result
        
    except Exception as e:
        print(f"❌ AI 분석 실패: {e}")
        return None

def demo_technical_analysis(stocks: List[StockData]):
    """기술적 분석 데모"""
    print("\n📈 기술적 분석 데모")
    print("-" * 30)
    
    analyzer = ChartAnalyzer()
    
    # 첫 번째 종목 분석
    stock = stocks[0]
    print(f"🔍 {stock.name} 기술적 분석:")
    print(f"   - 현재가: ${stock.price:.2f}")
    print(f"   - RSI: {stock.rsi:.1f}")
    print(f"   - MACD: {stock.macd:.2f}")
    print(f"   - 20일 이평: ${stock.moving_avg_20:.2f}")
    
    # 매매 신호 판단
    signals = []
    if stock.rsi < 30:
        signals.append("과매도 (매수 신호)")
    elif stock.rsi > 70:
        signals.append("과매수 (매도 신호)")
    
    if stock.macd > stock.macd_signal:
        signals.append("MACD 상승 (매수 신호)")
    
    if stock.price > stock.moving_avg_20:
        signals.append("20일선 상향 돌파")
    
    if signals:
        print(f"   📊 매매 신호: {', '.join(signals)}")
    else:
        print("   📊 매매 신호: 관망")

def demo_report_generation(gemini_result, strategy_results, market_data):
    """리포트 생성 데모"""
    print("\n📄 리포트 생성 데모")
    print("-" * 30)
    
    if not gemini_result:
        print("⚠️ AI 분석 결과가 없어 리포트를 생성할 수 없습니다.")
        return
    
    try:
        report_generator = ReportGenerator()
        report_content = report_generator.generate_text_report(
            gemini_result, strategy_results, {"NASDAQ": market_data}
        )
        
        # 리포트 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"demo_report_{timestamp}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"✅ 리포트가 생성되었습니다: {filename}")
        
    except Exception as e:
        print(f"❌ 리포트 생성 실패: {e}")

async def main():
    """메인 데모 함수"""
    print("🚀 Ultra Stock Analysis System - 데모")
    print("=" * 50)
    
    try:
        # 1. 데이터 수집
        stocks = await demo_data_collection()
        
        # 2. 전략 분석
        strategy_results = demo_strategy_analysis(stocks)
        
        # 3. 기술적 분석
        demo_technical_analysis(stocks)
        
        # 4. Gemini AI 분석
        gemini_result = await demo_gemini_analysis(strategy_results, stocks)
        
        # 5. 리포트 생성
        demo_report_generation(gemini_result, strategy_results, stocks)
        
        print("\n✅ 모든 데모가 완료되었습니다!")
        print("🎨 GUI 버전을 사용하려면 'python main.py' 또는 'python start.py'를 실행하세요.")
        
    except Exception as e:
        logger.error(f"데모 실행 중 오류: {e}")
        print(f"❌ 데모 실행 중 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 