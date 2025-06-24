#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 AI 기반 투자 분석 시스템 - 메인 진입점
코스피200·나스닥100·S&P500 전체 종목을 분석하여 
투자 대가 전략으로 Gemini AI가 Top5 종목을 자동 선정

🎯 표준 프로젝트 구조 기반 - 프로그램 개발 정석 적용
🔥 15명의 투자 대가 전략 + Gemini AI 고급 추론 시스템
"""

import os
import sys
import asyncio
import logging
from typing import List, Dict, Any
from datetime import datetime
import time

# 🔧 새로운 표준 모듈 구조 import
from src.investment_analyzer.core import StockData, BaseStrategy
from src.investment_analyzer.data import MultiDataCollector, DataCleaner
from src.investment_analyzer.strategies import StrategyManager
from src.investment_analyzer.ai import GeminiAnalyzer
from src.investment_analyzer.analysis import TechnicalAnalyzer
from src.investment_analyzer.reporting import ReportGenerator

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/logs/investment_analysis.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class InvestmentAnalysisSystem:
    """🚀 AI 기반 투자 분석 시스템 - 메인 클래스"""
    
    def __init__(self):
        """시스템 초기화"""
        logger.info("🚀 AI 기반 투자 분석 시스템 초기화 시작")
        
        # 🔧 핵심 컴포넌트 초기화
        self.data_collector = MultiDataCollector()
        self.data_cleaner = DataCleaner()
        self.technical_analyzer = TechnicalAnalyzer() 
        self.strategy_manager = StrategyManager()
        self.gemini_analyzer = GeminiAnalyzer()
        self.report_generator = ReportGenerator()
        
        # 📊 분석 결과 저장
        self.analysis_results = {}
        self.collected_stocks = []
        
        logger.info("✅ 시스템 초기화 완료")
    
    async def run_full_analysis(self) -> Dict[str, Any]:
        """🎯 전체 분석 프로세스 실행"""
        try:
            logger.info("🎯 전체 투자 분석 프로세스 시작")
            start_time = time.time()
            
            # 1단계: 데이터 수집
            logger.info("📊 1단계: 다중 시장 데이터 수집 시작")
            raw_stocks = await self.data_collector.collect_all_markets()
            logger.info(f"✅ 총 {len(raw_stocks)}개 종목 수집 완료")
            
            # 2단계: 데이터 정제
            logger.info("🧹 2단계: 데이터 정제 및 검증")
            cleaned_stocks = self.data_cleaner.clean_stock_data(raw_stocks)
            self.collected_stocks = cleaned_stocks
            logger.info(f"✅ {len(cleaned_stocks)}개 유효 종목 정제 완료")
            
            # 3단계: 기술적 분석
            logger.info("📈 3단계: 기술적 분석 실행")
            technical_results = self.technical_analyzer.analyze_all(cleaned_stocks)
            logger.info(f"✅ {len(technical_results)}개 종목 기술적 분석 완료")
            
            # 4단계: 투자 전략 적용
            logger.info("🎯 4단계: 15개 투자 대가 전략 적용")
        strategy_results = {}
            for strategy_name in self.strategy_manager.get_all_strategies():
                results = self.strategy_manager.apply_strategy(strategy_name, cleaned_stocks)
                strategy_results[strategy_name] = results
                logger.info(f"✅ {strategy_name} 전략 적용 완료")
            
            # 5단계: Gemini AI 종합 분석
            logger.info("🤖 5단계: Gemini AI 종합 분석 및 Top5 선정")
            ai_selection = await self.gemini_analyzer.select_top_stocks(
                stocks=cleaned_stocks,
                strategy_results=strategy_results,
                technical_analysis=technical_results
            )
            
            # 6단계: 결과 컴파일
            final_results = {
                'collection_summary': {
                    'total_stocks': len(raw_stocks),
                    'valid_stocks': len(cleaned_stocks),
                    'success_rate': (len(cleaned_stocks) / len(raw_stocks) * 100) if raw_stocks else 0,
                    'collection_time': datetime.now().isoformat()
                },
                'technical_analysis': technical_results,
                'strategy_results': strategy_results,
                'ai_selection': ai_selection,
                'execution_time': time.time() - start_time
            }
            
            # 7단계: 리포트 생성
            logger.info("📋 7단계: 종합 리포트 생성")
            report_path = await self.report_generator.generate_comprehensive_report(final_results)
            final_results['report_path'] = report_path
            
            logger.info(f"🎉 전체 분석 완료! 실행시간: {final_results['execution_time']:.2f}초")
            logger.info(f"📋 리포트 저장 위치: {report_path}")
            
            return final_results
            
        except Exception as e:
            logger.error(f"❌ 분석 중 오류 발생: {str(e)}")
            raise

    def display_results(self, results: Dict[str, Any]):
        """🖥️ 결과 화면 출력"""
        print("\n" + "="*80)
        print("🚀 AI 기반 투자 분석 시스템 - 결과 요약")
        print("="*80)
        
        # 수집 요약
        summary = results['collection_summary']
        print(f"\n📊 데이터 수집 요약:")
        print(f"   • 총 종목 수: {summary['total_stocks']}개")
        print(f"   • 유효 종목 수: {summary['valid_stocks']}개") 
        print(f"   • 수집 성공률: {summary['success_rate']:.1f}%")
        
        # AI 선정 결과
        if 'selected_stocks' in results['ai_selection']:
            print(f"\n🤖 Gemini AI 선정 Top5 종목:")
            for i, stock in enumerate(results['ai_selection']['selected_stocks'][:5], 1):
                print(f"   {i}. {stock['name']} ({stock['symbol']}) - 점수: {stock.get('score', 0):.1f}")
        
        # 실행 정보
        print(f"\n⏱️ 실행 시간: {results['execution_time']:.2f}초")
        print(f"📋 리포트: {results.get('report_path', 'N/A')}")
        print("\n" + "="*80)

def setup_environment():
    """🔧 실행 환경 설정"""
    # 로그 디렉토리 생성
    os.makedirs('data/logs', exist_ok=True)
    os.makedirs('data/reports', exist_ok=True)
    os.makedirs('data/cache', exist_ok=True)
    
    # Python 경로 추가
    if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))

async def main():
    """🎯 메인 실행 함수"""
    try:
        # 환경 설정
        setup_environment()
        
        # 시스템 생성 및 실행
        system = InvestmentAnalysisSystem()
        results = await system.run_full_analysis()
        
        # 결과 출력
        system.display_results(results)
        
        return results
        
    except KeyboardInterrupt:
        logger.info("🛑 사용자에 의해 중단됨")
    except Exception as e:
        logger.error(f"❌ 시스템 실행 중 오류: {str(e)}")
        raise

if __name__ == "__main__":
    """🚀 프로그램 진입점"""
    print("""
🚀 AI 기반 투자 분석 시스템 v2.0
===============================================
📊 코스피200·나스닥100·S&P500 전체 종목 분석
🎯 15명 투자 대가 전략 + Gemini AI 선정
🏗️ 표준 프로젝트 구조 기반 (프로그램 개발 정석)
===============================================
    """)
    
    # 비동기 실행
    asyncio.run(main()) 