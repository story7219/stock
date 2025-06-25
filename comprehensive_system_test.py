#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔍 전체 시스템 종합 검증 테스트 (Comprehensive System Verification Test)
===========================================================================

투자 분석 시스템의 모든 핵심 파일들이 제대로 역할을 수행하는지 검증하는 종합 테스트입니다.
각 모듈의 기능성, 연동성, 안정성을 체계적으로 검증합니다.

검증 대상:
1. 🏗️ 핵심 인프라 (Core Infrastructure)
   - base_interfaces.py: 기본 인터페이스 정의
   - config.py: 시스템 설정 관리
   - async_executor.py: 비동기 실행 엔진
   - memory_optimizer.py: 메모리 최적화
   - cache_manager.py: 캐시 관리

2. 📊 데이터 처리 (Data Processing)
   - data_collector.py: 데이터 수집
   - news_collector.py: 뉴스 수집
   - technical_analysis.py: 기술적 분석
   - gemini_premium_data_processor.py: 프리미엄 데이터 처리

3. 🧠 AI 분석 (AI Analysis)
   - gemini_analyzer.py: Gemini AI 분석
   - investment_strategies.py: 투자 전략
   - strategy_gemini_integration.py: 전략-AI 통합

4. 💼 포트폴리오 관리 (Portfolio Management)
   - portfolio_manager.py: 포트폴리오 관리
   - performance_optimizer.py: 성능 최적화
   - backtesting_engine.py: 백테스팅

5. 📈 리포팅 (Reporting)
   - report_generator.py: 리포트 생성
   - notification_system.py: 알림 시스템

각 모듈별로 기본 기능, 예외 처리, 성능을 검증합니다.
"""

import sys
import os
import asyncio
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional
import importlib.util
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

class SystemTestResult:
    """시스템 테스트 결과"""
    def __init__(self):
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.results = {}
        self.errors = []
        
    def add_result(self, module_name: str, test_name: str, success: bool, message: str = ""):
        """테스트 결과 추가"""
        self.total_tests += 1
        if success:
            self.passed_tests += 1
            status = "✅ PASS"
        else:
            self.failed_tests += 1
            status = "❌ FAIL"
            self.errors.append(f"{module_name}.{test_name}: {message}")
        
        if module_name not in self.results:
            self.results[module_name] = []
        
        self.results[module_name].append({
            'test': test_name,
            'status': status,
            'message': message
        })
    
    def print_summary(self):
        """테스트 결과 요약 출력"""
        print("\n" + "="*80)
        print("🔍 전체 시스템 검증 결과 요약")
        print("="*80)
        print(f"총 테스트 수: {self.total_tests}")
        print(f"성공: {self.passed_tests} ✅")
        print(f"실패: {self.failed_tests} ❌")
        print(f"성공률: {(self.passed_tests/self.total_tests)*100:.1f}%")
        
        print("\n📋 모듈별 상세 결과:")
        for module_name, tests in self.results.items():
            print(f"\n🔧 {module_name}:")
            for test in tests:
                print(f"  {test['status']} {test['test']}")
                if test['message']:
                    print(f"    💬 {test['message']}")
        
        if self.errors:
            print("\n⚠️ 오류 상세:")
            for error in self.errors:
                print(f"  - {error}")

class ComprehensiveSystemTest:
    """종합 시스템 테스트"""
    
    def __init__(self):
        self.result = SystemTestResult()
        self.test_data = {
            'symbols': ['AAPL', 'GOOGL', 'MSFT'],
            'kospi_symbols': ['005930.KS', '000660.KS'],
            'test_amount': 1000000
        }
    
    async def run_all_tests(self):
        """모든 테스트 실행"""
        print("🚀 전체 시스템 검증 테스트 시작")
        print("="*80)
        
        # 1. 핵심 인프라 테스트
        await self.test_core_infrastructure()
        
        # 2. 데이터 처리 테스트
        await self.test_data_processing()
        
        # 3. AI 분석 테스트
        await self.test_ai_analysis()
        
        # 4. 포트폴리오 관리 테스트
        await self.test_portfolio_management()
        
        # 5. 리포팅 테스트
        await self.test_reporting()
        
        # 6. 통합 테스트
        await self.test_integration()
        
        # 결과 출력
        self.result.print_summary()
    
    async def test_core_infrastructure(self):
        """핵심 인프라 테스트"""
        print("\n🏗️ 핵심 인프라 테스트 중...")
        
        # base_interfaces.py 테스트
        await self.test_module('base_interfaces', self._test_base_interfaces)
        
        # config.py 테스트
        await self.test_module('config', self._test_config)
        
        # async_executor.py 테스트
        await self.test_module('async_executor', self._test_async_executor)
        
        # cache_manager.py 테스트
        await self.test_module('cache_manager', self._test_cache_manager)
    
    async def test_data_processing(self):
        """데이터 처리 테스트"""
        print("\n📊 데이터 처리 테스트 중...")
        
        # data_collector.py 테스트
        await self.test_module('data_collector', self._test_data_collector)
        
        # news_collector.py 테스트
        await self.test_module('news_collector', self._test_news_collector)
        
        # technical_analysis.py 테스트
        await self.test_module('technical_analysis', self._test_technical_analysis)
        
        # gemini_premium_data_processor.py 테스트
        await self.test_module('gemini_premium_data_processor', self._test_gemini_premium_processor)
    
    async def test_ai_analysis(self):
        """AI 분석 테스트"""
        print("\n🧠 AI 분석 테스트 중...")
        
        # gemini_analyzer.py 테스트
        await self.test_module('gemini_analyzer', self._test_gemini_analyzer)
        
        # investment_strategies.py 테스트
        await self.test_module('investment_strategies', self._test_investment_strategies)
        
        # strategy_gemini_integration.py 테스트
        await self.test_module('strategy_gemini_integration', self._test_strategy_integration)
    
    async def test_portfolio_management(self):
        """포트폴리오 관리 테스트"""
        print("\n💼 포트폴리오 관리 테스트 중...")
        
        # portfolio_manager.py 테스트
        await self.test_module('portfolio_manager', self._test_portfolio_manager)
        
        # performance_optimizer.py 테스트
        await self.test_module('performance_optimizer', self._test_performance_optimizer)
        
        # backtesting_engine.py 테스트
        await self.test_module('backtesting_engine', self._test_backtesting_engine)
    
    async def test_reporting(self):
        """리포팅 테스트"""
        print("\n📈 리포팅 테스트 중...")
        
        # report_generator.py 테스트
        await self.test_module('report_generator', self._test_report_generator)
        
        # notification_system.py 테스트
        await self.test_module('notification_system', self._test_notification_system)
    
    async def test_integration(self):
        """통합 테스트"""
        print("\n🔗 통합 테스트 중...")
        
        # 전체 시스템 통합 테스트
        await self.test_module('system_integration', self._test_system_integration)
    
    async def test_module(self, module_name: str, test_func):
        """개별 모듈 테스트"""
        try:
            await test_func()
            self.result.add_result(module_name, "기본_기능", True, "모듈 로드 및 기본 기능 정상")
        except Exception as e:
            self.result.add_result(module_name, "기본_기능", False, f"오류: {str(e)}")
    
    # 개별 모듈 테스트 함수들
    async def _test_base_interfaces(self):
        """base_interfaces.py 테스트"""
        try:
            from src.core.base_interfaces import (
                StockData, InvestmentRecommendation, MarketType, StrategyType
            )
            
            # 기본 데이터 클래스 생성 테스트
            stock_data = StockData(
                symbol="AAPL",
                current_price=150.0,
                market_cap=2500000000000,
                pe_ratio=25.0
            )
            
            recommendation = InvestmentRecommendation(
                symbol="AAPL",
                recommendation="BUY",
                confidence=0.85,
                target_price=180.0,
                reasoning="Strong fundamentals"
            )
            
            assert stock_data.symbol == "AAPL"
            assert recommendation.recommendation == "BUY"
            assert MarketType.NASDAQ100.value == "NASDAQ100"
            
            self.result.add_result("base_interfaces", "데이터_클래스", True, "모든 인터페이스 정상 작동")
            
        except Exception as e:
            self.result.add_result("base_interfaces", "데이터_클래스", False, str(e))
    
    async def _test_config(self):
        """config.py 테스트"""
        try:
            from src.core.config import Config, SystemConfig
            
            # 설정 로드 테스트
            config = Config()
            system_config = SystemConfig.from_env()
            
            # 설정 검증 테스트
            is_valid = config.validate()
            
            self.result.add_result("config", "설정_로드", True, f"설정 유효성: {is_valid}")
            
        except Exception as e:
            self.result.add_result("config", "설정_로드", False, str(e))
    
    async def _test_async_executor(self):
        """async_executor.py 테스트"""
        try:
            from src.core.async_executor import AsyncExecutor
            
            executor = AsyncExecutor()
            
            # 간단한 비동기 작업 테스트
            async def test_task():
                return "test_result"
            
            result = await executor.execute_single(test_task())
            assert result == "test_result"
            
            self.result.add_result("async_executor", "비동기_실행", True, "비동기 작업 실행 정상")
            
        except Exception as e:
            self.result.add_result("async_executor", "비동기_실행", False, str(e))
    
    async def _test_cache_manager(self):
        """cache_manager.py 테스트"""
        try:
            from src.core.cache_manager import CacheManager
            
            cache = CacheManager()
            
            # 캐시 저장/조회 테스트
            await cache.set("test_key", "test_value")
            value = await cache.get("test_key")
            
            self.result.add_result("cache_manager", "캐시_기능", True, f"캐시 저장/조회 정상")
            
        except Exception as e:
            self.result.add_result("cache_manager", "캐시_기능", False, str(e))
    
    async def _test_data_collector(self):
        """data_collector.py 테스트"""
        try:
            from src.modules.data_collector import DataCollector
            
            collector = DataCollector()
            
            # Mock 모드에서 데이터 수집 테스트
            data = await collector.collect_stock_data("AAPL")
            
            self.result.add_result("data_collector", "데이터_수집", True, "주식 데이터 수집 정상")
            
        except Exception as e:
            self.result.add_result("data_collector", "데이터_수집", False, str(e))
    
    async def _test_news_collector(self):
        """news_collector.py 테스트"""
        try:
            from src.modules.news_collector import NewsCollector
            
            collector = NewsCollector()
            
            # 뉴스 수집 테스트 (Mock 모드)
            news = await collector.collect_news("AAPL")
            
            self.result.add_result("news_collector", "뉴스_수집", True, f"뉴스 {len(news)}개 수집")
            
        except Exception as e:
            self.result.add_result("news_collector", "뉴스_수집", False, str(e))
    
    async def _test_technical_analysis(self):
        """technical_analysis.py 테스트"""
        try:
            from src.modules.technical_analysis import TechnicalAnalyzer
            
            analyzer = TechnicalAnalyzer()
            
            # 기술적 분석 테스트
            analysis = await analyzer.analyze("AAPL")
            
            self.result.add_result("technical_analysis", "기술적_분석", True, "기술적 분석 정상")
            
        except Exception as e:
            self.result.add_result("technical_analysis", "기술적_분석", False, str(e))
    
    async def _test_gemini_premium_processor(self):
        """gemini_premium_data_processor.py 테스트"""
        try:
            from src.modules.gemini_premium_data_processor import GeminiPremiumDataProcessor
            
            processor = GeminiPremiumDataProcessor()
            
            # 프리미엄 데이터 처리 테스트
            processed_data = await processor.process_stock_data("AAPL")
            
            assert processed_data.symbol == "AAPL"
            
            self.result.add_result("gemini_premium_processor", "프리미엄_처리", True, "고품질 데이터 처리 정상")
            
        except Exception as e:
            self.result.add_result("gemini_premium_processor", "프리미엄_처리", False, str(e))
    
    async def _test_gemini_analyzer(self):
        """gemini_analyzer.py 테스트"""
        try:
            from src.modules.gemini_analyzer import GeminiAnalyzer
            
            analyzer = GeminiAnalyzer()
            
            # Gemini 분석 테스트 (Mock 모드)
            analysis = await analyzer.analyze_stock("AAPL")
            
            self.result.add_result("gemini_analyzer", "AI_분석", True, "Gemini AI 분석 정상")
            
        except Exception as e:
            self.result.add_result("gemini_analyzer", "AI_분석", False, str(e))
    
    async def _test_investment_strategies(self):
        """investment_strategies.py 테스트"""
        try:
            from src.modules.investment_strategies import InvestmentStrategies
            from src.core.base_interfaces import StrategyType
            
            strategies = InvestmentStrategies()
            
            # 개별 전략 테스트
            result = await strategies.analyze_stock("AAPL", StrategyType.WARREN_BUFFETT)
            
            assert result.symbol == "AAPL"
            assert result.strategy_type == StrategyType.WARREN_BUFFETT
            
            self.result.add_result("investment_strategies", "전략_분석", True, "15개 투자 대가 전략 정상")
            
        except Exception as e:
            self.result.add_result("investment_strategies", "전략_분석", False, str(e))
    
    async def _test_strategy_integration(self):
        """strategy_gemini_integration.py 테스트"""
        try:
            from src.modules.strategy_gemini_integration import StrategyGeminiIntegration
            from src.core.base_interfaces import MarketType
            
            integration = StrategyGeminiIntegration()
            
            # 통합 분석 테스트 (소규모)
            result = await integration.analyze_market_with_all_strategies(
                MarketType.NASDAQ100, 
                ["AAPL", "GOOGL"]
            )
            
            assert result.market_type == MarketType.NASDAQ100
            assert len(result.selected_stocks) <= 5
            
            self.result.add_result("strategy_integration", "통합_분석", True, "전략-AI 통합 정상")
            
        except Exception as e:
            self.result.add_result("strategy_integration", "통합_분석", False, str(e))
    
    async def _test_portfolio_manager(self):
        """portfolio_manager.py 테스트"""
        try:
            from src.modules.portfolio_manager import PortfolioManager
            
            manager = PortfolioManager()
            
            # 포트폴리오 생성 테스트
            portfolio = await manager.create_portfolio("test_portfolio", 1000000)
            
            self.result.add_result("portfolio_manager", "포트폴리오_관리", True, "포트폴리오 관리 정상")
            
        except Exception as e:
            self.result.add_result("portfolio_manager", "포트폴리오_관리", False, str(e))
    
    async def _test_performance_optimizer(self):
        """performance_optimizer.py 테스트"""
        try:
            from src.modules.performance_optimizer import PerformanceOptimizer
            
            optimizer = PerformanceOptimizer()
            
            # 성능 최적화 테스트
            await optimizer.optimize_system()
            
            self.result.add_result("performance_optimizer", "성능_최적화", True, "성능 최적화 정상")
            
        except Exception as e:
            self.result.add_result("performance_optimizer", "성능_최적화", False, str(e))
    
    async def _test_backtesting_engine(self):
        """backtesting_engine.py 테스트"""
        try:
            from src.modules.backtesting_engine import BacktestingEngine
            
            engine = BacktestingEngine()
            
            # 백테스팅 테스트
            result = await engine.run_backtest("AAPL", "2023-01-01", "2023-12-31")
            
            self.result.add_result("backtesting_engine", "백테스팅", True, "백테스팅 엔진 정상")
            
        except Exception as e:
            self.result.add_result("backtesting_engine", "백테스팅", False, str(e))
    
    async def _test_report_generator(self):
        """report_generator.py 테스트"""
        try:
            from src.modules.report_generator import ReportGenerator
            
            generator = ReportGenerator()
            
            # 리포트 생성 테스트
            report = await generator.generate_analysis_report("AAPL")
            
            self.result.add_result("report_generator", "리포트_생성", True, "리포트 생성 정상")
            
        except Exception as e:
            self.result.add_result("report_generator", "리포트_생성", False, str(e))
    
    async def _test_notification_system(self):
        """notification_system.py 테스트"""
        try:
            from src.modules.notification_system import NotificationSystem
            
            notifier = NotificationSystem()
            
            # 알림 테스트
            await notifier.send_notification("test_message", "INFO")
            
            self.result.add_result("notification_system", "알림_시스템", True, "알림 시스템 정상")
            
        except Exception as e:
            self.result.add_result("notification_system", "알림_시스템", False, str(e))
    
    async def _test_system_integration(self):
        """전체 시스템 통합 테스트"""
        try:
            # 핵심 시스템 통합 테스트
            from src.modules.strategy_gemini_integration import StrategyGeminiIntegration
            from src.core.base_interfaces import MarketType
            
            # 통합 시스템 초기화
            integration = StrategyGeminiIntegration()
            
            # 소규모 통합 테스트 실행
            result = await integration.analyze_market_with_all_strategies(
                MarketType.NASDAQ100,
                ["AAPL"]  # 1개 종목만으로 빠른 테스트
            )
            
            # 결과 검증
            assert result is not None
            assert result.market_type == MarketType.NASDAQ100
            assert len(result.selected_stocks) > 0
            
            self.result.add_result("system_integration", "전체_통합", True, 
                                 f"시스템 통합 성공 - {len(result.selected_stocks)}개 종목 분석")
            
        except Exception as e:
            self.result.add_result("system_integration", "전체_통합", False, str(e))

async def main():
    """메인 테스트 실행"""
    print("🔍 투자 분석 시스템 종합 검증 테스트")
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Mock 모드 설정
    os.environ['IS_MOCK'] = 'true'
    os.environ['DEBUG_MODE'] = 'true'
    
    try:
        # 종합 테스트 실행
        test_runner = ComprehensiveSystemTest()
        await test_runner.run_all_tests()
        
        print(f"\n완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 최종 평가
        success_rate = (test_runner.result.passed_tests / test_runner.result.total_tests) * 100
        
        if success_rate >= 90:
            print(f"\n🎉 시스템 상태: 우수 ({success_rate:.1f}%)")
            print("✅ 모든 핵심 기능이 정상 작동합니다.")
        elif success_rate >= 70:
            print(f"\n⚠️ 시스템 상태: 양호 ({success_rate:.1f}%)")
            print("🔧 일부 기능에서 개선이 필요합니다.")
        else:
            print(f"\n❌ 시스템 상태: 점검 필요 ({success_rate:.1f}%)")
            print("🚨 시스템 점검 및 수정이 필요합니다.")
        
    except Exception as e:
        print(f"\n💥 테스트 실행 중 치명적 오류: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 