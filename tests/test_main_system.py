#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 메인 시스템 종합 테스트
AI 기반 투자 분석 시스템의 핵심 기능 테스트
"""

import pytest
import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict, Any

# 프로젝트 루트 경로 설정
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

# 테스트 대상 모듈
from src.main_optimized import InvestmentAnalysisSystem, setup_environment, setup_logging
from investment_strategies import StockData, StrategyScore
from data_collector import MultiDataCollector
from ai_analyzer import GeminiAnalysisResult

class TestInvestmentAnalysisSystem:
    """🚀 투자 분석 시스템 테스트 클래스"""
    
    @pytest.fixture
    def mock_stock_data(self) -> List[StockData]:
        """테스트용 모의 주식 데이터"""
        return [
            StockData(
                symbol="005930",
                name="삼성전자",
                current_price=75000,
                market="KOSPI200",
                pe_ratio=12.5,
                pb_ratio=1.2,
                roe=0.15,
                debt_ratio=0.3,
                dividend_yield=0.025,
                rsi=55.0,
                market_cap=4.5e14
            ),
            StockData(
                symbol="AAPL",
                name="Apple Inc",
                current_price=180.0,
                market="NASDAQ100",
                pe_ratio=28.5,
                pb_ratio=12.8,
                roe=0.35,
                debt_ratio=0.4,
                dividend_yield=0.005,
                rsi=62.0,
                market_cap=2.8e12
            ),
            StockData(
                symbol="MSFT",
                name="Microsoft Corp",
                current_price=340.0,
                market="S&P500",
                pe_ratio=32.1,
                pb_ratio=11.2,
                roe=0.42,
                debt_ratio=0.25,
                dividend_yield=0.007,
                rsi=58.5,
                market_cap=2.5e12
            )
        ]
    
    @pytest.fixture
    def mock_strategy_scores(self) -> Dict[str, List[StrategyScore]]:
        """테스트용 모의 전략 점수"""
        return {
            "005930": [
                StrategyScore(
                    symbol="005930",
                    name="삼성전자",
                    strategy_name="Benjamin Graham",
                    total_score=85.5,
                    criteria_scores={"value": 40, "safety": 30, "dividend": 15.5},
                    reasoning="우수한 가치투자 대상",
                    confidence=0.85
                )
            ],
            "AAPL": [
                StrategyScore(
                    symbol="AAPL",
                    name="Apple Inc",
                    strategy_name="Warren Buffett",
                    total_score=92.3,
                    criteria_scores={"profitability": 35, "quality": 30, "moat": 27.3},
                    reasoning="뛰어난 수익성과 경쟁우위",
                    confidence=0.92
                )
            ]
        }
    
    @pytest.fixture
    def mock_gemini_result(self) -> GeminiAnalysisResult:
        """테스트용 모의 Gemini AI 결과"""
        return GeminiAnalysisResult(
            top5_stocks=[
                {"symbol": "AAPL", "name": "Apple Inc", "score": 95.0},
                {"symbol": "MSFT", "name": "Microsoft Corp", "score": 92.0},
                {"symbol": "005930", "name": "삼성전자", "score": 88.0}
            ],
            reasoning="기술적 지표와 투자 대가 전략을 종합한 결과",
            market_outlook="긍정적",
            risk_assessment="중간",
            confidence_score=0.89,
            alternative_picks=[
                {"symbol": "GOOGL", "name": "Alphabet Inc", "score": 85.0}
            ]
        )
    
    @pytest.fixture
    async def system(self):
        """테스트용 시스템 인스턴스"""
        with patch.multiple(
            'src.main_optimized',
            MultiDataCollector=Mock(),
            StrategyManager=Mock(),
            TechnicalAnalyzer=Mock(),
            GeminiAIAnalyzer=Mock(),
            NewsAnalyzer=Mock()
        ):
            system = InvestmentAnalysisSystem()
            return system
    
    def test_system_initialization(self, system):
        """시스템 초기화 테스트"""
        # Given & When: 시스템이 초기화됨
        
        # Then: 모든 컴포넌트가 올바르게 초기화되어야 함
        assert system.data_collector is not None
        assert system.strategy_manager is not None
        assert system.technical_analyzer is not None
        assert system.ai_analyzer is not None
        assert system.news_analyzer is not None
        
        # 초기 상태 확인
        assert len(system.collected_stocks) == 0
        assert len(system.strategy_results) == 0
        assert len(system.technical_results) == 0
        assert len(system.final_results) == 0
        
        # 성능 메트릭 초기 상태
        assert 'start_time' in system.performance_metrics
        assert system.performance_metrics['start_time'] is None
    
    @pytest.mark.asyncio
    async def test_step1_collect_market_data(self, system, mock_stock_data):
        """1단계: 시장 데이터 수집 테스트"""
        # Given: 모의 데이터 수집기 설정
        system.data_collector.collect_all_markets = AsyncMock(return_value=mock_stock_data)
        system.data_collector.get_collection_stats = Mock(return_value={
            'success_rate': 0.95,
            'failed_count': 2
        })
        
        # When: 데이터 수집 실행
        await system._step1_collect_market_data()
        
        # Then: 데이터가 올바르게 수집되어야 함
        assert len(system.collected_stocks) == 3
        assert system.collected_stocks[0].symbol == "005930"
        assert system.collected_stocks[1].symbol == "AAPL"
        assert system.performance_metrics['data_collection_time'] > 0
        
        # 수집기 메서드 호출 확인
        system.data_collector.collect_all_markets.assert_called_once()
        system.data_collector.get_collection_stats.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_step2_clean_and_validate_data(self, system, mock_stock_data):
        """2단계: 데이터 정제 및 검증 테스트"""
        # Given: 초기 데이터 설정
        system.collected_stocks = mock_stock_data.copy()
        cleaned_data = mock_stock_data[:2]  # 1개 제거된 상황 시뮬레이션
        system.data_cleaner.clean_stock_data = Mock(return_value=cleaned_data)
        
        # When: 데이터 정제 실행
        await system._step2_clean_and_validate_data()
        
        # Then: 데이터가 올바르게 정제되어야 함
        assert len(system.collected_stocks) == 2
        system.data_cleaner.clean_stock_data.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_step3_apply_investment_strategies(self, system, mock_stock_data, mock_strategy_scores):
        """3단계: 투자 전략 적용 테스트"""
        # Given: 데이터와 전략 매니저 설정
        system.collected_stocks = mock_stock_data
        system.strategy_manager.apply_all_strategies = Mock(return_value=mock_strategy_scores)
        
        # When: 투자 전략 적용
        await system._step3_apply_investment_strategies()
        
        # Then: 전략 결과가 올바르게 저장되어야 함
        assert len(system.strategy_results) == 2
        assert "005930" in system.strategy_results
        assert "AAPL" in system.strategy_results
        assert system.performance_metrics['strategy_analysis_time'] > 0
        
        system.strategy_manager.apply_all_strategies.assert_called_once_with(mock_stock_data)
    
    @pytest.mark.asyncio
    async def test_step4_technical_analysis(self, system, mock_stock_data):
        """4단계: 기술적 분석 테스트"""
        # Given: 주식 데이터와 기술적 분석기 설정
        system.collected_stocks = mock_stock_data
        
        mock_tech_result = Mock()
        mock_tech_result.signals = [Mock(signal_type="BUY")]
        mock_tech_result.trend_direction = "UPTREND"
        mock_tech_result.volatility_score = 65.0
        
        system.technical_analyzer.analyze_stock = AsyncMock(return_value=mock_tech_result)
        
        # When: 기술적 분석 실행
        await system._step4_technical_analysis()
        
        # Then: 분석 결과가 올바르게 저장되어야 함
        assert len(system.technical_results) == 3
        assert system.performance_metrics['technical_analysis_time'] > 0
        
        # 각 종목에 대해 분석이 호출되었는지 확인
        assert system.technical_analyzer.analyze_stock.call_count == 3
    
    @pytest.mark.asyncio
    async def test_step5_ai_comprehensive_analysis(self, system, mock_stock_data, 
                                                  mock_strategy_scores, mock_gemini_result):
        """5단계: Gemini AI 종합 분석 테스트"""
        # Given: 모든 필요한 데이터 설정
        system.collected_stocks = mock_stock_data
        system.strategy_results = mock_strategy_scores
        system.technical_results = {"005930": Mock(), "AAPL": Mock()}
        
        system.ai_analyzer.analyze_and_select_top5 = Mock(return_value=mock_gemini_result)
        
        # When: AI 분석 실행
        await system._step5_ai_comprehensive_analysis()
        
        # Then: AI 분석 결과가 올바르게 저장되어야 함
        assert 'ai_analysis' in system.final_results
        assert system.final_results['ai_analysis'] == mock_gemini_result
        assert system.performance_metrics['ai_analysis_time'] > 0
        
        system.ai_analyzer.analyze_and_select_top5.assert_called_once_with(
            stocks=mock_stock_data,
            technical_results=system.technical_results,
            strategy_scores=mock_strategy_scores
        )
    
    @pytest.mark.asyncio
    async def test_step6_generate_final_report(self, system, mock_stock_data, mock_strategy_scores):
        """6단계: 최종 리포트 생성 테스트"""
        # Given: 분석 완료된 시스템 상태
        system.collected_stocks = mock_stock_data
        system.strategy_results = mock_strategy_scores
        system.technical_results = {"005930": Mock(), "AAPL": Mock()}
        
        # _save_reports 메서드 모킹
        system._save_reports = AsyncMock()
        
        # When: 리포트 생성 실행
        await system._step6_generate_final_report()
        
        # Then: 최종 결과가 올바르게 구성되어야 함
        assert 'analysis_summary' in system.final_results
        assert 'performance_metrics' in system.final_results
        assert 'market_breakdown' in system.final_results
        assert 'strategy_summary' in system.final_results
        
        # 분석 요약 검증
        summary = system.final_results['analysis_summary']
        assert summary['total_stocks_analyzed'] == 3
        assert summary['strategies_applied'] == 2
        assert summary['technical_analysis_count'] == 2
        assert summary['system_version'] == '3.0.0'
        
        # 리포트 저장 메서드 호출 확인
        system._save_reports.assert_called_once()
    
    def test_get_market_breakdown(self, system, mock_stock_data):
        """시장별 분포 계산 테스트"""
        # Given: 주식 데이터 설정
        system.collected_stocks = mock_stock_data
        
        # When: 시장 분포 계산
        breakdown = system._get_market_breakdown()
        
        # Then: 올바른 분포가 계산되어야 함
        assert breakdown["KOSPI200"] == 1
        assert breakdown["NASDAQ100"] == 1
        assert breakdown["S&P500"] == 1
    
    def test_get_strategy_summary(self, system, mock_strategy_scores):
        """전략별 요약 계산 테스트"""
        # Given: 전략 점수 설정
        system.strategy_results = mock_strategy_scores
        
        # When: 전략 요약 계산
        summary = system._get_strategy_summary()
        
        # Then: 올바른 요약이 계산되어야 함
        assert len(summary) == 2  # 2개 종목
        
        # 첫 번째 종목 요약 확인
        first_summary = list(summary.values())[0]
        assert 'analyzed_stocks' in first_summary
        assert 'average_score' in first_summary
        assert 'top_score' in first_summary
    
    @pytest.mark.asyncio
    async def test_run_complete_analysis_integration(self, system, mock_stock_data, 
                                                   mock_strategy_scores, mock_gemini_result):
        """전체 분석 프로세스 통합 테스트"""
        # Given: 모든 컴포넌트 모킹
        system.data_collector.collect_all_markets = AsyncMock(return_value=mock_stock_data)
        system.data_collector.get_collection_stats = Mock(return_value={'success_rate': 1.0, 'failed_count': 0})
        system.data_cleaner.clean_stock_data = Mock(return_value=mock_stock_data)
        system.strategy_manager.apply_all_strategies = Mock(return_value=mock_strategy_scores)
        system.technical_analyzer.analyze_stock = AsyncMock(return_value=Mock())
        system.ai_analyzer.analyze_and_select_top5 = Mock(return_value=mock_gemini_result)
        system._save_reports = AsyncMock()
        
        # When: 전체 분석 실행
        results = await system.run_complete_analysis()
        
        # Then: 모든 단계가 완료되고 결과가 반환되어야 함
        assert results is not None
        assert 'analysis_summary' in results
        assert 'ai_analysis' in results
        assert system.performance_metrics['total_execution_time'] > 0
        
        # 모든 주요 메서드가 호출되었는지 확인
        system.data_collector.collect_all_markets.assert_called_once()
        system.strategy_manager.apply_all_strategies.assert_called_once()
        system.ai_analyzer.analyze_and_select_top5.assert_called_once()
    
    def test_display_results(self, system, mock_gemini_result, capsys):
        """결과 출력 테스트"""
        # Given: 최종 결과 설정
        system.final_results = {
            'analysis_summary': {
                'total_stocks_analyzed': 100,
                'strategies_applied': 5,
                'technical_analysis_count': 95
            },
            'market_breakdown': {
                'KOSPI200': 50,
                'NASDAQ100': 30,
                'S&P500': 20
            },
            'ai_analysis': mock_gemini_result,
            'performance_metrics': {
                'total_execution_time': 120.5,
                'data_collection_time': 30.2,
                'strategy_analysis_time': 45.1,
                'technical_analysis_time': 25.8,
                'ai_analysis_time': 19.4
            }
        }
        
        # When: 결과 출력
        system.display_results()
        
        # Then: 올바른 내용이 출력되어야 함
        captured = capsys.readouterr()
        assert "AI 기반 투자 분석 시스템 v3.0" in captured.out
        assert "총 분석 종목: 100개" in captured.out
        assert "KOSPI200: 50개" in captured.out
        assert "Gemini AI Top5 선정 결과" in captured.out
        assert "총 실행시간: 120.50초" in captured.out


class TestEnvironmentSetup:
    """환경 설정 함수 테스트"""
    
    @patch('src.main_optimized.ROOT_DIR')
    def test_setup_environment_creates_directories(self, mock_root_dir, tmp_path):
        """환경 설정시 필요한 디렉토리 생성 테스트"""
        # Given: 임시 루트 디렉토리 설정
        mock_root_dir.__truediv__ = lambda self, path: tmp_path / path
        
        # When: 환경 설정 실행
        setup_environment()
        
        # Then: 필요한 디렉토리들이 생성되어야 함
        assert (tmp_path / "data" / "logs").exists()
        assert (tmp_path / "data" / "reports").exists()
        assert (tmp_path / "data" / "cache").exists()
        assert (tmp_path / "data" / "temp").exists()
    
    @patch.dict(os.environ, {}, clear=True)
    def test_setup_environment_missing_env_vars(self, caplog):
        """환경 변수 누락시 경고 로그 테스트"""
        # When: 환경 설정 실행 (환경 변수 없음)
        setup_environment()
        
        # Then: 경고 로그가 출력되어야 함
        assert "누락된 환경변수" in caplog.text
    
    @patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'})
    def test_setup_environment_with_env_vars(self, caplog):
        """환경 변수 존재시 정상 동작 테스트"""
        # When: 환경 설정 실행 (환경 변수 있음)
        setup_environment()
        
        # Then: 경고 로그가 없어야 함
        assert "누락된 환경변수" not in caplog.text


class TestPerformanceMetrics:
    """성능 메트릭 테스트"""
    
    def test_performance_metrics_initialization(self):
        """성능 메트릭 초기화 테스트"""
        with patch.multiple(
            'src.main_optimized',
            MultiDataCollector=Mock(),
            StrategyManager=Mock(),
            TechnicalAnalyzer=Mock(),
            GeminiAIAnalyzer=Mock(),
            NewsAnalyzer=Mock()
        ):
            system = InvestmentAnalysisSystem()
            
            # 모든 성능 메트릭이 초기화되어야 함
            expected_metrics = [
                'start_time', 'data_collection_time', 'strategy_analysis_time',
                'technical_analysis_time', 'ai_analysis_time', 'total_execution_time'
            ]
            
            for metric in expected_metrics:
                assert metric in system.performance_metrics
    
    @pytest.mark.asyncio
    async def test_performance_timing_accuracy(self):
        """성능 측정 정확도 테스트"""
        with patch.multiple(
            'src.main_optimized',
            MultiDataCollector=Mock(),
            StrategyManager=Mock(),
            TechnicalAnalyzer=Mock(),
            GeminiAIAnalyzer=Mock(),
            NewsAnalyzer=Mock()
        ):
            system = InvestmentAnalysisSystem()
            
            # 모의 지연 추가
            async def delayed_collect():
                await asyncio.sleep(0.1)  # 100ms 지연
                return []
            
            system.data_collector.collect_all_markets = delayed_collect
            system.data_collector.get_collection_stats = Mock(return_value={})
            
            # 데이터 수집 실행
            await system._step1_collect_market_data()
            
            # 측정된 시간이 실제 지연 시간과 비슷해야 함
            assert system.performance_metrics['data_collection_time'] >= 0.1


if __name__ == "__main__":
    """테스트 실행"""
    pytest.main([__file__, "-v", "--tb=short"]) 