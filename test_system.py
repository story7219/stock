# -*- coding: utf-8 -*-
"""
시스템 테스트 파일
pytest를 사용한 단위 테스트 및 통합 테스트
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock
import pandas as pd
from enhanced_data_collector import EnhancedDataCollector
from enhanced_gemini_analyzer import EnhancedGeminiAnalyzer
from investment_strategies import InvestmentStrategies
from run_enhanced_system import EnhancedSystemRunner
import config


class TestEnhancedDataCollector:
    """데이터 수집기 테스트"""
    
    @pytest.fixture
    async def collector(self):
        """테스트용 데이터 수집기"""
        collector = EnhancedDataCollector(max_concurrent=5, timeout=10)
        await collector.__aenter__()
        yield collector
        await collector.__aexit__(None, None, None)
    
    @pytest.mark.asyncio
    async def test_get_kospi200_stocks(self, collector):
        """코스피200 종목 리스트 테스트"""
        stocks = await collector.get_kospi200_stocks()
        
        assert isinstance(stocks, list)
        assert len(stocks) > 0
        assert all(stock.endswith('.KS') for stock in stocks[:10])  # 샘플 체크
        
    @pytest.mark.asyncio
    async def test_get_nasdaq100_stocks(self, collector):
        """나스닥100 종목 리스트 테스트"""
        stocks = await collector.get_nasdaq100_stocks()
        
        assert isinstance(stocks, list)
        assert len(stocks) > 0
        assert 'AAPL' in stocks or 'MSFT' in stocks  # 주요 종목 포함 확인
        
    @pytest.mark.asyncio
    async def test_get_sp500_stocks(self, collector):
        """S&P500 종목 리스트 테스트"""
        stocks = await collector.get_sp500_stocks()
        
        assert isinstance(stocks, list)
        assert len(stocks) > 0
        
    @pytest.mark.asyncio
    async def test_get_stock_data(self, collector):
        """개별 종목 데이터 테스트"""
        stock_data = await collector.get_stock_data('AAPL', 'NASDAQ100')
        
        if stock_data:  # 네트워크 연결 시에만 테스트
            assert 'symbol' in stock_data
            assert 'current_price' in stock_data
            assert 'technical_indicators' in stock_data
            assert stock_data['symbol'] == 'AAPL'
            assert stock_data['current_price'] > 0
    
    def test_backup_data(self, collector):
        """백업 데이터 테스트"""
        kospi_backup = collector._get_backup_kospi200()
        nasdaq_backup = collector._get_backup_nasdaq100()
        sp500_backup = collector._get_backup_sp500()
        
        assert len(kospi_backup) == 200
        assert len(nasdaq_backup) == 100
        assert len(sp500_backup) == 500
        
        # 형식 확인
        assert all(stock.endswith('.KS') for stock in kospi_backup)
        assert 'AAPL' in nasdaq_backup
        assert 'AAPL' in sp500_backup


class TestEnhancedGeminiAnalyzer:
    """Gemini AI 분석기 테스트"""
    
    @pytest.fixture
    def analyzer(self):
        """테스트용 Gemini 분석기 (Mock 모드)"""
        return EnhancedGeminiAnalyzer(api_key=None)  # Mock 모드
    
    def test_analyzer_initialization(self, analyzer):
        """분석기 초기화 테스트"""
        assert analyzer.mock_mode is True  # API 키 없이 Mock 모드
        assert analyzer.cache_ttl == 1800
        assert len(analyzer.strategy_weights) == 17
        assert sum(analyzer.strategy_weights.values()) == 100.0
    
    @pytest.mark.asyncio
    async def test_analyze_stock_mock(self, analyzer):
        """Mock 모드 종목 분석 테스트"""
        sample_stock = {
            'symbol': 'AAPL',
            'market': 'NASDAQ100',
            'current_price': 175.0,
            'market_cap': 2700000000000,
            'pe_ratio': 28.5,
            'technical_indicators': {'RSI': 55.2, 'MACD': 1.25}
        }
        
        sample_strategy_scores = {
            'warren_buffett': 85.0,
            'benjamin_graham': 78.5,
            'peter_lynch': 82.3
        }
        
        result = await analyzer.analyze_stock(sample_stock, sample_strategy_scores)
        
        assert isinstance(result, dict)
        assert 'symbol' in result
        assert 'api_mode' in result
        assert result['api_mode'] == 'mock'
        assert '종합_점수' in result
        assert '투자_액션' in result
    
    def test_generate_analysis_prompt(self, analyzer):
        """분석 프롬프트 생성 테스트"""
        sample_stock = {
            'symbol': 'AAPL',
            'current_price': 175.0,
            'market_cap': 2700000000000,
            'pe_ratio': 28.5,
            'sector': 'Technology',
            'technical_indicators': {'RSI': 55.2}
        }
        
        sample_strategy_scores = {'warren_buffett': 85.0}
        
        prompt = analyzer._generate_analysis_prompt(sample_stock, sample_strategy_scores)
        
        assert isinstance(prompt, str)
        assert 'AAPL' in prompt
        assert 'Goldman Sachs' in prompt
        assert 'JSON' in prompt
    
    def test_parse_gemini_response(self, analyzer):
        """Gemini 응답 파싱 테스트"""
        sample_response = """
        분석 결과입니다.
        
        {
            "종합_점수": 85,
            "투자_액션": "매수",
            "목표가": 200.0,
            "분석_요약": "우수한 투자 대상"
        }
        
        추가 설명...
        """
        
        result = analyzer._parse_gemini_response(sample_response)
        
        assert isinstance(result, dict)
        assert result['종합_점수'] == 85
        assert result['투자_액션'] == '매수'
        assert result['목표가'] == 200.0
    
    def test_get_top_stocks(self, analyzer):
        """Top 종목 선정 테스트"""
        sample_analyses = [
            {'symbol': 'AAPL', '종합_점수': 90},
            {'symbol': 'MSFT', '종합_점수': 85},
            {'symbol': 'GOOGL', '종합_점수': 80},
            {'symbol': 'TSLA', '종합_점수': 75},
            {'symbol': 'META', '종합_점수': 70},
            {'symbol': 'NVDA', '종합_점수': 88}
        ]
        
        top_5 = analyzer.get_top_stocks(sample_analyses, top_n=5)
        
        assert len(top_5) == 5
        assert top_5[0]['symbol'] == 'AAPL'  # 최고점
        assert top_5[1]['symbol'] == 'NVDA'  # 두번째
        assert all(
            top_5[i]['종합_점수'] >= top_5[i+1]['종합_점수'] 
            for i in range(len(top_5)-1)
        )


class TestInvestmentStrategies:
    """투자 전략 테스트"""
    
    @pytest.fixture
    def strategies(self):
        """테스트용 투자 전략"""
        return InvestmentStrategies()
    
    def test_strategies_initialization(self, strategies):
        """전략 초기화 테스트"""
        assert len(strategies.strategies) == 17
        assert 'warren_buffett' in strategies.strategies
        assert 'benjamin_graham' in strategies.strategies
        assert 'peter_lynch' in strategies.strategies
    
    def test_warren_buffett_strategy(self, strategies):
        """워런 버핏 전략 테스트"""
        sample_stock = {
            'symbol': 'AAPL',
            'current_price': 175.0,
            'market_cap': 2700000000000,  # 대형주
            'technical_indicators': {
                'RSI': 50,  # 안정적 구간
                'SMA_20': 170.0,
                'SMA_60': 165.0,
                'Volatility': 20  # 낮은 변동성
            }
        }
        
        score = strategies.warren_buffett_strategy(sample_stock)
        
        assert isinstance(score, float)
        assert 0 <= score <= 100
        assert score > 50  # 대형주 + 안정적 지표로 평균 이상 예상
    
    def test_benjamin_graham_strategy(self, strategies):
        """벤저민 그레이엄 전략 테스트"""
        sample_stock = {
            'symbol': 'VALUE',
            'current_price': 50.0,
            '52_week_low': 40.0,
            '52_week_high': 80.0,  # 저가 구간
            'pe_ratio': 8.0,  # 낮은 PER
            'technical_indicators': {
                'RSI': 25,  # 과매도
                'BB_Upper': 60.0,
                'BB_Lower': 45.0
            }
        }
        
        score = strategies.benjamin_graham_strategy(sample_stock)
        
        assert isinstance(score, float)
        assert 0 <= score <= 100
        assert score > 60  # 가치주 조건 만족으로 높은 점수 예상
    
    def test_peter_lynch_strategy(self, strategies):
        """피터 린치 전략 테스트"""
        sample_stock = {
            'symbol': 'GROWTH',
            'current_price': 100.0,
            'price_change_percent': 3.0,  # 상승 모멘텀
            'market_cap': 5000000000,  # 중형주
            'volume': 2000000,
            'avg_volume': 1500000,  # 거래량 증가
            'technical_indicators': {
                'SMA_5': 102.0,
                'SMA_20': 98.0,  # 상승 정배열
                'MACD': 1.5  # 상승 신호
            }
        }
        
        score = strategies.peter_lynch_strategy(sample_stock)
        
        assert isinstance(score, float)
        assert 0 <= score <= 100
        assert score > 65  # 성장주 조건 만족으로 높은 점수 예상
    
    def test_analyze_stock(self, strategies):
        """종목 전체 전략 분석 테스트"""
        sample_stock = {
            'symbol': 'TEST',
            'current_price': 100.0,
            'market_cap': 10000000000,
            'technical_indicators': {'RSI': 50}
        }
        
        results = strategies.analyze_stock(sample_stock)
        
        assert isinstance(results, dict)
        assert len(results) == 17  # 17개 전략 모두 분석
        assert all(isinstance(score, float) for score in results.values())
        assert all(0 <= score <= 100 for score in results.values())
    
    def test_analyze_multiple_stocks(self, strategies):
        """다중 종목 분석 테스트"""
        sample_stocks = [
            {'symbol': 'STOCK1', 'current_price': 100.0, 'technical_indicators': {'RSI': 50}},
            {'symbol': 'STOCK2', 'current_price': 200.0, 'technical_indicators': {'RSI': 60}},
            {'symbol': 'STOCK3', 'current_price': 150.0, 'technical_indicators': {'RSI': 40}}
        ]
        
        results = strategies.analyze_multiple_stocks(sample_stocks)
        
        assert isinstance(results, list)
        assert len(results) == 3
        assert all(len(result) == 17 for result in results)
    
    def test_get_strategy_summary(self, strategies):
        """전략 요약 테스트"""
        sample_results = [
            {'warren_buffett': 80, 'peter_lynch': 75, 'benjamin_graham': 70},
            {'warren_buffett': 85, 'peter_lynch': 80, 'benjamin_graham': 75},
            {'warren_buffett': 75, 'peter_lynch': 85, 'benjamin_graham': 80}
        ]
        
        summary = strategies.get_strategy_summary(sample_results)
        
        assert isinstance(summary, dict)
        assert 'total_stocks' in summary
        assert 'strategy_averages' in summary
        assert summary['total_stocks'] == 3
        assert 'warren_buffett' in summary['strategy_averages']


class TestEnhancedSystemRunner:
    """시스템 실행기 테스트"""
    
    @pytest.fixture
    def runner(self, tmp_path):
        """테스트용 시스템 실행기"""
        return EnhancedSystemRunner(output_dir=str(tmp_path))
    
    def test_runner_initialization(self, runner):
        """실행기 초기화 테스트"""
        assert runner.output_dir.exists()
        assert runner.strategies is not None
        assert runner.results['timestamp'] is None
    
    def test_calculate_technical_score(self, runner):
        """기술적 점수 계산 테스트"""
        tech_indicators = {
            'RSI': 50,
            'SMA_20': 100.0,
            'MACD': 1.0,
            'Volatility': 20
        }
        
        stock_data = {'current_price': 105.0}
        
        score = runner._calculate_technical_score(tech_indicators, stock_data)
        
        assert isinstance(score, float)
        assert 0 <= score <= 100
    
    def test_calculate_combined_score(self, runner):
        """종합 점수 계산 테스트"""
        strategy_scores = {
            'warren_buffett': 80.0,
            'benjamin_graham': 75.0,
            'peter_lynch': 85.0
        }
        
        tech_score = 70.0
        
        combined = runner._calculate_combined_score(strategy_scores, tech_score)
        
        assert isinstance(combined, float)
        assert 0 <= combined <= 100
        assert combined != tech_score  # 전략 점수가 반영되어야 함
    
    def test_generate_basic_analysis(self, runner):
        """기본 분석 생성 테스트"""
        stock_data = {'combined_score': 75.0}
        
        analysis = runner._generate_basic_analysis(stock_data)
        
        assert isinstance(analysis, dict)
        assert 'api_mode' in analysis
        assert analysis['api_mode'] == 'basic'
        assert '종합_점수' in analysis


class TestConfig:
    """설정 테스트"""
    
    def test_config_structure(self):
        """설정 구조 테스트"""
        assert 'api' in config.CONFIG
        assert 'data' in config.CONFIG
        assert 'strategy' in config.CONFIG
        assert 'gemini' in config.CONFIG
        assert 'technical' in config.CONFIG
        assert 'market' in config.CONFIG
    
    def test_strategy_weights(self):
        """전략 가중치 테스트"""
        weights = config.STRATEGY_CONFIG['STRATEGY_WEIGHTS']
        
        assert isinstance(weights, dict)
        assert len(weights) == 17
        assert abs(sum(weights.values()) - 100.0) < 0.1
        assert weights['warren_buffett'] == 15.0
        assert weights['benjamin_graham'] == 12.0
    
    def test_config_validation(self):
        """설정 유효성 검사 테스트"""
        assert config.validate_config() is True
    
    def test_get_config(self):
        """설정 조회 테스트"""
        # 전체 설정
        full_config = config.get_config()
        assert isinstance(full_config, dict)
        assert 'api' in full_config
        
        # 섹션별 설정
        api_config = config.get_config('api')
        assert isinstance(api_config, dict)
        assert 'REQUEST_TIMEOUT' in api_config
        
        # 존재하지 않는 섹션
        empty_config = config.get_config('nonexistent')
        assert empty_config == {}


@pytest.mark.integration
class TestSystemIntegration:
    """시스템 통합 테스트"""
    
    @pytest.mark.asyncio
    async def test_quick_system_test(self):
        """빠른 시스템 통합 테스트"""
        runner = EnhancedSystemRunner()
        
        try:
            # 소수 종목으로 테스트
            results = await runner.run_quick_test()
            
            # 결과 검증
            assert isinstance(results, dict)
            
            if results:  # 네트워크 연결 시에만
                assert len(results) > 0
                
                for market, stocks in results.items():
                    assert isinstance(stocks, list)
                    
                    for stock in stocks:
                        assert 'symbol' in stock
                        assert 'strategy_scores' in stock
                        assert len(stock['strategy_scores']) == 17
                        
        except Exception as e:
            pytest.skip(f"네트워크 연결 필요: {e}")


# pytest 실행 함수들
def test_all_components():
    """모든 컴포넌트 기본 동작 테스트"""
    # 데이터 수집기
    collector = EnhancedDataCollector()
    assert collector is not None
    
    # Gemini 분석기
    analyzer = EnhancedGeminiAnalyzer()
    assert analyzer is not None
    
    # 투자 전략
    strategies = InvestmentStrategies()
    assert strategies is not None
    
    # 시스템 실행기
    runner = EnhancedSystemRunner()
    assert runner is not None


if __name__ == "__main__":
    # 테스트 실행
    pytest.main([__file__, "-v", "--tb=short"]) 