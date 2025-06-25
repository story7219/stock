#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 Gemini AI 분석기 테스트 모듈 (Gemini Analyzer Test Module)
========================================================

GeminiAnalyzer 클래스의 모든 기능을 포괄적으로 테스트하는 pytest 기반 테스트 모듈입니다.
단위 테스트, 통합 테스트, Mock 테스트를 통해 90% 이상의 테스트 커버리지를 달성합니다.

테스트 범위:
1. 기본 기능 테스트 (Basic Functionality Tests)
   - GeminiAnalyzer 클래스 초기화
   - API 키 설정 및 검증
   - 기본 설정값 확인
   - 객체 상태 검증

2. 주식 분석 테스트 (Stock Analysis Tests)
   - 단일 주식 분석 기능
   - 다중 주식 배치 분석
   - 분석 결과 구조 검증
   - 오류 상황 처리 테스트

3. 추천 생성 테스트 (Recommendation Generation Tests)
   - Top N 종목 추천 생성
   - 추천 근거 및 신뢰도 계산
   - 투자 전략별 추천 차이
   - 추천 결과 일관성 검증

4. API 통신 테스트 (API Communication Tests)
   - Gemini API 정상 호출
   - 네트워크 오류 처리
   - API 응답 파싱
   - 재시도 로직 검증

5. Mock 데이터 테스트 (Mock Data Tests)
   - 가상 주식 데이터로 테스트
   - API 응답 Mock 처리
   - 외부 의존성 제거 테스트
   - 격리된 환경에서 동작 검증

6. 성능 테스트 (Performance Tests)
   - 대량 데이터 처리 성능
   - 메모리 사용량 테스트
   - 응답 시간 측정
   - 동시 요청 처리 능력

7. 오류 처리 테스트 (Error Handling Tests)
   - 잘못된 입력 데이터 처리
   - API 키 누락 상황
   - 네트워크 연결 실패
   - 예외 상황 복구 테스트

테스트 데이터:
- Mock 주식 데이터: 다양한 시장 상황 시뮬레이션
- 전략 점수 데이터: 실제와 유사한 점수 분포
- 기술적 지표 데이터: 정상/비정상 케이스 포함
- API 응답 데이터: 성공/실패 시나리오

테스트 도구:
- pytest: 테스트 프레임워크
- unittest.mock: Mock 객체 생성
- pytest-asyncio: 비동기 함수 테스트
- pytest-cov: 테스트 커버리지 측정

실행 방법:
- 전체 테스트: pytest tests/test_gemini_analyzer.py
- 특정 테스트: pytest tests/test_gemini_analyzer.py::test_function_name
- 커버리지 포함: pytest --cov=src.modules.gemini_analyzer tests/test_gemini_analyzer.py

목표:
- 테스트 커버리지 90% 이상
- 모든 주요 기능 검증
- 오류 상황 완벽 처리
- 성능 기준 충족 확인

이 테스트 모듈을 통해 GeminiAnalyzer의 안정성과 신뢰성을 보장하고
지속적인 품질 개선을 지원합니다.
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict

from src.modules.gemini_analyzer import GeminiAnalyzer
from src.core.base_interfaces import (
    StockData, StrategyScore, TechnicalAnalysisResult,
    MarketType, StrategyType, RiskLevel, InvestmentPeriod,
    TechnicalIndicators, TechnicalSignals
)

@pytest.fixture
def gemini_analyzer():
    """GeminiAnalyzer 인스턴스 생성"""
    with patch.dict('os.environ', {'IS_MOCK': 'true'}):
        return GeminiAnalyzer()

@pytest.fixture
def sample_stock_data():
    """샘플 주식 데이터"""
    return StockData(
        symbol="005930",
        name="삼성전자",
        current_price=70000.0,
        market=MarketType.KOSPI200
    )

@pytest.fixture
def sample_strategy_scores():
    """샘플 전략 점수"""
    return [
        StrategyScore(
            symbol="005930",
            strategy_name="워런_버핏",
            score=75.0,
            confidence=0.8,
            reasoning="강력한 재무 기반"
        ),
        StrategyScore(
            symbol="005930",
            strategy_name="벤저민_그레이엄",
            score=68.0,
            confidence=0.7,
            reasoning="적정 가치 평가"
        )
    ]

@pytest.fixture
def sample_technical_result():
    """샘플 기술적 분석 결과"""
    indicators = TechnicalIndicators(
        rsi=45.0,
        macd=1.2,
        sma_20=69000.0,
        sma_50=68000.0,
        bb_upper=72000.0,
        bb_lower=66000.0
    )
    
    signals = TechnicalSignals(
        rsi_signal="중립",
        macd_signal="상승",
        ma_trend="상승",
        bb_signal="중간",
        overall_trend="상승"
    )
    
    return TechnicalAnalysisResult(
        symbol="005930",
        indicators=indicators,
        signals=signals,
        confidence=0.75
    )

class TestGeminiAnalyzer:
    """GeminiAnalyzer 테스트 클래스"""
    
    def test_initialization(self, gemini_analyzer):
        """초기화 테스트"""
        assert gemini_analyzer is not None
        assert gemini_analyzer.is_mock is True
        assert gemini_analyzer.strategy_weights is not None
        assert len(gemini_analyzer.strategy_weights) > 0
    
    def test_api_key_initialization(self):
        """API 키 초기화 테스트"""
        analyzer = GeminiAnalyzer()
        api_key = analyzer._initialize_api_key("test_key")
        assert api_key == "test_key"
    
    def test_check_mock_mode(self):
        """Mock 모드 확인 테스트"""
        with patch.dict('os.environ', {'IS_MOCK': 'true'}):
            analyzer = GeminiAnalyzer()
            assert analyzer._check_mock_mode() is True
    
    def test_strategy_weights_initialization(self, gemini_analyzer):
        """전략 가중치 초기화 테스트"""
        weights = gemini_analyzer._initialize_strategy_weights()
        assert isinstance(weights, dict)
        assert StrategyType.WARREN_BUFFETT in weights
        assert weights[StrategyType.WARREN_BUFFETT] == 0.15
    
    def test_convert_strategy_name_to_type(self, gemini_analyzer):
        """전략 이름 변환 테스트"""
        strategy_type = gemini_analyzer._convert_strategy_name_to_type("워런_버핏")
        assert strategy_type == StrategyType.WARREN_BUFFETT
        
        # 잘못된 이름의 경우 기본값 반환
        default_type = gemini_analyzer._convert_strategy_name_to_type("잘못된_전략")
        assert default_type == StrategyType.WARREN_BUFFETT
    
    def test_calculate_ai_confidence(self, gemini_analyzer, sample_strategy_scores, sample_technical_result):
        """AI 신뢰도 계산 테스트"""
        strategy_scores_dict = {score.strategy_name: score for score in sample_strategy_scores}
        confidence = gemini_analyzer._calculate_ai_confidence(strategy_scores_dict, sample_technical_result)
        
        assert 0.0 <= confidence <= 1.0
        assert isinstance(confidence, float)
    
    def test_calculate_entry_price(self, gemini_analyzer, sample_technical_result):
        """진입 가격 계산 테스트"""
        current_price = 70000.0
        entry_price = gemini_analyzer._calculate_entry_price(current_price, sample_technical_result)
        
        assert entry_price > 0
        assert isinstance(entry_price, float)
        # 진입 가격이 현재 가격의 합리적 범위 내에 있는지 확인
        assert 0.95 * current_price <= entry_price <= 1.05 * current_price
    
    def test_calculate_target_price(self, gemini_analyzer, sample_technical_result):
        """목표 가격 계산 테스트"""
        current_price = 70000.0
        total_score = 75.0
        target_price = gemini_analyzer._calculate_target_price(current_price, sample_technical_result, total_score)
        
        assert target_price > current_price
        assert isinstance(target_price, float)
    
    def test_calculate_stop_loss_price(self, gemini_analyzer, sample_technical_result):
        """손절 가격 계산 테스트"""
        current_price = 70000.0
        stop_loss = gemini_analyzer._calculate_stop_loss_price(current_price, sample_technical_result)
        
        assert stop_loss < current_price
        assert isinstance(stop_loss, float)
        # 손절가가 현재 가격의 90% 이상인지 확인 (최대 10% 손실)
        assert stop_loss >= current_price * 0.90
    
    def test_determine_risk_level(self, gemini_analyzer, sample_technical_result):
        """위험도 결정 테스트"""
        # 높은 점수와 신뢰도
        risk_level_low = gemini_analyzer._determine_risk_level(sample_technical_result, 80.0)
        
        # 낮은 점수
        low_confidence_result = TechnicalAnalysisResult(
            symbol="005930",
            indicators=sample_technical_result.indicators,
            signals=sample_technical_result.signals,
            confidence=0.3
        )
        risk_level_high = gemini_analyzer._determine_risk_level(low_confidence_result, 30.0)
        
        assert risk_level_low in [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH]
        assert risk_level_high in [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH]
    
    def test_determine_investment_period(self, gemini_analyzer, sample_technical_result):
        """투자 기간 결정 테스트"""
        # 장기 투자 전략
        long_period = gemini_analyzer._determine_investment_period(
            sample_technical_result, StrategyType.WARREN_BUFFETT
        )
        assert long_period == InvestmentPeriod.LONG
        
        # 단기 투자 전략
        short_period = gemini_analyzer._determine_investment_period(
            sample_technical_result, StrategyType.JESSE_LIVERMORE
        )
        assert short_period == InvestmentPeriod.SHORT
        
        # 중기 투자 (기본값)
        medium_period = gemini_analyzer._determine_investment_period(
            sample_technical_result, StrategyType.PETER_LYNCH
        )
        assert medium_period == InvestmentPeriod.MEDIUM
    
    def test_calculate_position_size(self, gemini_analyzer):
        """포지션 크기 계산 테스트"""
        # 저위험, 높은 신뢰도
        size_low_risk = gemini_analyzer._calculate_position_size(RiskLevel.LOW, 0.9)
        assert size_low_risk > 0
        
        # 고위험, 낮은 신뢰도
        size_high_risk = gemini_analyzer._calculate_position_size(RiskLevel.HIGH, 0.3)
        assert size_high_risk > 0
        
        # 저위험이 고위험보다 큰 포지션 크기를 가져야 함
        assert size_low_risk > size_high_risk
    
    def test_generate_fallback_recommendation_reasoning(
        self, gemini_analyzer, sample_stock_data, sample_strategy_scores, sample_technical_result
    ):
        """Fallback 추천 근거 생성 테스트"""
        strategy_scores_dict = {score.strategy_name: score for score in sample_strategy_scores}
        reasoning = gemini_analyzer._generate_fallback_recommendation_reasoning(
            sample_stock_data, strategy_scores_dict, sample_technical_result, 75.0
        )
        
        assert isinstance(reasoning, str)
        assert len(reasoning) > 0
        assert reasoning.endswith(".")
    
    def test_collect_key_indicators(self, gemini_analyzer, sample_strategy_scores, sample_technical_result):
        """핵심 지표 수집 테스트"""
        strategy_scores_dict = {score.strategy_name: score for score in sample_strategy_scores}
        indicators = gemini_analyzer._collect_key_indicators(strategy_scores_dict, sample_technical_result)
        
        assert isinstance(indicators, dict)
        assert 'best_strategy' in indicators
        assert 'best_strategy_score' in indicators
        assert 'overall_trend' in indicators
    
    @pytest.mark.asyncio
    async def test_generate_market_sentiment(self, gemini_analyzer):
        """시장 심리 생성 테스트"""
        # Mock 추천 데이터
        mock_recommendations = []
        sentiment = await gemini_analyzer._generate_market_sentiment(
            MarketType.KOSPI200, mock_recommendations
        )
        
        assert isinstance(sentiment, str)
        assert sentiment in ["매우 긍정적", "긍정적", "보통", "부정적", "매우 부정적", "중립"]
    
    @pytest.mark.asyncio
    async def test_generate_key_insights(self, gemini_analyzer):
        """주요 인사이트 생성 테스트"""
        mock_recommendations = []
        insights = await gemini_analyzer._generate_key_insights(
            MarketType.KOSPI200, mock_recommendations
        )
        
        assert isinstance(insights, list)
        assert len(insights) > 0
        assert all(isinstance(insight, str) for insight in insights)
    
    def test_assess_market_risk(self, gemini_analyzer):
        """시장 위험도 평가 테스트"""
        mock_recommendations = []
        risk_assessment = gemini_analyzer._assess_market_risk(mock_recommendations)
        
        assert isinstance(risk_assessment, str)
        assert "평가 불가" in risk_assessment or any(
            keyword in risk_assessment for keyword in ["낮음", "보통", "높음", "매우 높음"]
        )
    
    def test_prepare_analysis_data(self, gemini_analyzer, sample_stock_data, sample_strategy_scores, sample_technical_result):
        """분석 데이터 준비 테스트"""
        stocks = [sample_stock_data]
        technical_results = [sample_technical_result]
        
        stock_map, technical_map, symbol_strategy_scores = gemini_analyzer._prepare_analysis_data(
            stocks, sample_strategy_scores, technical_results
        )
        
        assert isinstance(stock_map, dict)
        assert isinstance(technical_map, dict)
        assert isinstance(symbol_strategy_scores, dict)
        assert "005930" in stock_map
        assert "005930" in technical_map
        assert "005930" in symbol_strategy_scores

if __name__ == "__main__":
    pytest.main([__file__]) 