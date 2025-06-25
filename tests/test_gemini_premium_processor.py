"""
Gemini 프리미엄 데이터 처리 시스템 테스트
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
import pandas as pd
import sys
import os

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.modules.gemini_premium_data_processor import (
    GeminiPremiumDataProcessor,
    NewsData,
    ChartData,
    ProcessedData
)

class TestGeminiPremiumDataProcessor:
    """Gemini 프리미엄 데이터 처리기 테스트"""
    
    @pytest.fixture
    def processor(self):
        """테스트용 프로세서 인스턴스"""
        with patch.dict(os.environ, {'IS_MOCK': 'true'}):
            return GeminiPremiumDataProcessor()
    
    @pytest.fixture
    def sample_news_data(self):
        """샘플 뉴스 데이터"""
        return [
            NewsData(
                title="Apple 주가 상승 전망",
                content="애플이 새로운 제품 출시로 주가 상승이 예상됩니다.",
                source="Yahoo Finance",
                published_time=datetime.now(),
                url="https://example.com",
                sentiment=0.8,
                relevance_score=0.9,
                keywords=["Apple", "상승", "제품"]
            ),
            NewsData(
                title="기술주 전반 하락",
                content="기술주 전반이 금리 인상 우려로 하락했습니다.",
                source="Reuters",
                published_time=datetime.now(),
                url="https://example.com",
                sentiment=-0.3,
                relevance_score=0.7,
                keywords=["기술주", "하락", "금리"]
            )
        ]
    
    @pytest.fixture
    def sample_chart_data(self):
        """샘플 차트 데이터"""
        return ChartData(
            symbol="AAPL",
            image_base64="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
            technical_indicators={
                'rsi': 65.5,
                'macd': 0.25,
                'bb_position': 0.7
            },
            price_data={
                'current': 150.0,
                'change_pct': 2.5,
                'high_52w': 180.0,
                'low_52w': 120.0
            },
            volume_data={
                'current': 50000000,
                'avg_volume': 45000000
            },
            chart_analysis="상승 추세, 보통 변동성"
        )
    
    def test_processor_initialization(self):
        """프로세서 초기화 테스트"""
        with patch.dict(os.environ, {'IS_MOCK': 'true'}):
            processor = GeminiPremiumDataProcessor()
            assert processor.is_mock == True
            assert processor.model is None
            assert len(processor.news_sources) > 0
    
    @pytest.mark.asyncio
    async def test_collect_real_time_news(self, processor):
        """실시간 뉴스 수집 테스트"""
        with patch.object(processor, '_fetch_yahoo_finance_news') as mock_yahoo, \
             patch.object(processor, '_fetch_marketwatch_news') as mock_market, \
             patch.object(processor, '_fetch_reuters_news') as mock_reuters:
            
            # Mock 응답 설정
            mock_yahoo.return_value = [NewsData(
                title="Test Yahoo News",
                content="Test content",
                source="Yahoo Finance",
                published_time=datetime.now(),
                url="https://test.com",
                sentiment=0.0,
                relevance_score=0.8,
                keywords=[]
            )]
            mock_market.return_value = []
            mock_reuters.return_value = []
            
            news_data = await processor._collect_real_time_news("AAPL")
            
            assert len(news_data) >= 0
            mock_yahoo.assert_called_once_with("AAPL")
            mock_market.assert_called_once_with("AAPL")
            mock_reuters.assert_called_once_with("AAPL")
    
    @pytest.mark.asyncio
    async def test_generate_chart_image(self, processor):
        """차트 이미지 생성 테스트"""
        # yfinance Mock 데이터
        mock_hist = pd.DataFrame({
            'Close': [100, 102, 104, 103, 105],
            'High': [101, 103, 105, 104, 106],
            'Low': [99, 101, 103, 102, 104],
            'Volume': [1000000, 1100000, 1200000, 1050000, 1150000]
        }, index=pd.date_range('2024-01-01', periods=5))
        
        with patch('yfinance.Ticker') as mock_ticker:
            mock_ticker.return_value.history.return_value = mock_hist
            
            chart_data = await processor._generate_chart_image("AAPL")
            
            assert chart_data.symbol == "AAPL"
            assert chart_data.image_base64 != ""
            assert 'rsi' in chart_data.technical_indicators
            assert 'current' in chart_data.price_data
    
    def test_verify_news_quality(self, processor, sample_news_data):
        """뉴스 품질 검증 테스트"""
        verified_news = processor._verify_news_quality(sample_news_data)
        
        assert len(verified_news) <= len(sample_news_data)
        assert all(news.relevance_score > 0 for news in verified_news)
        
        # 관련성 점수로 정렬되었는지 확인
        scores = [news.relevance_score for news in verified_news]
        assert scores == sorted(scores, reverse=True)
    
    def test_calculate_relevance_score(self, processor):
        """관련성 점수 계산 테스트"""
        news = NewsData(
            title="Test news",
            content="Test content",
            source="Reuters",
            published_time=datetime.now(),
            url="https://test.com",
            sentiment=0.0,
            relevance_score=0.0,
            keywords=[]
        )
        
        score = processor._calculate_relevance_score(news)
        
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Reuters는 높은 가중치
    
    def test_calculate_technical_indicators(self, processor):
        """기술적 지표 계산 테스트"""
        # 테스트 데이터
        hist = pd.DataFrame({
            'Close': [100 + i + (i % 3) for i in range(50)]
        })
        
        indicators = processor._calculate_technical_indicators(hist)
        
        assert 'rsi' in indicators
        assert 0 <= indicators['rsi'] <= 100
        
        if 'macd' in indicators:
            assert isinstance(indicators['macd'], float)
        
        if 'bb_position' in indicators:
            assert 0 <= indicators['bb_position'] <= 1
    
    def test_analyze_chart_pattern(self, processor):
        """차트 패턴 분석 테스트"""
        # 상승 추세 데이터
        hist_up = pd.DataFrame({
            'Close': [100 + i for i in range(20)]
        })
        
        pattern_up = processor._analyze_chart_pattern(hist_up)
        assert "상승 추세" in pattern_up
        
        # 하락 추세 데이터
        hist_down = pd.DataFrame({
            'Close': [120 - i for i in range(20)]
        })
        
        pattern_down = processor._analyze_chart_pattern(hist_down)
        assert "하락 추세" in pattern_down
    
    def test_summarize_news(self, processor, sample_news_data):
        """뉴스 요약 테스트"""
        summary = processor._summarize_news(sample_news_data)
        
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert "Apple" in summary or "기술주" in summary
    
    def test_analyze_market_sentiment(self, processor, sample_news_data, sample_chart_data):
        """시장 심리 분석 테스트"""
        sentiment = processor._analyze_market_sentiment(sample_news_data, sample_chart_data)
        
        assert sentiment in ["긍정적", "부정적", "중립적"]
    
    def test_extract_risk_factors(self, processor, sample_news_data):
        """리스크 요인 추출 테스트"""
        technical_data = {
            'pe_ratio': 35.0,  # 높은 PER
            'beta': 1.8        # 높은 베타
        }
        
        risk_factors = processor._extract_risk_factors(sample_news_data, technical_data)
        
        assert isinstance(risk_factors, list)
        assert len(risk_factors) >= 0
        assert any("PER" in factor for factor in risk_factors)
        assert any("베타" in factor for factor in risk_factors)
    
    def test_extract_opportunities(self, processor, sample_news_data, sample_chart_data):
        """기회 요인 추출 테스트"""
        technical_data = {'sector': 'Technology'}
        
        opportunities = processor._extract_opportunities(
            sample_news_data, sample_chart_data, technical_data
        )
        
        assert isinstance(opportunities, list)
        assert len(opportunities) >= 0
    
    def test_create_gemini_prompt(self, processor, sample_chart_data):
        """Gemini 프롬프트 생성 테스트"""
        prompt = processor._create_gemini_prompt(
            symbol="AAPL",
            news_summary="테스트 뉴스 요약",
            chart_data=sample_chart_data,
            technical_data={'pe_ratio': 25.0},
            market_sentiment="긍정적",
            risk_factors=["테스트 리스크"],
            opportunities=["테스트 기회"]
        )
        
        assert isinstance(prompt, str)
        assert "AAPL" in prompt
        assert "투자 분석 요청" in prompt
        assert "테스트 뉴스 요약" in prompt
        assert "긍정적" in prompt
    
    @pytest.mark.asyncio
    async def test_process_stock_data(self, processor):
        """주식 데이터 처리 통합 테스트"""
        with patch.object(processor, '_collect_real_time_news') as mock_news, \
             patch.object(processor, '_generate_chart_image') as mock_chart, \
             patch.object(processor, '_collect_technical_data') as mock_tech:
            
            # Mock 응답 설정
            mock_news.return_value = []
            mock_chart.return_value = ChartData(
                symbol="AAPL",
                image_base64="test",
                technical_indicators={},
                price_data={},
                volume_data={},
                chart_analysis="test"
            )
            mock_tech.return_value = {}
            
            processed_data = await processor.process_stock_data("AAPL")
            
            assert isinstance(processed_data, ProcessedData)
            assert processed_data.symbol == "AAPL"
            assert processed_data.gemini_prompt != ""
    
    @pytest.mark.asyncio
    async def test_send_to_gemini_mock(self, processor):
        """Gemini AI 전송 테스트 (Mock 모드)"""
        processed_data = ProcessedData(
            symbol="AAPL",
            news_summary="테스트",
            chart_analysis="테스트",
            technical_data={},
            market_sentiment="긍정적",
            risk_factors=[],
            opportunities=[],
            gemini_prompt="테스트 프롬프트"
        )
        
        result = await processor.send_to_gemini(processed_data)
        
        assert result['symbol'] == "AAPL"
        assert 'gemini_analysis' in result
        assert 'recommendation' in result
        assert result['recommendation'] in ['BUY', 'SELL', 'HOLD']
    
    def test_extract_recommendation(self, processor):
        """추천도 추출 테스트"""
        assert processor._extract_recommendation("매수 추천합니다") == 'BUY'
        assert processor._extract_recommendation("매도 권장") == 'SELL'
        assert processor._extract_recommendation("보유 추천") == 'HOLD'
        assert processor._extract_recommendation("BUY recommendation") == 'BUY'
    
    def test_extract_target_price(self, processor):
        """목표가 추출 테스트"""
        text = "목표가: 150,000원으로 설정합니다"
        price = processor._extract_target_price(text)
        assert price == 150000.0
        
        text_no_price = "목표가 정보가 없습니다"
        price_no = processor._extract_target_price(text_no_price)
        assert price_no == 0.0
    
    def test_extract_risk_level(self, processor):
        """리스크 레벨 추출 테스트"""
        assert processor._extract_risk_level("높은 위험도") == 'HIGH'
        assert processor._extract_risk_level("낮은 리스크") == 'LOW'
        assert processor._extract_risk_level("보통 수준") == 'MEDIUM'
    
    def test_extract_investment_period(self, processor):
        """투자 기간 추출 테스트"""
        assert processor._extract_investment_period("단기 투자 추천") == 'SHORT_TERM'
        assert processor._extract_investment_period("장기 보유") == 'LONG_TERM'
        assert processor._extract_investment_period("중기 투자") == 'MEDIUM_TERM'

@pytest.mark.asyncio
async def test_full_integration():
    """전체 통합 테스트"""
    with patch.dict(os.environ, {'IS_MOCK': 'true'}):
        processor = GeminiPremiumDataProcessor()
        
        # 실제 처리 플로우 테스트
        test_symbols = ['AAPL', 'GOOGL']
        
        for symbol in test_symbols:
            try:
                # 데이터 처리
                processed_data = await processor.process_stock_data(symbol)
                assert processed_data.symbol == symbol
                
                # Gemini 분석
                gemini_result = await processor.send_to_gemini(processed_data)
                assert gemini_result['symbol'] == symbol
                assert 'recommendation' in gemini_result
                
                print(f"✅ {symbol} 통합 테스트 성공")
                
            except Exception as e:
                print(f"❌ {symbol} 통합 테스트 실패: {e}")
                raise

if __name__ == "__main__":
    # 직접 실행 시 통합 테스트 수행
    asyncio.run(test_full_integration()) 