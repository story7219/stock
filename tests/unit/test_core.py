#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: test_core.py
모듈: Core 모듈 단위 테스트
목적: 설정, 로깅, 모델에 대한 테스트

Author: Trading AI System
Created: 2025-01-27
Version: 1.0.0
"""

import pytest
import tempfile
import os
from datetime import datetime
from unittest.mock import patch, MagicMock

from core.config import Config, DatabaseConfig, APIConfig, TradingConfig
from core.models import Signal, TradeType, StrategyType, Stock, News, NewsCategory, SentimentType


class TestConfig:
    """설정 테스트"""

    def test_database_config_defaults(self):
        """데이터베이스 설정 기본값 테스트"""
        config = DatabaseConfig()
        assert config.url == "sqlite:///./trading_data.db"
        assert config.pool_size == 10
        assert config.max_overflow == 20
        assert config.echo is False
        assert config.pool_pre_ping is True

    def test_api_config_validation(self):
        """API 설정 검증 테스트"""
        with pytest.raises(ValueError):
            APIConfig(
                KIS_APP_KEY="",
                KIS_APP_SECRET="",
                KIS_ACCESS_TOKEN="",
                KIS_REAL_APP_KEY="",
                KIS_REAL_APP_SECRET="",
                KIS_REAL_ACCESS_TOKEN="",
                DART_API_KEY="",
                max_requests_per_minute=-1  # 음수는 허용되지 않음
            )

    def test_trading_config_weights_validation(self):
        """거래 설정 가중치 검증 테스트"""
        # 가중치 합이 1을 초과하는 경우
        with pytest.raises(ValueError):
            TradingConfig(
                news_momentum_weight=0.5,
                technical_pattern_weight=0.4,
                theme_rotation_weight=0.3,
                risk_management_weight=0.2
            )

    def test_config_load_from_env(self):
        """환경 변수에서 설정 로드 테스트"""
        with patch.dict(os.environ, {
            'DATABASE_URL': 'postgresql://test:test@localhost:5432/test',
            'REDIS_URL': 'redis://localhost:6379/1'
        }):
            config = Config.load_from_env()
            assert config.database.url == 'postgresql://test:test@localhost:5432/test'
            assert config.data_dir == "./data"

    def test_config_save_and_load(self):
        """설정 저장 및 로드 테스트"""
        config = Config()
        config.environment = "test"
        config.debug = True
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            config.save_to_file(temp_path)
            loaded_config = Config.load_from_file(temp_path)
            assert loaded_config.environment == "test"
            assert loaded_config.debug is True
        finally:
            os.unlink(temp_path)


class TestModels:
    """모델 테스트"""

    def test_stock_validation(self):
        """주식 모델 검증 테스트"""
        # 유효한 주식 데이터
        stock = Stock(
            code="005930",
            name="삼성전자",
            market="KOSPI",
            sector="전기전자",
            market_cap=500000000000000,
            price=75000,
            volume=1000000,
            change_pct=2.5
        )
        assert stock.code == "005930"
        assert stock.name == "삼성전자"

        # 잘못된 종목 코드
        with pytest.raises(ValueError):
            Stock(
                code="005930!",  # 특수문자 포함
                name="삼성전자",
                market="KOSPI"
            )

    def test_news_validation(self):
        """뉴스 모델 검증 테스트"""
        news = News(
            id="news_001",
            title="삼성전자 실적 발표",
            content="삼성전자가 예상치를 상회하는 실적을 발표했습니다.",
            url="https://example.com/news/001",
            source="경제일보",
            published_at=datetime.now(),
            category=NewsCategory.FINANCIAL,
            sentiment=SentimentType.POSITIVE,
            sentiment_score=0.8,
            importance_score=0.9
        )
        assert news.sentiment_score == 0.8
        assert news.category == NewsCategory.FINANCIAL

        # 잘못된 감성 점수
        with pytest.raises(ValueError):
            News(
                id="news_002",
                title="테스트 뉴스",
                content="테스트 내용",
                url="https://example.com",
                source="테스트",
                published_at=datetime.now(),
                category=NewsCategory.FINANCIAL,
                sentiment=SentimentType.POSITIVE,
                sentiment_score=1.5,  # 범위 초과
                importance_score=0.5
            )

    def test_signal_creation(self):
        """신호 모델 생성 테스트"""
        signal = Signal(
            id="signal_001",
            stock_code="005930",
            strategy_type=StrategyType.NEWS_MOMENTUM,
            signal_type=TradeType.BUY,
            confidence_score=0.8,
            target_price=80000,
            stop_loss=70000,
            take_profit=90000,
            reasoning="뉴스 모멘텀 기반 매수 신호"
        )
        assert signal.stock_code == "005930"
        assert signal.confidence_score == 0.8
        assert signal.signal_type == TradeType.BUY

        # 잘못된 신뢰도 점수
        with pytest.raises(ValueError):
            Signal(
                id="signal_002",
                stock_code="005930",
                strategy_type=StrategyType.NEWS_MOMENTUM,
                signal_type=TradeType.BUY,
                confidence_score=1.5,  # 범위 초과
                reasoning="테스트"
            )


class TestLogger:
    """로거 테스트"""

    @patch('core.logger.get_logger')
    def test_logger_initialization(self, mock_get_logger):
        """로거 초기화 테스트"""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        from core.logger import initialize_logging
        initialize_logging()
        
        mock_get_logger.assert_called()

    def test_log_function_call_decorator(self):
        """함수 호출 로깅 데코레이터 테스트"""
        from core.logger import log_function_call
        
        @log_function_call
        def test_function(arg1, arg2):
            return arg1 + arg2
        
        result = test_function(1, 2)
        assert result == 3


if __name__ == "__main__":
    pytest.main([__file__]) 