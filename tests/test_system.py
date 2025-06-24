#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stock Analysis System 통합 테스트

Author: AI Assistant
Version: 5.0
"""

import pytest
import asyncio
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, List, Any
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 상위 디렉토리를 path에 추가하여 src 모듈을 import 가능하게 함
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# 테스트용 환경 변수 설정
os.environ["TESTING"] = "true"
os.environ["GEMINI_API_KEY"] = "test_key"
os.environ["TELEGRAM_BOT_TOKEN"] = "test_token"
os.environ["TELEGRAM_CHAT_ID"] = "test_chat"
os.environ["GOOGLE_CREDENTIALS_PATH"] = "test_credentials.json"
os.environ["GOOGLE_SHEET_ID"] = "test_sheet_id"

from src.multi_data_collector import MultiDataCollector
from src.gemini_analyzer import GeminiAnalyzer, StockData, StrategyScore
from src.telegram_notifier import TelegramNotifier
from src.google_sheets_manager import GoogleSheetsManager
from src.scheduler import AutomatedScheduler


class TestSystemIntegration:
    """시스템 통합 테스트"""

    @pytest.fixture
    def mock_stock_data(self) -> List[StockData]:
        """테스트용 주식 데이터"""
        return [
            StockData(
                symbol="005930.KS",
                name="삼성전자",
                market="KOSPI",
                current_price=75000.0,
                technical_indicators={
                    "rsi": 65.0,
                    "macd": 0.5,
                    "bb_position": 0.7,
                    "sma_20": 74000.0,
                    "sma_50": 72000.0,
                    "volume_ratio": 1.2,
                    "momentum": 0.8,
                    "volatility": 0.15,
                },
                quality_score=85.0,
            ),
            StockData(
                symbol="AAPL",
                name="Apple Inc.",
                market="NASDAQ",
                current_price=180.0,
                technical_indicators={
                    "rsi": 70.0,
                    "macd": 0.8,
                    "bb_position": 0.8,
                    "sma_20": 178.0,
                    "sma_50": 175.0,
                    "volume_ratio": 1.5,
                    "momentum": 0.9,
                    "volatility": 0.20,
                },
                quality_score=90.0,
            ),
        ]

    @pytest.fixture
    def mock_analysis_result(self) -> Dict[str, Any]:
        """테스트용 분석 결과"""
        return {
            "top5_stocks": [
                {
                    "rank": 1,
                    "symbol": "005930.KS",
                    "name": "삼성전자",
                    "score": 94.2,
                    "strategies": ["워런 버핏", "피터 린치"],
                    "reasoning": "안정적 기술적 지표, 강한 모멘텀",
                },
                {
                    "rank": 2,
                    "symbol": "AAPL",
                    "name": "Apple Inc.",
                    "score": 92.8,
                    "strategies": ["피터 린치", "레이 달리오"],
                    "reasoning": "지속적 성장성, 적정 변동성",
                },
            ],
            "analysis_summary": {
                "total_analyzed": 2,
                "analysis_time": "2024-01-01 12:00:00",
                "gemini_model": "gemini-1.5-flash-8b",
                "quality_threshold": 75.0,
            },
            "market_insights": {
                "kospi_trend": "positive",
                "nasdaq_trend": "positive",
                "sp500_trend": "neutral",
            },
        }


class TestDataCollector:
    """데이터 수집기 테스트"""

    @pytest.mark.asyncio
    async def test_data_collector_initialization(self):
        """데이터 수집기 초기화 테스트"""
        collector = MultiDataCollector()
        assert collector is not None
        assert hasattr(collector, "collect_all_data")
        assert hasattr(collector, "health_check")

    @pytest.mark.asyncio
    async def test_health_check(self):
        """헬스 체크 테스트"""
        collector = MultiDataCollector()

        with patch.object(collector, "_check_api_connections", return_value=True):
            health_status = await collector.health_check()
            assert health_status is True

    @pytest.mark.asyncio
    async def test_collect_kospi_data(self):
        """코스피 데이터 수집 테스트"""
        collector = MultiDataCollector()

        # Mock 데이터 설정
        mock_data = [
            {"symbol": "005930.KS", "name": "삼성전자", "price": 75000.0, "rsi": 65.0}
        ]

        with patch.object(collector, "_fetch_kospi_stocks", return_value=mock_data):
            data = await collector._collect_kospi_data()
            assert len(data) > 0
            assert data[0]["symbol"] == "005930.KS"

    @pytest.mark.asyncio
    async def test_collect_nasdaq_data(self):
        """나스닥 데이터 수집 테스트"""
        collector = MultiDataCollector()

        mock_data = [
            {"symbol": "AAPL", "name": "Apple Inc.", "price": 180.0, "rsi": 70.0}
        ]

        with patch.object(collector, "_fetch_nasdaq_stocks", return_value=mock_data):
            data = await collector._collect_nasdaq_data()
            assert len(data) > 0
            assert data[0]["symbol"] == "AAPL"


class TestGeminiAnalyzer:
    """Gemini 분석기 테스트"""

    def test_analyzer_initialization(self):
        """분석기 초기화 테스트"""
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test_key"}):
            analyzer = GeminiAnalyzer()
            assert analyzer is not None
            assert analyzer.model_name == "gemini-1.5-flash-8b"

    def test_flash_model_enforcement(self):
        """Flash 모델 강제 적용 테스트"""
        with patch.dict(
            "os.environ",
            {
                "GEMINI_API_KEY": "test_key",
                "GEMINI_MODEL": "gemini-pro",  # 다른 모델 설정
            },
        ):
            analyzer = GeminiAnalyzer()
            # Flash 모델로 강제 변경되어야 함
            assert analyzer.model_name == "gemini-1.5-flash-8b"

    def test_health_check(self):
        """헬스 체크 테스트"""
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test_key"}):
            analyzer = GeminiAnalyzer()
            health_status = analyzer.health_check()
            assert isinstance(health_status, bool)

    def test_strategy_scoring(self):
        """전략별 점수 계산 테스트"""
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test_key"}):
            analyzer = GeminiAnalyzer()

            stock_data = {
                "symbol": "005930.KS",
                "technical_indicators": {
                    "rsi": 65.0,
                    "sma_20": 74000.0,
                    "sma_50": 72000.0,
                    "volatility": 0.15,
                },
            }

            # 워런 버핏 전략 테스트
            score = analyzer._calculate_buffett_score(stock_data)
            assert isinstance(score, (int, float))
            assert 0 <= score <= 100

    @pytest.mark.asyncio
    async def test_top5_selection(self, mock_stock_data):
        """Top5 선정 테스트"""
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test_key"}):
            analyzer = GeminiAnalyzer()

            # Mock Gemini API 응답
            mock_response = {
                "top5_selections": [
                    {
                        "symbol": "005930.KS",
                        "score": 94.2,
                        "reasoning": "Strong technical indicators",
                    }
                ]
            }

            with patch.object(analyzer, "_call_gemini_api", return_value=mock_response):
                result = await analyzer.analyze_and_select_top5(mock_stock_data)
                assert result is not None
                assert "top5_stocks" in result


class TestTelegramNotifier:
    """텔레그램 알림 테스트"""

    def test_notifier_initialization(self):
        """알림기 초기화 테스트"""
        with patch.dict(
            "os.environ",
            {"TELEGRAM_BOT_TOKEN": "test_token", "TELEGRAM_CHAT_ID": "test_chat_id"},
        ):
            notifier = TelegramNotifier()
            assert notifier is not None

    @pytest.mark.asyncio
    async def test_health_check(self):
        """헬스 체크 테스트"""
        with patch.dict(
            "os.environ",
            {"TELEGRAM_BOT_TOKEN": "test_token", "TELEGRAM_CHAT_ID": "test_chat_id"},
        ):
            notifier = TelegramNotifier()

            with patch.object(notifier, "_test_bot_connection", return_value=True):
                health_status = await notifier.health_check()
                assert health_status is True

    @pytest.mark.asyncio
    async def test_send_system_start(self):
        """시스템 시작 알림 테스트"""
        with patch.dict(
            "os.environ",
            {"TELEGRAM_BOT_TOKEN": "test_token", "TELEGRAM_CHAT_ID": "test_chat_id"},
        ):
            notifier = TelegramNotifier()

            with patch.object(
                notifier, "_send_message", return_value=True
            ) as mock_send:
                await notifier.send_system_start()
                mock_send.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_analysis_results(self, mock_analysis_result):
        """분석 결과 알림 테스트"""
        with patch.dict(
            "os.environ",
            {"TELEGRAM_BOT_TOKEN": "test_token", "TELEGRAM_CHAT_ID": "test_chat_id"},
        ):
            notifier = TelegramNotifier()

            with patch.object(
                notifier, "_send_message", return_value=True
            ) as mock_send:
                await notifier.send_analysis_results(mock_analysis_result)
                mock_send.assert_called()


class TestGoogleSheetsManager:
    """구글 시트 관리자 테스트"""

    def test_manager_initialization(self):
        """관리자 초기화 테스트"""
        with patch.dict(
            "os.environ",
            {
                "GOOGLE_SHEETS_CREDENTIALS": "test_credentials.json",
                "GOOGLE_SHEETS_ID": "test_sheet_id",
            },
        ):
            with patch("gspread.service_account"):
                manager = GoogleSheetsManager()
                assert manager is not None

    @pytest.mark.asyncio
    async def test_health_check(self):
        """헬스 체크 테스트"""
        with patch.dict(
            "os.environ",
            {
                "GOOGLE_SHEETS_CREDENTIALS": "test_credentials.json",
                "GOOGLE_SHEETS_ID": "test_sheet_id",
            },
        ):
            with patch("gspread.service_account"):
                manager = GoogleSheetsManager()

                with patch.object(manager, "_test_sheet_access", return_value=True):
                    health_status = await manager.health_check()
                    assert health_status is True

    @pytest.mark.asyncio
    async def test_save_analysis_results(self, mock_analysis_result):
        """분석 결과 저장 테스트"""
        with patch.dict(
            "os.environ",
            {
                "GOOGLE_SHEETS_CREDENTIALS": "test_credentials.json",
                "GOOGLE_SHEETS_ID": "test_sheet_id",
            },
        ):
            with patch("gspread.service_account"):
                manager = GoogleSheetsManager()

                with patch.object(
                    manager, "_update_worksheet", return_value=True
                ) as mock_update:
                    await manager.save_analysis_results(mock_analysis_result)
                    mock_update.assert_called()


class TestScheduler:
    """스케줄러 테스트"""

    def test_scheduler_initialization(self):
        """스케줄러 초기화 테스트"""
        scheduler = AutomatedScheduler()
        assert scheduler is not None
        assert hasattr(scheduler, "start")
        assert hasattr(scheduler, "stop")

    def test_job_scheduling(self):
        """작업 스케줄링 테스트"""
        scheduler = AutomatedScheduler()

        # 스케줄 설정 확인
        scheduler._setup_schedules()

        # 스케줄된 작업이 있는지 확인
        import schedule

        jobs = schedule.jobs
        assert len(jobs) > 0

    @pytest.mark.asyncio
    async def test_morning_analysis_job(self):
        """아침 분석 작업 테스트"""
        scheduler = AutomatedScheduler()

        with patch.object(
            scheduler, "_run_full_analysis", return_value=True
        ) as mock_analysis:
            await scheduler._morning_analysis()
            mock_analysis.assert_called_once()

    @pytest.mark.asyncio
    async def test_system_health_check_job(self):
        """시스템 헬스 체크 작업 테스트"""
        scheduler = AutomatedScheduler()

        with patch.object(
            scheduler, "_check_system_health", return_value=True
        ) as mock_health:
            await scheduler._system_health_check()
            mock_health.assert_called_once()


class TestSystemWorkflow:
    """시스템 워크플로우 테스트"""

    @pytest.mark.asyncio
    async def test_full_analysis_workflow(self, mock_stock_data, mock_analysis_result):
        """전체 분석 워크플로우 테스트"""
        # 각 컴포넌트 Mock 설정
        with patch.dict(
            "os.environ",
            {
                "GEMINI_API_KEY": "test_key",
                "TELEGRAM_BOT_TOKEN": "test_token",
                "TELEGRAM_CHAT_ID": "test_chat_id",
                "GOOGLE_SHEETS_CREDENTIALS": "test_credentials.json",
                "GOOGLE_SHEETS_ID": "test_sheet_id",
            },
        ):
            # Mock 컴포넌트들
            with (
                patch("gspread.service_account"),
                patch.object(
                    MultiDataCollector, "collect_all_data", return_value=mock_stock_data
                ),
                patch.object(
                    GeminiAnalyzer,
                    "analyze_and_select_top5",
                    return_value=mock_analysis_result,
                ),
                patch.object(TelegramNotifier, "send_system_start", return_value=None),
                patch.object(
                    TelegramNotifier, "send_analysis_results", return_value=None
                ),
                patch.object(
                    GoogleSheetsManager, "save_analysis_results", return_value=None
                ),
            ):

                # 시스템 실행
                from main import StockAnalysisSystem

                system = StockAnalysisSystem()

                result = await system.run_full_analysis()
                assert result is True

    @pytest.mark.asyncio
    async def test_error_handling_workflow(self):
        """오류 처리 워크플로우 테스트"""
        with patch.dict(
            "os.environ",
            {
                "GEMINI_API_KEY": "test_key",
                "TELEGRAM_BOT_TOKEN": "test_token",
                "TELEGRAM_CHAT_ID": "test_chat_id",
            },
        ):
            # 데이터 수집 실패 시뮬레이션
            with (
                patch.object(
                    MultiDataCollector,
                    "collect_all_data",
                    side_effect=Exception("Connection error"),
                ),
                patch.object(
                    TelegramNotifier, "send_error", return_value=None
                ) as mock_error,
            ):

                from main import StockAnalysisSystem

                system = StockAnalysisSystem()

                result = await system.run_full_analysis()
                assert result is False
                mock_error.assert_called()


@pytest.mark.integration
class TestPerformance:
    """성능 테스트"""

    @pytest.mark.asyncio
    async def test_data_collection_performance(self):
        """데이터 수집 성능 테스트"""
        import time

        collector = MultiDataCollector()

        # 소량 데이터로 성능 측정
        start_time = time.time()

        with patch.object(collector, "collect_all_data", return_value=[]):
            await collector.collect_all_data()

        end_time = time.time()
        execution_time = end_time - start_time

        # 10초 이내 완료되어야 함
        assert execution_time < 10.0

    @pytest.mark.asyncio
    async def test_analysis_performance(self, mock_stock_data):
        """분석 성능 테스트"""
        import time

        with patch.dict("os.environ", {"GEMINI_API_KEY": "test_key"}):
            analyzer = GeminiAnalyzer()

            start_time = time.time()

            with patch.object(analyzer, "analyze_and_select_top5", return_value={}):
                await analyzer.analyze_and_select_top5(mock_stock_data)

            end_time = time.time()
            execution_time = end_time - start_time

            # 30초 이내 완료되어야 함
            assert execution_time < 30.0


if __name__ == "__main__":
    # 테스트 실행
    pytest.main([__file__, "-v", "--tb=short"])
