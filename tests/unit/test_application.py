#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: test_application.py
모듈: Application 레이어 단위 테스트
목적: CLI 서비스와 명령 패턴에 대한 테스트

Author: Trading AI System
Created: 2025-01-27
Version: 1.0.0
"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime

from application.cli import CLIService
from application.commands import (
    CommandHandler, 
    GenerateSignalCommand, 
    ExecuteTradeCommand, 
    UpdateRiskCommand
)
from core.models import Signal, TradeType, StrategyType


class TestCLIService:
    """CLI 서비스 테스트"""

    @pytest.fixture
    def cli_service(self):
        """CLI 서비스 인스턴스"""
        return CLIService()

    def test_cli_service_initialization(self, cli_service):
        """CLI 서비스 초기화 테스트"""
        assert cli_service.running is False
        assert 'start' in cli_service.commands
        assert 'stop' in cli_service.commands
        assert 'status' in cli_service.commands
        assert 'backtest' in cli_service.commands
        assert 'dashboard' in cli_service.commands

    @patch('builtins.input')
    @patch('application.cli.get_logger')
    async def test_cli_start_stop(self, mock_logger, mock_input, cli_service):
        """CLI 시작/종료 테스트"""
        mock_input.return_value = "quit"
        mock_log = MagicMock()
        mock_logger.return_value = mock_log
        
        # 시작 테스트
        await cli_service.start()
        
        # 로그 확인
        mock_log.info.assert_called_with("CLI 서비스 시작")
        mock_log.info.assert_called_with("사용자에 의해 중단됨")

    @patch('builtins.input')
    @patch('application.cli.get_logger')
    async def test_cli_command_processing(self, mock_logger, mock_input, cli_service):
        """CLI 명령 처리 테스트"""
        mock_input.side_effect = ["start", "status", "quit"]
        mock_log = MagicMock()
        mock_logger.return_value = mock_log
        
        await cli_service.start()
        
        # 명령 처리 로그 확인
        assert mock_log.info.call_count >= 3

    @patch('builtins.input')
    @patch('application.cli.get_logger')
    async def test_cli_unknown_command(self, mock_logger, mock_input, cli_service):
        """알 수 없는 명령 처리 테스트"""
        mock_input.side_effect = ["unknown_command", "quit"]
        mock_log = MagicMock()
        mock_logger.return_value = mock_log
        
        # print 함수 모킹
        with patch('builtins.print') as mock_print:
            await cli_service.start()
            
            # 알 수 없는 명령 메시지 확인
            mock_print.assert_called_with("알 수 없는 명령: unknown_command")


class TestCommands:
    """명령 패턴 테스트"""

    @pytest.fixture
    def command_handler(self):
        """명령 핸들러 인스턴스"""
        return CommandHandler()

    def test_generate_signal_command_creation(self):
        """신호 생성 명령 생성 테스트"""
        command = GenerateSignalCommand(
            symbol="005930",
            strategy_type="news_momentum",
            confidence_threshold=0.8
        )
        
        assert command.symbol == "005930"
        assert command.strategy_type == "news_momentum"
        assert command.confidence_threshold == 0.8

    @patch('application.commands.get_logger')
    async def test_generate_signal_command_execution(self, mock_logger):
        """신호 생성 명령 실행 테스트"""
        command = GenerateSignalCommand(
            symbol="005930",
            strategy_type="news_momentum",
            confidence_threshold=0.8
        )
        
        mock_log = MagicMock()
        mock_logger.return_value = mock_log
        
        result = await command.execute()
        
        assert result['success'] is True
        assert 'signal' in result
        assert 'message' in result
        mock_log.info.assert_called_with("신호 생성: 005930, 전략: news_momentum")

    def test_execute_trade_command_creation(self):
        """거래 실행 명령 생성 테스트"""
        signal = Signal(
            id="signal_001",
            stock_code="005930",
            strategy_type=StrategyType.NEWS_MOMENTUM,
            signal_type=TradeType.BUY,
            confidence_score=0.8,
            reasoning="테스트 신호"
        )
        
        command = ExecuteTradeCommand(
            signal=signal,
            trade_type=TradeType.BUY,
            quantity=100,
            price=75000
        )
        
        assert command.signal == signal
        assert command.trade_type == TradeType.BUY
        assert command.quantity == 100
        assert command.price == 75000

    @patch('application.commands.get_logger')
    async def test_execute_trade_command_execution(self, mock_logger):
        """거래 실행 명령 실행 테스트"""
        signal = Signal(
            id="signal_001",
            stock_code="005930",
            strategy_type=StrategyType.NEWS_MOMENTUM,
            signal_type=TradeType.BUY,
            confidence_score=0.8,
            reasoning="테스트 신호"
        )
        
        command = ExecuteTradeCommand(
            signal=signal,
            trade_type=TradeType.BUY,
            quantity=100,
            price=75000
        )
        
        mock_log = MagicMock()
        mock_logger.return_value = mock_log
        
        result = await command.execute()
        
        assert result['success'] is True
        assert 'trade' in result
        assert 'message' in result
        mock_log.info.assert_called_with("거래 실행: 005930, 수량: 100")

    def test_update_risk_command_creation(self):
        """리스크 업데이트 명령 생성 테스트"""
        risk_parameters = {
            'max_position_size': 0.1,
            'stop_loss_pct': 0.05,
            'take_profit_pct': 0.15
        }
        
        command = UpdateRiskCommand(
            portfolio_id="portfolio_001",
            risk_parameters=risk_parameters
        )
        
        assert command.portfolio_id == "portfolio_001"
        assert command.risk_parameters == risk_parameters

    @patch('application.commands.get_logger')
    async def test_update_risk_command_execution(self, mock_logger):
        """리스크 업데이트 명령 실행 테스트"""
        risk_parameters = {
            'max_position_size': 0.1,
            'stop_loss_pct': 0.05,
            'take_profit_pct': 0.15
        }
        
        command = UpdateRiskCommand(
            portfolio_id="portfolio_001",
            risk_parameters=risk_parameters
        )
        
        mock_log = MagicMock()
        mock_logger.return_value = mock_log
        
        result = await command.execute()
        
        assert result['success'] is True
        assert 'risk_update' in result
        assert 'message' in result
        mock_log.info.assert_called_with("리스크 업데이트: 포트폴리오 portfolio_001")


class TestCommandHandler:
    """명령 핸들러 테스트"""

    @pytest.fixture
    def command_handler(self):
        """명령 핸들러 인스턴스"""
        return CommandHandler()

    @patch('application.commands.get_logger')
    async def test_command_handler_success(self, mock_logger, command_handler):
        """명령 핸들러 성공 테스트"""
        command = GenerateSignalCommand(
            symbol="005930",
            strategy_type="news_momentum",
            confidence_threshold=0.8
        )
        
        mock_log = MagicMock()
        mock_logger.return_value = mock_log
        
        result = await command_handler.handle(command)
        
        assert result['success'] is True
        mock_log.info.assert_called_with("명령 실행 성공: GenerateSignalCommand")

    @patch('application.commands.get_logger')
    async def test_command_handler_failure(self, mock_logger, command_handler):
        """명령 핸들러 실패 테스트"""
        # 실패하는 명령 생성
        class FailingCommand:
            async def execute(self):
                raise Exception("테스트 오류")
        
        command = FailingCommand()
        
        mock_log = MagicMock()
        mock_logger.return_value = mock_log
        
        result = await command_handler.handle(command)
        
        assert result['success'] is False
        assert 'error' in result
        assert result['error'] == "테스트 오류"
        mock_log.error.assert_called_with("명령 실행 실패: 테스트 오류")


if __name__ == "__main__":
    pytest.main([__file__]) 