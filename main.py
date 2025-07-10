#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: main.py
모듈: 통합 메인 엔트리 포인트
목적: 모든 시스템 기능을 통합하여 단일 진입점 제공

Author: Trading AI System
Created: 2025-01-27
Version: 2.0.0

Dependencies:
    - Python 3.11+
    - asyncio
    - argparse
    - signal
    - sys

Architecture:
    - Clean Architecture
    - Dependency Injection
    - Event-Driven Architecture
    - Command Pattern

License: MIT
"""

import argparse
import asyncio
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from core.config import config
    from core.logger import initialize_logging, get_logger, performance_monitor, error_tracker
    from core.models import Signal, StrategyType, TradeType
    from application.cli import CLIService
    from application.commands import CommandHandler, GenerateSignalCommand, ExecuteTradeCommand, UpdateRiskCommand
    from domain.events import event_bus
    from infrastructure.di import DependencyContainer
    from service.command_service import CommandService
    from service.query_service import QueryService
    from data.auto_data_collector import AutoDataCollector
    from src.agile_trading_strategy import AgileTradingStrategy
    from src.main_integrated import IntegratedTradingSystem
    CORE_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ 핵심 모듈 import 실패: {e}")
    CORE_MODULES_AVAILABLE = False

# 로거 정의
logger = None
if CORE_MODULES_AVAILABLE:
    logger = get_logger(__name__)
else:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


class UnifiedTradingSystem:
    """통합 트레이딩 시스템"""

    def __init__(self) -> None:
        if not CORE_MODULES_AVAILABLE:
            raise ImportError("핵심 모듈들이 설치되지 않았습니다.")

        self.logger = get_logger(__name__)
        self.container = DependencyContainer()
        self.cli_service: Optional[CLIService] = None
        self.command_handler: Optional[CommandHandler] = None
        self.data_collector: Optional[AutoDataCollector] = None
        self.trading_strategy: Optional[AgileTradingStrategy] = None
        self.integrated_system: Optional[IntegratedTradingSystem] = None
        self._shutdown_event = asyncio.Event()
        self._tasks: list[asyncio.Task] = []

    async def initialize(self) -> None:
        """시스템 초기화"""
        try:
            self.logger.info("🚀 통합 트레이딩 시스템 초기화 시작")

            # 로깅 시스템 초기화
            initialize_logging()

            # 의존성 컨테이너 초기화
            await self.container.initialize()

            # 서비스들 초기화
            self.cli_service = self.container.get(CLIService)
            self.command_handler = CommandHandler()
            self.data_collector = AutoDataCollector()
            self.trading_strategy = AgileTradingStrategy(config.trading.__dict__ if config.trading else {})
            self.integrated_system = IntegratedTradingSystem()

            # 이벤트 버스 시작
            await event_bus.start()

            # 데이터 수집기 초기화
            await self.data_collector.initialize()

            # 통합 시스템 초기화
            await self.integrated_system.initialize()

            self.logger.info("✅ 통합 트레이딩 시스템 초기화 완료")

        except Exception as e:
            self.logger.critical(f"❌ 시스템 초기화 실패: {e}")
            error_tracker.track_error(e, context={'operation': 'system_initialization'})
            raise

    async def run(self, mode: str = "interactive") -> None:
        """시스템 실행"""
        try:
            self.logger.info(f"🚀 통합 트레이딩 시스템 시작 (모드: {mode})")

            if mode == "interactive":
                await self._run_interactive_mode()
            elif mode == "automated":
                await self._run_automated_mode()
            elif mode == "backtest":
                await self._run_backtest_mode()
            elif mode == "dashboard":
                await self._run_dashboard_mode()
            else:
                raise ValueError(f"지원하지 않는 모드: {mode}")

        except Exception as e:
            self.logger.critical(f"❌ 시스템 실행 오류: {e}")
            error_tracker.track_error(e, context={'operation': 'system_runtime'})
            raise

    async def _run_interactive_mode(self) -> None:
        """대화형 모드 실행"""
        self.logger.info("💬 대화형 모드 시작")
        
        if self.cli_service:
            await self.cli_service.start()

    async def _run_automated_mode(self) -> None:
        """자동화 모드 실행"""
        self.logger.info("🤖 자동화 모드 시작")
        
        # 메인 루프
        while not self._shutdown_event.is_set():
            try:
                # 실시간 데이터 수집
                await self._collect_real_time_data()
                
                # 트레이딩 신호 생성
                signals = await self._generate_trading_signals()
                
                # 거래 실행
                if signals:
                    await self._execute_trades(signals)
                
                # 포트폴리오 업데이트
                await self._update_portfolio()
                
                # 대기
                await asyncio.sleep(config.trading.REALTIME_UPDATE_INTERVAL if config.trading else 1)
                
            except Exception as e:
                self.logger.error(f"자동화 모드 오류: {e}")
                await asyncio.sleep(5)

    async def _run_backtest_mode(self) -> None:
        """백테스트 모드 실행"""
        self.logger.info("📊 백테스트 모드 시작")
        
        # 백테스트 실행 로직
        command = GenerateSignalCommand(
            symbol="005930",  # 삼성전자
            strategy_type="news_momentum",
            confidence_threshold=0.7
        )
        
        result = await self.command_handler.handle(command)
        self.logger.info(f"백테스트 결과: {result}")

    async def _run_dashboard_mode(self) -> None:
        """대시보드 모드 실행"""
        self.logger.info("📈 대시보드 모드 시작")
        
        # 대시보드 실행 로직
        if self.integrated_system:
            await self.integrated_system._update_dashboard()

    async def _collect_real_time_data(self) -> None:
        """실시간 데이터 수집"""
        try:
            if self.data_collector:
                await self.data_collector._collect_realtime_prices()
        except Exception as e:
            self.logger.error(f"실시간 데이터 수집 오류: {e}")

    async def _generate_trading_signals(self) -> List[Dict[str, Any]]:
        """트레이딩 신호 생성"""
        try:
            if self.trading_strategy:
                # 실제 구현에서는 신호 생성 로직
                return []
        except Exception as e:
            self.logger.error(f"신호 생성 오류: {e}")
        return []

    async def _execute_trades(self, signals: List[Dict[str, Any]]) -> None:
        """거래 실행"""
        try:
            for signal_data in signals:
                # 실제 구현에서는 거래 실행 로직
                self.logger.info(f"거래 실행: {signal_data}")
        except Exception as e:
            self.logger.error(f"거래 실행 오류: {e}")

    async def _update_portfolio(self) -> None:
        """포트폴리오 업데이트"""
        try:
            # 실제 구현에서는 포트폴리오 업데이트 로직
            pass
        except Exception as e:
            self.logger.error(f"포트폴리오 업데이트 오류: {e}")

    async def shutdown(self, signal_name: Optional[str] = None) -> None:
        """시스템 종료"""
        try:
            self.logger.info(f"🛑 통합 트레이딩 시스템 종료{f' (시그널: {signal_name})' if signal_name else ''}")

            # 종료 이벤트 설정
            self._shutdown_event.set()

            # 실행 중인 태스크들 취소
            for task in self._tasks:
                if not task.done():
                    task.cancel()

            # 서비스들 종료
            if self.cli_service:
                await self.cli_service.stop()

            if self.data_collector:
                await self.data_collector.stop()

            if self.integrated_system:
                await self.integrated_system.stop()

            # 이벤트 버스 종료
            await event_bus.stop()

            # 의존성 컨테이너 정리
            await self.container.cleanup()

            self.logger.info("✅ 통합 트레이딩 시스템 종료 완료")

        except Exception as e:
            self.logger.error(f"❌ 종료 중 오류: {e}")
            error_tracker.track_error(e, context={'operation': 'system_shutdown'})

    def setup_signal_handlers(self) -> None:
        """시그널 핸들러 설정"""
        def signal_handler(signum: int, frame) -> None:
            signal_name = signal.Signals(signum).name
            self.logger.info(f"📡 시그널 수신: {signal_name}")
            asyncio.create_task(self.shutdown(signal_name))

        # 종료 시그널들 등록
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Windows에서 SIGBREAK 지원
        if hasattr(signal, 'SIGBREAK'):
            signal.signal(signal.SIGBREAK, signal_handler)


def create_parser() -> argparse.ArgumentParser:
    """명령행 인수 파서 생성"""
    parser = argparse.ArgumentParser(
        description="통합 트레이딩 시스템",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python main.py                    # 대화형 모드
  python main.py --mode automated   # 자동화 모드
  python main.py --mode backtest    # 백테스트 모드
  python main.py --mode dashboard   # 대시보드 모드
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['interactive', 'automated', 'backtest', 'dashboard'],
        default='interactive',
        help='실행 모드 (기본값: interactive)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='설정 파일 경로'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='로그 레벨 (기본값: INFO)'
    )
    
    return parser


async def main() -> None:
    """메인 함수"""
    parser = create_parser()
    args = parser.parse_args()
    
    # 로그 레벨 설정
    if CORE_MODULES_AVAILABLE:
        import logging
        logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # 시스템 생성 및 초기화
    system = UnifiedTradingSystem()
    
    try:
        # 시그널 핸들러 설정
        system.setup_signal_handlers()
        
        # 시스템 초기화
        await system.initialize()
        
        # 시스템 실행
        await system.run(args.mode)
        
    except KeyboardInterrupt:
        logger.info("👆 사용자에 의해 중단됨")
    except Exception as e:
        logger.critical(f"❌ 치명적 오류: {e}")
        sys.exit(1)
    finally:
        await system.shutdown()


if __name__ == "__main__":
    asyncio.run(main()) 