#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: main_v2.py
모듈: 새로운 아키텍처 기반 메인 엔트리 포인트
목적: 도메인 중심 설계, 의존성 주입, 이벤트 기반 아키텍처

Author: Trading Strategy System
Created: 2025-01-27
Modified: 2025-01-27
Version: 2.0.0

Dependencies:
    - Python 3.11+
    - asyncio
    - argparse
    - signal
    - sys

Architecture:
    - Domain-Driven Design (DDD)
    - Clean Architecture
    - Dependency Injection
    - Event-Driven Architecture
    - SOLID Principles

License: MIT
"""

from __future__ import annotations

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
    from core.models import Signal, StrategyType, TradeType
    from application.cli import CLIService
    from application.dashboard import DashboardService
    from application.services import TradingSystemService
    from backtest.engine import BacktestEngine
    from core.config import config
    from core.logger import initialize_logging, get_logger, performance_monitor, error_tracker
    from core.logger import log_function_call
    from core.settings import settings
    from domain.events import event_bus
    from infrastructure.di import DependencyContainer
    from monitoring.realtime_monitor import RealtimeMonitor
    from service.command_service import CommandService
    from service.query_service import QueryService
    CORE_MODULES_AVAILABLE = True
except ImportError:
    CORE_MODULES_AVAILABLE = False

# 로거 정의
logger = None
if CORE_MODULES_AVAILABLE:
    logger = get_logger(__name__)
else:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


class TradingSystem:
    """트레이딩 시스템 메인 클래스"""

    def __init__(self) -> None:
        if not CORE_MODULES_AVAILABLE:
            raise ImportError("핵심 모듈들이 설치되지 않았습니다.")

        self.logger = get_logger(__name__)
        self.container = DependencyContainer()
        self.trading_service: Optional[TradingSystemService] = None
        self.cli_service: Optional[CLIService] = None
        self.dashboard_service: Optional[DashboardService] = None
        self._shutdown_event = asyncio.Event()
        self._tasks: list[asyncio.Task] = []

    async def initialize(self) -> None:
        """시스템 초기화"""
        try:
            self.logger.info("Trading system initialization started")

            # 로깅 시스템 초기화
            initialize_logging()

            # 의존성 컨테이너 초기화
            await self.container.initialize()

            # 서비스들 초기화
            self.trading_service = self.container.get(TradingSystemService)
            self.cli_service = self.container.get(CLIService)
            self.dashboard_service = self.container.get(DashboardService)

            # 이벤트 버스 시작
            await event_bus.start()

            # 서비스들 시작
            if self.trading_service:
                await self.trading_service.start()
            if self.cli_service:
                await self.cli_service.start()

            if settings.monitoring.dashboard_enabled and self.dashboard_service:
                await self.dashboard_service.start()

            self.logger.info("Trading system initialization completed successfully")

        except Exception as e:
            self.logger.critical(f"System initialization failed: {e}")
            error_tracker.track_error(e, context={'operation': 'system_initialization'})
            raise

    async def run(self) -> None:
        """시스템 실행"""
        try:
            self.logger.info("Trading system started")

            # 메인 루프
            while not self._shutdown_event.is_set():
                try:
                    await asyncio.sleep(1)
                except asyncio.CancelledError:
                    break

            self.logger.info("Trading system main loop completed")

        except Exception as e:
            self.logger.critical(f"System runtime error: {e}")
            error_tracker.track_error(e, context={'operation': 'system_runtime'})
            raise

    async def shutdown(self, signal_name: Optional[str] = None) -> None:
        """시스템 종료"""
        try:
            self.logger.info(f"Trading system shutdown initiated{f' by {signal_name}' if signal_name else ''}")

            # 종료 이벤트 설정
            self._shutdown_event.set()

            # 실행 중인 태스크들 취소
            for task in self._tasks:
                if not task.done():
                    task.cancel()

            # 서비스들 종료
            if self.dashboard_service:
                await self.dashboard_service.stop()

            if self.cli_service:
                await self.cli_service.stop()

            if self.trading_service:
                await self.trading_service.stop()

            # 이벤트 버스 종료
            await event_bus.stop()

            # 의존성 컨테이너 정리
            await self.container.cleanup()

            self.logger.info("Trading system shutdown completed")

        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            error_tracker.track_error(e, context={'operation': 'system_shutdown'})

    def setup_signal_handlers(self) -> None:
        """시그널 핸들러 설정"""
        def signal_handler(signum: int, frame) -> None:
            signal_name = signal.Signals(signum).name
            self.logger.info(f"Received signal: {signal_name}")
            asyncio.create_task(self.shutdown(signal_name))

        # 종료 시그널들 등록
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Windows에서 SIGBREAK 지원
        if hasattr(signal, 'SIGBREAK'):
            signal.signal(signal.SIGBREAK, signal_handler)

    async def run_with_monitoring(self) -> None:
        """모니터링과 함께 시스템 실행"""
        try:
            # 성능 모니터링 시작
            with performance_monitor.measure("system_startup"):
                await self.initialize()

            # 시그널 핸들러 설정
            self.setup_signal_handlers()

            # 메인 실행
            with performance_monitor.measure("system_runtime"):
                await self.run()

        except Exception as e:
            self.logger.critical(f"Fatal system error: {e}")
            error_tracker.track_error(e, context={'operation': 'fatal_error'})
            raise
        finally:
            # 종료 처리
            with performance_monitor.measure("system_shutdown"):
                await self.shutdown()

    async def run_realtime_trading(self) -> None:
        """실시간 AI 트레이딩 실행"""
        try:
            if not CORE_MODULES_AVAILABLE:
                raise ImportError("핵심 모듈들이 설치되지 않았습니다.")

            logger.info("실시간 AI 트레이딩 시작")

            # 실시간 모니터링 시스템 초기화
            monitor = RealtimeMonitor()

            # 모니터링할 종목 설정 (KOSPI 200 + KOSDAQ 150)
            target_symbols = [
                "005930", "000660", "035420", "051910", "006400",  # 삼성전자, SK하이닉스, NAVER, LG화학, 삼성SDI
                "035720", "207940", "068270", "323410", "051900",  # 카카오, 삼성바이오로직스, 셀트리온, 카카오뱅크, LG생활건강
                "006380", "017670", "035460", "035600", "068760"   # 카프리, SK텔레콤, 기아, 삼성전기, 셀트리온제약
            ]

            # 테마 키워드 설정
            theme_keywords = [
                "AI", "반도체", "바이오", "전기차", "배터리", "메타버스", "블록체인",
                "ESG", "친환경", "디지털", "클라우드", "5G", "로봇", "드론"
            ]

            # 실시간 모니터링 시작
            await monitor.start_monitoring(
                symbols=target_symbols,
                theme_keywords=theme_keywords
            )

            # AI 트레이딩 루프
            while True:
                try:
                    # 최근 알림 확인
                    recent_alerts = monitor.get_recent_alerts(hours=1)
                    high_priority_alerts = monitor.get_high_priority_alerts(min_priority=4)

                    if high_priority_alerts:
                        logger.info(f"고우선순위 알림 발견: {len(high_priority_alerts)}개")

                        for alert in high_priority_alerts:
                            # AI 분석 및 매매 신호 생성
                            await self.process_ai_trading_signal(alert)

                    # 1분 대기
                    await asyncio.sleep(60)

                except KeyboardInterrupt:
                    logger.info("사용자에 의해 중단됨")
                    break
                except Exception as e:
                    logger.error(f"AI 트레이딩 루프 오류: {e}")
                    await asyncio.sleep(10)

            # 모니터링 중지
            await monitor.stop_monitoring()
            logger.info("실시간 AI 트레이딩 종료")

        except Exception as e:
            logger.error(f"실시간 AI 트레이딩 실행 실패: {e}")
            raise

    async def process_ai_trading_signal(self, alert: Any) -> None:
        """AI 매매 신호 처리"""
        try:
            logger.info(f"AI 매매 신호 처리: {alert.alert_type} - {alert.symbol}")

            # 1. 실시간 데이터 수집
            realtime_data = await self.collect_realtime_data(alert.symbol)

            # 2. AI 분석 (ML/DL 모델)
            ai_analysis = await self.run_ai_analysis(realtime_data, alert)

            # 3. 알고리즘 판단
            trading_decision = await self.make_trading_decision(ai_analysis)

            # 4. 매매 신호 생성
            if trading_decision['should_trade']:
                signal = await self.generate_trading_signal(
                    symbol=alert.symbol,
                    decision=trading_decision,
                    confidence=trading_decision['confidence']
                )

                # 5. 거래 실행
                await self.execute_trade(signal)

                logger.info(f"AI 매매 실행 완료: {alert.symbol} - {trading_decision['action']}")

        except Exception as e:
            logger.error(f"AI 매매 신호 처리 오류: {e}")

    async def collect_realtime_data(self, symbol: str) -> Dict[str, Any]:
        """실시간 데이터 수집"""
        # 실시간 가격, 거래량, 호가 데이터 수집
        return {
            'symbol': symbol,
            'price': 50000,  # 실제로는 실시간 API에서 가져옴
            'volume': 1000000,
            'timestamp': datetime.now()
        }

    async def run_ai_analysis(self, data: Dict[str, Any], alert: Any) -> Dict[str, Any]:
        """AI 분석 (ML/DL 모델)"""
        # 실제로는 학습된 모델로 분석
        return {
            'sentiment_score': 0.7,
            'trend_prediction': 'UP',
            'confidence': 0.8,
            'risk_score': 0.3
        }

    async def make_trading_decision(self, ai_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """알고리즘 판단"""
        # AI 분석 결과를 바탕으로 매매 판단
        confidence = ai_analysis['confidence']
        sentiment = ai_analysis['sentiment_score']
        risk = ai_analysis['risk_score']

        should_trade = confidence > 0.7 and risk < 0.5
        action = 'BUY' if sentiment > 0.6 else 'SELL' if sentiment < 0.4 else 'HOLD'

        return {
            'should_trade': should_trade,
            'action': action,
            'confidence': confidence,
            'reasoning': f"AI 신뢰도: {confidence:.2f}, 감정점수: {sentiment:.2f}, 리스크: {risk:.2f}"
        }

    async def generate_trading_signal(self, symbol: str, decision: Dict[str, Any], confidence: float) -> Signal:
        """매매 신호 생성"""
        return Signal(
            id=f"ai_{symbol}_{int(time.time())}",
            stock_code=symbol,
            strategy_type=StrategyType.COMBINED,
            signal_type=TradeType.BUY if decision['action'] == 'BUY' else TradeType.SELL,
            confidence_score=confidence,
            target_price=None,
            stop_loss=None,
            take_profit=None,
            reasoning=decision['reasoning'],
            created_at=datetime.now()
        )

    async def execute_trade(self, signal: Signal) -> None:
        """거래 실행"""
        # 실제 거래 실행 로직
        logger.info(f"거래 실행: {signal.stock_code} {signal.signal_type} (신뢰도: {signal.confidence_score})")

    async def run_backtest(self, args: argparse.Namespace) -> None:
        """백테스트 실행"""
        try:
            if not CORE_MODULES_AVAILABLE:
                raise ImportError("핵심 모듈들이 설치되지 않았습니다.")

            logger.info("백테스트 모드 시작")
            # 백테스트 로직 구현
            pass
        except Exception as e:
            logger.error(f"백테스트 실행 실패: {e}")

    async def run_dashboard(self) -> None:
        """대시보드 실행"""
        try:
            if not CORE_MODULES_AVAILABLE:
                raise ImportError("핵심 모듈들이 설치되지 않았습니다.")

            logger.info("대시보드 모드 시작")
            # 대시보드 로직 구현
            pass
        except Exception as e:
            logger.error(f"대시보드 실행 실패: {e}")


async def main() -> None:
    """메인 함수"""
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description="Trading Strategy System v2.0")
    parser.add_argument(
        "--config",
        type=str,
        default=".env",
        help="설정 파일 경로 (기본값: .env)"
    )
    parser.add_argument(
        "--mode",
        choices=["live", "mock", "backtest"],
        default="mock",
        help="실행 모드 (기본값: mock)"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="로깅 레벨 (기본값: INFO)"
    )
    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="대시보드 비활성화"
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="백테스트 모드 활성화"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="백테스트 시작 날짜 (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="백테스트 종료 날짜 (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--initial-capital",
        type=int,
        help="백테스트 초기자본"
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        help="백테스트할 전략들"
    )
    parser.add_argument('--realtime', action='store_true', help='실시간 AI 트레이딩 모드')

    args = parser.parse_args()

    if not CORE_MODULES_AVAILABLE:
        print("❌ 핵심 모듈들이 설치되지 않았습니다.")
        print("다음 명령어로 필요한 패키지들을 설치하세요:")
        print("pip install -r requirements.txt")
        return

    # 설정 파일 로드
    if args.config != ".env":
        settings.model_config["env_file"] = args.config

    # 로깅 레벨 설정
    settings.logging.level = args.log_level

    # 대시보드 비활성화
    if args.no_dashboard:
        settings.monitoring.dashboard_enabled = False

    # 실행 모드 설정
    settings.environment = "production" if args.mode == "live" else "development"

    # 백테스트 실행
    if args.backtest:
        logger.info("백테스트 모드 시작")
        backtest_config = {
            "start_date": args.start_date or "1990-01-01",
            "end_date": args.end_date or "2024-12-31",
            "initial_capital": args.initial_capital or 10000000,
            "enable_compression": True,
            "cache_enabled": True,
            "batch_size": 500
        }
        strategies = args.strategies or ["news", "technical", "theme", "sentiment"]
        logger.info(f"백테스트 기간: {backtest_config['start_date']} ~ {backtest_config['end_date']}")
        logger.info(f"초기자본: {backtest_config['initial_capital']:,}원")
        logger.info(f"전략: {strategies}")
        try:
            engine = BacktestEngine(
                start_date=backtest_config["start_date"],
                end_date=backtest_config["end_date"],
                initial_capital=backtest_config["initial_capital"],
                enable_compression=backtest_config["enable_compression"],
                cache_enabled=backtest_config["cache_enabled"],
                batch_size=backtest_config["batch_size"]
            )
            for strategy in strategies:
                result = await engine.run_backtest(
                    strategy_type=strategy,
                    start_date=engine.start_date,
                    end_date=engine.end_date,
                    initial_capital=engine.initial_capital
                )
                # 결과 출력 (기존 print 코드 활용)
                print("\n" + "="*60)
                print(f"🎯 백테스트 결과 - 전략: {strategy}")
                print("="*60)
                print(f"📅 백테스트 기간: {backtest_config['start_date']} ~ {backtest_config['end_date']}")
                print(f"💰 초기자본: {backtest_config['initial_capital']:,}원")
                print(f"💵 최종자본: {getattr(result, 'final_capital', 'N/A'):,}원")
                print(f"📈 총 수익률: {getattr(result, 'total_return', 0):.2%}")
                print(f"📊 연평균 수익률: {getattr(result, 'annual_return', 0):.2%}")
                print(f"📉 최대 낙폭: {getattr(result, 'max_drawdown', 0):.2%}")
                print(f"📊 샤프 비율: {getattr(result, 'sharpe_ratio', 0):.2f}")
                print(f"🎯 승률: {getattr(result, 'win_rate', 0):.1%}")
                print(f"🔄 총 거래 횟수: {getattr(result, 'total_trades', 0):,}회")
                print(f"📈 평균 수익: {getattr(result, 'avg_profit', 0):,.0f}원")
                print(f"📉 평균 손실: {getattr(result, 'avg_loss', 0):,.0f}원")
                print(f"📊 수익/손실 비율: {getattr(result, 'profit_loss_ratio', 0):.2f}")
                print(f"🔥 최대 연속 손실: {getattr(result, 'max_consecutive_losses', 0)}회")
                print("="*60)
        except Exception as e:
            logger.error(f"백테스트 실행 실패: {e}")
            print(f"❌ 백테스트 실행 실패: {e}")
            return None

    # 시스템 생성 및 실행
    system = TradingSystem()

    try:
        # 모드별 실행
        if args.realtime:
            logger.info("실시간 AI 트레이딩 모드 시작")
            await system.run_realtime_trading()
        elif args.backtest:
            logger.info("백테스트 모드 시작")
            await system.run_backtest(args)
        else:
            logger.info("대시보드 모드 시작")
            await system.run_dashboard()
    except KeyboardInterrupt:
        print("\n시스템이 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"시스템 실행 중 오류가 발생했습니다: {e}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        # Windows에서 asyncio 이벤트 루프 정책 설정
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

        # 메인 함수 실행
        asyncio.run(main())

    except KeyboardInterrupt:
        print("\n프로그램이 중단되었습니다.")
        sys.exit(0)
    except Exception as e:
        print(f"프로그램 실행 중 치명적 오류가 발생했습니다: {e}")
        sys.exit(1)

