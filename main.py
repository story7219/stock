"""
🤖 통합 자동매매 시스템 v2.0 (Orchestrator)
===========================================

시스템의 모든 모듈을 조립하고 전체 실행 흐름을 관장하는 중앙 관제소입니다.
이 파일은 시스템의 유일한 진입점(Entry Point) 역할을 합니다.

주요 기능:
- 설정 및 로거 초기화
- 핵심 컴포넌트(Trader, Provider, Analyzer, Manager) 생성 및 의존성 주입
- 스케줄러를 이용한 주기적 작업 실행 (척후병 전략)
- 안전한 시스템 시작 및 종료 처리

실행: python main.py
"""
import asyncio
import logging
import signal
import sys
import traceback
import argparse
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

import config
from utils.logger_config import setup_logging
from core_trader import CoreTrader
from market_data_provider import AIDataCollector, StockFilter
from ai_analyzer import AIAnalyzer
from scout_strategy_manager import ScoutStrategyManager
from google_sheet_logger import GoogleSheetLogger

# --- 로거 설정 ---
# 다른 모듈보다 먼저 설정되어야 전역적으로 적용됩니다.
setup_logging()
logger = logging.getLogger(__name__)


class TradingSystemOrchestrator:
    """시스템 전체를 조율하고 관리하는 오케스트레이터 클래스"""

    def __init__(self):
        self.scheduler = AsyncIOScheduler()
        self.shutdown_event = asyncio.Event()
        self.trader = None
        self.strategy_manager = None
        self.sheet_logger = None
        self.mode = "scout" # 기본 모드 설정

    async def initialize(self, mode: str = "scout") -> bool:
        """시스템의 모든 핵심 컴포넌트를 초기화하고 의존성을 주입합니다."""
        self.mode = mode
        logger.info("==================================================")
        logger.info(f"🤖 통합 자동매매 시스템 초기화를 시작합니다... (모드: {self.mode})")
        logger.info("==================================================")
        try:
            # --- 1. 환경변수 및 설정 검증 ---
            missing_configs, _ = config.validate_config()
            if missing_configs:
                logger.critical(f"❌ 필수 환경변수가 설정되지 않았습니다: {missing_configs}")
                return False

            # --- 1.5. 구글 시트 로거 초기화 (안전 모드) ---
            logger.info("🔧 [1/5] 구글 시트 로거를 초기화합니다...")
            try:
                if config.GOOGLE_SERVICE_ACCOUNT_FILE and config.GOOGLE_SPREADSHEET_ID:
                    self.sheet_logger = GoogleSheetLogger(
                        credentials_path=config.GOOGLE_SERVICE_ACCOUNT_FILE,
                        spreadsheet_key=config.GOOGLE_SPREADSHEET_ID
                    )
                    await self.sheet_logger.async_initialize()
                    
                    # async_initialize가 성공적으로 완료되었는지 한번 더 확인
                    if not self.sheet_logger.initialized:
                        logger.warning("⚠️ 구글 시트 로거 초기화에 실패했습니다. 로깅 기능이 비활성화됩니다.")
                        self.sheet_logger = None
                else:
                    logger.warning("⚠️ 구글 시트 관련 환경변수가 설정되지 않아 로깅 기능이 비활성화됩니다.")
                    self.sheet_logger = None
            except Exception as e:
                logger.error(f"💥 구글 시트 로거 초기화 중 심각한 예외 발생: {e}. 로깅 기능이 비활성화됩니다.", exc_info=True)
                self.sheet_logger = None

            # --- 2. 핵심 컴포넌트 생성 (의존성 주입 준비) ---
            logger.info("🔧 [2/5] 코어 트레이더를 초기화합니다...")
            self.trader = CoreTrader(sheet_logger=self.sheet_logger)
            if not await self.trader.async_initialize():
                logger.critical("❌ 코어 트레이더 초기화 실패. 시스템을 시작할 수 없습니다.")
                return False

            logger.info("🔧 [3/5] 마켓 데이터 제공자를 초기화합니다...")
            data_provider = AIDataCollector(self.trader)
            stock_filter = StockFilter(self.trader)

            logger.info("🔧 [4/5] AI 분석기를 초기화합니다...")
            ai_analyzer = AIAnalyzer(trader=self.trader, data_provider=data_provider)

            logger.info("🔧 [5/5] 전략 관리자를 초기화합니다...")
            self.strategy_manager = ScoutStrategyManager(
                trader=self.trader,
                data_provider=data_provider,
                stock_filter=stock_filter,
                ai_analyzer=ai_analyzer,
            )
            
            logger.info("✅ 모든 컴포넌트가 성공적으로 초기화되었습니다.")
            return True

        except Exception as e:
            logger.critical("❌ 시스템 초기화 중 심각한 오류 발생", exc_info=True)
            return False

    def setup_schedules(self):
        """스케줄러에 주기적인 작업을 등록합니다."""
        if not self.strategy_manager:
            logger.error("전략 관리자가 초기화되지 않아 스케줄을 설정할 수 없습니다.")
            return

        logger.info(f"⏰ 스케줄러를 설정합니다: {config.SCOUT_RUN_INTERVAL_MIN}분마다 '{self.mode}' 전략 실행")
        
        # 'run' 메서드에 모드를 인자로 전달
        job_func = lambda: self.strategy_manager.run(mode=self.mode)
        
        self.scheduler.add_job(
            job_func,
            trigger=IntervalTrigger(minutes=config.SCOUT_RUN_INTERVAL_MIN),
            id="strategy_run",
            name=f"{self.mode} 전략 실행",
            max_instances=1, # 동시에 여러 작업이 실행되지 않도록 보장
        )
        # 일일 리포트 생성 (오후 3시 40분)
        self.scheduler.add_job(
            self.strategy_manager.generate_daily_report,
            trigger='cron',
            hour=15,
            minute=40,
            id="daily_report_generation",
            name="AI 코치 일일 리포트 생성"
        )


    async def run(self, mode: str = "scout"):
        """시스템을 시작하고 종료 시그널을 대기합니다."""
        if not await self.initialize(mode):
            logger.critical("시스템 초기화 실패. 프로그램을 종료합니다.")
            return

        self.setup_schedules()
        self.scheduler.start()
        
        # 초기 즉시 실행 (테스트 및 빠른 피드백 용도)
        logger.info(f"🚀 시스템 시작 즉시 첫 번째 '{mode}' 전략 실행을 예약합니다.")
        self.scheduler.add_job(lambda: self.strategy_manager.run(mode=mode), 'date', run_date=None)

        initial_message = (
            "==================================================\n"
            f"✅ 자동매매 시스템이 성공적으로 시작되었습니다. (모드: {mode})\n"
            f"🕒 {config.SCOUT_RUN_INTERVAL_MIN}분 간격으로 전략을 실행합니다.\n"
            "🛑 종료하려면 Ctrl+C를 누르세요.\n"
            "=================================================="
        )
        logger.info(initial_message)
        
        if self.trader and self.trader.notifier:
            await self.trader.notifier.send_message("🚀 자동매매 시스템이 시작되었습니다.")

        # 종료 시그널 대기
        await self.shutdown_event.wait()

    async def shutdown(self):
        """시스템을 안전하게 종료합니다."""
        if self.shutdown_event.is_set():
            return
            
        logger.info("==================================================")
        logger.info("👋 시스템 종료를 시작합니다...")
        logger.info("==================================================")

        # 스케줄러 종료
        if self.scheduler.running:
            logger.info("⏰ 스케줄러를 중지합니다...")
            self.scheduler.shutdown(wait=True)
            logger.info("✅ 스케줄러가 성공적으로 중지되었습니다.")

        # 리포트 생성 (종료 시)
        if self.strategy_manager:
            try:
                logger.info("📊 최종 일일 리포트를 생성합니다...")
                # generate_daily_report가 동기 함수이므로, 비동기 루프에서 안전하게 실행
                await asyncio.to_thread(self.strategy_manager.generate_daily_report)
                logger.info("✅ 최종 리포트 생성이 완료되었습니다.")
            except Exception:
                logger.error("💥 최종 리포트 생성 중 오류 발생.", exc_info=True)
        
        final_message = "👋 자동매매 시스템이 안전하게 종료되었습니다."
        logger.info(final_message)
        if self.trader and self.trader.notifier:
            await self.trader.notifier.send_message(final_message)

        self.shutdown_event.set()


def handle_signal(sig, loop, system):
    """운영체제 시그널을 처리하여 안전한 종료를 유도합니다."""
    logger.info(f"🛑 시그널 {sig} 수신. 시스템 종료를 시작합니다.")
    asyncio.create_task(system.shutdown())


async def main():
    """애플리케이션의 메인 진입점"""
    parser = argparse.ArgumentParser(description="AI 기반 자동매매 시스템")
    parser.add_argument(
        "mode",
        type=str,
        nargs='?',
        default="scout",
        choices=["scout", "advanced"],
        help="실행할 트레이딩 모드 ('scout' 또는 'advanced')"
    )
    args = parser.parse_args()
    
    orchestrator = TradingSystemOrchestrator()

    # 시그널 핸들러 설정
    try:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, handle_signal, sig, loop, orchestrator)
    except RuntimeError: # 'no running event loop'
        loop = None

    try:
        await orchestrator.run(mode=args.mode)
    except Exception:
        logger.critical("💥 오케스트레이터에서 처리되지 않은 예외 발생", exc_info=True)
    finally:
        # 이미 종료 프로세스가 진행 중일 수 있으므로, 재진입을 방지합니다.
        if not orchestrator.shutdown_event.is_set():
            logger.info("🏁 최종 정리 작업을 수행합니다.")
            await orchestrator.shutdown()
        logger.info("프로그램이 완전히 종료되었습니다.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("프로그램 실행이 중단되었습니다.")
    sys.exit(0)