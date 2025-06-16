# main.py
# 자동매매 시스템의 전체 흐름을 제어하는 메인 컨트롤러 (비동기)

import sys
import os
import asyncio
from datetime import datetime
from dotenv import load_dotenv

# --- 시스템 경로 설정 ---
# 현재 파일의 디렉토리를 기준으로 프로젝트 루트 디렉토리를 시스템 경로에 추가
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

# --- 모듈 임포트 ---
import config
from trading.kis_api import KIS_API
from trading.trader import Trader
from portfolio import PortfolioManager
from utils.logger import log_event
from utils.telegram_bot import TelegramBot

async def run_system():
    """'실시간 오디션' 전략 기반의 비동기 자동매매 메인 루프"""
    log_event("INFO", "🚀 자동매매 시스템을 시작합니다...")

    # --- API 및 주요 모듈 초기화 ---
    app_key = config.LIVE_KIS_APP_KEY if not config.IS_MOCK_TRADING else config.MOCK_KIS_APP_KEY
    app_secret = config.LIVE_KIS_APP_SECRET if not config.IS_MOCK_TRADING else config.MOCK_KIS_APP_SECRET
    account_number = config.LIVE_KIS_ACCOUNT_NUMBER if not config.IS_MOCK_TRADING else config.MOCK_KIS_ACCOUNT_NUMBER

    if not all([app_key, app_secret, account_number]):
        log_event("CRITICAL", "API 키 또는 계좌번호가 .env 파일에 설정되지 않았습니다. 프로그램을 종료합니다.")
        return

    telegram_bot = TelegramBot()
    kis_api = KIS_API(app_key, app_secret, account_number, mock=config.IS_MOCK_TRADING, telegram_bot=telegram_bot)
    portfolio_manager = PortfolioManager(capital=config.TOTAL_CAPITAL, kis_api=kis_api, telegram_bot=telegram_bot)
    trader = Trader(portfolio_manager=portfolio_manager, kis_api=kis_api, telegram_bot=telegram_bot)
    
    mode = '모의투자' if config.IS_MOCK_TRADING else '실전투자'
    telegram_bot.send_message(f"✅ 자동매매 시스템 'Phoenix' 시작 ({mode})")

    # --- 메인 실행 루프 ---
    while True:
        try:
            now = datetime.now()
            market_open = now.replace(hour=9, minute=0, second=0)
            market_close = now.replace(hour=15, minute=30, second=0)

            if not (market_open <= now <= market_close):
                log_event("INFO", f"⏳ 장 운영 시간이 아닙니다. 다음 개장까지 대기... (현재: {now:%H:%M})")
                await asyncio.sleep(60)
                continue
            
            # 1. 포트폴리오 관리 (청산 및 오디션 심사)
            await trader._manage_portfolio()
            
            # 2. 신규 오디션 시작
            await trader._start_new_audition()

            log_event("INFO", f"--- ✅ 1 사이클 완료. {config.SYSTEM_CHECK_INTERVAL_MINUTES}분 후 다음 사이클 시작... ---")
            await asyncio.sleep(config.SYSTEM_CHECK_INTERVAL_MINUTES * 60)

        except KeyboardInterrupt:
            telegram_bot.send_message("⏹️ 자동매매 시스템을 수동으로 종료합니다.")
            log_event("INFO", "사용자에 의해 프로그램이 중단되었습니다.")
            break
        except Exception as e:
            import traceback
            error_msg = f"🔥 [메인 루프 오류] {e}"
            log_event("CRITICAL", f"{error_msg}\n{traceback.format_exc()}")
            telegram_bot.send_message(error_msg)
            await asyncio.sleep(60) # 오류 발생 시 잠시 대기 후 재시도

if __name__ == '__main__':
    load_dotenv()
    try:
        asyncio.run(run_system())
    except KeyboardInterrupt:
        log_event("INFO", "프로그램이 종료되었습니다.") 