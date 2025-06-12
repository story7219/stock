# main.py
# 자동매매 시스템의 전체 흐름을 제어하는 메인 컨트롤러 (비동기)

import sys
import os

# --- 시스템 경로 설정 ---
# 현재 파일의 디렉토리를 기준으로 프로젝트 루트 디렉토리를 시스템 경로에 추가
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

import asyncio
import time
from datetime import datetime
import config
from dotenv import load_dotenv

# 클래스 정의는 각자의 파일로 분리
from portfolio import PortfolioManager
from trading.kis_api import KIS_API
from trading.trader import Trader
from utils.logger import log_event
from utils.telegram_bot import TelegramBot
from utils.gspread_client import gspread_client # 구글 시트 클라이언트 import
from utils.system_utils import get_public_ip # 공인 IP 확인 유틸리티 import

async def main():
    """메인 실행 함수 (비동기)"""
    # --- .env 파일에서 환경 변수 로드 ---
    load_dotenv()
    
    # --- 공인 IP 확인 및 안내 ---
    public_ip = get_public_ip()
    ip_check_message = (
        f"🖥️ 현재 공인 IP: {public_ip}\n"
        f"이 IP가 한국투자증권 개발자 포털에 등록되어 있는지 확인하세요.\n"
        f"IP가 다르거나 미등록 시, 실전투자 API 접속이 거부됩니다."
    )
    if public_ip:
        log_event("INFO", ip_check_message)
    else:
        log_event("WARNING", "공인 IP 주소를 확인하지 못했습니다. KIS API 접속에 문제가 발생할 수 있습니다.")
    
    # config.py의 설정에 따라 사용할 키와 모드를 결정
    if config.IS_MOCK_TRADING:
        log_event("INFO", "=== 모의투자 환경으로 시작합니다. ===")
        APP_KEY = os.getenv("MOCK_KIS_APP_KEY")
        APP_SECRET = os.getenv("MOCK_KIS_APP_SECRET")
        ACCOUNT_NUMBER = os.getenv("MOCK_KIS_ACCOUNT_NUMBER")
        is_mock = True
    else:
        log_event("INFO", "=== 실전투자 환경으로 시작합니다. ===")
        APP_KEY = os.getenv("LIVE_KIS_APP_KEY")
        APP_SECRET = os.getenv("LIVE_KIS_APP_SECRET")
        ACCOUNT_NUMBER = os.getenv("LIVE_KIS_ACCOUNT_NUMBER")
        is_mock = False

    if not all([APP_KEY, APP_SECRET, ACCOUNT_NUMBER]):
        log_event("CRITICAL", f"{'모의' if is_mock else '실전'}투자용 API 키 또는 계좌번호가 .env 파일에 설정되지 않았습니다. 프로그램을 종료합니다.")
        return

    # --- 초기화 ---
    kis_api = KIS_API(app_key=APP_KEY, app_secret=APP_SECRET, account_number=ACCOUNT_NUMBER, mock=is_mock)
    telegram_bot = TelegramBot()
    portfolio_manager = PortfolioManager(capital=config.TOTAL_CAPITAL, kis_api=kis_api, telegram_bot=telegram_bot)
    trader = Trader(portfolio_manager=portfolio_manager, kis_api=kis_api, telegram_bot=telegram_bot)

    telegram_bot.send_message(f"✅ 자동매매 V5 '오디션 전략' 시작\n- 총 자본: {config.TOTAL_CAPITAL:,.0f}원")
    if public_ip:
        telegram_bot.send_message(ip_check_message)
    
    # --- 구글 시트 연동 상태 확인 및 알림 ---
    if gspread_client.client:
        telegram_bot.send_message("✅ 구글 시트 연동 성공. 거래 내역이 스프레드시트에 기록됩니다.")
    else:
        telegram_bot.send_message("⚠️ [경고] 구글 시트 연동 실패. `credentials.json` 파일을 확인하세요. 거래 내역 기록이 비활성화됩니다.")
    
    while True:
        try:
            # --- 시간 변수 정의 ---
            now = datetime.now()
            market_open_time = now.replace(hour=9, minute=0, second=0, microsecond=0)
            market_close_time = now.replace(hour=15, minute=30, second=0, microsecond=0)

            # --- 로직 분기 ---
            # 1. 장중: 매매 실행
            if market_open_time <= now <= market_close_time:
                summary = portfolio_manager.get_portfolio_summary()
                log_event("INFO", f"--- [상태] {summary} ---")
                
                await trader.run_trading_cycle() # 비동기 거래 사이클 실행

                sleep_duration = config.SYSTEM_CHECK_INTERVAL_MINUTES * 60
                log_event("INFO", f"사이클 종료. {config.SYSTEM_CHECK_INTERVAL_MINUTES}분 후 다음 사이클을 시작합니다.")
                await asyncio.sleep(sleep_duration)

            # 2. 장 마감 후: 대기
            else:
                log_event("INFO", f"장 운영 시간이 아닙니다. 개장 시간(09:00)까지 대기합니다. (현재: {now.strftime('%H:%M')})")
                await asyncio.sleep(300) # 5분 대기

        except KeyboardInterrupt:
            telegram_bot.send_message("⏹️ 자동매매 봇을 수동으로 종료합니다...")
            break
        except Exception as e:
            error_message = f"🔥 [메인 루프 오류] {e}"
            log_event("CRITICAL", error_message)
            telegram_bot.send_message(error_message)
            await asyncio.sleep(60)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log_event("INFO", "프로그램 실행이 중단되었습니다.")

# ... 실제 구현은 각 모듈 개발 후 추가 ... 