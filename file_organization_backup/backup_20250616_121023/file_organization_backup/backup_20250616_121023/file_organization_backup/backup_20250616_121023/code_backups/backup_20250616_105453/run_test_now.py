# run_test_now.py
# 자동매매 시스템의 전체 흐름을 시간 제약 없이 즉시 테스트하기 위한 일회성 스크립트

import config
import os
from dotenv import load_dotenv

from portfolio import PortfolioManager
from trading.kis_api import KIS_API
from trading.trader import Trader
from utils.logger import log_event
from utils.telegram_bot import TelegramBot
from reporting.reporter import generate_reports

def run_test():
    """테스트 실행 함수"""
    load_dotenv()
    
    log_event("INFO", "=== [수동 테스트] 자동매매 전체 사이클 테스트를 시작합니다. ===")

    # 모의투자 환경으로 강제 설정
    APP_KEY = os.getenv("MOCK_KIS_APP_KEY")
    APP_SECRET = os.getenv("MOCK_KIS_APP_SECRET")
    ACCOUNT_NUMBER = os.getenv("MOCK_KIS_ACCOUNT_NUMBER")
    
    if not all([APP_KEY, APP_SECRET, ACCOUNT_NUMBER]):
        log_event("CRITICAL", "모의투자용 API 키 또는 계좌번호가 .env 파일에 설정되지 않았습니다. 테스트를 종료합니다.")
        return

    # --- 초기화 ---
    kis_api = KIS_API(app_key=APP_KEY, app_secret=APP_SECRET, account_number=ACCOUNT_NUMBER, mock=True)
    telegram_bot = TelegramBot()
    portfolio_manager = PortfolioManager(capital=config.TOTAL_CAPITAL, kis_api=kis_api, telegram_bot=telegram_bot)
    trader = Trader(portfolio_manager=portfolio_manager, kis_api=kis_api, telegram_bot=telegram_bot)

    telegram_bot.send_message("⚙️ [수동 테스트] 자동매매 전체 사이클 테스트를 시작합니다.")
    
    try:
        # 1. 장 시작 전 준비 단계를 강제로 실행
        trader.prepare_for_market_open()
        
        # 2. 기존 보유 종목 관리 (현재는 보유 종목 없으므로 로그만 확인)
        trader.manage_existing_holdings()
        
        # 3. 신규 투자처 탐색 및 실행 (사전 준비된 후보군 사용)
        trader.find_and_execute_new_investments()

        summary = portfolio_manager.get_portfolio_summary()
        log_event("INFO", f"=== [수동 테스트] 모든 사이클이 완료되었습니다. 최종 포트폴리오 상태: {summary} ===")
        telegram_bot.send_message(f"✅ [수동 테스트] 완료. 최종 상태: {summary}")

    except Exception as e:
        error_message = f"🔥 [수동 테스트 오류] {e}"
        log_event("CRITICAL", error_message)
        telegram_bot.send_message(error_message)

if __name__ == "__main__":
    run_test() 