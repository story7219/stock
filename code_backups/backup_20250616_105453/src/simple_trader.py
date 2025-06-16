"""
초보자용 간단한 자동매매 봇
기존 trader.py의 간단 버전
"""

import os
import sys
import logging
from datetime import datetime
from dotenv import load_dotenv

# 기존 모듈 import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trader import AdvancedTrader  # 기존 코드 활용

load_dotenv()

class SimpleTrader:
    """초보자용 래퍼 클래스"""
    
    def __init__(self):
        self.is_mock = os.getenv('IS_MOCK', 'true').lower() == 'true'
        
        # 기존 AdvancedTrader 활용
        try:
            env_prefix = "MOCK" if self.is_mock else "LIVE"
            app_key = os.getenv(f'{env_prefix}_KIS_APP_KEY')
            app_secret = os.getenv(f'{env_prefix}_KIS_APP_SECRET')
            account_no = os.getenv(f'{env_prefix}_KIS_ACCOUNT_NUMBER')
            gemini_api_key = os.getenv('GEMINI_API_KEY')
            
            if not all([app_key, app_secret, account_no, gemini_api_key]):
                raise ValueError("필수 환경변수가 설정되지 않았습니다.")
            
            self.trader = AdvancedTrader(
                app_key=app_key,
                app_secret=app_secret, 
                account_no=account_no,
                gemini_api_key=gemini_api_key,
                is_mock=self.is_mock
            )
            
            logging.info("🚀 기존 고급 트레이더 연결 성공!")
            
        except Exception as e:
            logging.error(f"❌ 트레이더 초기화 실패: {e}")
            self.trader = None
    
    async def run_simple_cycle(self):
        """간단한 실행 사이클"""
        if not self.trader:
            logging.error("❌ 트레이더가 초기화되지 않았습니다.")
            return
        
        try:
            logging.info("🔄 일일 매매 사이클 시작...")
            
            # 기존 고급 기능 활용
            await self.trader.rebalance_portfolio()
            
            # 간단한 리포트
            await self.send_simple_report()
            
            logging.info("✅ 일일 사이클 완료!")
            
        except Exception as e:
            logging.error(f"❌ 실행 중 오류: {e}")
            await self.send_error_alert(str(e))
    
    async def send_simple_report(self):
        """간단한 일일 리포트"""
        try:
            portfolio_info = self.trader.portfolio_manager
            
            message = f"""
📊 <b>일일 자동매매 리포트</b>

⏰ 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
🎯 모드: {'모의투자' if self.is_mock else '실전투자'}
💰 현금잔고: {portfolio_info.cash_balance:,}원
💎 총자산: {portfolio_info.total_assets:,}원
📊 보유종목: {len(portfolio_info.portfolio)}개

✅ 오늘의 자동매매가 완료되었습니다!
"""
            
            self.trader.telegram_notifier.send_message(message)
            
        except Exception as e:
            logging.error(f"❌ 리포트 전송 실패: {e}")
    
    async def send_error_alert(self, error_msg):
        """오류 알림"""
        try:
            alert_message = f"""
🚨 <b>자동매매 오류 발생</b>

⏰ 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
❌ 오류: {error_msg[:200]}...

🔧 관리자 확인이 필요합니다.
"""
            
            if self.trader and self.trader.telegram_notifier:
                self.trader.telegram_notifier.send_message(alert_message)
            
        except Exception as e:
            logging.error(f"❌ 오류 알림 전송 실패: {e}")

async def main():
    """메인 실행 함수"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('simple_trading.log'),
            logging.StreamHandler()
        ]
    )
    
    simple_trader = SimpleTrader()
    await simple_trader.run_simple_cycle()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 