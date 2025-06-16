"""
📢 알림 시스템 - 텔레그램 알림
"""

import asyncio
import logging
import aiohttp
from typing import Optional
from datetime import datetime

class TelegramNotifier:
    """텔레그램 알림자"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.is_initialized = False
        self.session = None
        self.bot_token = None
        self.chat_id = None
    
    async def initialize(self):
        """알림자 초기화"""
        if self.is_initialized:
            return
            
        self.logger.info("📢 텔레그램 알림자 초기화")
        
        # 환경변수에서 설정 로드
        import os
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        if self.bot_token and self.chat_id:
            self.session = aiohttp.ClientSession()
            self.logger.info("✅ 텔레그램 설정 완료")
        else:
            self.logger.warning("⚠️ 텔레그램 설정이 없습니다")
        
        self.is_initialized = True
    
    async def cleanup(self):
        """정리"""
        if self.session:
            await self.session.close()
        self.logger.info("📢 텔레그램 알림자 정리 완료")
    
    async def send_trade_alert(self, message: str):
        """거래 알림 전송"""
        try:
            full_message = f"🤖 자동매매 알림\n\n{message}\n\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            await self._send_message(full_message)
            
        except Exception as e:
            self.logger.error(f"❌ 거래 알림 전송 실패: {e}")
    
    async def send_error_alert(self, error_message: str):
        """오류 알림 전송"""
        try:
            full_message = f"🚨 시스템 오류 알림\n\n{error_message}\n\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            await self._send_message(full_message)
            
        except Exception as e:
            self.logger.error(f"❌ 오류 알림 전송 실패: {e}")
    
    async def _send_message(self, message: str):
        """메시지 전송"""
        if not self.session or not self.bot_token or not self.chat_id:
            self.logger.info(f"📢 알림 (텔레그램 미설정): {message}")
            return
        
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            data = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            
            async with self.session.post(url, data=data) as response:
                if response.status == 200:
                    self.logger.info("📢 텔레그램 알림 전송 완료")
                else:
                    self.logger.warning(f"⚠️ 텔레그램 전송 실패: {response.status}")
                    
        except Exception as e:
            self.logger.error(f"❌ 텔레그램 전송 오류: {e}") 