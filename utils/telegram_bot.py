# utils/telegram_bot.py
# 텔레그램 봇으로 메시지를 보내는 클래스 (최신 버전 v20+ 호환)

import os
from dotenv import load_dotenv
import asyncio
from telegram import Bot
from telegram.constants import ParseMode
from telegram.error import TelegramError
from utils.logger import log_event

class TelegramBot:
    """
    텔레그램 메시지 발송을 담당하는 클래스 (python-telegram-bot v20+).
    """
    def __init__(self):
        load_dotenv()
        self.token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        
        if not self.token or not self.chat_id:
            log_event("CRITICAL", "[텔레그램 설정 오류] TELEGRAM_BOT_TOKEN 또는 TELEGRAM_CHAT_ID가 .env 파일에 없습니다.")
            self.bot = None
        else:
            try:
                self.bot = Bot(token=self.token)
            except Exception as e:
                log_event("CRITICAL", f"[텔레그램 초기화 실패] {e}")
                self.bot = None
    
    async def _send_async(self, text: str):
        """비동기로 메시지를 발송하는 내부 메서드"""
        if not self.bot:
            return
        try:
            # 최신 버전에서는 parse_mode를 ParseMode.MARKDOWN 으로 명확히 지정
            await self.bot.send_message(chat_id=self.chat_id, text=text, parse_mode=ParseMode.MARKDOWN)
        except TelegramError as e:
            log_event("ERROR", f"[텔레그램 발송 실패] {e}")
        except Exception as e:
            # 예상치 못한 다른 오류 처리
            log_event("ERROR", f"[텔레그램 발송 중 알 수 없는 오류] {e}")

    def send_message(self, text: str):
        """동기 환경에서 메시지를 보낼 수 있는 메서드"""
        if not self.bot:
            log_event("WARNING", f"[텔레그램 메시지] 발송 안됨 (봇 초기화 실패). 내용: {text}")
            return
        
        try:
            # asyncio.run()은 루프를 닫아버려 다른 라이브러리와 충돌할 수 있음
            # 기존 루프를 얻거나 새로 만들어 실행하는 방식으로 변경
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 이미 루프가 실행 중이면 task를 생성하여 실행
                loop.create_task(self._send_async(text))
            else:
                # 루프가 실행 중이 아니면 루프를 통해 실행
                loop.run_until_complete(self._send_async(text))
        except RuntimeError as e:
            # 'Event loop is closed'와 같은 런타임 에러 처리
            if "Event loop is closed" in str(e):
                log_event("WARNING", "기존 이벤트 루프가 닫혀있어 새 루프를 생성하여 메시지를 발송합니다.")
                # 새 루프를 명시적으로 생성하여 실행
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                new_loop.run_until_complete(self._send_async(text))
            else:
                log_event("ERROR", f"[텔레그램 발송 런타임 오류] {e}")
        except Exception as e:
            log_event("ERROR", f"[텔레그램 래퍼 오류] {e}") 