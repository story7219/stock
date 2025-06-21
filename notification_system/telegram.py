"""
텔레그램 메시지 발송을 위한 비동기 봇 모듈
"""
import aiohttp
import logging

class TelegramBot:
    def __init__(self, bot_token: str, chat_id: str):
        if not bot_token or not chat_id:
            raise ValueError("텔레그램 봇 토큰과 채팅 ID가 필요합니다.")
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
        logging.info("텔레그램 봇이 초기화되었습니다.")

    async def send_message(self, message: str) -> bool:
        """
        지정된 채팅 ID로 비동기 메시지를 보냅니다.
        """
        url = f"{self.base_url}/sendMessage"
        payload = {
            'chat_id': self.chat_id,
            'text': message,
            'parse_mode': 'Markdown'  # Markdown 형식 지원
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        logging.info("텔레그램 메시지를 성공적으로 보냈습니다.")
                        return True
                    else:
                        response_text = await response.text()
                        logging.error(f"텔레그램 메시지 발송 실패: {response.status} - {response_text}")
                        return False
        except Exception as e:
            logging.error(f"텔레그램 메시지 발송 중 예외 발생: {e}")
            return False 