"""
텔레그램 봇 알림 기능
"""
import requests
import logging

logger = logging.getLogger(__name__)

class TelegramNotifier:
    """텔레그램 봇을 통한 알림 전송 클래스"""
    
    def __init__(self, bot_token, chat_id):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        
    def send_message(self, message):
        """텔레그램으로 메시지 전송"""
        if not self.bot_token or not self.chat_id:
            logger.warning("텔레그램 설정이 없어 메시지를 전송하지 않습니다.")
            return False
            
        try:
            url = f"{self.base_url}/sendMessage"
            data = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "HTML"
            }
            response = requests.post(url, data=data, timeout=10)
            response.raise_for_status()
            logger.info("✅ 텔레그램 메시지 전송 성공")
            return True
        except Exception as e:
            logger.error(f"❌ 텔레그램 메시지 전송 실패: {e}")
            return False 