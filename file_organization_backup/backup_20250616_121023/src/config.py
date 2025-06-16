"""
⚙️ 통합 설정 관리 (절대 import)
모든 설정을 한 곳에서 관리
"""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

@dataclass
class TradingConfig:
    """매매 설정"""
    # 기본 설정
    is_mock: bool = os.getenv('IS_MOCK', 'true').lower() == 'true'
    max_stocks: int = int(os.getenv('MAX_STOCKS', '4'))
    initial_investment: int = int(os.getenv('INITIAL_INVESTMENT', '1000000'))
    
    # API 설정
    mock_app_key: str = os.getenv('MOCK_KIS_APP_KEY', '')
    mock_app_secret: str = os.getenv('MOCK_KIS_APP_SECRET', '')
    mock_account: str = os.getenv('MOCK_KIS_ACCOUNT_NUMBER', '')
    
    live_app_key: str = os.getenv('LIVE_KIS_APP_KEY', '')
    live_app_secret: str = os.getenv('LIVE_KIS_APP_SECRET', '')
    live_account: str = os.getenv('LIVE_KIS_ACCOUNT_NUMBER', '')
    
    # AI 설정
    gemini_api_key: str = os.getenv('GEMINI_API_KEY', '')
    
    # 알림 설정
    telegram_token: str = os.getenv('TELEGRAM_BOT_TOKEN', '')
    telegram_chat_id: str = os.getenv('TELEGRAM_CHAT_ID', '')
    
    # 전략 설정
    fibonacci_sequence: list = None
    scout_candidates: int = 5
    final_selections: int = 2
    
    def __post_init__(self):
        if self.fibonacci_sequence is None:
            self.fibonacci_sequence = [1, 1, 2, 3, 5, 8, 13, 21]
    
    @property
    def current_api_config(self) -> dict:
        """현재 모드에 따른 API 설정 반환"""
        if self.is_mock:
            return {
                'app_key': self.mock_app_key,
                'app_secret': self.mock_app_secret,
                'account_number': self.mock_account
            }
        else:
            return {
                'app_key': self.live_app_key,
                'app_secret': self.live_app_secret,
                'account_number': self.live_account
            }

# 전역 설정 인스턴스
config = TradingConfig() 