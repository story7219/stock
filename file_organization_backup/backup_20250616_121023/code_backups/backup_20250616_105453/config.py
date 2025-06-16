"""
⚙️ 통합 설정 관리 (수정됨)
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

# 전역 설정 인스턴스 (이 부분이 누락되었음!)
config = TradingConfig()

# --- 기본 환경 설정 ---
IS_MOCK_TRADING = True  # True: 모의투자, False: 실전투자
SYSTEM_CHECK_INTERVAL_MINUTES = 5  # 메인 루프의 사이클 간 대기 시간 (분)

# --- 투자 전략: '실시간 오디션' & '듀얼 스탑' ---
MAX_STOCKS_TO_HOLD = 2           # 최종 본대 편입 종목 수
AUDITION_CANDIDATE_COUNT = 4     # 오디션 후보 종목 수 (4종목)
AUDITION_SCOUT_BUY_AMOUNT = 1    # 척후병(정찰병) 매수 수량 (1주)
AUDITION_WINNER_MIN_PROFIT_PCT = 3.0 # 오디션 통과 최소 수익률 (%)
EXIT_STRATEGY_STOP_LOSS_PCT = -5.0   # 가격 기반 손절매 비율 (%)
EXIT_STRATEGY_TIME_LIMIT_MINUTES = 60 # 시간 기반 손절매 시간 (분)

# --- AI 분석 설정 ---
USE_GEMINI_ANALYSIS = True            # 장 초반 Gemini 분석 사용 여부
GEMINI_ANALYSIS_START_TIME = "09:05"  # AI 분석 시작 시간
GEMINI_ANALYSIS_END_TIME = "10:00"    # AI 분석 종료 시간
GEMINI_MODEL_NAME = "gemini-1.5-flash" # AI 분석에 사용할 Gemini 모델

# --- 포트폴리오 설정 ---
TOTAL_CAPITAL = 500000000        # 총 자본 5억
MIN_CASH_RATIO = 0.25            # 현금 25% 반드시 유지

# --- 포트폴리오 기본 설정 ---
MIN_STOCK_PRICE = 5000  # 최소 주가
MAX_STOCK_PRICE = 100000  # 최대 주가
MIN_VOLUME = 1000000  # 최소 거래량

# API 설정
API_RATE_LIMIT = 10  # 초당 API 호출 제한
REQUEST_TIMEOUT = 30  # API 요청 타임아웃 (초)

# 포트폴리오 관리
MAX_POSITION_SIZE = 1000000  # 최대 포지션 크기 (100만원)
MAX_POSITIONS = 3  # 최대 동시 보유 종목 수
TARGET_PROFIT_RATE = 1.5  # 목표 수익률 (%)
STOP_LOSS_RATE = 0.5  # 손절 기준 (%)

# --- 거래 환경 설정 ---
# IS_MOCK_TRADING = True  # 모의투자로 전환하려면 이 줄의 주석을 해제하고 위 줄을 주석 처리하세요.

# --- 장기 투자 전략 설정 --- (장기 투자는 현재 논의에서 제외되었으므로 주석 처리 또는 확인)
# LONG_TERM_BUY_SPLIT_DAYS = 30 # 분할 매수 기간 (영업일 기준)
# LONG_TERM_SELL_PROFIT_TARGET = 0.20 # 수익실현 시작 목표 수익률 (예: 20%)
# LONG_TERM_SELL_SHARES_PER_DAY = 5   # 수익실현 시 일일 매도 수량 (주)

# --- 대상 종목 리스트 ---
# 동적으로 로드되므로 설정 파일에서는 제거합니다.
# SHORT_TERM_TARGET_STOCKS = fetch_kospi_tickers()

# --- 기타 설정 ---
# from data.fetcher import fetch_kospi_tickers # 순환 참조 오류를 유발하므로 제거합니다.

# Google Sheet
# (참고용으로 남겨둠)

# 거래 제한
MAX_DAILY_TRADES = 5  # 일일 최대 거래 횟수

# API 설정
# (참고용으로 남겨둠)

# 거래 제한
# (참고용으로 남겨둠)

LONG_TERM_STOCK = "005930"  # 삼성전자
LONG_TERM_BUY_AMOUNT = 0.25  # 25% 비중 