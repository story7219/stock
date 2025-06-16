"""
모든 설정을 관리하는 중앙 파일
API 키, 거래 전략, 경로 등 모든 설정을 여기서 변경합니다.
"""
import os

# --- 💡 API 설정 ---
# 한국투자증권 API (실전/모의투자)
KIS_APP_KEY = "YOUR_APP_KEY"  # 여기에 실제 App Key를 입력하세요.
KIS_APP_SECRET = "YOUR_APP_SECRET" # 여기에 실제 App Secret을 입력하세요.
KIS_ACCOUNT_NO = "YOUR_ACCOUNT_NUMBER" # 계좌번호 전체를 입력하세요. (예: 12345678-01)

# 구글 Gemini API
# (환경 변수 'GOOGLE_API_KEY'에 설정하는 것을 권장)
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY', 'YOUR_GOOGLE_API_KEY')

# --- 💰 거래 전략 설정 ---
# 개인 투자자 현실형 전략
TOTAL_CAPITAL = 10_000_000  # 총 투자 자본금 (예: 1천만원)
MAX_STOCK_COUNT = 5        # 최대 보유 종목 수
PER_STOCK_INVESTMENT = TOTAL_CAPITAL // MAX_STOCK_COUNT # 종목당 투자 금액

# 리스크 관리
STOP_LOSS_PCT = -0.10      # 손절매 퍼센트 (예: -10%)
TAKE_PROFIT_PCT = 0.20     # 익절 퍼센트 (예: +20%)

# --- ⚙️ 시스템 설정 ---
# 모의투자 여부 (True: 모의투자, False: 실전투자)
IS_MOCK_TRADING = True

# --- 기본 환경 설정 ---
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