# config.py
# 이 파일은 전체 자동매매 시스템의 주요 설정값을 관리합니다.
# 사용자는 이 파일의 변수들을 수정하여 투자 전략을 변경할 수 있습니다.

# from data.fetcher import fetch_kospi_tickers # 순환 참조 오류를 유발하므로 제거합니다.

import os

# --- 거래 환경 설정 ---
IS_MOCK_TRADING = False  # True: 모의투자 서버 사용, False: 실전투자 서버 사용
# IS_MOCK_TRADING = True  # 모의투자로 전환하려면 이 줄의 주석을 해제하고 위 줄을 주석 처리하세요.

# --- 포트폴리오 기본 설정 ---
TOTAL_CAPITAL = 500000000  # 총 자본금 5억 원으로 설정
MAX_STOCKS_TO_HOLD = 2     # "실시간 오디션" 후 최종적으로 보유할 최대 종목 수 (2종목으로 변경)

# --- 장기 투자 전략 설정 --- (장기 투자는 현재 논의에서 제외되었으므로 주석 처리 또는 확인)
# LONG_TERM_BUY_SPLIT_DAYS = 30 # 분할 매수 기간 (영업일 기준)
# LONG_TERM_SELL_PROFIT_TARGET = 0.20 # 수익실현 시작 목표 수익률 (예: 20%)
# LONG_TERM_SELL_SHARES_PER_DAY = 5   # 수익실현 시 일일 매도 수량 (주)

# --- 단기 투자 '실시간 오디션' 진입 전략 설정 ---
AUDITION_CANDIDATE_COUNT = 3      # 오디션에 참가할 후보 종목 수
AUDITION_SCOUT_BUY_AMOUNT = 1     # 오디션 진행을 위해 매수할 척후병 수량 (1주)
AUDITION_WINNER_MIN_PROFIT_PCT = 3.0  # 오디션 통과를 위한 최소 수익률 (%)

# --- 단기 투자 '듀얼 스탑' 청산 전략 설정 ---
EXIT_STRATEGY_STOP_LOSS_PCT = -5.0      # 가격 기반 손절매 비율 (%)
EXIT_STRATEGY_TIME_LIMIT_MINUTES = 60 # 시간 기반 손절매 시간 (분)

# --- 대상 종목 리스트 ---
# 동적으로 로드되므로 설정 파일에서는 제거합니다.
# SHORT_TERM_TARGET_STOCKS = fetch_kospi_tickers()

# --- API 키 ---
# API 키는 .env 파일을 통해 관리되므로 이 파일에서 직접 수정하지 않습니다.
# (참고용으로 남겨둠)

# --- 기타 설정 ---
SYSTEM_CHECK_INTERVAL_MINUTES = 1 # 전체 로직을 반복 실행할 주기 (분)
GEMINI_MODEL_NAME = "gemini-1.5-flash" # AI 분석에 사용할 Gemini 모델 이름

# --- Gemini AI 기반 분석 시간 설정 ---
USE_GEMINI_ANALYSIS = True  # 장 초반 Gemini 분석 사용 여부
GEMINI_ANALYSIS_START_TIME = "09:05"  # AI 분석 시작 시간 (오전 9시 5분)
GEMINI_ANALYSIS_END_TIME = "10:00"    # AI 분석 종료 시간 (오전 10시)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Google Sheet
SPREADSHEET_ID = os.getenv("SPREADSHEET_ID", "YOUR_SPREADSHEET_ID") 