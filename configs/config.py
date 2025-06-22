#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 주식 분석 시스템 고급 설정 파일 (Gemini AI 최적화)
"""

import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 🎯 모드 확인 (먼저 설정)
IS_MOCK = os.getenv('IS_MOCK', 'True').lower() == 'true'

# 🏦 KIS API 설정 (모드에 따른 동적 선택)
if IS_MOCK:
    # 모의투자 모드
    KIS_APP_KEY = os.getenv('MOCK_KIS_APP_KEY', '')
    KIS_APP_SECRET = os.getenv('MOCK_KIS_APP_SECRET', '')
    KIS_ACCOUNT_NUMBER = os.getenv('MOCK_KIS_ACCOUNT_NUMBER', '')
    print("🧪 모의투자 모드로 KIS API 설정됨")
else:
    # 실전투자 모드
    KIS_APP_KEY = os.getenv('LIVE_KIS_APP_KEY', '')
    KIS_APP_SECRET = os.getenv('LIVE_KIS_APP_SECRET', '')
    KIS_ACCOUNT_NUMBER = os.getenv('LIVE_KIS_ACCOUNT_NUMBER', '')
    print("🚀 실전투자 모드로 KIS API 설정됨")

# KIS API 기본 설정
KIS_BASE_URL = os.getenv('KIS_BASE_URL', 'https://openapi.koreainvestment.com:9443')

# API 제한 설정
ORDER_API_CALLS_PER_SEC = int(os.getenv('ORDER_API_CALLS_PER_SEC', '20'))
TOTAL_API_CALLS_PER_SEC = int(os.getenv('TOTAL_API_CALLS_PER_SEC', '100'))
MARKET_DATA_API_CALLS_PER_SEC = int(os.getenv('MARKET_DATA_API_CALLS_PER_SEC', '50'))
DAILY_API_LIMIT = int(os.getenv('DAILY_API_LIMIT', '10000'))

# 🧠 Gemini AI 고급 설정 (최적화)
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
GEMINI_MODEL_VERSION = os.getenv('GEMINI_MODEL_VERSION', 'gemini-1.5-pro')  # pro 모델 사용
GEMINI_TEMPERATURE = float(os.getenv('GEMINI_TEMPERATURE', '0.3'))  # 더 정확한 분석을 위해 낮은 온도
GEMINI_TOP_P = float(os.getenv('GEMINI_TOP_P', '0.8'))
GEMINI_TOP_K = int(os.getenv('GEMINI_TOP_K', '40'))
GEMINI_MAX_OUTPUT_TOKENS = int(os.getenv('GEMINI_MAX_OUTPUT_TOKENS', '4096'))  # 더 긴 응답 허용
GEMINI_CANDIDATE_COUNT = int(os.getenv('GEMINI_CANDIDATE_COUNT', '1'))
GEMINI_STOP_SEQUENCES = os.getenv('GEMINI_STOP_SEQUENCES', '').split(',') if os.getenv('GEMINI_STOP_SEQUENCES') else []

# 🚀 Gemini AI 성능 최적화 설정
GEMINI_BATCH_SIZE = int(os.getenv('GEMINI_BATCH_SIZE', '10'))  # 배치 크기 최적화
GEMINI_MAX_CONCURRENT = int(os.getenv('GEMINI_MAX_CONCURRENT', '15'))  # 동시 요청 수 최적화
GEMINI_REQUEST_DELAY = float(os.getenv('GEMINI_REQUEST_DELAY', '0.1'))  # 요청 간격
GEMINI_RETRY_ATTEMPTS = int(os.getenv('GEMINI_RETRY_ATTEMPTS', '5'))  # 재시도 횟수 증가
GEMINI_TIMEOUT = int(os.getenv('GEMINI_TIMEOUT', '60'))  # 타임아웃 설정

# 🧠 AI 분석 품질 향상 설정
AI_ANALYSIS_DEPTH = os.getenv('AI_ANALYSIS_DEPTH', 'EXPERT')  # BASIC, ADVANCED, EXPERT
AI_CONTEXT_WINDOW = int(os.getenv('AI_CONTEXT_WINDOW', '32000'))  # 컨텍스트 윈도우 크기
AI_MULTI_STRATEGY_ANALYSIS = os.getenv('AI_MULTI_STRATEGY_ANALYSIS', 'True').lower() == 'true'
AI_CROSS_VALIDATION = os.getenv('AI_CROSS_VALIDATION', 'True').lower() == 'true'

# DART API 설정
DART_API_KEY = os.getenv('DART_API_KEY', '')

# 텔레그램 봇 설정
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')

# 데이터 경로 설정
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
LOGS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')

# 시스템 설정
DEBUG_MODE = os.getenv('DEBUG_MODE', 'False').lower() == 'true'

# 📊 API 제한 및 성능 설정 (최적화)
API_RATE_LIMIT = int(os.getenv('API_RATE_LIMIT', '200'))  # 분당 요청 수 증가
CACHE_TTL = int(os.getenv('CACHE_TTL', '1800'))  # 캐시 유효 시간 (30분)
CACHE_MAX_SIZE = int(os.getenv('CACHE_MAX_SIZE', '1000'))  # 캐시 최대 크기

# 🎯 분석 설정 (고도화)
TOP_STOCKS_COUNT = int(os.getenv('TOP_STOCKS_COUNT', '10'))  # TOP N 종목 증가
ANALYSIS_TIMEOUT = int(os.getenv('ANALYSIS_TIMEOUT', '300'))  # 분석 타임아웃 (초)
DETAILED_ANALYSIS = os.getenv('DETAILED_ANALYSIS', 'True').lower() == 'true'
INCLUDE_NEWS_ANALYSIS = os.getenv('INCLUDE_NEWS_ANALYSIS', 'True').lower() == 'true'
INCLUDE_TECHNICAL_ANALYSIS = os.getenv('INCLUDE_TECHNICAL_ANALYSIS', 'True').lower() == 'true'
INCLUDE_FUNDAMENTAL_ANALYSIS = os.getenv('INCLUDE_FUNDAMENTAL_ANALYSIS', 'True').lower() == 'true'

# 🔧 시스템 최적화 설정
ENABLE_ASYNC_PROCESSING = os.getenv('ENABLE_ASYNC_PROCESSING', 'True').lower() == 'true'
THREAD_POOL_SIZE = int(os.getenv('THREAD_POOL_SIZE', '16'))  # 스레드 풀 크기
MEMORY_OPTIMIZATION = os.getenv('MEMORY_OPTIMIZATION', 'True').lower() == 'true'

# 📈 백테스팅 및 검증 설정
ENABLE_BACKTESTING = os.getenv('ENABLE_BACKTESTING', 'True').lower() == 'true'
BACKTEST_PERIOD_DAYS = int(os.getenv('BACKTEST_PERIOD_DAYS', '365'))
CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', '0.7'))

print("🚀 고급 설정 파일 로드 완료 - Gemini AI 최적화 모드 활성화") 