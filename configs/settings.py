#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
⚙️ Investing TOP5 시스템 설정
전체 시스템의 설정값을 관리하는 중앙 설정 파일
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

# 프로젝트 루트 경로
PROJECT_ROOT = Path(__file__).parent.parent

# 환경 설정
ENV = os.getenv('ENVIRONMENT', 'development')
DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'

# API 설정
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-1.5-pro')

# Yahoo Finance API
YAHOO_FINANCE_ENABLED = True
YAHOO_FINANCE_TIMEOUT = 30

# 데이터 설정
DATA_CACHE_TTL = 3600  # 1시간 (초)

# Gemini AI 캐시 설정
GEMINI_CACHE_TTL = int(os.getenv('GEMINI_CACHE_TTL', '7200'))  # .env에서 읽기, 기본값 2시간 (초) - 토큰 발급 시간

MAX_STOCKS_PER_ANALYSIS = 1000
BATCH_SIZE = 20

# AI 분석 설정
AI_ANALYSIS_ENABLED = True
AI_MAX_RETRIES = 3
AI_TIMEOUT = 60
AI_TEMPERATURE = 0.2
AI_MAX_TOKENS = 8192

# 투자 전략 가중치
STRATEGY_WEIGHTS = {
    'buffett': 0.35,    # 워렌 버핏 전략
    'lynch': 0.35,      # 피터 린치 전략
    'greenblatt': 0.30  # 그린블라트 전략
}

# 점수 계산 가중치
SCORING_WEIGHTS = {
    'financial_health': 0.25,  # 재무건전성
    'profitability': 0.25,     # 수익성
    'growth': 0.20,            # 성장성
    'valuation': 0.20,         # 밸류에이션
    'momentum': 0.10           # 모멘텀
}

# 시장별 설정
MARKET_SETTINGS = {
    'KR': {
        'name': '한국',
        'currency': 'KRW',
        'major_indices': ['KOSPI', 'KOSDAQ'],
        'trading_hours': {
            'start': '09:00',
            'end': '15:30',
            'timezone': 'Asia/Seoul'
        },
        'top_stocks': [
            '005930',  # 삼성전자
            '000660',  # SK하이닉스
            '035420',  # NAVER
            '051910',  # LG화학
            '068270',  # 셀트리온
            '207940',  # 삼성바이오로직스
            '005380',  # 현대차
            '006400',  # 삼성SDI
            '035720',  # 카카오
            '028260'   # 삼성물산
        ]
    },
    'US': {
        'name': '미국',
        'currency': 'USD',
        'major_indices': ['S&P500', 'NASDAQ', 'DOW'],
        'trading_hours': {
            'start': '09:30',
            'end': '16:00',
            'timezone': 'America/New_York'
        },
        'top_stocks': [
            'AAPL',   # Apple
            'MSFT',   # Microsoft
            'GOOGL',  # Alphabet
            'AMZN',   # Amazon
            'NVDA',   # NVIDIA
            'TSLA',   # Tesla
            'META',   # Meta
            'BRK-B',  # Berkshire Hathaway
            'V',      # Visa
            'JNJ'     # Johnson & Johnson
        ]
    }
}

# 로깅 설정
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
        'detailed': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'standard',
            'stream': 'ext://sys.stdout'
        },
        'file': {
            'class': 'logging.FileHandler',
            'level': 'DEBUG' if DEBUG else 'INFO',
            'formatter': 'detailed',
            'filename': str(PROJECT_ROOT / 'logs' / 'investing_top5.log'),
            'encoding': 'utf-8'
        }
    },
    'loggers': {
        '': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG' if DEBUG else 'INFO',
            'propagate': False
        }
    }
}

# 디렉토리 설정
DIRECTORIES = {
    'data': PROJECT_ROOT / 'data',
    'data_raw': PROJECT_ROOT / 'data' / 'raw',
    'data_processed': PROJECT_ROOT / 'data' / 'processed',
    'data_external': PROJECT_ROOT / 'data' / 'external',
    'logs': PROJECT_ROOT / 'logs',
    'cache': PROJECT_ROOT / 'cache',
    'notebooks': PROJECT_ROOT / 'notebooks',
    'configs': PROJECT_ROOT / 'configs'
}

# 성능 설정
PERFORMANCE_SETTINGS = {
    'max_concurrent_requests': 25,
    'request_delay': 0.03,  # 초
    'cache_enabled': True,
    'cache_size': 1000,
    'memory_limit_mb': 2048
}

# 알림 설정
NOTIFICATION_SETTINGS = {
    'telegram_enabled': False,
    'telegram_bot_token': os.getenv('TELEGRAM_BOT_TOKEN', ''),
    'telegram_chat_id': os.getenv('TELEGRAM_CHAT_ID', ''),
    'email_enabled': False,
    'email_smtp_server': os.getenv('EMAIL_SMTP_SERVER', ''),
    'email_smtp_port': int(os.getenv('EMAIL_SMTP_PORT', '587')),
    'email_username': os.getenv('EMAIL_USERNAME', ''),
    'email_password': os.getenv('EMAIL_PASSWORD', ''),
    'email_recipients': os.getenv('EMAIL_RECIPIENTS', '').split(',')
}

# 투자 등급 기준
INVESTMENT_GRADES = {
    'A+': {'min_score': 90, 'description': '최우수 투자 종목', 'color': '#00C851'},
    'A':  {'min_score': 80, 'description': '우수 투자 종목', 'color': '#2BBBAD'},
    'B+': {'min_score': 70, 'description': '양호한 투자 종목', 'color': '#4285F4'},
    'B':  {'min_score': 60, 'description': '보통 투자 종목', 'color': '#FF6F00'},
    'C+': {'min_score': 50, 'description': '주의 필요 종목', 'color': '#FF8800'},
    'C':  {'min_score': 40, 'description': '위험 종목', 'color': '#FF4444'},
    'D':  {'min_score': 0,  'description': '투자 부적합 종목', 'color': '#CC0000'}
}

# 리스크 레벨 정의
RISK_LEVELS = {
    1: {'name': '매우 안전', 'description': '안정적인 대형주, 배당주 중심'},
    2: {'name': '안전', 'description': '우량 중대형주 중심'},
    3: {'name': '보통', 'description': '성장주와 가치주 균형'},
    4: {'name': '공격적', 'description': '고성장 중소형주 중심'},
    5: {'name': '매우 공격적', 'description': '고위험 고수익 종목 중심'}
}

# 섹터 분류
SECTOR_MAPPING = {
    'KR': {
        '기술': ['IT', '소프트웨어', '반도체', '전자'],
        '금융': ['은행', '증권', '보험', '카드'],
        '소비재': ['식품', '의류', '화장품', '유통'],
        '헬스케어': ['제약', '바이오', '의료기기'],
        '산업재': ['건설', '기계', '조선', '항공'],
        '소재': ['철강', '화학', '종이', '유리'],
        '에너지': ['석유', '가스', '전력', '신재생'],
        '통신': ['통신서비스', '미디어'],
        '부동산': ['건설', '부동산개발', 'REITs']
    },
    'US': {
        'Technology': ['Software', 'Hardware', 'Semiconductors'],
        'Healthcare': ['Pharmaceuticals', 'Biotechnology', 'Medical Devices'],
        'Financials': ['Banks', 'Insurance', 'Investment Services'],
        'Consumer Discretionary': ['Retail', 'Automotive', 'Media'],
        'Consumer Staples': ['Food & Beverages', 'Household Products'],
        'Industrials': ['Aerospace', 'Machinery', 'Transportation'],
        'Materials': ['Chemicals', 'Metals & Mining', 'Paper'],
        'Energy': ['Oil & Gas', 'Renewable Energy'],
        'Utilities': ['Electric', 'Gas', 'Water'],
        'Real Estate': ['REITs', 'Real Estate Services'],
        'Communication Services': ['Telecom', 'Media & Entertainment']
    }
}

# 통합 시스템 설정
SYSTEM_CONFIG = {
    'environment': ENV,
    'debug': DEBUG,
    'project_root': PROJECT_ROOT,
    'api': {
        'gemini_api_key': GEMINI_API_KEY,
        'gemini_model': GEMINI_MODEL,
        'yahoo_finance_enabled': YAHOO_FINANCE_ENABLED,
        'yahoo_finance_timeout': YAHOO_FINANCE_TIMEOUT
    },
    'data': {
        'cache_ttl': DATA_CACHE_TTL,
        'max_stocks_per_analysis': MAX_STOCKS_PER_ANALYSIS,
        'batch_size': BATCH_SIZE
    },
    'ai': {
        'analysis_enabled': AI_ANALYSIS_ENABLED,
        'max_retries': AI_MAX_RETRIES,
        'timeout': AI_TIMEOUT,
        'temperature': AI_TEMPERATURE,
        'max_tokens': AI_MAX_TOKENS
    },
    'strategies': {
        'weights': STRATEGY_WEIGHTS
    },
    'scoring': {
        'weights': SCORING_WEIGHTS
    },
    'markets': MARKET_SETTINGS,
    'directories': DIRECTORIES,
    'performance': PERFORMANCE_SETTINGS,
    'notifications': NOTIFICATION_SETTINGS,
    'investment_grades': INVESTMENT_GRADES,
    'risk_levels': RISK_LEVELS,
    'sectors': SECTOR_MAPPING
}

def get_setting(key: str, default=None):
    """설정값 조회"""
    return globals().get(key, default)

def update_setting(key: str, value):
    """설정값 업데이트"""
    globals()[key] = value

def validate_settings():
    """설정값 검증"""
    errors = []
    
    # API 키 검증
    if AI_ANALYSIS_ENABLED and not GEMINI_API_KEY:
        errors.append("GEMINI_API_KEY가 설정되지 않았습니다.")
    
    # 디렉토리 생성
    for name, path in DIRECTORIES.items():
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"디렉토리 생성 실패 ({name}): {e}")
    
    # 가중치 합계 검증
    if abs(sum(STRATEGY_WEIGHTS.values()) - 1.0) > 0.01:
        errors.append("STRATEGY_WEIGHTS의 합계가 1.0이 아닙니다.")
    
    if abs(sum(SCORING_WEIGHTS.values()) - 1.0) > 0.01:
        errors.append("SCORING_WEIGHTS의 합계가 1.0이 아닙니다.")
    
    return errors

def get_market_config(market: str) -> Dict:
    """시장별 설정 조회"""
    return MARKET_SETTINGS.get(market, {})

def get_investment_grade(score: float) -> Dict:
    """점수에 따른 투자 등급 조회"""
    for grade, config in INVESTMENT_GRADES.items():
        if score >= config['min_score']:
            return {'grade': grade, **config}
    return {'grade': 'D', **INVESTMENT_GRADES['D']}

# 초기화 시 설정 검증
if __name__ == "__main__":
    errors = validate_settings()
    if errors:
        print("❌ 설정 검증 오류:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("✅ 모든 설정이 정상입니다.") 