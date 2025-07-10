#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: settings.py
모듈: AI 트레이딩 시스템 설정
목적: 시스템 전체 설정 중앙화 관리

Author: Trading AI System
Created: 2025-07-08
Version: 1.0.0

설정 항목:
- 데이터베이스 연결
- API 키 및 인증
- ML 모델 파라미터
- 트레이딩 파라미터
- 시스템 성능 설정
- 모니터링 설정

License: MIT
"""

from __future__ import annotations
import os
from typing import Dict, List, Optional, Any
from pydantic import BaseSettings, Field
from pydantic_settings import BaseSettings
from enum import Enum

class Environment(str, Enum):
    """환경 설정"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

class DatabaseType(str, Enum):
    """데이터베이스 타입"""
    POSTGRESQL = "postgresql"
    TIMESCALEDB = "timescaledb"
    MONGODB = "mongodb"

class ModelType(str, Enum):
    """모델 타입"""
    LSTM = "lstm"
    TRANSFORMER = "transformer"
    CNN = "cnn"
    RNN = "rnn"
    GRU = "gru"
    ENSEMBLE = "ensemble"
    REINFORCEMENT_LEARNING = "rl"

class TradingStrategy(str, Enum):
    """트레이딩 전략"""
    DAY_TRADING = "day_trading"
    SWING_TRADING = "swing_trading"
    MEDIUM_TERM = "medium_term"
    LONG_TERM = "long_term"
    SCALPING = "scalping"

class Settings(BaseSettings):
    """AI 트레이딩 시스템 설정"""
    
    # 기본 설정
    ENVIRONMENT: Environment = Field(default=Environment.DEVELOPMENT, env="ENVIRONMENT")
    DEBUG: bool = Field(default=True, env="DEBUG")
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    
    # 애플리케이션 설정
    APP_NAME: str = Field(default="AI Trading System", env="APP_NAME")
    APP_VERSION: str = Field(default="1.0.0", env="APP_VERSION")
    API_PREFIX: str = Field(default="/api/v1", env="API_PREFIX")
    
    # 서버 설정
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8000, env="PORT")
    WORKERS: int = Field(default=4, env="WORKERS")
    
    # 데이터베이스 설정
    DATABASE_TYPE: DatabaseType = Field(default=DatabaseType.POSTGRESQL, env="DATABASE_TYPE")
    DATABASE_URL: str = Field(default="postgresql://user:pass@localhost:5432/ai_trading", env="DATABASE_URL")
    TIMESCALE_URL: str = Field(default="postgresql://user:pass@localhost:5433/timeseries", env="TIMESCALE_URL")
    MONGODB_URL: str = Field(default="mongodb://localhost:27017/ai_trading", env="MONGODB_URL")
    
    # Redis 설정
    REDIS_URL: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    REDIS_PASSWORD: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    REDIS_DB: int = Field(default=0, env="REDIS_DB")
    
    # Kafka 설정
    KAFKA_BROKERS: List[str] = Field(default=["localhost:9092"], env="KAFKA_BROKERS")
    KAFKA_TOPICS: Dict[str, str] = Field(default={
        "market_data": "market_data",
        "signals": "trading_signals",
        "orders": "orders",
        "executions": "executions"
    })
    
    # API 키 설정
    KIS_APP_KEY: Optional[str] = Field(default=None, env="KIS_APP_KEY")
    KIS_APP_SECRET: Optional[str] = Field(default=None, env="KIS_APP_SECRET")
    KIS_REAL_APP_KEY: Optional[str] = Field(default=None, env="KIS_REAL_APP_KEY")
    KIS_REAL_APP_SECRET: Optional[str] = Field(default=None, env="KIS_REAL_APP_SECRET")
    KIS_ACCOUNT_CODE: Optional[str] = Field(default=None, env="KIS_ACCOUNT_CODE")
    
    DART_API_KEY: Optional[str] = Field(default=None, env="DART_API_KEY")
    ALPHA_VANTAGE_API_KEY: Optional[str] = Field(default=None, env="ALPHA_VANTAGE_API_KEY")
    FRED_API_KEY: Optional[str] = Field(default=None, env="FRED_API_KEY")
    NEWS_API_KEY: Optional[str] = Field(default=None, env="NEWS_API_KEY")
    TWITTER_API_KEY: Optional[str] = Field(default=None, env="TWITTER_API_KEY")
    TWITTER_API_SECRET: Optional[str] = Field(default=None, env="TWITTER_API_SECRET")
    
    # ML 모델 설정
    MODEL_TYPE: ModelType = Field(default=ModelType.LSTM, env="MODEL_TYPE")
    MODEL_DIR: str = Field(default="./models", env="MODEL_DIR")
    MODEL_VERSION: str = Field(default="latest", env="MODEL_VERSION")
    
    # 모델 하이퍼파라미터
    SEQUENCE_LENGTH: int = Field(default=60, env="SEQUENCE_LENGTH")
    BATCH_SIZE: int = Field(default=32, env="BATCH_SIZE")
    EPOCHS: int = Field(default=100, env="EPOCHS")
    LEARNING_RATE: float = Field(default=0.001, env="LEARNING_RATE")
    DROPOUT_RATE: float = Field(default=0.2, env="DROPOUT_RATE")
    
    # LSTM 설정
    LSTM_UNITS: List[int] = Field(default=[128, 64, 32], env="LSTM_UNITS")
    LSTM_LAYERS: int = Field(default=3, env="LSTM_LAYERS")
    
    # Transformer 설정
    TRANSFORMER_HEADS: int = Field(default=8, env="TRANSFORMER_HEADS")
    TRANSFORMER_LAYERS: int = Field(default=6, env="TRANSFORMER_LAYERS")
    TRANSFORMER_D_MODEL: int = Field(default=512, env="TRANSFORMER_D_MODEL")
    
    # CNN 설정
    CNN_FILTERS: List[int] = Field(default=[64, 128, 256], env="CNN_FILTERS")
    CNN_KERNEL_SIZE: int = Field(default=3, env="CNN_KERNEL_SIZE")
    
    # 트레이딩 설정
    TRADING_STRATEGY: TradingStrategy = Field(default=TradingStrategy.DAY_TRADING, env="TRADING_STRATEGY")
    TRADING_SYMBOLS: List[str] = Field(default=["005930", "000660", "035420"], env="TRADING_SYMBOLS")
    TRADING_HOURS: Dict[str, str] = Field(default={
        "market_open": "09:00",
        "market_close": "15:30",
        "pre_market": "08:30",
        "post_market": "16:00"
    })
    
    # 리스크 관리 설정
    MAX_POSITION_SIZE: float = Field(default=0.1, env="MAX_POSITION_SIZE")  # 포트폴리오의 10%
    STOP_LOSS_PERCENT: float = Field(default=0.02, env="STOP_LOSS_PERCENT")  # 2%
    TAKE_PROFIT_PERCENT: float = Field(default=0.04, env="TAKE_PROFIT_PERCENT")  # 4%
    MAX_DRAWDOWN: float = Field(default=0.15, env="MAX_DRAWDOWN")  # 15%
    
    # 포지션 사이징
    POSITION_SIZING_METHOD: str = Field(default="kelly", env="POSITION_SIZING_METHOD")
    KELLY_FRACTION: float = Field(default=0.25, env="KELLY_FRACTION")
    FIXED_POSITION_SIZE: float = Field(default=0.05, env="FIXED_POSITION_SIZE")
    
    # 기술적 분석 설정
    TECHNICAL_INDICATORS: List[str] = Field(default=[
        "sma", "ema", "rsi", "macd", "bollinger_bands", "stochastic",
        "atr", "adx", "cci", "williams_r", "mfi", "obv"
    ], env="TECHNICAL_INDICATORS")
    
    # 일목균형표 설정
    ICHIMOKU_SETTINGS: Dict[str, int] = Field(default={
        "tenkan_period": 9,
        "kijun_period": 26,
        "senkou_span_b_period": 52,
        "displacement": 26
    }, env="ICHIMOKU_SETTINGS")
    
    # 시간론 설정
    TIME_ANALYSIS_SETTINGS: Dict[str, Any] = Field(default={
        "fibonacci_levels": [0.236, 0.382, 0.5, 0.618, 0.786],
        "time_cycles": [5, 8, 13, 21, 34, 55, 89],
        "seasonal_patterns": True
    }, env="TIME_ANALYSIS_SETTINGS")
    
    # 대등수치 설정
    EQUAL_NUMBERS_SETTINGS: Dict[str, Any] = Field(default={
        "price_levels": [1000, 2000, 3000, 5000, 10000, 20000, 30000, 50000],
        "volume_levels": [1000000, 5000000, 10000000, 50000000],
        "support_resistance": True
    }, env="EQUAL_NUMBERS_SETTINGS")
    
    # 실시간 처리 설정
    REALTIME_UPDATE_INTERVAL: int = Field(default=1, env="REALTIME_UPDATE_INTERVAL")  # 초
    BATCH_UPDATE_INTERVAL: int = Field(default=60, env="BATCH_UPDATE_INTERVAL")  # 초
    MODEL_UPDATE_INTERVAL: int = Field(default=3600, env="MODEL_UPDATE_INTERVAL")  # 초
    
    # 성능 설정
    MAX_WORKERS: int = Field(default=8, env="MAX_WORKERS")
    CHUNK_SIZE: int = Field(default=1000, env="CHUNK_SIZE")
    CACHE_TTL: int = Field(default=300, env="CACHE_TTL")  # 초
    
    # GPU 설정
    GPU_ENABLED: bool = Field(default=False, env="GPU_ENABLED")
    GPU_MEMORY_FRACTION: float = Field(default=0.8, env="GPU_MEMORY_FRACTION")
    
    # 모니터링 설정
    PROMETHEUS_ENABLED: bool = Field(default=True, env="PROMETHEUS_ENABLED")
    PROMETHEUS_PORT: int = Field(default=9090, env="PROMETHEUS_PORT")
    
    GRAFANA_ENABLED: bool = Field(default=True, env="GRAFANA_ENABLED")
    GRAFANA_URL: str = Field(default="http://localhost:3000", env="GRAFANA_URL")
    
    # 로깅 설정
    LOG_FORMAT: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s", env="LOG_FORMAT")
    LOG_FILE: str = Field(default="./logs/trading.log", env="LOG_FILE")
    LOG_MAX_SIZE: int = Field(default=100, env="LOG_MAX_SIZE")  # MB
    LOG_BACKUP_COUNT: int = Field(default=5, env="LOG_BACKUP_COUNT")
    
    # 알림 설정
    ALERT_ENABLED: bool = Field(default=True, env="ALERT_ENABLED")
    TELEGRAM_BOT_TOKEN: Optional[str] = Field(default=None, env="TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID: Optional[str] = Field(default=None, env="TELEGRAM_CHAT_ID")
    
    SLACK_WEBHOOK_URL: Optional[str] = Field(default=None, env="SLACK_WEBHOOK_URL")
    EMAIL_SMTP_SERVER: Optional[str] = Field(default=None, env="EMAIL_SMTP_SERVER")
    EMAIL_SMTP_PORT: int = Field(default=587, env="EMAIL_SMTP_PORT")
    EMAIL_USERNAME: Optional[str] = Field(default=None, env="EMAIL_USERNAME")
    EMAIL_PASSWORD: Optional[str] = Field(default=None, env="EMAIL_PASSWORD")
    
    # 백테스팅 설정
    BACKTEST_START_DATE: str = Field(default="2020-01-01", env="BACKTEST_START_DATE")
    BACKTEST_END_DATE: str = Field(default="2024-01-01", env="BACKTEST_END_DATE")
    BACKTEST_INITIAL_CAPITAL: float = Field(default=100000000, env="BACKTEST_INITIAL_CAPITAL")  # 1억원
    BACKTEST_COMMISSION: float = Field(default=0.00015, env="BACKTEST_COMMISSION")  # 0.015%
    
    # 보안 설정
    SECRET_KEY: str = Field(default="your-secret-key-here", env="SECRET_KEY")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(default=7, env="REFRESH_TOKEN_EXPIRE_DAYS")
    
    # CORS 설정
    CORS_ORIGINS: List[str] = Field(default=["http://localhost:3000"], env="CORS_ORIGINS")
    CORS_ALLOW_CREDENTIALS: bool = Field(default=True, env="CORS_ALLOW_CREDENTIALS")
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = Field(default=60, env="RATE_LIMIT_PER_MINUTE")
    RATE_LIMIT_PER_HOUR: int = Field(default=1000, env="RATE_LIMIT_PER_HOUR")
    
    # 파일 저장 설정
    DATA_DIR: str = Field(default="./data", env="DATA_DIR")
    BACKUP_DIR: str = Field(default="./backups", env="BACKUP_DIR")
    LOG_DIR: str = Field(default="./logs", env="LOG_DIR")
    
    # 백업 설정
    BACKUP_ENABLED: bool = Field(default=True, env="BACKUP_ENABLED")
    BACKUP_SCHEDULE: str = Field(default="0 2 * * *", env="BACKUP_SCHEDULE")  # 매일 새벽 2시
    BACKUP_RETENTION_DAYS: int = Field(default=30, env="BACKUP_RETENTION_DAYS")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

# 전역 설정 인스턴스
settings = Settings()

# 환경별 설정 오버라이드
if settings.ENVIRONMENT == Environment.PRODUCTION:
    settings.DEBUG = False
    settings.LOG_LEVEL = "WARNING"
    settings.GPU_ENABLED = True
    settings.ALERT_ENABLED = True
elif settings.ENVIRONMENT == Environment.STAGING:
    settings.DEBUG = False
    settings.LOG_LEVEL = "INFO"
    settings.GPU_ENABLED = True
    settings.ALERT_ENABLED = True
else:  # DEVELOPMENT
    settings.DEBUG = True
    settings.LOG_LEVEL = "DEBUG"
    settings.GPU_ENABLED = False
    settings.ALERT_ENABLED = False

# 설정 검증
def validate_settings() -> None:
    """설정 유효성 검증"""
    # 필수 API 키 검증
    if not settings.KIS_APP_KEY:
        raise ValueError("KIS_APP_KEY is required")
    
    if not settings.KIS_APP_SECRET:
        raise ValueError("KIS_APP_SECRET is required")
    
    # 데이터베이스 URL 검증
    if not settings.DATABASE_URL:
        raise ValueError("DATABASE_URL is required")
    
    # Redis URL 검증
    if not settings.REDIS_URL:
        raise ValueError("REDIS_URL is required")
    
    # 모델 디렉토리 생성
    os.makedirs(settings.MODEL_DIR, exist_ok=True)
    os.makedirs(settings.DATA_DIR, exist_ok=True)
    os.makedirs(settings.BACKUP_DIR, exist_ok=True)
    os.makedirs(settings.LOG_DIR, exist_ok=True)

# 설정 초기화
validate_settings() 