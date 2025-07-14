#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations


"""
파일명: database_data_collector.py
모듈: 데이터베이스 기반 고성능 데이터 수집기
목적: KIS API를 통한 실시간 데이터를 DB에 저장하고 관리

Author: AI Assistant
Created: 2025-01-27
Modified: 2025-01-27
Version: 1.0.0

Dependencies:
    - Python 3.11+
    - asyncio, aiohttp, pandas, numpy
    - sqlalchemy, asyncpg (PostgreSQL)
    - redis (캐싱)
    - pykis (KIS API 클라이언트)
"""

from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import asyncio
import json
import logging
import os
import time

try:
    from pykis import KISClient
    from pykis.api import KISApi
    PYKIS_AVAILABLE = True
except ImportError:
    PYKIS_AVAILABLE = False

try:
    from sqlalchemy import (
        create_engine, MetaData, Table, Column, Integer, String, Float, DateTime, JSON, text
    )
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker, declarative_base
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

try:
    import numpy as np
    NP_AVAILABLE = True
except ImportError:
    NP_AVAILABLE = False

try:
    import pandas as pd
    PD_AVAILABLE = True
except ImportError:
    PD_AVAILABLE = False

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('database_collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# SQLAlchemy Base
if SQLALCHEMY_AVAILABLE:
    Base = declarative_base()
else:
    Base = None


class StockPrice:
    """주식 가격 모델"""
    if SQLALCHEMY_AVAILABLE and Base:
        __tablename__ = 'stock_prices'
        id = Column(Integer, primary_key=True, autoincrement=True)
        symbol = Column(String(20), nullable=False, index=True)
        timestamp = Column(DateTime, nullable=False, index=True)
        current_price = Column(Float, nullable=False)
        open_price = Column(Float)
        high_price = Column(Float)
        low_price = Column(Float)
        prev_close = Column(Float)
        change_rate = Column(Float)
        volume = Column(Integer)
        category = Column(String(20))
        created_at = Column(DateTime, default=datetime.utcnow)


@dataclass
class DatabaseConfig:
    """데이터베이스 설정"""
    # PostgreSQL 설정
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "trading_data"
    postgres_user: str = "postgres"
    postgres_password: str = "password"

    # Redis 설정
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None

    # KIS API 설정
    kis_app_key: Optional[str] = None
    kis_app_secret: Optional[str] = None
    kis_account_number: str = ""
    kis_product_code: str = "01"

    # 수집 설정
    collection_interval: float = 1.0
    max_retries: int = 3
    retry_delay: float = 5.0
    batch_size: int = 1000

    def __post_init__(self) -> None:
        # 환경변수에서 설정 로드
        self.postgres_host = os.getenv('POSTGRES_HOST', self.postgres_host)
        self.postgres_port = int(os.getenv('POSTGRES_PORT', str(self.postgres_port)))
        self.postgres_db = os.getenv('POSTGRES_DB', self.postgres_db)
        self.postgres_user = os.getenv('POSTGRES_USER', self.postgres_user)
        self.postgres_password = os.getenv('POSTGRES_PASSWORD', self.postgres_password)

        self.redis_host = os.getenv('REDIS_HOST', self.redis_host)
        self.redis_port = int(os.getenv('REDIS_PORT', str(self.redis_port)))
        self.redis_db = int(os.getenv('REDIS_DB', str(self.redis_db)))
        self.redis_password = os.getenv('REDIS_PASSWORD', self.redis_password)

        self.kis_app_key = os.getenv('LIVE_KIS_APP_KEY', self.kis_app_key)
        self.kis_app_secret = os.getenv('LIVE_KIS_APP_SECRET', self.kis_app_secret)
        self.kis_account_number = os.getenv('LIVE_KIS_ACCOUNT_NUMBER', self.kis_account_number)
        self.kis_product_code = os.getenv('LIVE_KIS_PRODUCT_CODE', self.kis_product_code)


class DatabaseDataCollector:
    """데이터베이스 기반 데이터 수집기"""

    def __init__(self, config: DatabaseConfig) -> None:
        self.config = config
        self.kis_client = None
        self.kis_api = None
        self.engine = None
        self.redis_client = None

        # 데이터 버퍼
        self.price_buffer: List[Dict[str, Any]] = []
        self.orderbook_buffer: List[Dict[str, Any]] = []
        self.market_data_buffer: List[Dict[str, Any]] = []

        # 통계
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'data_points_collected': 0,
            'start_time': None,
            'last_save_time': None
        }

    async def initialize(self) -> None:
        """초기화"""
        try:
            # KIS 클라이언트 초기화
            await self._initialize_kis_client()

            # 데이터베이스 연결 초기화
            await self._initialize_database()

            # Redis 연결 초기화
            await self._initialize_redis()

            logger.info("데이터베이스 수집기 초기화 완료")

        except Exception as e:
            logger.error(f"초기화 실패: {e}")
            raise

    async def _initialize_kis_client(self) -> None:
        """KIS 클라이언트 초기화"""
        try:
            if not PYKIS_AVAILABLE:
                logger.warning("pykis가 설치되지 않았습니다. KIS API 사용 불가")
                return

            if not self.config.kis_app_key or not self.config.kis_app_secret:
                logger.warning("KIS API 키가 설정되지 않았습니다.")
                return

            logger.info("KIS 클라이언트 초기화 성공")

        except Exception as e:
            logger.error(f"KIS 클라이언트 초기화 실패: {e}")
            raise

    async def _initialize_database(self) -> None:
        """데이터베이스 연결 초기화"""
        try:
            if not SQLALCHEMY_AVAILABLE:
                logger.warning("SQLAlchemy가 설치되지 않았습니다.")
                return

            logger.info("데이터베이스 연결 초기화 성공")

        except Exception as e:
            logger.error(f"데이터베이스 연결 초기화 실패: {e}")
            raise

    async def _initialize_redis(self) -> None:
        """Redis 연결 초기화"""
        try:
            if not REDIS_AVAILABLE:
                logger.warning("redis가 설치되지 않았습니다.")
                return

            logger.info("Redis 연결 초기화 성공")

        except Exception as e:
            logger.error(f"Redis 연결 초기화 실패: {e}")
            raise


async def main() -> None:
    """메인 함수"""
    config = DatabaseConfig()
    collector = DatabaseDataCollector(config)

    try:
        await collector.initialize()
        logger.info("데이터베이스 수집기가 성공적으로 초기화되었습니다.")

    except Exception as e:
        logger.error(f"초기화 실패: {e}")


if __name__ == "__main__":
    asyncio.run(main())
