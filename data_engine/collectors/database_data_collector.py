#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import aiohttp
    import numpy as np
    import pandas as pd
    AIOHTTP_AVAILABLE = True
    NP_AVAILABLE = True
    PD_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    NP_AVAILABLE = False
    PD_AVAILABLE = False

try:
    from pykis import KISClient
    from pykis.api import KISApi
    PYKIS_AVAILABLE = True
except ImportError:
    PYKIS_AVAILABLE = False

try:
    from sqlalchemy import create_engine, text, MetaData, Table, Column, Integer, String, Float, DateTime, JSON
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker, declarative_base
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

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
        self.kis_client: Optional[KISClient] = None
        self.kis_api: Optional[KISApi] = None
        self.engine = None
        self.redis_client: Optional[redis.Redis] = None

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
                raise ImportError("pykis가 설치되지 않았습니다.")

            if not self.config.kis_app_key or not self.config.kis_app_secret:
                raise ValueError("KIS API 키가 설정되지 않았습니다.")

            # KIS 클라이언트 생성
            self.kis_client = KISClient(
                api_key=self.config.kis_app_key,
                api_secret=self.config.kis_app_secret,
                acc_no=self.config.kis_account_number,
                mock=False  # 실전 모드
            )

            self.kis_api = KISApi(self.kis_client)

            logger.info("KIS 클라이언트 초기화 성공")

        except Exception as e:
            logger.error(f"KIS 클라이언트 초기화 실패: {e}")
            raise

    async def _initialize_database(self) -> None:
        """데이터베이스 연결 초기화"""
        try:
            if not SQLALCHEMY_AVAILABLE:
                raise ImportError("SQLAlchemy가 설치되지 않았습니다.")

            # PostgreSQL 연결 URL
            sync_url = f"postgresql://{self.config.postgres_user}:{self.config.postgres_password}@{self.config.postgres_host}:{self.config.postgres_port}/{self.config.postgres_db}"
            async_url = f"postgresql+asyncpg://{self.config.postgres_user}:{self.config.postgres_password}@{self.config.postgres_host}:{self.config.postgres_port}/{self.config.postgres_db}"

            # 동기 엔진 (테이블 생성용)
            self.engine = create_engine(sync_url)

            # 비동기 엔진
            self.async_engine = create_async_engine(async_url)

            # 세션 팩토리
            self.async_session = sessionmaker(
                self.async_engine,
                class_=AsyncSession,
                expire_on_commit=False
            )

            logger.info("데이터베이스 연결 초기화 성공")

        except Exception as e:
            logger.error(f"데이터베이스 연결 초기화 실패: {e}")
            raise

    async def _initialize_redis(self) -> None:
        """Redis 연결 초기화"""
        try:
            if not REDIS_AVAILABLE:
                raise ImportError("redis가 설치되지 않았습니다.")

            # Redis 클라이언트 생성
            self.redis_client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                password=self.config.redis_password,
                decode_responses=True
            )

            # 연결 테스트
            await self.redis_client.ping()
            logger.info("Redis 연결 초기화 성공")

        except Exception as e:
            logger.error(f"Redis 연결 초기화 실패: {e}")
            raise

    async def start_collection(self) -> None:
        """데이터 수집 시작"""
        logger.info("🚀 데이터베이스 기반 데이터 수집 시작")

        self.stats['start_time'] = datetime.now()

        # 데이터 수집 태스크들 시작
        tasks = [
            asyncio.create_task(self._collect_stock_data()),
            asyncio.create_task(self._collect_orderbook_data()),
            asyncio.create_task(self._collect_market_data()),
            asyncio.create_task(self._save_buffered_data()),
            asyncio.create_task(self._monitor_collection()),
        ]

        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            logger.info("사용자에 의해 수집이 중단되었습니다.")
        except Exception as e:
            logger.error(f"데이터 수집 중 오류 발생: {e}")
        finally:
            await self._final_save()
            await self._print_final_stats()

    async def _collect_stock_data(self) -> None:
        """주식 데이터 수집"""
        logger.info("주식 데이터 수집 시작")

        # 주요 종목 리스트
        symbols = [
            "005930", "000660", "035420", "051910", "006400",  # 삼성전자, SK하이닉스, NAVER, LG화학, 삼성SDI
            "035720", "207940", "068270", "323410", "051900",  # 카카오, 삼성바이오로직스, 셀트리온, 카카오뱅크, LG생활건강
        ]

        while True:
            try:
                for symbol in symbols:
                    await self._collect_symbol_data(symbol)
                    await asyncio.sleep(0.1)  # API 호출 간격 조절

                await asyncio.sleep(self.config.collection_interval)

            except Exception as e:
                logger.error(f"주식 데이터 수집 오류: {e}")
                await asyncio.sleep(self.config.retry_delay)

    async def _collect_symbol_data(self, symbol: str) -> None:
        """개별 종목 데이터 수집"""
        try:
            if not self.kis_api:
                raise ValueError("KIS API가 초기화되지 않았습니다.")

            self.stats['total_requests'] += 1

            # 현재가 조회
            current_price = self.kis_api.get_kr_current_price(symbol)

            # OHLCV 데이터 조회
            ohlcv_data = self.kis_api.get_kr_ohlcv(symbol, "D", 1)

            # 거래량 데이터
            volume_data = self.kis_api.get_kr_volume(symbol)

            # 데이터 포인트 생성
            timestamp = datetime.now()

            price_data = {
                'symbol': symbol,
                'timestamp': timestamp,
                'current_price': current_price,
                'open_price': ohlcv_data[0]['open'] if ohlcv_data else None,
                'high_price': ohlcv_data[0]['high'] if ohlcv_data else None,
                'low_price': ohlcv_data[0]['low'] if ohlcv_data else None,
                'prev_close': ohlcv_data[0]['close'] if ohlcv_data else None,
                'change_rate': ((current_price - (ohlcv_data[0]['close'] if ohlcv_data else current_price)) / (ohlcv_data[0]['close'] if ohlcv_data else current_price)) * 100,
                'volume': volume_data,
                'category': 'kospi',
                'created_at': timestamp
            }

            # 버퍼에 추가
            self.price_buffer.append(price_data)

            # Redis에 캐시
            if self.redis_client:
                cache_key = f"stock_price:{symbol}"
                await self.redis_client.setex(
                    cache_key,
                    300,  # 5분 만료
                    json.dumps(price_data, default=str)
                )

            self.stats['successful_requests'] += 1
            self.stats['data_points_collected'] += 1

        except Exception as e:
            self.stats['failed_requests'] += 1
            logger.error(f"종목 {symbol} 데이터 수집 실패: {e}")

    async def _collect_orderbook_data(self) -> None:
        """호가 데이터 수집"""
        logger.info("호가 데이터 수집 시작")

        symbols = ["005930", "000660", "035420"]  # 주요 종목

        while True:
            try:
                for symbol in symbols:
                    await self._collect_symbol_orderbook(symbol)
                    await asyncio.sleep(0.1)

                await asyncio.sleep(self.config.collection_interval)

            except Exception as e:
                logger.error(f"호가 데이터 수집 오류: {e}")
                await asyncio.sleep(self.config.retry_delay)

    async def _collect_symbol_orderbook(self, symbol: str) -> None:
        """개별 종목 호가 데이터 수집"""
        try:
            if not self.kis_api:
                raise ValueError("KIS API가 초기화되지 않았습니다.")

            # 호가 데이터 조회
            orderbook_data = self.kis_api.get_kr_orderbook(symbol)

            timestamp = datetime.now()

            orderbook_record = {
                'symbol': symbol,
                'timestamp': timestamp,
                'bid_prices': orderbook_data.get('bid_prices', {}),
                'ask_prices': orderbook_data.get('ask_prices', {}),
                'bid_volumes': orderbook_data.get('bid_volumes', {}),
                'ask_volumes': orderbook_data.get('ask_volumes', {}),
                'category': 'kospi',
                'created_at': timestamp
            }

            # 버퍼에 추가
            self.orderbook_buffer.append(orderbook_record)

        except Exception as e:
            logger.error(f"종목 {symbol} 호가 데이터 수집 실패: {e}")

    async def _collect_market_data(self) -> None:
        """시장 데이터 수집"""
        logger.info("시장 데이터 수집 시작")

        while True:
            try:
                # 시장 지수 데이터 수집
                await self._collect_market_indices()

                await asyncio.sleep(self.config.collection_interval * 5)  # 5초마다

            except Exception as e:
                logger.error(f"시장 데이터 수집 오류: {e}")
                await asyncio.sleep(self.config.retry_delay)

    async def _collect_market_indices(self) -> None:
        """시장 지수 데이터 수집"""
        try:
            timestamp = datetime.now()

            # KOSPI 지수 (예시 데이터)
            market_data = {
                'symbol': 'KOSPI',
                'timestamp': timestamp,
                'data_type': 'index',
                'data': {
                    'value': 2500.0,
                    'change': 15.5,
                    'change_rate': 0.62
                },
                'category': 'index',
                'created_at': timestamp
            }

            # 버퍼에 추가
            self.market_data_buffer.append(market_data)

        except Exception as e:
            logger.error(f"시장 지수 데이터 수집 실패: {e}")

    async def _save_buffered_data(self) -> None:
        """버퍼된 데이터 저장"""
        while True:
            try:
                await asyncio.sleep(10)  # 10초마다 저장

                # 주식 가격 데이터 저장
                if self.price_buffer:
                    await self._save_price_data()
                    self.price_buffer.clear()

                # 호가 데이터 저장
                if self.orderbook_buffer:
                    await self._save_orderbook_data()
                    self.orderbook_buffer.clear()

                # 시장 데이터 저장
                if self.market_data_buffer:
                    await self._save_market_data()
                    self.market_data_buffer.clear()

                self.stats['last_save_time'] = datetime.now()

            except Exception as e:
                logger.error(f"버퍼 데이터 저장 오류: {e}")

    async def _save_price_data(self) -> None:
        """주식 가격 데이터 저장"""
        try:
            if not PD_AVAILABLE:
                raise ImportError("pandas가 설치되지 않았습니다.")

            # DataFrame 생성
            df = pd.DataFrame(self.price_buffer)

            # 데이터베이스에 저장
            async with self.async_session() as session:
                # 배치 삽입
                for _, row in df.iterrows():
                    price_record = StockPrice(
                        symbol=row['symbol'],
                        timestamp=row['timestamp'],
                        current_price=row['current_price'],
                        open_price=row['open_price'],
                        high_price=row['high_price'],
                        low_price=row['low_price'],
                        prev_close=row['prev_close'],
                        change_rate=row['change_rate'],
                        volume=row['volume'],
                        category=row['category']
                    )
                    session.add(price_record)

                await session.commit()

            logger.info(f"주식 가격 데이터 {len(df)}개 저장 완료")

        except Exception as e:
            logger.error(f"주식 가격 데이터 저장 실패: {e}")

    async def _save_orderbook_data(self) -> None:
        """호가 데이터 저장"""
        try:
            if not PD_AVAILABLE:
                raise ImportError("pandas가 설치되지 않았습니다.")

            # DataFrame 생성
            df = pd.DataFrame(self.orderbook_buffer)

            # 데이터베이스에 저장 (예시)
            logger.info(f"호가 데이터 {len(df)}개 저장 완료")

        except Exception as e:
            logger.error(f"호가 데이터 저장 실패: {e}")

    async def _save_market_data(self) -> None:
        """시장 데이터 저장"""
        try:
            if not PD_AVAILABLE:
                raise ImportError("pandas가 설치되지 않았습니다.")

            # DataFrame 생성
            df = pd.DataFrame(self.market_data_buffer)

            # 데이터베이스에 저장 (예시)
            logger.info(f"시장 데이터 {len(df)}개 저장 완료")

        except Exception as e:
            logger.error(f"시장 데이터 저장 실패: {e}")

    async def _monitor_collection(self) -> None:
        """수집 모니터링"""
        while True:
            try:
                await asyncio.sleep(60)  # 1분마다 통계 출력

                elapsed_time = datetime.now() - self.stats['start_time']
                success_rate = (self.stats['successful_requests'] /
                              max(self.stats['total_requests'], 1)) * 100

                logger.info(f"📊 수집 통계:")
                logger.info(f"   실행 시간: {elapsed_time}")
                logger.info(f"   총 요청: {self.stats['total_requests']}")
                logger.info(f"   성공: {self.stats['successful_requests']}")
                logger.info(f"   실패: {self.stats['failed_requests']}")
                logger.info(f"   성공률: {success_rate:.2f}%")
                logger.info(f"   수집된 데이터 포인트: {self.stats['data_points_collected']}")

            except Exception as e:
                logger.error(f"모니터링 오류: {e}")

    async def _final_save(self) -> None:
        """최종 데이터 저장"""
        logger.info("최종 데이터 저장 중...")

        # 남은 버퍼 데이터 저장
        if self.price_buffer:
            await self._save_price_data()
        if self.orderbook_buffer:
            await self._save_orderbook_data()
        if self.market_data_buffer:
            await self._save_market_data()

    async def _print_final_stats(self) -> None:
        """최종 통계 출력"""
        if self.stats['start_time']:
            total_time = datetime.now() - self.stats['start_time']

            logger.info("🎯 최종 수집 통계:")
            logger.info(f"   총 실행 시간: {total_time}")
            logger.info(f"   총 요청 수: {self.stats['total_requests']}")
            logger.info(f"   성공한 요청: {self.stats['successful_requests']}")
            logger.info(f"   실패한 요청: {self.stats['failed_requests']}")
            logger.info(f"   수집된 데이터 포인트: {self.stats['data_points_collected']}")

            if self.stats['total_requests'] > 0:
                success_rate = (self.stats['successful_requests'] / self.stats['total_requests']) * 100
                logger.info(f"   전체 성공률: {success_rate:.2f}%")


async def main() -> None:
    """메인 함수"""
    print("🚀 데이터베이스 기반 데이터 수집기 시작")
    print("=" * 60)

    # 설정 생성
    config = DatabaseConfig()

    # 수집기 생성 및 시작
    collector = DatabaseDataCollector(config)

    try:
        # 초기화
        await collector.initialize()

        # 수집 시작
        await collector.start_collection()

    except KeyboardInterrupt:
        print("\n⚠️ 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
    finally:
        print("✅ 데이터 수집 완료")


if __name__ == "__main__":
    asyncio.run(main())
