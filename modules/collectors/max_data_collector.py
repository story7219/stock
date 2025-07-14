#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
파일명: max_data_collector.py
모듈: MAX API 기반 데이터 수집기
목적: MAX API를 통한 실시간 데이터 수집

Author: AI Assistant
Created: 2025-01-27
Modified: 2025-01-27
Version: 1.0.0

Dependencies:
    - Python 3.11+
    - asyncio, aiohttp, pandas, numpy
    - sqlalchemy, asyncpg (PostgreSQL)
    - redis (캐싱)
"""

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import json
import logging
import os
import time
from pathlib import Path
import asyncio

try:
    from pykis import *
    PYKIS_AVAILABLE = True
except ImportError as e:
    PYKIS_AVAILABLE = False
    PYKIS_IMPORT_ERROR = e

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

try:
    from pykrx import stock
    PYKRX_AVAILABLE = True
except ImportError:
    PYKRX_AVAILABLE = False

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('max_collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# SQLAlchemy Base
if SQLALCHEMY_AVAILABLE:
    Base = declarative_base()
else:
    Base = None


@dataclass
class DataCollectionConfig:
    """데이터 수집 설정 (최대치 자동 확장)"""
    kospi_symbols: List[str] = field(default_factory=list)
    kosdaq_symbols: List[str] = field(default_factory=list)
    futures_symbols: List[str] = field(default_factory=list)
    etf_symbols: List[str] = field(default_factory=list)
    start_date: str = "20000101"  # 최대한 과거
    end_date: str = datetime.now().strftime("%Y%m%d")
    collection_interval: float = 1.0
    max_retries: int = 3
    retry_delay: float = 5.0
    data_save_interval: int = 60
    max_data_points: int = 1000000
    data_dir: str = "./collected_data"
    backup_dir: str = "./data_backup"

    def __post_init__(self):
        if PYKRX_AVAILABLE and PD_AVAILABLE:
            # 기준일: 오늘
            today = datetime.now().strftime("%Y%m%d")
            # KOSPI 시총 필터
            kospi_cap = stock.get_market_cap(today, market="KOSPI")
            kospi_filtered = kospi_cap[kospi_cap['시가총액'] >= 500_000_000_000]
            self.kospi_symbols = [str(code) for code in kospi_filtered.index.tolist()]
            # KOSDAQ 시총 필터
            kosdaq_cap = stock.get_market_cap(today, market="KOSDAQ")
            kosdaq_filtered = kosdaq_cap[kosdaq_cap['시가총액'] >= 500_000_000_000]
            self.kosdaq_symbols = [str(code) for code in kosdaq_filtered.index.tolist()]
            # ETF 전체
            self.etf_symbols = stock.get_etf_ticker_list()
            self.futures_symbols = []
        else:
            # pykrx/pandas 미설치 시 기본값(20개)
            self.kospi_symbols = [
                "005930", "000660", "035420", "051910", "006400", "035720", "207940", "068270", "323410", "373220",
                "005380", "000270", "015760", "017670", "032830", "086790", "105560", "055550", "138930", "316140"
            ]
            self.kosdaq_symbols = [
                "091990", "122870", "086520", "096770", "018260", "091810", "036570", "079370", "053160", "058470",
                "214150", "039030", "036830", "053290", "054780", "036460", "039340", "036010", "054620", "036420"
            ]
            self.etf_symbols = []
            self.futures_symbols = []


class MaxDataCollector:
    """최대 실시간 데이터 수집기"""

    def __init__(self, config: DataCollectionConfig) -> None:
        self.config = config
        self.kis_client: Optional[KIS] = None
        self.kis_api: Optional[KISApi] = None

        # 데이터 저장소
        self.price_data: Dict[str, List[Dict[str, Any]]] = {}
        self.volume_data: Dict[str, List[Dict[str, Any]]] = {}
        self.orderbook_data: Dict[str, List[Dict[str, Any]]] = {}
        self.market_data: Dict[str, List[Dict[str, Any]]] = {}

        # 통계
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'data_points_collected': 0,
            'start_time': None,
            'last_save_time': None
        }

        # 디렉토리 생성
        self._create_directories()

    def _create_directories(self) -> None:
        """필요한 디렉토리 생성"""
        Path(self.config.data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.backup_dir).mkdir(parents=True, exist_ok=True)

        # 종목별 하위 디렉토리 생성
        for category in ['kospi', 'kosdaq', 'futures', 'options']:
            Path(f"{self.config.data_dir}/{category}").mkdir(exist_ok=True)

    async def initialize_kis_client(self) -> bool:
        """KIS 클라이언트 초기화"""
        try:
            if not PYKIS_AVAILABLE:
                raise ImportError(f"pykis가 설치되지 않았습니다: {PYKIS_IMPORT_ERROR}")
            app_key = os.getenv('LIVE_KIS_APP_KEY')
            app_secret = os.getenv('LIVE_KIS_APP_SECRET')
            account_code = os.getenv('LIVE_KIS_ACCOUNT_NUMBER', '')
            product_code = os.getenv('LIVE_KIS_PRODUCT_CODE', '01')
            if not app_key or not app_secret:
                raise ValueError("KIS API 키가 설정되지 않았습니다.")
            if 'KIS' not in globals() or 'KISApi' not in globals():
                raise AttributeError("pykis에서 KIS/KISApi 클래스를 찾을 수 없습니다. pykis 버전을 확인하세요.")
            self.kis_client = KIS(
                appkey=app_key,
                appsecret=app_secret,
                account=account_code,
                use_mock=False
            )
            self.kis_api = KISApi(self.kis_client)
            logger.info("KIS 클라이언트 초기화 성공")
            return True
        except Exception as e:
            logger.error(f"KIS 클라이언트 초기화 실패: {e}")
            return False

    async def start_collection(self) -> None:
        """데이터 수집 시작"""
        logger.info("🚀 최대 실시간 데이터 수집 시작")
        logger.info(f"수집 대상: KOSPI {len(self.config.kospi_symbols)}개, KOSDAQ {len(self.config.kosdaq_symbols)}개")

        self.stats['start_time'] = datetime.now()

        # KIS 클라이언트 초기화
        if not await self.initialize_kis_client():
            logger.error("KIS 클라이언트 초기화 실패로 수집을 중단합니다.")
            return

        # 모든 종목 데이터 초기화
        all_symbols = (self.config.kospi_symbols +
                      self.config.kosdaq_symbols +
                      self.config.futures_symbols)

        for symbol in all_symbols:
            self.price_data[symbol] = []
            self.volume_data[symbol] = []
            self.orderbook_data[symbol] = []
            self.market_data[symbol] = []

        # 데이터 수집 태스크들 시작
        tasks = [
            asyncio.create_task(self._collect_kospi_data()),
            asyncio.create_task(self._collect_kosdaq_data()),
            asyncio.create_task(self._collect_futures_data()),
            asyncio.create_task(self._auto_save_data()),
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

    async def _collect_kospi_data(self) -> None:
        """KOSPI 종목 데이터 수집"""
        logger.info("KOSPI 종목 데이터 수집 시작")

        while True:
            try:
                for symbol in self.config.kospi_symbols:
                    await self._collect_symbol_data(symbol, 'kospi')
                    await asyncio.sleep(0.1)  # API 호출 간격 조절

                await asyncio.sleep(self.config.collection_interval)

            except Exception as e:
                logger.error(f"KOSPI 데이터 수집 오류: {e}")
                await asyncio.sleep(self.config.retry_delay)

    async def _collect_kosdaq_data(self) -> None:
        """KOSDAQ 종목 데이터 수집"""
        logger.info("KOSDAQ 종목 데이터 수집 시작")

        while True:
            try:
                for symbol in self.config.kosdaq_symbols:
                    await self._collect_symbol_data(symbol, 'kosdaq')
                    await asyncio.sleep(0.1)  # API 호출 간격 조절

                await asyncio.sleep(self.config.collection_interval)

            except Exception as e:
                logger.error(f"KOSDAQ 데이터 수집 오류: {e}")
                await asyncio.sleep(self.config.retry_delay)

    async def _collect_futures_data(self) -> None:
        """선물 데이터 수집"""
        logger.info("선물 데이터 수집 시작")

        while True:
            try:
                for symbol in self.config.futures_symbols:
                    await self._collect_futures_symbol_data(symbol)
                    await asyncio.sleep(0.1)

                await asyncio.sleep(self.config.collection_interval)

            except Exception as e:
                logger.error(f"선물 데이터 수집 오류: {e}")
                await asyncio.sleep(self.config.retry_delay)

    async def _collect_symbol_data(self, symbol: str, category: str) -> None:
        """개별 종목 데이터 수집"""
        try:
            if not self.kis_api:
                raise ValueError("KIS API가 초기화되지 않았습니다.")
            self.stats['total_requests'] += 1
            # 과거부터 최신까지 반복 수집 (최대치)
            for date in self._date_range(self.config.start_date, self.config.end_date):
                ohlcv_data = self.kis_api.get_kr_ohlcv(symbol, "D", 100, date)
                # 현재가 조회
                current_price = self.kis_api.get_kr_current_price(symbol)

                # 거래량 데이터
                volume_data = self.kis_api.get_kr_volume(symbol)

                # 호가 데이터 (실시간)
                orderbook_data = self.kis_api.get_kr_orderbook(symbol)

                # 데이터 포인트 생성
                timestamp = datetime.now()

                price_point = {
                    'timestamp': timestamp.isoformat(),
                    'symbol': symbol,
                    'current_price': current_price,
                    'category': category
                }

                volume_point = {
                    'timestamp': timestamp.isoformat(),
                    'symbol': symbol,
                    'volume': volume_data,
                    'category': category
                }

                orderbook_point = {
                    'timestamp': timestamp.isoformat(),
                    'symbol': symbol,
                    'orderbook': orderbook_data,
                    'category': category
                }

                # 데이터 저장
                self.price_data[symbol].append(price_point)
                self.volume_data[symbol].append(volume_point)
                self.orderbook_data[symbol].append(orderbook_point)

                # 데이터 포인트 수 제한
                if len(self.price_data[symbol]) > self.config.max_data_points:
                    self.price_data[symbol] = self.price_data[symbol][-self.config.max_data_points:]
                    self.volume_data[symbol] = self.volume_data[symbol][-self.config.max_data_points:]
                    self.orderbook_data[symbol] = self.orderbook_data[symbol][-self.config.max_data_points:]

                self.stats['successful_requests'] += 1
                self.stats['data_points_collected'] += 3  # price, volume, orderbook
                await asyncio.sleep(0.01)
        except Exception as e:
            self.stats['failed_requests'] += 1
            logger.error(f"종목 {symbol} 데이터 수집 실패: {e}")

    async def _collect_futures_symbol_data(self, symbol: str) -> None:
        """선물 종목 데이터 수집"""
        try:
            if not self.kis_api:
                raise ValueError("KIS API가 초기화되지 않았습니다.")

            self.stats['total_requests'] += 1

            # 선물 현재가 조회
            futures_price = self.kis_api.get_futures_current_price(symbol)

            # 선물 OHLCV 데이터
            futures_ohlcv = self.kis_api.get_futures_ohlcv(symbol, "D", 100)

            timestamp = datetime.now()

            futures_point = {
                'timestamp': timestamp.isoformat(),
                'symbol': symbol,
                'price': futures_price,
                'ohlcv': futures_ohlcv,
                'category': 'futures'
            }

            self.market_data[symbol].append(futures_point)

            if len(self.market_data[symbol]) > self.config.max_data_points:
                self.market_data[symbol] = self.market_data[symbol][-self.config.max_data_points:]

            self.stats['successful_requests'] += 1
            self.stats['data_points_collected'] += 1

        except Exception as e:
            self.stats['failed_requests'] += 1
            logger.error(f"선물 {symbol} 데이터 수집 실패: {e}")

    async def _auto_save_data(self) -> None:
        """자동 데이터 저장"""
        while True:
            try:
                await asyncio.sleep(self.config.data_save_interval)
                await self._save_all_data()
                self.stats['last_save_time'] = datetime.now()

            except Exception as e:
                logger.error(f"자동 저장 오류: {e}")

    async def _save_all_data(self) -> None:
        """모든 데이터 저장"""
        try:
            if not PD_AVAILABLE:
                raise ImportError("pandas가 설치되지 않았습니다.")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # KOSPI 데이터 저장
            for symbol in self.config.kospi_symbols:
                await self._save_symbol_data(symbol, 'kospi', timestamp)

            # KOSDAQ 데이터 저장
            for symbol in self.config.kosdaq_symbols:
                await self._save_symbol_data(symbol, 'kosdaq', timestamp)

            # 선물 데이터 저장
            for symbol in self.config.futures_symbols:
                await self._save_futures_data(symbol, timestamp)

            logger.info(f"데이터 저장 완료: {timestamp}")

        except Exception as e:
            logger.error(f"데이터 저장 오류: {e}")

    async def _save_symbol_data(self, symbol: str, category: str, timestamp: str) -> None:
        """개별 종목 데이터 저장"""
        try:
            if not PD_AVAILABLE:
                raise ImportError("pandas가 설치되지 않았습니다.")

            # 가격 데이터 저장
            if self.price_data[symbol]:
                price_df = pd.DataFrame(self.price_data[symbol])
                price_file = f"{self.config.data_dir}/{category}/{symbol}_price_{timestamp}.csv"
                price_df.to_csv(price_file, index=False)

            # 거래량 데이터 저장
            if self.volume_data[symbol]:
                volume_df = pd.DataFrame(self.volume_data[symbol])
                volume_file = f"{self.config.data_dir}/{category}/{symbol}_volume_{timestamp}.csv"
                volume_df.to_csv(volume_file, index=False)

            # 호가 데이터 저장
            if self.orderbook_data[symbol]:
                orderbook_df = pd.DataFrame(self.orderbook_data[symbol])
                orderbook_file = f"{self.config.data_dir}/{category}/{symbol}_orderbook_{timestamp}.csv"
                orderbook_df.to_csv(orderbook_file, index=False)

        except Exception as e:
            logger.error(f"종목 {symbol} 데이터 저장 실패: {e}")

    async def _save_futures_data(self, symbol: str, timestamp: str) -> None:
        """선물 데이터 저장"""
        try:
            if not PD_AVAILABLE:
                raise ImportError("pandas가 설치되지 않았습니다.")

            if self.market_data[symbol]:
                futures_df = pd.DataFrame(self.market_data[symbol])
                futures_file = f"{self.config.data_dir}/futures/{symbol}_{timestamp}.csv"
                futures_df.to_csv(futures_file, index=False)

        except Exception as e:
            logger.error(f"선물 {symbol} 데이터 저장 실패: {e}")

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
        await self._save_all_data()

        # 백업 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{self.config.backup_dir}/backup_{timestamp}"

        try:
            import shutil
            shutil.copytree(self.config.data_dir, backup_path)
            logger.info(f"백업 생성 완료: {backup_path}")
        except Exception as e:
            logger.error(f"백업 생성 실패: {e}")

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

    def _date_range(self, start: str, end: str) -> List[str]:
        """YYYYMMDD 문자열 기준 일자 리스트 반환"""
        s = datetime.strptime(start, "%Y%m%d")
        e = datetime.strptime(end, "%Y%m%d")
        return [(s + timedelta(days=i)).strftime("%Y%m%d") for i in range((e-s).days+1)]


async def main() -> None:
    """메인 함수"""
    print("🚀 KIS API 최대 실시간 데이터 수집기 시작")
    print("=" * 60)

    # 설정 생성
    config = DataCollectionConfig()

    # 수집기 생성 및 시작
    collector = MaxDataCollector(config)

    try:
        await collector.start_collection()
    except KeyboardInterrupt:
        print("\n⚠️ 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
    finally:
        print("✅ 데이터 수집 완료")


if __name__ == "__main__":
    asyncio.run(main())

