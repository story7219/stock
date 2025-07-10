#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: max_data_collector.py
모듈: 최대 실시간 데이터 수집기
목적: KIS API를 통한 최대한 많은 실시간 데이터 수집

Author: AI Assistant
Created: 2025-01-27
Modified: 2025-01-27
Version: 1.0.0

Dependencies:
    - Python 3.11+
    - aiohttp, asyncio, pandas, numpy
    - pykis (KIS API 클라이언트)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import time
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

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('max_data_collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class DataCollectionConfig:
    """데이터 수집 설정"""
    # 주요 종목 리스트 (KOSPI, KOSDAQ 대표 종목들)
    kospi_symbols: List[str] = field(default_factory=lambda: [
        "005930",  # 삼성전자
        "000660",  # SK하이닉스
        "035420",  # NAVER
        "051910",  # LG화학
        "006400",  # 삼성SDI
        "035720",  # 카카오
        "207940",  # 삼성바이오로직스
        "068270",  # 셀트리온
        "323410",  # 카카오뱅크
        "373220",  # LG에너지솔루션
        "005380",  # 현대차
        "000270",  # 기아
        "015760",  # 한국전력
        "017670",  # SK텔레콤
        "032830",  # 삼성생명
        "086790",  # 하나금융지주
        "105560",  # KB금융
        "055550",  # 신한지주
        "138930",  # BNK금융지주
        "316140",  # 우리금융지주
    ])

    kosdaq_symbols: List[str] = field(default_factory=lambda: [
        "091990",  # 셀트리온헬스케어
        "122870",  # 와이지엔터테인먼트
        "086520",  # 에코프로
        "096770",  # SK이노베이션
        "018260",  # 삼성에스디에스
        "091810",  # 테크윈
        "036570",  # 엔씨소프트
        "079370",  # 제우스
        "053160",  # 프리엠스
        "058470",  # 리노공업
        "214150",  # 클래시스
        "039030",  # 이오테크닉스
        "036830",  # 솔브레인
        "053290",  # NE능률
        "054780",  # 키이스트
        "036460",  # 한국국제협력단
        "039340",  # 한건
        "036010",  # 아비코전자
        "054620",  # 화진
        "036420",  # 진양제약
    ])

    # 선물/옵션 종목
    futures_symbols: List[str] = field(default_factory=lambda: [
        "KOSPI200",  # KOSPI200 선물
        "KOSPI200MINI",  # KOSPI200 미니선물
    ])

    # 수집 설정
    collection_interval: float = 1.0  # 1초마다 수집
    max_retries: int = 3
    retry_delay: float = 5.0
    data_save_interval: int = 60  # 60초마다 저장
    max_data_points: int = 10000  # 종목당 최대 데이터 포인트

    # 저장 경로
    data_dir: str = "./collected_data"
    backup_dir: str = "./data_backup"


class MaxDataCollector:
    """최대 실시간 데이터 수집기"""

    def __init__(self, config: DataCollectionConfig) -> None:
        self.config = config
        self.kis_client: Optional[KISClient] = None
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
                raise ImportError("pykis가 설치되지 않았습니다.")

            # 환경변수에서 API 키 가져오기
            app_key = os.getenv('LIVE_KIS_APP_KEY')
            app_secret = os.getenv('LIVE_KIS_APP_SECRET')
            account_code = os.getenv('LIVE_KIS_ACCOUNT_NUMBER', '')
            product_code = os.getenv('LIVE_KIS_PRODUCT_CODE', '01')

            if not app_key or not app_secret:
                raise ValueError("KIS API 키가 설정되지 않았습니다.")

            # KIS 클라이언트 생성
            self.kis_client = KISClient(
                api_key=app_key,
                api_secret=app_secret,
                acc_no=account_code,
                mock=False  # 실전 모드
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

            # 현재가 조회
            current_price = self.kis_api.get_kr_current_price(symbol)

            # OHLCV 데이터 조회 (최근 100개)
            ohlcv_data = self.kis_api.get_kr_ohlcv(symbol, "D", 100)

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

