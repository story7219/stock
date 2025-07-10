#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: optimized_data_pipeline.py
모듈: 최적화된 데이터 파이프라인
목적: 병렬 처리, 멀티스레딩, GPU 활용을 통한 고성능 데이터 수집 및 전처리

Author: AI Assistant
Created: 2025-01-27
Modified: 2025-01-27
Version: 1.0.0

Dependencies:
    - Python 3.11+
    - asyncio, multiprocessing, threading
    - pandas, numpy, pyarrow, dask
    - torch, cudf (GPU 가속)
    - pykis, aiohttp
    - redis, sqlalchemy
"""

from __future__ import annotations

import asyncio
import json
import logging
import multiprocessing as mp
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import aiohttp
    import dask.dataframe as dd
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    import redis.asyncio as redis
    import torch
    from dask.distributed import Client, LocalCluster
    from pykis import KISClient
    from pykis.api import KISApi
    from sqlalchemy import create_engine, text
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
except ImportError:
    pass

# GPU_AVAILABLE은 실제로 torch, cudf import 성공 여부로 결정
try:
    import cudf
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('optimized_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class PipelineConfig:
    """파이프라인 설정"""
    # 데이터 수집 설정
    batch_size: int = 10000
    max_workers: int = mp.cpu_count()
    chunk_size: int = 100000

    # 병렬 처리 설정
    use_multiprocessing: bool = True
    use_multithreading: bool = True
    use_gpu: bool = GPU_AVAILABLE

    # 캐싱 설정
    cache_enabled: bool = True
    cache_ttl: int = 3600  # 1시간
    local_cache_dir: str = "./cache"

    # 저장 설정
    storage_format: str = "parquet"  # parquet, arrow, hdf5
    compression: str = "snappy"

    # 성능 모니터링
    enable_profiling: bool = True
    log_performance: bool = True

@dataclass
class DataCollectionConfig:
    """데이터 수집 설정"""
    # 과거 데이터 수집
    historical_start_date: str = "2020-01-01"
    historical_end_date: str = "2025-01-27"
    historical_batch_days: int = 30  # 30일씩 배치 처리

    # 실시간 데이터 수집
    realtime_interval: float = 1.0
    realtime_buffer_size: int = 1000

    # 종목 설정
    kospi_symbols: List[str] = field(default_factory=lambda: [
        "005930", "000660", "035420", "051910", "006400",  # 삼성전자, SK하이닉스, NAVER, LG화학, 삼성SDI
        "035720", "207940", "068270", "323410", "373220",  # 카카오, 삼성바이오로직스, 셀트리온, 카카오뱅크, LG에너지솔루션
        "005380", "000270", "015760", "017670", "032830",  # 현대차, 기아, 한국전력, SK텔레콤, 삼성생명
    ])

    kosdaq_symbols: List[str] = field(default_factory=lambda: [
        "091990", "122870", "086520", "096770", "018260",  # 셀트리온헬스케어, 와이지엔터테인먼트, 에코프로, SK이노베이션, 삼성에스디에스
    ])

class OptimizedDataPipeline:
    """최적화된 데이터 파이프라인"""

    def __init__(self, pipeline_config: PipelineConfig, collection_config: DataCollectionConfig):
        self.pipeline_config = pipeline_config
        self.collection_config = collection_config

        # 클라이언트 초기화
        self.kis_client = None
        self.kis_api = None
        self.redis_client = None
        self.dask_client = None

        # 성능 모니터링
        self.performance_stats = {
            'collection_time': [],
            'processing_time': [],
            'storage_time': [],
            'total_records': 0,
            'start_time': None
        }

        # 캐시 디렉토리 생성
        Path(self.pipeline_config.local_cache_dir).mkdir(parents=True, exist_ok=True)

    async def initialize(self):
        """초기화"""
        logger.info("최적화된 데이터 파이프라인 초기화 시작")

        # KIS API 초기화
        await self._initialize_kis_client()

        # Redis 클라이언트 초기화
        await self._initialize_redis()

        # Dask 클라이언트 초기화
        await self._initialize_dask()

        # GPU 확인
        if self.pipeline_config.use_gpu and GPU_AVAILABLE:
            logger.info("GPU 가속 활성화")
        else:
            logger.info("CPU 모드로 실행")

        self.performance_stats['start_time'] = datetime.now()
        logger.info("초기화 완료")

    async def _initialize_kis_client(self):
        """KIS API 클라이언트 초기화"""
        try:
            app_key = os.getenv('LIVE_KIS_APP_KEY')
            app_secret = os.getenv('LIVE_KIS_APP_SECRET')
            account_code = os.getenv('LIVE_KIS_ACCOUNT_NUMBER', '')

            if not app_key or not app_secret:
                raise ValueError("KIS API 키가 설정되지 않았습니다.")

            self.kis_client = KISClient(
                api_key=app_key,
                api_secret=app_secret,
                acc_no=account_code,
                mock=False
            )

            self.kis_api = KISApi(self.kis_client)
            logger.info("KIS API 클라이언트 초기화 성공")

        except Exception as e:
            logger.error(f"KIS API 클라이언트 초기화 실패: {e}")
            raise

    async def _initialize_redis(self):
        """Redis 클라이언트 초기화"""
        try:
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            await self.redis_client.ping()
            logger.info("Redis 클라이언트 초기화 성공")

        except Exception as e:
            logger.error(f"Redis 클라이언트 초기화 실패: {e}")
            raise

    async def _initialize_dask(self):
        """Dask 클라이언트 초기화"""
        try:
            # 로컬 클러스터 생성
            cluster = LocalCluster(
                n_workers=self.pipeline_config.max_workers,
                threads_per_worker=2,
                memory_limit='2GB'
            )

            self.dask_client = Client(cluster)
            logger.info(f"Dask 클러스터 초기화 성공: {self.dask_client}")

        except Exception as e:
            logger.error(f"Dask 클라이언트 초기화 실패: {e}")
            raise

    async def collect_historical_data_parallel(self):
        """병렬 처리로 과거 데이터 수집"""
        logger.info("병렬 과거 데이터 수집 시작")

        start_time = time.time()

        # 날짜 범위 생성
        start_date = datetime.strptime(self.collection_config.historical_start_date, "%Y-%m-%d")
        end_date = datetime.strptime(self.collection_config.historical_end_date, "%Y-%m-%d")

        # 배치 단위로 날짜 분할
        date_batches = []
        current_date = start_date
        while current_date <= end_date:
            batch_end = min(
                current_date + timedelta(days=self.collection_config.historical_batch_days),
                end_date
            )
            date_batches.append((current_date, batch_end))
            current_date = batch_end + timedelta(days=1)

        logger.info(f"총 {len(date_batches)}개 배치로 분할")

        # 병렬 처리로 데이터 수집
        all_symbols = (self.collection_config.kospi_symbols +
                      self.collection_config.kosdaq_symbols)

        total_records = 0

        # ProcessPoolExecutor로 병렬 처리
        with ProcessPoolExecutor(max_workers=self.pipeline_config.max_workers) as executor:
            # 각 배치별로 태스크 생성
            futures = []
            for batch_start, batch_end in date_batches:
                for symbol in all_symbols:
                    future = executor.submit(
                        self._collect_symbol_historical_data,
                        symbol,
                        batch_start.strftime("%Y-%m-%d"),
                        batch_end.strftime("%Y-%m-%d")
                    )
                    futures.append(future)

            # 결과 수집
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        total_records += len(result)
                        # 캐시에 저장
                        await self._save_to_cache(result)
                except Exception as e:
                    logger.error(f"데이터 수집 실패: {e}")

        collection_time = time.time() - start_time
        self.performance_stats['collection_time'].append(collection_time)
        self.performance_stats['total_records'] += total_records

        logger.info(f"과거 데이터 수집 완료: {total_records:,}개 레코드, {collection_time:.2f}초")

        return total_records

    def _collect_symbol_historical_data(self, symbol: str, start_date: str, end_date: str) -> List[Dict]:
        """개별 종목 과거 데이터 수집 (프로세스별 실행)"""
        try:
            # KIS API 호출 (동기 방식)
            app_key = os.getenv('LIVE_KIS_APP_KEY')
            app_secret = os.getenv('LIVE_KIS_APP_SECRET')
            account_code = os.getenv('LIVE_KIS_ACCOUNT_NUMBER', '')

            kis_client = KISClient(
                api_key=app_key,
                api_secret=app_secret,
                acc_no=account_code,
                mock=False
            )
            kis_api = KISApi(kis_client)

            # OHLCV 데이터 수집
            ohlcv_data = kis_api.get_kr_ohlcv(symbol, "D", 1000)

            # 데이터 변환
            records = []
            if hasattr(ohlcv_data, 'to_dict'):
                df = ohlcv_data
            else:
                df = pd.DataFrame(ohlcv_data)

            for _, row in df.iterrows():
                record = {
                    'symbol': symbol,
                    'date': row.get('date', start_date),
                    'open': row.get('open', 0),
                    'high': row.get('high', 0),
                    'low': row.get('low', 0),
                    'close': row.get('close', 0),
                    'volume': row.get('volume', 0),
                    'category': 'kospi' if symbol.startswith('00') else 'kosdaq',
                    'data_type': 'historical'
                }
                records.append(record)

            logger.info(f"종목 {symbol} 과거 데이터 수집 완료: {len(records)}개 레코드")
            return records

        except Exception as e:
            logger.error(f"종목 {symbol} 과거 데이터 수집 실패: {e}")
            return []

    async def collect_realtime_data_async(self):
        """비동기 실시간 데이터 수집"""
        logger.info("비동기 실시간 데이터 수집 시작")

        start_time = time.time()

        # 세마포어로 동시 요청 수 제한
        semaphore = asyncio.Semaphore(self.pipeline_config.max_workers)

        all_symbols = (self.collection_config.kospi_symbols +
                      self.collection_config.kosdaq_symbols)

        # 비동기 태스크 생성
        tasks = []
        for symbol in all_symbols:
            task = asyncio.create_task(
                self._collect_symbol_realtime_data(symbol, semaphore)
            )
            tasks.append(task)

        # 모든 태스크 실행
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 결과 집계
        total_records = sum(len(result) for result in results if isinstance(result, list))

        collection_time = time.time() - start_time
        self.performance_stats['collection_time'].append(collection_time)
        self.performance_stats['total_records'] += total_records

        logger.info(f"실시간 데이터 수집 완료: {total_records:,}개 레코드, {collection_time:.2f}초")

        return total_records

    async def _collect_symbol_realtime_data(self, symbol: str, semaphore: asyncio.Semaphore) -> List[Dict]:
        """개별 종목 실시간 데이터 수집"""
        async with semaphore:
            try:
                # 캐시 확인
                cache_key = f"realtime:{symbol}:{int(time.time())}"
                cached_data = await self.redis_client.get(cache_key)

                if cached_data and self.pipeline_config.cache_enabled:
                    return []

                # 현재가 조회
                current_price = self.kis_api.get_kr_current_price(symbol)

                # 호가 데이터 조회
                orderbook = self.kis_api.get_kr_orderbook(symbol)

                # 데이터 포인트 생성
                timestamp = datetime.now()

                record = {
                    'symbol': symbol,
                    'timestamp': timestamp.isoformat(),
                    'current_price': current_price,
                    'orderbook': orderbook,
                    'category': 'kospi' if symbol.startswith('00') else 'kosdaq',
                    'data_type': 'realtime'
                }

                # 캐시에 저장
                if self.pipeline_config.cache_enabled:
                    await self.redis_client.setex(
                        cache_key,
                        self.pipeline_config.cache_ttl,
                        str(record)
                    )

                return [record]

            except Exception as e:
                logger.error(f"종목 {symbol} 실시간 데이터 수집 실패: {e}")
                return []

    async def process_data_parallel(self, data: List[Dict]) -> pd.DataFrame:
        """병렬 데이터 전처리"""
        logger.info("병렬 데이터 전처리 시작")

        start_time = time.time()

        # Dask DataFrame으로 변환
        df = pd.DataFrame(data)
        ddf = dd.from_pandas(df, npartitions=self.pipeline_config.max_workers)

        # GPU 가속 사용 시
        if self.pipeline_config.use_gpu and GPU_AVAILABLE:
            # cuDF로 변환
            gdf = cudf.DataFrame.from_pandas(df)

            # GPU에서 전처리
            processed_gdf = await self._preprocess_gpu(gdf)

            # 다시 pandas로 변환
            processed_df = processed_gdf.to_pandas()
        else:
            # CPU에서 전처리
            processed_df = await self._preprocess_cpu(ddf)

        processing_time = time.time() - start_time
        self.performance_stats['processing_time'].append(processing_time)

        logger.info(f"데이터 전처리 완료: {len(processed_df):,}개 레코드, {processing_time:.2f}초")

        return processed_df

    async def _preprocess_cpu(self, ddf: dd.DataFrame) -> pd.DataFrame:
        """CPU 기반 데이터 전처리"""
        # 데이터 타입 변환
        ddf['timestamp'] = dd.to_datetime(ddf['timestamp'])
        ddf['current_price'] = ddf['current_price'].astype(float)

        # 기술적 지표 계산
        ddf = ddf.map_partitions(self._calculate_technical_indicators)

        # 결측값 처리
        ddf = ddf.fillna(0)

        # 계산 실행
        return ddf.compute()

    async def _preprocess_gpu(self, gdf: 'cudf.DataFrame') -> 'cudf.DataFrame':
        """GPU 기반 데이터 전처리"""
        # GPU에서 기술적 지표 계산
        gdf = self._calculate_technical_indicators_gpu(gdf)

        # 결측값 처리
        gdf = gdf.fillna(0)

        return gdf

    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """기술적 지표 계산 (CPU)"""
        if len(df) == 0:
            return df

        # 이동평균
        df['ma_5'] = df['current_price'].rolling(window=5).mean()
        df['ma_20'] = df['current_price'].rolling(window=20).mean()

        # RSI 계산
        delta = df['current_price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # 변동성
        df['volatility'] = df['current_price'].rolling(window=20).std()

        return df

    def _calculate_technical_indicators_gpu(self, gdf: 'cudf.DataFrame') -> 'cudf.DataFrame':
        """기술적 지표 계산 (GPU)"""
        if len(gdf) == 0:
            return gdf

        # GPU에서 이동평균 계산
        gdf['ma_5'] = gdf['current_price'].rolling(window=5).mean()
        gdf['ma_20'] = gdf['current_price'].rolling(window=20).mean()

        # GPU에서 RSI 계산
        delta = gdf['current_price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        gdf['rsi'] = 100 - (100 / (1 + rs))

        # GPU에서 변동성 계산
        gdf['volatility'] = gdf['current_price'].rolling(window=20).std()

        return gdf

    async def save_data_optimized(self, df: pd.DataFrame, filename: str):
        """최적화된 데이터 저장"""
        logger.info(f"최적화된 데이터 저장 시작: {filename}")

        start_time = time.time()

        # 저장 형식별 최적화
        if self.pipeline_config.storage_format == "parquet":
            await self._save_parquet_optimized(df, filename)
        elif self.pipeline_config.storage_format == "arrow":
            await self._save_arrow_optimized(df, filename)
        elif self.pipeline_config.storage_format == "hdf5":
            await self._save_hdf5_optimized(df, filename)
        else:
            await self._save_parquet_optimized(df, filename)

        storage_time = time.time() - start_time
        self.performance_stats['storage_time'].append(storage_time)

        logger.info(f"데이터 저장 완료: {filename}, {storage_time:.2f}초")

    async def _save_parquet_optimized(self, df: pd.DataFrame, filename: str):
        """Parquet 최적화 저장"""
        # PyArrow 테이블로 변환
        table = pa.Table.from_pandas(df)

        # 압축 설정
        compression = self.pipeline_config.compression

        # 청크 단위로 저장
        chunk_size = self.pipeline_config.chunk_size

        if len(df) > chunk_size:
            # 대용량 데이터는 청크 단위로 분할 저장
            chunks = [df[i:i+chunk_size] for i in range(0, len(df), chunk_size)]

            for i, chunk in enumerate(chunks):
                chunk_table = pa.Table.from_pandas(chunk)
                chunk_filename = f"{filename}_part_{i:04d}.parquet"

                pq.write_table(
                    chunk_table,
                    chunk_filename,
                    compression=compression,
                    row_group_size=10000
                )
        else:
            # 소용량 데이터는 단일 파일로 저장
            pq.write_table(
                table,
                f"{filename}.parquet",
                compression=compression,
                row_group_size=10000
            )

    async def _save_arrow_optimized(self, df: pd.DataFrame, filename: str):
        """Arrow 최적화 저장"""
        table = pa.Table.from_pandas(df)

        # Arrow 파일로 저장
        with pa.OSFile(f"{filename}.arrow", 'wb') as sink:
            with pa.RecordBatchStreamWriter(sink, table.schema) as writer:
                writer.write_table(table)

    async def _save_hdf5_optimized(self, df: pd.DataFrame, filename: str):
        """HDF5 최적화 저장"""
        # HDF5로 저장 (압축 포함)
        df.to_hdf(
            f"{filename}.h5",
            key='data',
            mode='w',
            complevel=9,
            complib='blosc'
        )

    async def _save_to_cache(self, data: List[Dict]):
        """캐시에 데이터 저장"""
        if not self.pipeline_config.cache_enabled:
            return

        try:
            # Redis에 저장
            cache_key = f"batch:{int(time.time())}"
            await self.redis_client.setex(
                cache_key,
                self.pipeline_config.cache_ttl,
                str(data)
            )

            # 로컬 캐시에도 저장
            cache_file = Path(self.pipeline_config.local_cache_dir) / f"{cache_key}.json"
            with open(cache_file, 'w') as f:
                json.dump(data, f)

        except Exception as e:
            logger.error(f"캐시 저장 실패: {e}")

    async def run_optimized_pipeline(self):
        """최적화된 파이프라인 실행"""
        logger.info("🚀 최적화된 데이터 파이프라인 시작")

        try:
            # 1. 과거 데이터 수집 (병렬 처리)
            historical_records = await self.collect_historical_data_parallel()

            # 2. 실시간 데이터 수집 (비동기)
            realtime_records = await self.collect_realtime_data_async()

            # 3. 데이터 통합
            all_data = []

            # 캐시에서 데이터 로드
            if self.pipeline_config.cache_enabled:
                cache_data = await self._load_from_cache()
                all_data.extend(cache_data)

            # 4. 병렬 전처리
            if all_data:
                processed_df = await self.process_data_parallel(all_data)

                # 5. 최적화된 저장
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"optimized_data_{timestamp}"
                await self.save_data_optimized(processed_df, filename)

            # 6. 성능 통계 출력
            await self._print_performance_stats()

        except Exception as e:
            logger.error(f"파이프라인 실행 실패: {e}")
            raise

    async def _load_from_cache(self) -> List[Dict]:
        """캐시에서 데이터 로드"""
        try:
            all_data = []

            # Redis 캐시에서 로드
            keys = await self.redis_client.keys("batch:*")
            for key in keys:
                data_str = await self.redis_client.get(key)
                if data_str:
                    data = json.loads(data_str)
                    all_data.extend(data)

            # 로컬 캐시에서 로드
            cache_dir = Path(self.pipeline_config.local_cache_dir)
            for cache_file in cache_dir.glob("*.json"):
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    all_data.extend(data)

            logger.info(f"캐시에서 {len(all_data)}개 레코드 로드")
            return all_data

        except Exception as e:
            logger.error(f"캐시 로드 실패: {e}")
            return []

    async def _print_performance_stats(self):
        """성능 통계 출력"""
        if not self.performance_stats['start_time']:
            return

        total_time = datetime.now() - self.performance_stats['start_time']

        avg_collection_time = np.mean(self.performance_stats['collection_time']) if self.performance_stats['collection_time'] else 0
        avg_processing_time = np.mean(self.performance_stats['processing_time']) if self.performance_stats['processing_time'] else 0
        avg_storage_time = np.mean(self.performance_stats['storage_time']) if self.performance_stats['storage_time'] else 0

        logger.info("🎯 최적화된 파이프라인 성능 통계:")
        logger.info(f"   총 실행 시간: {total_time}")
        logger.info(f"   총 레코드 수: {self.performance_stats['total_records']:,}")
        logger.info(f"   평균 수집 시간: {avg_collection_time:.2f}초")
        logger.info(f"   평균 처리 시간: {avg_processing_time:.2f}초")
        logger.info(f"   평균 저장 시간: {avg_storage_time:.2f}초")
        logger.info(f"   처리 속도: {self.performance_stats['total_records'] / total_time.total_seconds():.0f} 레코드/초")

        if self.pipeline_config.use_gpu and GPU_AVAILABLE:
            logger.info("   GPU 가속: 활성화")
        else:
            logger.info("   GPU 가속: 비활성화")

async def main():
    """메인 함수"""
    print("🚀 최적화된 데이터 파이프라인 시작")
    print("=" * 60)

    # 설정 생성
    pipeline_config = PipelineConfig()
    collection_config = DataCollectionConfig()

    # 파이프라인 생성 및 실행
    pipeline = OptimizedDataPipeline(pipeline_config, collection_config)

    try:
        await pipeline.initialize()
        await pipeline.run_optimized_pipeline()
    except KeyboardInterrupt:
        print("\n⚠️ 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
    finally:
        print("✅ 최적화된 파이프라인 완료")

if __name__ == "__main__":
    asyncio.run(main())

