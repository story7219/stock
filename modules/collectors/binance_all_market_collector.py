#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: binance_all_market_collector.py
목적: 바이낸스 현물/선물/옵션 전체 마켓을 설립일(상장일)부터 현재까지 비동기 고속 병렬, 멀티레벨 캐싱, 커넥션풀링, 성능/메모리 최적화, 유지보수성 최우선으로 수집하는 운영 품질의 통합 수집기

Author: World-Class Python
Created: 2025-01-27
Version: 1.0.0

Dependencies:
    - Python 3.11+
    - aiohttp>=3.9.0
    - pandas>=2.1.0
    - pyarrow>=14.0.0
    - python-binance>=1.0.19
    - aiocache>=0.12.2
    - tqdm>=4.66.0
    - pydantic>=2.5.0
    - tenacity>=8.2.3

"""

import asyncio
import logging
import os
import sys
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set, Literal, cast, Union
from functools import lru_cache, wraps
import pandas as pd
import aiohttp
from aiohttp import ClientSession, TCPConnector, ClientResponseError
from aiocache import cached, Cache
from pydantic import BaseModel, Field, ValidationError
from tqdm.asyncio import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError
import json
import sqlite3
import tables  # pytables(HDF5)
import time
import websockets
from asyncio import Queue, create_task, gather
from typing import Dict, Set, Optional, Callable, Any
import numpy as np

# 고급 로깅 설정 (커서룰 100%)
logging.basicConfig(
    level=logging.DEBUG,  # DEBUG 레벨로 변경
    format='%(asctime)s %(levelname)s [%(name)s:%(lineno)d] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('binance_collector.log', encoding='utf-8')
    ]
)
logger = logging.getLogger("binance_all_market_collector")

# 환경변수에서 API 키 로드
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET = os.getenv("BINANCE_SECRET")

# 초고속 성능 설정 (3마켓 동시 수집 최적화)
ULTRA_FAST_CONFIG = {
    "max_concurrent_requests": 5,  # 더 보수적으로 감소
    "chunk_size": 30,  # 청크 크기 감소
    "request_timeout": 60,  # 타임아웃 증가
    "connection_timeout": 20,  # 연결 타임아웃 증가
    "max_connections": 30,  # 연결 풀 크기 감소
    "rate_limit_delay": 0.2,  # Rate limit 지연 시간 증가
    "memory_optimization": True,
    "batch_processing": True,
    "use_compression": True,  # 압축 사용
    "keepalive_timeout": 120,  # Keep-alive 시간
    "enable_cleanup_closed": True,
    "limit_per_host": 10,  # 호스트당 연결 수 감소
    "ttl_dns_cache": 1200,  # DNS 캐시 시간
    "use_dns_cache": True,
    "force_close": False,
    "market_delay": 2.0  # 마켓 간 지연 시간
}

# 바이낸스 공식 Rate Limit 설정 (3마켓 안전 설정)
BINANCE_RATE_LIMITS = {
    "klines": {"requests_per_second": 2, "requests_per_10min": 300},  # 더 보수적
    "exchange_info": {"requests_per_second": 3, "requests_per_10min": 30},  # 더 보수적
    "futures_klines": {"requests_per_second": 2, "requests_per_10min": 200},  # 선물 전용
    "options_klines": {"requests_per_second": 2, "requests_per_10min": 200}   # 옵션 전용
}

# Weight-based Rate Limiter (개선)
class BinanceRateLimiter:
    def __init__(self):
        self.request_times: List[float] = []
        self.last_request: float = 0
        self.lock = asyncio.Lock()
        self.request_count: int = 0
        self.last_reset: float = time.time()
    
    async def wait_if_needed(self, endpoint: str) -> None:
        """Rate limit 체크 및 대기 (최적화된 버전)"""
        async with self.lock:
            now = time.time()
            
            # 10분마다 카운터 리셋
            if now - self.last_reset > 600:
                self.request_count = 0
                self.last_reset = now
            
            # 10분 윈도우에서 오래된 요청 제거 (최적화)
            cutoff_time = now - 600
            self.request_times = [t for t in self.request_times if t > cutoff_time]
            
            limit = BINANCE_RATE_LIMITS.get(endpoint, {"requests_per_second": 5, "requests_per_10min": 1000})
            
            # 초당 제한 체크 (퍼블릭 API 최적화)
            recent_requests = len([t for t in self.request_times if now - t < 1])
            if recent_requests >= limit["requests_per_second"]:
                wait_time = ULTRA_FAST_CONFIG["rate_limit_delay"]
                if wait_time > 0:
                    logger.debug(f"Rate limit: waiting {wait_time:.3f}s for {endpoint}")
                    await asyncio.sleep(wait_time)
            
            # 10분 제한 체크 (보수적 설정)
            if self.request_count >= limit["requests_per_10min"] * 0.8:  # 80%에서 경고
                wait_time = 600 - (now - self.last_reset)
                if wait_time > 0:
                    logger.warning(f"10min rate limit approaching: {self.request_count}/{limit['requests_per_10min']}, waiting {wait_time:.2f}s")
                    await asyncio.sleep(wait_time)
                    self.request_count = 0
                    self.last_reset = time.time()
            
            self.request_times.append(now)
            self.request_count += 1

# 전역 Rate Limiter 인스턴스
rate_limiter = BinanceRateLimiter()

# 상수
BINANCE_BASE_URL = "https://api.binance.com"
BINANCE_FUTURES_URL = "https://fapi.binance.com"
BINANCE_OPTIONS_URL = "https://eapi.binance.com"
MAX_CONCURRENCY = 10  # 동시성 대폭 증가
BATCH_SIZE = 1000
CACHE_DIR = Path(".binance_cache")
CACHE_DIR.mkdir(exist_ok=True)

# Rate limiting을 위한 세마포어
rate_limit_semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

# LRU 메모리 캐시 (심볼/상장일 등)
@lru_cache(maxsize=256)
def lru_json_cache(key: str) -> Optional[Any]:
    """LRU 캐시에서 JSON 데이터 조회"""
    try:
        cache_path = CACHE_DIR / f"{key}.json"
        if cache_path.exists():
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"캐시 읽기 오류 {key}: {e}")
    return None

def save_json_cache(key: str, data: Any) -> None:
    """JSON 데이터를 캐시에 저장"""
    try:
        cache_path = CACHE_DIR / f"{key}.json"
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception as e:
        logger.error(f"캐시 저장 오류 {key}: {e}")

# Pydantic 모델 (커서룰 100%)
class SymbolInfo(BaseModel):
    """심볼 정보 모델"""
    symbol: str = Field(..., description="심볼명")
    market_type: Literal["spot", "futures", "options"] = Field(..., description="마켓 타입")
    onboard_date: datetime = Field(..., description="상장일")
    status: str = Field(..., description="상태")
    base_asset: str = Field(..., description="기본 자산")
    quote_asset: str = Field(..., description="견적 자산")
    
    class Config:
        validate_assignment = True

# 심볼/상장일 조회
async def fetch_all_symbols(session: ClientSession) -> List[SymbolInfo]:
    """모든 심볼 정보 수집"""
    results: List[SymbolInfo] = []
    
    try:
        # 현물
        spot_url = f"{BINANCE_BASE_URL}/api/v3/exchangeInfo"
        # 선물
        fut_url = f"{BINANCE_FUTURES_URL}/fapi/v1/exchangeInfo"
        # 옵션
        opt_url = f"{BINANCE_OPTIONS_URL}/eapi/v1/exchangeInfo"
        
        for url, mtype in [
            (spot_url, "spot"),
            (fut_url, "futures"),
            (opt_url, "options")
        ]:
            try:
                await rate_limiter.wait_if_needed("exchange_info")
                cache_key = f"exchangeinfo_{mtype}"
                cached_data = lru_json_cache(cache_key)
                
                if cached_data:
                    data = cached_data
                    logger.debug(f"캐시에서 {mtype} 데이터 로드")
                else:
                    logger.info(f"{mtype} 마켓 정보 수집 중...")
                    async with session.get(url) as resp:
                        if resp.status == 418:
                            logger.warning(f"{mtype} 마켓 API 제한 (418): IP 차단 또는 API 제한. 건너뜀.")
                            continue
                        elif resp.status != 200:
                            logger.warning(f"{mtype} 마켓 API 오류 ({resp.status}): {resp.reason}. 건너뜀.")
                            continue
                        
                        data = await resp.json()
                        save_json_cache(cache_key, data)
                        logger.info(f"{mtype} 마켓 정보 수집 완료")
                
                for s in data.get("symbols", []):
                    try:
                        # 거래 중단된 심볼 필터링
                        if s.get("status") != "TRADING":
                            continue
                        
                        onboard = s.get("onboardDate")
                        if onboard:
                            onboard_dt = datetime.utcfromtimestamp(onboard/1000).replace(tzinfo=timezone.utc)
                        else:
                            onboard_dt = datetime(2017,7,14, tzinfo=timezone.utc)
                        
                        symbol_info = SymbolInfo(
                            symbol=s["symbol"],
                            market_type=cast(Literal["spot", "futures", "options"], mtype),
                            onboard_date=onboard_dt,
                            status=s.get("status", "UNKNOWN"),
                            base_asset=s.get("baseAsset", ""),
                            quote_asset=s.get("quoteAsset", "")
                        )
                        results.append(symbol_info)
                        
                    except ValidationError as e:
                        logger.error(f"심볼 정보 검증 오류 {s.get('symbol', 'unknown')}: {e}")
                    except Exception as e:
                        logger.error(f"심볼 처리 오류 {s.get('symbol', 'unknown')}: {e}")
                        
            except Exception as e:
                logger.error(f"마켓 {mtype} 수집 오류: {e}")
                
    except Exception as e:
        logger.error(f"심볼 수집 전체 오류: {e}")
        raise
    
    logger.info(f"총 {len(results)}개 유효한 심볼 수집 완료")
    return results

# 멀티레벨 캐시 데코레이터 (aiocache + 디스크)
def multi_level_cache(key_builder: Callable) -> Callable:
    """멀티레벨 캐시 데코레이터"""
    def decorator(func: Callable) -> Callable:
        @cached(ttl=3600, cache=Cache.MEMORY)
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                key = key_builder(*args, **kwargs)
                disk_path = CACHE_DIR / f"{key}.parquet"
                if disk_path.exists():
                    return pd.read_parquet(disk_path)
                result = await func(*args, **kwargs)
                if isinstance(result, pd.DataFrame):
                    result.to_parquet(disk_path)
                return result
            except Exception as e:
                logger.error(f"캐시 처리 오류: {e}")
                return await func(*args, **kwargs)
        return wrapper
    return decorator

# K라인 데이터 수집 (비동기, 병렬, 캐싱)
@multi_level_cache(
    lambda *args, **kwargs: (
        f"klines_{args[5]}_{args[1]}_{args[2]}_{args[3]:%Y%m%d}_{args[4]:%Y%m%d}"
        if len(args) >= 6 else "klines_unknown"
    )
)
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def fetch_klines(
    session: ClientSession,
    symbol: str,
    interval: str,
    start: datetime,
    end: datetime,
    market: str
) -> pd.DataFrame:
    """K라인 데이터 수집 (기본)"""
    try:
        # 시간대 일관성 보장
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        
        async with rate_limit_semaphore:
            await rate_limiter.wait_if_needed("klines")
            
            if market == "spot":
                url = f"{BINANCE_BASE_URL}/api/v3/klines"
            elif market == "futures":
                url = f"{BINANCE_FUTURES_URL}/fapi/v1/klines"
            elif market == "options":
                url = f"{BINANCE_OPTIONS_URL}/eapi/v1/klines"
            else:
                raise ValueError(f"Unknown market: {market}")
            
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": int(start.timestamp() * 1000),
                "endTime": int(end.timestamp() * 1000),
                "limit": BATCH_SIZE
            }
            
            all_rows = []
            cur_start = start
            
            while cur_start < end:
                params["startTime"] = int(cur_start.timestamp() * 1000)
                
                try:
                    async with session.get(url, params=params) as resp:
                        if resp.status == 429:
                            logger.warning(f"Rate limit for {symbol}, waiting...")
                            await asyncio.sleep(5)
                            continue
                        elif resp.status == 418:
                            logger.warning(f"Symbol {symbol} is invalid (418) - IP 차단 가능성, skipping...")
                            await asyncio.sleep(1)  # 잠시 대기
                            return pd.DataFrame()
                        
                        resp.raise_for_status()
                        rows = await resp.json()
                        
                        if not rows:
                            break
                        
                        all_rows.extend(rows)
                        last_time = rows[-1][0] / 1000
                        cur_start = datetime.utcfromtimestamp(last_time).replace(tzinfo=timezone.utc) + timedelta(milliseconds=1)
                        
                        if len(rows) < BATCH_SIZE:
                            break
                        
                        await asyncio.sleep(0.2)
                        
                except ClientResponseError as e:
                    if e.status == 429:
                        await asyncio.sleep(5)
                        continue
                    elif e.status == 418:
                        return pd.DataFrame()
                    else:
                        logger.error(f"HTTP error for {symbol}: {e.status} - {e.message}")
                        raise
                except Exception as e:
                    logger.error(f"Request error for {symbol}: {type(e).__name__}: {str(e)}")
                    raise
            
            if not all_rows:
                return pd.DataFrame()
            
            df = pd.DataFrame(all_rows, columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_asset_volume", "number_of_trades",
                "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
            ])
            
            df["symbol"] = symbol
            df["market"] = market
            df["interval"] = interval
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
            df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
            
            return df
            
    except Exception as e:
        logger.error(f"fetch_klines 오류 {symbol}: {type(e).__name__}: {str(e)}")
        logger.error(f"스택 트레이스: {traceback.format_exc()}")
        raise

def save_data_multi_format(df: pd.DataFrame, base_path: Path, table_name: str = "binance_data") -> None:
    """데이터 성격/용도별로 Parquet, SQLite, HDF5로 저장"""
    try:
        base_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Parquet (ML/DL 최적화, 컬럼 기반, 대용량/AI)
        pq_path = base_path / f"{table_name}.parquet"
        df.to_parquet(pq_path, index=False)
        logger.info(f"Parquet 저장 완료: {pq_path}")
        
        # 2. SQLite (관계형 쿼리, 복잡 분석/검색)
        sqlite_path = base_path / f"{table_name}.sqlite"
        with sqlite3.connect(sqlite_path) as conn:
            df.to_sql(table_name, conn, if_exists="replace", index=False)
        logger.info(f"SQLite 저장 완료: {sqlite_path}")
        
        # 3. HDF5 (시계열/행렬, 계층적, 과학/금융)
        hdf5_path = base_path / f"{table_name}.h5"
        df.to_hdf(hdf5_path, key=table_name, mode="w", format="table", complevel=9, complib="blosc")
        logger.info(f"HDF5 저장 완료: {hdf5_path}")
        
    except Exception as e:
        logger.error(f"데이터 저장 오류: {type(e).__name__}: {str(e)}")
        logger.error(f"스택 트레이스: {traceback.format_exc()}")
        raise

# 고성능 연결 풀
class UltraFastConnectionPool:
    def __init__(self, max_connections: int = 30):
        self.connector = TCPConnector(
            limit=max_connections,
            limit_per_host=ULTRA_FAST_CONFIG["limit_per_host"],
            ttl_dns_cache=ULTRA_FAST_CONFIG["ttl_dns_cache"],
            use_dns_cache=ULTRA_FAST_CONFIG["use_dns_cache"],
            keepalive_timeout=ULTRA_FAST_CONFIG["keepalive_timeout"],
            enable_cleanup_closed=ULTRA_FAST_CONFIG["enable_cleanup_closed"],
            force_close=ULTRA_FAST_CONFIG["force_close"]
        )
        self.session: Optional[ClientSession] = None
        self.semaphore = asyncio.Semaphore(ULTRA_FAST_CONFIG["max_concurrent_requests"])
    
    async def get_session(self) -> ClientSession:
        """세션 가져오기 (최적화된 버전)"""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(
                total=ULTRA_FAST_CONFIG["request_timeout"],
                connect=ULTRA_FAST_CONFIG["connection_timeout"]
            )
            self.session = aiohttp.ClientSession(
                connector=self.connector,
                timeout=timeout,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "Accept-Encoding": "gzip, deflate, br",  # 압축 지원
                    "Accept": "application/json",
                    "Connection": "keep-alive"
                },
                skip_auto_headers=["Accept-Encoding"]  # 자동 헤더 비활성화
            )
        return self.session
    
    async def close(self) -> None:
        """연결 풀 정리"""
        if self.session:
            await self.session.close()

# 전역 연결 풀 (지연 초기화)
connection_pool: Optional[UltraFastConnectionPool] = None

def get_connection_pool() -> UltraFastConnectionPool:
    """연결 풀 지연 초기화"""
    global connection_pool
    if connection_pool is None:
        connection_pool = UltraFastConnectionPool()
    return connection_pool

# 초고속 배치 처리 (최적화된 버전)
async def process_batch_ultra_fast(
    symbols: List[SymbolInfo],
    intervals: List[str],
    session: ClientSession
) -> List[pd.DataFrame]:
    """초고속 배치 처리 (메모리 최적화 + 병렬 처리 최적화)"""
    
    try:
        logger.info(f"=== 초고속 배치 처리 시작 ===")
        logger.info(f"심볼 수: {len(symbols)}, 인터벌: {intervals}")
        
        # 메모리 최적화: 청크 단위로 처리하여 메모리 사용량 제한
        chunk_size = ULTRA_FAST_CONFIG["chunk_size"]
        max_concurrent = ULTRA_FAST_CONFIG["max_concurrent_requests"]
        
        all_results: List[pd.DataFrame] = []
        total_processed = 0
        
        # 3마켓 모두 처리 (현물, 선물, 옵션)
        all_symbols = symbols
        logger.info(f"전체 {len(all_symbols)}개 심볼 처리 (3마켓)")
        
        # 마켓별로 분리
        spot_symbols = [s for s in all_symbols if s.market_type == "spot"]
        futures_symbols = [s for s in all_symbols if s.market_type == "futures"]
        options_symbols = [s for s in all_symbols if s.market_type == "options"]
        
        logger.info(f"현물: {len(spot_symbols)}개, 선물: {len(futures_symbols)}개, 옵션: {len(options_symbols)}개")
        
        # 마켓별로 순차 처리 (418 에러 방지)
        all_market_symbols = []
        if spot_symbols:
            all_market_symbols.extend(spot_symbols[:10])  # 현물 10개
            logger.info("현물 마켓 추가")
        if futures_symbols:
            all_market_symbols.extend(futures_symbols[:5])  # 선물 5개
            logger.info("선물 마켓 추가")
        if options_symbols:
            all_market_symbols.extend(options_symbols[:5])  # 옵션 5개
            logger.info("옵션 마켓 추가")
        
        test_symbols = all_market_symbols
        logger.info(f"총 {len(test_symbols)}개 심볼 처리 (3마켓)")
        
        # 심볼을 청크로 분할
        symbol_chunks = [test_symbols[i:i + chunk_size] for i in range(0, len(test_symbols), chunk_size)]
        logger.info(f"총 {len(symbol_chunks)}개 청크로 분할")
        
        # 세마포어로 동시 요청 제한
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_symbol_chunk(chunk: List[SymbolInfo], chunk_idx: int) -> List[pd.DataFrame]:
            """심볼 청크 처리"""
            chunk_results: List[pd.DataFrame] = []
            
            for symbol in chunk:
                for interval in intervals:
                    try:
                        # 시간 범위 계산
                        start = symbol.onboard_date
                        end = datetime.now(timezone.utc)
                        
                        # 시간대 일관성 보장
                        if start.tzinfo is None:
                            start = start.replace(tzinfo=timezone.utc)
                        if end.tzinfo is None:
                            end = end.replace(tzinfo=timezone.utc)
                        
                        # 연도별로 분할하여 처리 (메모리 효율성)
                        current_start = start
                        while current_start < end:
                            current_end = min(current_start + timedelta(days=365), end)
                            
                            async with semaphore:
                                try:
                                    result = await fetch_klines(
                                        session, symbol.symbol, interval, 
                                        current_start, current_end, symbol.market_type
                                    )
                                    if result is not None and not result.empty:
                                        chunk_results.append(result)
                                        logger.debug(f"데이터 수집: {symbol.symbol} {interval} {len(result)}행")
                                except Exception as e:
                                    logger.warning(f"데이터 수집 실패: {symbol.symbol} {interval}: {e}")
                            
                            current_start = current_end
                            
                    except Exception as e:
                        logger.error(f"심볼 처리 오류 {symbol.symbol}: {e}")
                        continue
            
            logger.info(f"청크 {chunk_idx + 1} 완료: {len(chunk_results)}개 데이터프레임")
            return chunk_results
        
        # 마켓별로 순차 처리 (418 에러 방지)
        chunk_results = []
        for idx, chunk in enumerate(symbol_chunks):
            logger.info(f"청크 {idx + 1}/{len(symbol_chunks)} 처리 중...")
            try:
                result = await process_symbol_chunk(chunk, idx)
                chunk_results.append(result)
                # 마켓 간 지연 시간
                await asyncio.sleep(ULTRA_FAST_CONFIG["market_delay"])
            except Exception as e:
                logger.error(f"청크 {idx + 1} 처리 실패: {e}")
                chunk_results.append([])
        
        logger.info(f"순차 처리 완료: {len(chunk_results)}개 청크")
        
        # 결과 수집
        success_chunks = 0
        error_chunks = 0
        
        for idx, result in enumerate(chunk_results):
            if isinstance(result, Exception):
                error_chunks += 1
                logger.error(f"청크 {idx + 1} 처리 실패: {result}")
            elif isinstance(result, list):
                success_chunks += 1
                all_results.extend(result)
                total_processed += sum(len(df) for df in result if isinstance(df, pd.DataFrame))
                logger.info(f"청크 {idx + 1} 성공: {len(result)}개 데이터프레임")
            else:
                logger.warning(f"청크 {idx + 1} 예상치 못한 결과 타입: {type(result)}")
        
        logger.info(f"=== 배치 처리 완료 ===")
        logger.info(f"성공 청크: {success_chunks}, 실패 청크: {error_chunks}")
        logger.info(f"총 데이터프레임: {len(all_results)}개")
        logger.info(f"총 처리된 행: {total_processed}개")
        
        return all_results
        
    except Exception as e:
        logger.error(f"배치 처리 전체 오류: {type(e).__name__}: {str(e)}")
        logger.error(f"스택 트레이스: {traceback.format_exc()}")
        raise

# 통합 데이터 수집 파이프라인
async def run_integrated_pipeline(
    intervals: List[str] = ["1d"],  # 테스트용으로 1d만
    output_dir: Path = Path("data/binance_all_markets"),
    table_name: str = "binance_data",
    enable_realtime: bool = False
) -> None:
    """통합 파이프라인 (과거 + 실시간)"""
    try:
        logger.info("=== 파이프라인 시작 ===")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 과거 데이터 수집 (REST API)
        logger.info("=== 과거 데이터 수집 시작 ===")
        pool = get_connection_pool()
        session = await pool.get_session()
        logger.info("세션 생성 완료")
        
        symbols = await fetch_all_symbols(session)
        logger.info(f"총 {len(symbols)}개 심볼 과거 데이터 수집")
        
        logger.info("배치 처리 시작...")
        results = await process_batch_ultra_fast(symbols, intervals, session)
        logger.info("배치 처리 완료")
        
        if results:
            logger.info("데이터프레임 병합 시작...")
            df_all = pd.concat(results, ignore_index=True)
            logger.info(f"병합 완료: {len(df_all)}개 행")
            
            logger.info("다중 포맷 저장 시작...")
            save_data_multi_format(df_all, output_dir, table_name)
            logger.info(f"과거 데이터 수집 완료: {len(df_all)}개 행")
        else:
            logger.warning("수집된 데이터 없음")
        
        logger.info("=== 통합 파이프라인 완료 ===")
        
    except Exception as e:
        logger.error(f"통합 파이프라인 오류: {type(e).__name__}: {str(e)}")
        logger.error(f"스택 트레이스: {traceback.format_exc()}")
        raise
    finally:
        if connection_pool:
            await connection_pool.close()
            logger.info("연결 풀 정리 완료")

# Binance 성능 평가 기준 (KRX와 동일한 기준 적용)
BINANCE_PERFORMANCE_CRITERIA = {
    "excellent": {"min_r2": 0.8, "max_rmse": 0.1, "min_excellent_folds": 3},
    "good": {"min_r2": 0.6, "max_rmse": 0.2, "min_excellent_folds": 2},
    "fair": {"min_r2": 0.4, "max_rmse": 0.3, "min_excellent_folds": 1},
    "poor": {"min_r2": 0.0, "max_rmse": float('inf'), "min_excellent_folds": 0}
}

# Binance 자동매매 가능성 판단 기준
BINANCE_TRADING_CRITERIA = {
    "high_confidence": {
        "min_r2": 0.85,
        "max_rmse": 0.08,
        "min_excellent_folds": 4,
        "max_poor_folds": 0,
        "min_data_quality": 0.9
    },
    "medium_confidence": {
        "min_r2": 0.7,
        "max_rmse": 0.15,
        "min_excellent_folds": 3,
        "max_poor_folds": 1,
        "min_data_quality": 0.8
    },
    "low_confidence": {
        "min_r2": 0.5,
        "max_rmse": 0.25,
        "min_excellent_folds": 2,
        "max_poor_folds": 2,
        "min_data_quality": 0.7
    },
    "not_tradeable": {
        "min_r2": 0.0,
        "max_rmse": float('inf'),
        "min_excellent_folds": 0,
        "max_poor_folds": 5,
        "min_data_quality": 0.0
    }
}

# Binance 데이터 성격별 저장 전략
BINANCE_STORAGE_STRATEGIES = {
    "high_frequency_trading": {
        "storage_format": "parquet",
        "compression": "snappy",
        "partition_by": ["date", "symbol"],
        "retention_days": 30,
        "backup_frequency": "daily",
        "description": "고빈도 거래 - 빠른 읽기/쓰기, 압축 최적화"
    },
    "medium_frequency_analysis": {
        "storage_format": "parquet",
        "compression": "gzip",
        "partition_by": ["month", "symbol"],
        "retention_days": 90,
        "backup_frequency": "weekly",
        "description": "중빈도 분석 - 균형잡힌 성능과 용량"
    },
    "long_term_research": {
        "storage_format": "parquet",
        "compression": "brotli",
        "partition_by": ["year", "symbol"],
        "retention_days": 365,
        "backup_frequency": "monthly",
        "description": "장기 연구 - 최대 압축, 장기 보관"
    },
    "real_time_monitoring": {
        "storage_format": "parquet",
        "compression": "snappy",
        "partition_by": ["hour", "symbol"],
        "retention_days": 7,
        "backup_frequency": "hourly",
        "description": "실시간 모니터링 - 최소 지연, 빠른 처리"
    }
}

# 데이터 유형별 권장 설정 (Binance용)
BINANCE_DATA_TYPE_CONFIGS = {
    "financial_timeseries": {
        "max_iterations": 8,  # 5-10회 중간값
        "max_no_improvement": 3,
        "target_excellent_folds": 3,
        "description": "Binance 금융 시계열 데이터 - 노이즈 많음, 예측 어려움"
    },
    "general_ml": {
        "max_iterations": 4,  # 3-5회 중간값
        "max_no_improvement": 2,
        "target_excellent_folds": 3,
        "description": "일반 ML 데이터 - 안정적 패턴, 빠른 수렴"
    },
    "image_text": {
        "max_iterations": 5,  # 3-7회 중간값
        "max_no_improvement": 2,
        "target_excellent_folds": 3,
        "description": "이미지/텍스트 데이터 - 복잡하지만 패턴 존재"
    },
    "experimental": {
        "max_iterations": 2,  # 2-3회 중간값
        "max_no_improvement": 1,
        "target_excellent_folds": 2,
        "description": "실험적 데이터 - 빠른 검증 필요"
    }
}

def evaluate_binance_performance(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Binance 성능 평가 (KRX와 동일한 기준 적용)"""
    avg_r2 = analysis.get('avg_r2', 0)
    avg_rmse = analysis.get('avg_rmse', float('inf'))
    excellent_folds = analysis.get('excellent_folds', 0)
    poor_folds = analysis.get('poor_folds', 0)
    
    # 성능 등급 평가
    performance_grade = "🔴 Poor"
    if avg_r2 >= BINANCE_PERFORMANCE_CRITERIA["excellent"]["min_r2"] and excellent_folds >= BINANCE_PERFORMANCE_CRITERIA["excellent"]["min_excellent_folds"]:
        performance_grade = "🟢 Excellent"
    elif avg_r2 >= BINANCE_PERFORMANCE_CRITERIA["good"]["min_r2"] and excellent_folds >= BINANCE_PERFORMANCE_CRITERIA["good"]["min_excellent_folds"]:
        performance_grade = "🟡 Good"
    elif avg_r2 >= BINANCE_PERFORMANCE_CRITERIA["fair"]["min_r2"] and excellent_folds >= BINANCE_PERFORMANCE_CRITERIA["fair"]["min_excellent_folds"]:
        performance_grade = "🟠 Fair"
    
    # 자동매매 가능성 판단
    trading_confidence = "not_tradeable"
    data_quality = 1.0 - (poor_folds / (excellent_folds + poor_folds + 1))
    
    for confidence, criteria in BINANCE_TRADING_CRITERIA.items():
        if (avg_r2 >= criteria["min_r2"] and 
            avg_rmse <= criteria["max_rmse"] and
            excellent_folds >= criteria["min_excellent_folds"] and
            poor_folds <= criteria["max_poor_folds"] and
            data_quality >= criteria["min_data_quality"]):
            trading_confidence = confidence
            break
    
    return {
        "performance_grade": performance_grade,
        "trading_confidence": trading_confidence,
        "data_quality_score": data_quality,
        "improvement_needed": performance_grade.startswith("🔴") or trading_confidence == "not_tradeable",
        "trading_recommendation": _get_binance_trading_recommendation(trading_confidence)
    }

def _get_binance_trading_recommendation(confidence: str) -> str:
    """Binance 자동매매 권장사항"""
    recommendations = {
        "high_confidence": "✅ 자동매매 권장 - 높은 신뢰도",
        "medium_confidence": "⚠️ 제한적 자동매매 - 중간 신뢰도",
        "low_confidence": "❌ 자동매매 비권장 - 낮은 신뢰도",
        "not_tradeable": "🚫 자동매매 불가 - 개선 필요"
    }
    return recommendations.get(confidence, "❓ 평가 불가")

def detect_binance_data_characteristics(df: pd.DataFrame) -> str:
    """Binance 데이터 성격 감지"""
    # 데이터 크기 및 빈도 분석
    data_size = len(df)
    time_columns = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
    
    # 거래량 패턴 분석
    volume_columns = [col for col in df.columns if 'volume' in col.lower()]
    has_volume_data = len(volume_columns) > 0
    
    # 가격 변동성 분석
    price_columns = [col for col in df.columns if any(price in col.lower() for price in ['open', 'high', 'low', 'close'])]
    has_price_data = len(price_columns) > 0
    
    # 데이터 성격 판단
    if data_size > 100000 and has_volume_data and has_price_data:
        return "high_frequency_trading"
    elif data_size > 10000 and has_price_data:
        return "medium_frequency_analysis"
    elif data_size > 1000:
        return "long_term_research"
    else:
        return "real_time_monitoring"

def get_binance_storage_strategy(df: pd.DataFrame, trading_confidence: str) -> Dict[str, Any]:
    """Binance 데이터 저장 전략 결정"""
    data_characteristics = detect_binance_data_characteristics(df)
    base_strategy = BINANCE_STORAGE_STRATEGIES[data_characteristics].copy()
    
    # 자동매매 신뢰도에 따른 저장 전략 조정
    if trading_confidence == "high_confidence":
        base_strategy["backup_frequency"] = "hourly"
        base_strategy["retention_days"] = 60
        base_strategy["description"] += " (자동매매 활성화)"
    elif trading_confidence == "medium_confidence":
        base_strategy["backup_frequency"] = "daily"
        base_strategy["retention_days"] = 45
        base_strategy["description"] += " (제한적 자동매매)"
    elif trading_confidence == "low_confidence":
        base_strategy["backup_frequency"] = "weekly"
        base_strategy["retention_days"] = 30
        base_strategy["description"] += " (연구용)"
    else:
        base_strategy["backup_frequency"] = "monthly"
        base_strategy["retention_days"] = 15
        base_strategy["description"] += " (개선 필요)"
    
    return base_strategy

def detect_binance_data_type(df: pd.DataFrame) -> str:
    """Binance 데이터 유형 자동 감지"""
    # Binance 데이터 특성 확인
    binance_indicators = [
        'open', 'high', 'low', 'close', 'volume',
        'quote_asset_volume', 'taker_buy_base_asset_volume'
    ]
    
    has_binance_cols = any(col in df.columns for col in binance_indicators)
    has_time_cols = any('time' in col.lower() for col in df.columns)
    has_symbol_cols = any('symbol' in col.lower() for col in df.columns)
    
    # 데이터 크기 확인
    data_size = len(df)
    feature_count = len(df.select_dtypes(include=[np.number]).columns)
    
    # 데이터 유형 판단
    if has_binance_cols and has_time_cols and has_symbol_cols:
        return "financial_timeseries"
    elif data_size > 10000 and feature_count > 20:
        return "image_text"
    elif data_size < 5000 or feature_count < 10:
        return "experimental"
    else:
        return "general_ml"

def get_binance_optimized_config(df: pd.DataFrame) -> Dict[str, Any]:
    """Binance 데이터 유형에 따른 최적 설정 반환"""
    data_type = detect_binance_data_type(df)
    config = BINANCE_DATA_TYPE_CONFIGS[data_type].copy()
    
    logger.info(f"Binance 데이터 유형 감지: {data_type}")
    logger.info(f"설정 적용: {config['description']}")
    logger.info(f"최대 반복: {config['max_iterations']}회")
    logger.info(f"조기 종료: 연속 {config['max_no_improvement']}회 개선 없음")
    
    return config

# 전역 변수 (우수 등급 달성 추적)
achieved_excellent_grade = False

def np_encoder(obj):
    """numpy 타입 JSON 직렬화용"""
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return str(obj)

if __name__ == "__main__":
    try:
        asyncio.run(run_integrated_pipeline(enable_realtime=False))
    except KeyboardInterrupt:
        logger.info("사용자에 의해 중단됨")
    except Exception as e:
        logger.error(f"프로그램 실행 오류: {type(e).__name__}: {str(e)}")
        logger.error(f"스택 트레이스: {traceback.format_exc()}")
        sys.exit(1) 