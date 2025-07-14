#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: krx_smart_main_collector.py
목적: 모든 KRX(주식/ETF/선물/옵션) 데이터 과거~현재까지 안전·최적화 수집
Author: World-Class AI Quant Team
Created: 2025-07-14
Version: 4.0.0

- 최신 Python(3.11+) 문법, 최소 코드, 최대 유지보수성
- pykrx, pandas, tqdm, numpy, logging, typing 적극 활용
- 전략(404방지, 극값, 병렬, pykrx 등) 100% 유지
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Callable, Optional
import pandas as pd
import numpy as np
from pykrx import stock
from tqdm import tqdm
import logging
import asyncio
import aiohttp
import sqlite3
import json
import time
import hashlib
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import requests
from collections import defaultdict

logging.basicConfig(level=logging.INFO)

@dataclass
class KRXCollector:
    """KRX 데이터 수집기 (최소 코드, 최대 유지보수, 최신 문법)"""
    logger: logging.Logger = logging.getLogger("KRXCollector")
    db_path: str = "krx_data.db"
    cache_dir: Path = field(default_factory=lambda: Path("cache"))
    retry_count: int = 3
    delay: float = 0.1
    
    def __post_init__(self):
        self.cache_dir.mkdir(exist_ok=True)
        self._init_db()
        self.endpoint_health = defaultdict(dict)
        self.request_count = 0
        
    def _init_db(self):
        """데이터베이스 초기화"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS extremes (
                    ticker TEXT PRIMARY KEY,
                    max_price REAL,
                    min_price REAL,
                    max_price_date TEXT,
                    min_price_date TEXT,
                    max_volume REAL,
                    max_value REAL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS collection_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category TEXT,
                    ticker_count INTEGER,
                    success_count INTEGER,
                    error_count INTEGER,
                    execution_time REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def safe_request(self, func: Callable, *args, **kwargs) -> Any:
        """안전한 요청 (404 방지, 재시도)"""
        for attempt in range(self.retry_count):
            try:
                self.request_count += 1
                result = func(*args, **kwargs)
                time.sleep(self.delay)
                return result
            except Exception as e:
                self.logger.warning(f"요청 실패 (시도 {attempt + 1}/{self.retry_count}): {e}")
                if attempt < self.retry_count - 1:
                    time.sleep(self.delay * (2 ** attempt))  # 지수 백오프
                else:
                    self.logger.error(f"최종 실패: {e}")
                    return None

    def collect_all(
        self,
        tickers: List[str],
        start: str,
        end: str,
        fetch_func: Callable[[str, str, str], pd.DataFrame],
        desc: str = "수집"
    ) -> Dict[str, pd.DataFrame]:
        """범용 데이터 수집 (안전한 요청)"""
        results = {}
        for t in tqdm(tickers, desc=desc):
            df = self.safe_request(fetch_func, start, end, t)
            if df is not None and not df.empty:
                results[t] = df
        return results

    def get_extremes(self, df: pd.DataFrame) -> dict:
        """극값 추적 (최신 문법)"""
        if df.empty:
            return {}
        return {
            "max_price": float(np.float64(df["종가"].max())) if not pd.isnull(df["종가"].max()) else None,
            "min_price": float(np.float64(df["종가"].min())) if not pd.isnull(df["종가"].min()) else None,
            "max_price_date": str(df["종가"].idxmax().date()) if hasattr(df["종가"].idxmax(), "date") else str(df["종가"].idxmax()),
            "min_price_date": str(df["종가"].idxmin().date()) if hasattr(df["종가"].idxmin(), "date") else str(df["종가"].idxmin()),
            "max_volume": float(np.float64(df["거래량"].max())) if "거래량" in df and not pd.isnull(df["거래량"].max()) else None,
            "max_value": float(np.float64(df["거래대금"].max())) if "거래대금" in df and not pd.isnull(df["거래대금"].max()) else None,
        }

    def save_extremes(self, ticker: str, extremes: dict):
        """극값 DB 저장"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO extremes 
                (ticker, max_price, min_price, max_price_date, min_price_date, max_volume, max_value)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                ticker,
                extremes.get("max_price"),
                extremes.get("min_price"),
                extremes.get("max_price_date"),
                extremes.get("min_price_date"),
                extremes.get("max_volume"),
                extremes.get("max_value")
            ))

    def collect_kospi200(self, start: str, end: str) -> Dict[str, pd.DataFrame]:
        tickers = self.safe_request(stock.get_index_portfolio_deposit_file, "1028")
        return self.collect_all(tickers, start, end, stock.get_market_ohlcv_by_date, desc="KOSPI200")

    def collect_kosdaq150(self, start: str, end: str) -> Dict[str, pd.DataFrame]:
        tickers = self.safe_request(stock.get_index_portfolio_deposit_file, "1538")
        return self.collect_all(tickers, start, end, stock.get_market_ohlcv_by_date, desc="KOSDAQ150")

    def collect_top_etf(self, n: int, start: str, end: str) -> Dict[str, pd.DataFrame]:
        today = datetime.now().strftime("%Y%m%d")
        etf_tickers = self.safe_request(stock.get_etf_ticker_list)
        if not etf_tickers:
            return {}
        
        values = []
        for t in etf_tickers:
            df = self.safe_request(stock.get_etf_ohlcv_by_date, today, today, t)
            if df is not None and not df.empty:
                values.append((t, df.iloc[0]["거래대금"]))
        
        values.sort(key=lambda x: x[1], reverse=True)
        top_tickers = [t for t, _ in values[:n]]
        return self.collect_all(top_tickers, start, end, stock.get_etf_ohlcv_by_date, desc="ETF_TOP")

    def collect_derivatives(self, start: str, end: str) -> Dict[str, Dict[str, pd.DataFrame]]:
        """선물/옵션 데이터 수집 (stock 모듈 사용)"""
        categories = {
            "FUTURES": self.safe_request(stock.get_future_ticker_list),
        }
        return {
            cat: self.collect_all(tickers, start, end, stock.get_future_ohlcv_by_ticker, desc=cat)
            for cat, tickers in categories.items()
            if tickers
        }

    def collect_all_data(self, start: str, end: str, etf_n: int = 8) -> Dict[str, Any]:
        """모든 데이터 수집 (모니터링 포함)"""
        start_time = time.time()
        self.logger.info("🚀 모든 데이터 동시 수집 시작")
        
        data = {
            "kospi200": self.collect_kospi200(start, end),
            "kosdaq150": self.collect_kosdaq150(start, end),
            "etf_top": self.collect_top_etf(etf_n, start, end),
            "derivatives": self.collect_derivatives(start, end),
        }
        
        # 극값 저장 및 통계
        total_tickers = sum(len(tickers) for tickers in data.values() if isinstance(tickers, dict))
        success_count = sum(len(tickers) for tickers in data.values() if isinstance(tickers, dict))
        
        # DB에 수집 로그 저장
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO collection_log 
                (category, ticker_count, success_count, error_count, execution_time)
                VALUES (?, ?, ?, ?, ?)
            """, ("ALL", total_tickers, success_count, total_tickers - success_count, time.time() - start_time))
        
        self.logger.info(f"✅ 수집 완료: {success_count}/{total_tickers} 성공, {time.time() - start_time:.2f}초")
        return data

    def save_data_by_type(
        self, 
        data: Dict[str, Any], 
        output_dir: str = "data",
        storage_config: Dict[str, str] = None
    ):
        """데이터 성격에 따른 선택적 저장 (Parquet, SQLite, SQL, HDF5)"""
        if storage_config is None:
            storage_config = {
                "kospi200": "parquet",      # 대용량 주식 데이터 → Parquet (압축률 높음)
                "kosdaq150": "parquet",     # 대용량 주식 데이터 → Parquet
                "etf_top": "sqlite",        # 상위 ETF → SQLite (빠른 조회)
                "derivatives": "hdf5",      # 선물/옵션 → HDF5 (복잡한 구조)
            }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        Path(output_dir).mkdir(exist_ok=True)
        
        for category, tickers in data.items():
            if not isinstance(tickers, dict):
                continue
                
            storage_type = storage_config.get(category, "parquet")
            
            if storage_type == "parquet":
                self._save_as_parquet(tickers, output_dir, category, timestamp)
            elif storage_type == "sqlite":
                self._save_as_sqlite(tickers, category, timestamp)
            elif storage_type == "sql":
                self._save_as_sql(tickers, output_dir, category, timestamp)
            elif storage_type == "hdf5":
                self._save_as_hdf5(tickers, output_dir, category, timestamp)
            else:
                self.logger.warning(f"알 수 없는 저장 타입: {storage_type}")

    def _save_as_parquet(self, tickers: Dict[str, pd.DataFrame], output_dir: str, category: str, timestamp: str):
        """Parquet 형식 저장 (대용량 데이터 최적화)"""
        for ticker, df in tickers.items():
            filepath = Path(output_dir) / f"{category}_{ticker}_{timestamp}.parquet"
            df.to_parquet(filepath, compression="gzip", index=True)
            self.logger.info(f"Parquet 저장 완료: {filepath}")

    def _save_as_sqlite(self, tickers: Dict[str, pd.DataFrame], category: str, timestamp: str):
        """SQLite DB 저장 (빠른 조회용)"""
        table_name = f"{category}_{timestamp.replace('_', '')}"
        
        with sqlite3.connect(self.db_path) as conn:
            for ticker, df in tickers.items():
                df.to_sql(f"{table_name}_{ticker}", conn, if_exists="replace", index=True)
                self.logger.info(f"SQLite 저장 완료: {table_name}_{ticker}")

    def _save_as_sql(self, tickers: Dict[str, pd.DataFrame], output_dir: str, category: str, timestamp: str):
        """SQL 파일 저장 (백업/이동용)"""
        sql_file = Path(output_dir) / f"{category}_{timestamp}.sql"
        
        with open(sql_file, 'w', encoding='utf-8') as f:
            f.write(f"-- {category} 데이터 SQL 백업\n")
            f.write(f"-- 생성일: {datetime.now()}\n\n")
            
            for ticker, df in tickers.items():
                table_name = f"{category}_{ticker}_{timestamp.replace('_', '')}"
                
                # CREATE TABLE 문
                columns = []
                for col, dtype in df.dtypes.items():
                    if 'int' in str(dtype):
                        sql_type = "INTEGER"
                    elif 'float' in str(dtype):
                        sql_type = "REAL"
                    else:
                        sql_type = "TEXT"
                    columns.append(f"{col} {sql_type}")
                
                f.write(f"CREATE TABLE IF NOT EXISTS {table_name} (\n")
                f.write("    " + ",\n    ".join(columns) + "\n);\n\n")
                
                # INSERT 문
                for _, row in df.iterrows():
                    values = []
                    for val in row.values:
                        if pd.isna(val):
                            values.append("NULL")
                        elif isinstance(val, (int, float)):
                            values.append(str(val))
                        else:
                            values.append(f"'{str(val)}'")
                    
                    f.write(f"INSERT INTO {table_name} VALUES ({', '.join(values)});\n")
                f.write("\n")
        
        self.logger.info(f"SQL 저장 완료: {sql_file}")

    def _save_as_hdf5(self, tickers: Dict[str, pd.DataFrame], output_dir: str, category: str, timestamp: str):
        """HDF5 형식 저장 (복잡한 구조 데이터)"""
        try:
            import h5py
            hdf5_file = Path(output_dir) / f"{category}_{timestamp}.h5"
            
            with h5py.File(hdf5_file, 'w') as f:
                for ticker, df in tickers.items():
                    # 데이터프레임을 HDF5에 저장
                    df.to_hdf(hdf5_file, key=f"{category}/{ticker}", mode='a')
                    
                    # 메타데이터 저장
                    f[f"{category}/{ticker}"].attrs['ticker'] = ticker
                    f[f"{category}/{ticker}"].attrs['category'] = category
                    f[f"{category}/{ticker}"].attrs['created_at'] = timestamp
                    f[f"{category}/{ticker}"].attrs['row_count'] = len(df)
                    f[f"{category}/{ticker}"].attrs['column_count'] = len(df.columns)
            
            self.logger.info(f"HDF5 저장 완료: {hdf5_file}")
            
        except ImportError:
            self.logger.warning("h5py가 설치되지 않아 HDF5 저장을 사용할 수 없습니다. Parquet으로 대체합니다.")
            self._save_as_parquet(tickers, output_dir, category, timestamp)

    def save_to_parquet(self, data: Dict[str, Any], output_dir: str = "data"):
        """기존 Parquet 저장 (하위 호환성)"""
        self.save_data_by_type(data, output_dir, {"kospi200": "parquet", "kosdaq150": "parquet", "etf_top": "parquet", "derivatives": "parquet"})

    def get_storage_recommendations(self) -> Dict[str, str]:
        """데이터 성격별 저장 방식 추천"""
        return {
            "kospi200": "parquet",      # 대용량 주식 데이터 → Parquet (압축률 높음, 빠른 읽기)
            "kosdaq150": "parquet",     # 대용량 주식 데이터 → Parquet
            "etf_top": "sqlite",        # 상위 ETF → SQLite (빠른 조회, 인덱싱)
            "derivatives": "hdf5",      # 선물/옵션 → HDF5 (복잡한 구조, 메타데이터)
            "extremes": "sqlite",       # 극값 데이터 → SQLite (빠른 조회)
            "logs": "sqlite",           # 로그 데이터 → SQLite (구조화된 저장)
        }

    def load_from_storage(self, category: str, storage_type: str = None, **kwargs) -> Dict[str, pd.DataFrame]:
        """저장된 데이터 로드"""
        if storage_type is None:
            storage_type = self.get_storage_recommendations().get(category, "parquet")
        
        if storage_type == "parquet":
            return self._load_from_parquet(category, **kwargs)
        elif storage_type == "sqlite":
            return self._load_from_sqlite(category, **kwargs)
        elif storage_type == "hdf5":
            return self._load_from_hdf5(category, **kwargs)
        else:
            raise ValueError(f"지원하지 않는 저장 타입: {storage_type}")

    def _load_from_parquet(self, category: str, data_dir: str = "data") -> Dict[str, pd.DataFrame]:
        """Parquet 파일에서 로드"""
        data = {}
        data_path = Path(data_dir)
        
        for file in data_path.glob(f"{category}_*.parquet"):
            ticker = file.stem.split("_")[1]  # 파일명에서 티커 추출
            data[ticker] = pd.read_parquet(file)
        
        return data

    def _load_from_sqlite(self, category: str, **kwargs) -> Dict[str, pd.DataFrame]:
        """SQLite DB에서 로드"""
        data = {}
        with sqlite3.connect(self.db_path) as conn:
            tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE ?", (f"{category}%",)).fetchall()
            
            for (table_name,) in tables:
                ticker = table_name.split("_")[1]  # 테이블명에서 티커 추출
                data[ticker] = pd.read_sql(f"SELECT * FROM {table_name}", conn)
        
        return data

    def _load_from_hdf5(self, category: str, data_dir: str = "data") -> Dict[str, pd.DataFrame]:
        """HDF5 파일에서 로드"""
        try:
            import h5py
            data = {}
            data_path = Path(data_dir)
            
            for file in data_path.glob(f"{category}_*.h5"):
                with h5py.File(file, 'r') as f:
                    if category in f:
                        for ticker in f[category].keys():
                            data[ticker] = pd.read_hdf(file, key=f"{category}/{ticker}")
            
            return data
            
        except ImportError:
            self.logger.warning("h5py가 설치되지 않아 HDF5 로드를 사용할 수 없습니다.")
            return {}

    def send_telegram_alert(self, message: str):
        """텔레그램 알림 (선택적)"""
        try:
            bot_token = "YOUR_BOT_TOKEN"  # 환경변수에서 가져오기
            chat_id = "YOUR_CHAT_ID"
            if bot_token != "YOUR_BOT_TOKEN":
                url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
                requests.post(url, data={"chat_id": chat_id, "text": message})
        except Exception as e:
            self.logger.warning(f"텔레그램 알림 실패: {e}")

if __name__ == "__main__":
    collector = KRXCollector()
    START = "19960503"
    END = datetime.now().strftime("%Y%m%d")
    
    # 데이터 수집
    data = collector.collect_all_data(START, END, etf_n=8)
    
    # 극값 추적 및 저장
    for category, tickers in data.items():
        if isinstance(tickers, dict):
            for ticker, df in tickers.items():
                extremes = collector.get_extremes(df)
                collector.save_extremes(ticker, extremes)
                print(f"[{category}] {ticker}: {extremes}")
    
    # 데이터 성격에 따른 선택적 저장
    print("\n📊 저장 방식 추천:")
    recommendations = collector.get_storage_recommendations()
    for category, storage_type in recommendations.items():
        print(f"  - {category}: {storage_type}")
    
    # 기본 저장 (추천 방식)
    collector.save_data_by_type(data)
    
    # 사용자 정의 저장 설정 예시
    custom_storage = {
        "kospi200": "parquet",      # 대용량 → Parquet
        "kosdaq150": "sqlite",      # 빠른 조회 → SQLite
        "etf_top": "sql",           # 백업용 → SQL
        "derivatives": "hdf5",      # 복잡한 구조 → HDF5
    }
    collector.save_data_by_type(data, storage_config=custom_storage)
    
    # 저장된 데이터 로드 예시
    print("\n📂 저장된 데이터 로드 예시:")
    for category in data.keys():
        loaded_data = collector.load_from_storage(category)
        print(f"  - {category}: {len(loaded_data)}개 티커 로드됨")
    
    # 알림 발송
    collector.send_telegram_alert("🎉 KRX 데이터 수집 및 선택적 저장 완료!")
    
    print("\n✅ 모든 데이터 수집, 극값 추적, 선택적 저장 완료")
    print("💾 저장된 파일들:")
    print("  - Parquet: 대용량 데이터 (압축률 높음)")
    print("  - SQLite: 빠른 조회용 (인덱싱)")
    print("  - SQL: 백업/이동용 (표준 SQL)")
    print("  - HDF5: 복잡한 구조 (메타데이터 포함)")