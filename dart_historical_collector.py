#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: dart_historical_collector.py
목적: DART Open API를 활용한 전체 상장사 과거 공시 데이터 수집 및 CSV 저장

Author: Trading AI System
Created: 2025-07-10
Version: 1.0.0

Dependencies:
    - Python 3.11+
    - OpenDartReader>=0.2.3
    - pandas>=2.0.0
    - pydantic>=2.5.0
    - python-dotenv>=1.0.0

License: MIT
"""

from __future__ import annotations

import os
import logging
from typing import List, Optional
from datetime import datetime
from pathlib import Path
import pandas as pd
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv

try:
    import OpenDartReader
except ImportError as e:
    raise ImportError("OpenDartReader 패키지가 설치되어 있지 않습니다. pip install OpenDartReader") from e

# 환경변수 로드
load_dotenv()
DART_API_KEY: str = os.getenv("DART_API_KEY", "")

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class DisclosureModel(BaseModel):
    """공시 데이터 모델"""
    rcept_no: str = Field(..., min_length=1)
    corp_code: str = Field(..., min_length=1)
    corp_name: str = Field(..., min_length=1)
    stock_code: str = Field(default="")
    report_nm: str = Field(..., min_length=1)
    rcept_dt: str = Field(..., min_length=8, max_length=8)
    flr_nm: str = Field(default="")
    rcept_url: str = Field(default="")

class DartHistoricalCollector:
    """DART 전체 상장사 과거 공시 데이터 수집기"""
    def __init__(self, api_key: str, start_year: int = 2015, output_dir: str = "dart_historical_data"):
        if not api_key:
            raise ValueError("DART_API_KEY 환경변수가 설정되어 있지 않습니다.")
        self.api_key = api_key
        self.start_year = start_year
        self.end_year = datetime.now().year
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        OpenDartReader.set_api_key(self.api_key)
        logger.info("DART API 연결 성공")

    def get_corp_list(self) -> pd.DataFrame:
        """전체 상장사 목록 조회 및 저장"""
        corp_list = OpenDartReader.corp_codes()
        corp_df = pd.DataFrame(corp_list)
        corp_csv = self.output_dir / "corp_list.csv"
        corp_df.to_csv(corp_csv, index=False, encoding="utf-8-sig")
        logger.info(f"상장사 목록 저장 완료: {corp_csv} ({len(corp_df)}개)")
        return corp_df

    def collect_all_disclosures(self) -> None:
        """전체 상장사 과거 공시 데이터 수집 및 저장"""
        corp_df = self.get_corp_list()
        total = len(corp_df)
        for idx, row in corp_df.iterrows():
            corp_code = str(row["corp_code"])
            corp_name = str(row["corp_name"])
            try:
                logger.info(f"[{idx+1}/{total}] {corp_name}({corp_code}) 공시 수집 시작")
                self.collect_corp_disclosures(corp_code, corp_name)
            except Exception as e:
                logger.error(f"{corp_name}({corp_code}) 수집 실패: {e}")

    def collect_corp_disclosures(self, corp_code: str, corp_name: str) -> None:
        """개별 기업 과거 공시 데이터 수집 및 저장"""
        start_date = f"{self.start_year}0101"
        end_date = datetime.now().strftime("%Y%m%d")
        try:
            df = OpenDartReader.list(corp_code, start_date, end_date)
        except Exception as e:
            logger.error(f"DART API 조회 실패: {corp_name}({corp_code}): {e}")
            return
        if df is None or df.empty:
            logger.warning(f"{corp_name}({corp_code}) 데이터 없음")
            return
        disclosures: List[dict] = []
        for _, row in df.iterrows():
            try:
                disclosure = DisclosureModel(
                    rcept_no=str(row.get("rcept_no", "") or ""),
                    corp_code=str(row.get("corp_code", "") or ""),
                    corp_name=str(row.get("corp_name", "") or ""),
                    stock_code=str(row.get("stock_code", "") or ""),
                    report_nm=str(row.get("report_nm", "") or ""),
                    rcept_dt=str(row.get("rcept_dt", "") or ""),
                    flr_nm=str(row.get("flr_nm", "") or ""),
                    rcept_url=str(row.get("rcept_url", "") or "")
                )
                disclosures.append(disclosure.dict())
            except ValidationError as ve:
                logger.warning(f"{corp_name}({corp_code}) 데이터 검증 실패: {ve}")
        corp_dir = self.output_dir / corp_code
        corp_dir.mkdir(parents=True, exist_ok=True)
        csv_path = corp_dir / f"{corp_code}_disclosures.csv"
        pd.DataFrame(disclosures).to_csv(csv_path, index=False, encoding="utf-8-sig")
        logger.info(f"{corp_name}({corp_code}) CSV 저장 완료: {csv_path} ({len(disclosures)}건)")

if __name__ == "__main__":
    try:
        collector = DartHistoricalCollector(
            api_key=DART_API_KEY,
            start_year=2015,
            output_dir="dart_historical_data"
        )
        collector.collect_all_disclosures()
    except Exception as e:
        logger.error(f"실행 오류: {e}") 