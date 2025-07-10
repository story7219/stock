#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: dart_filtered_collector.py
모듈: 시가총액 필터링 DART 공시 수집기
목적: 시가총액 2000억원 미만 기업의 전기간 공시 데이터 병렬 수집

Author: AI Assistant
Created: 2025-07-10
Version: 1.0.0

Dependencies:
    - aiohttp>=3.9.0
    - pydantic>=2.11.0
    - structlog>=24.1.0
    - pandas>=2.2.0
    - python-dotenv>=1.0.0

License: MIT
"""

from __future__ import annotations
import os
import sys
import zipfile
import io
import xml.etree.ElementTree as ET
import asyncio
import structlog
import pandas as pd
import requests
from dotenv import load_dotenv
from typing import List, Dict, Set
from domain.dart_models import DartCorpInfo
from service.dart_collector import DartDisclosureCollector

logger = structlog.get_logger(__name__)

CORPCODE_URL = "https://opendart.fss.or.kr/api/corpCode.xml"
MARKET_CAP_LIMIT = 200000000000  # 2000억원 (원 단위)


def get_market_cap_data() -> Dict[str, float]:
    """KRX에서 시가총액 데이터를 수집합니다."""
    logger.info("Collecting market cap data from KRX...")
    
    # KRX API로 시가총액 데이터 수집
    url = "http://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    market_caps = {}
    
    # KOSPI 시가총액
    try:
        kospi_data = {
            'bld': 'dbms/MDC/STAT/standard/MDCSTAT01501',
            'mktId': 'STK',
            'trdDd': pd.Timestamp.now().strftime('%Y%m%d'),
            'share': '1',
            'money': '1',
            'csvxls_isNo': 'false'
        }
        resp = requests.post(url, data=kospi_data, headers=headers, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            for item in data.get('OutBlock_1', []):
                stock_code = item.get('ISU_CD', '').strip()
                market_cap = float(item.get('TDD_CLSPRC', 0)) * float(item.get('LIST_SHRS', 0))
                if stock_code and market_cap > 0:
                    market_caps[stock_code] = market_cap
        logger.info("Collected KOSPI market cap data", count=len(market_caps))
    except Exception as e:
        logger.warning("Failed to collect KOSPI data", error=str(e))
    
    # KOSDAQ 시가총액
    try:
        kosdaq_data = {
            'bld': 'dbms/MDC/STAT/standard/MDCSTAT01501',
            'mktId': 'KSQ',
            'trdDd': pd.Timestamp.now().strftime('%Y%m%d'),
            'share': '1',
            'money': '1',
            'csvxls_isNo': 'false'
        }
        resp = requests.post(url, data=kosdaq_data, headers=headers, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            for item in data.get('OutBlock_1', []):
                stock_code = item.get('ISU_CD', '').strip()
                market_cap = float(item.get('TDD_CLSPRC', 0)) * float(item.get('LIST_SHRS', 0))
                if stock_code and market_cap > 0:
                    market_caps[stock_code] = market_cap
        logger.info("Collected KOSDAQ market cap data", total_count=len(market_caps))
    except Exception as e:
        logger.warning("Failed to collect KOSDAQ data", error=str(e))
    
    return market_caps


def filter_corps_by_market_cap(corps: List[DartCorpInfo], market_caps: Dict[str, float]) -> List[DartCorpInfo]:
    """시가총액 2000억원 미만 기업만 필터링합니다."""
    filtered_corps = []
    matched_count = 0
    
    for corp in corps:
        if corp.stock_code and corp.stock_code.strip():
            stock_code = corp.stock_code.strip()
            market_cap = market_caps.get(stock_code, 0)
            
            if market_cap > 0 and market_cap < MARKET_CAP_LIMIT:
                filtered_corps.append(corp)
                matched_count += 1
                if matched_count <= 10:  # 처음 10개만 로그
                    logger.info(
                        "Matched corp with market cap",
                        corp_name=corp.corp_name,
                        stock_code=stock_code,
                        market_cap=market_cap,
                        market_cap_billion=market_cap / 1000000000
                    )
    
    logger.info(
        "Filtered corps by market cap",
        total_corps=len(corps),
        matched_corps=matched_count,
        filtered_corps=len(filtered_corps),
        market_cap_limit_billion=MARKET_CAP_LIMIT / 1000000000
    )
    
    return filtered_corps


def load_corp_list(api_key: str) -> List[DartCorpInfo]:
    """DART에서 corpCode.xml을 받아 기업 리스트를 파싱합니다."""
    logger.info("Downloading corpCode.xml from DART API...")
    resp = requests.get(f"{CORPCODE_URL}?crtfc_key={api_key}", timeout=60)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        xml_content = z.read("CORPCODE.xml")
    root = ET.fromstring(xml_content)
    corps = []
    for el in root.findall(".//list"):
        try:
            corp = DartCorpInfo(
                corp_code=el.findtext("corp_code", "").strip(),
                corp_name=el.findtext("corp_name", "").strip(),
                stock_code=el.findtext("stock_code", "").strip() or None,
                modify_date=el.findtext("modify_date", "").strip() or None,
            )
            corps.append(corp)
        except Exception as e:
            logger.warning("Failed to parse corp info", error=str(e))
    logger.info("Loaded corp list", count=len(corps))
    return corps


def main() -> None:
    """시가총액 필터링 DART 공시 수집 파이프라인 실행"""
    load_dotenv()
    api_key = os.getenv("DART_API_KEY")
    if not api_key:
        logger.error("DART_API_KEY not found in .env")
        sys.exit(1)
    
    # 1. 시가총액 데이터 수집
    market_caps = get_market_cap_data()
    
    # 2. DART 기업 목록 수집
    all_corps = load_corp_list(api_key)
    
    # 3. 시가총액 필터링
    filtered_corps = filter_corps_by_market_cap(all_corps, market_caps)
    
    if not filtered_corps:
        logger.error("No corps found after market cap filtering")
        sys.exit(1)
    
    # 4. 필터링된 기업으로 공시 수집
    collector = DartDisclosureCollector(api_key, filtered_corps)
    collector.CSV_PATH = "data/historical/dart_disclosures_under_2000b.csv"
    asyncio.run(collector.collect_all())


if __name__ == "__main__":
    main() 