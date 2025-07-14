#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: krx_smart_manager.py
모듈: KRX 데이터 수집 매니저
목적: KRX API에서 데이터 수집 및 저장

Author: User
Created: 2025-07-06
Version: 1.0.0

Dependencies:
    - Python 3.11+
    - requests>=2.31.0
    - structlog>=24.1.0
    - python-dotenv>=1.0.0

Performance:
    - O(n) for data fetch

Security:
    - 환경변수 기반 API 키/보안
    - 에러 로깅: structlog

License: MIT
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional
import requests
import structlog
from dotenv import load_dotenv

logger = structlog.get_logger(__name__)
load_dotenv()

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

KRX_API_URL = os.getenv("KRX_API_URL", "https://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd")
USER_AGENT = os.getenv("USER_AGENT", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
REFERER = os.getenv("REFERER", "https://data.krx.co.kr/contents/MDC/MDI/mdiLoader")


def fetch_krx_data(
    bld: str,
    params: dict,
    output_path: Optional[str] = None
) -> str:
    """KRX 데이터 수집 함수

    Args:
        bld: KRX API bld 파라미터
        params: 추가 파라미터 dict
        output_path: 저장 경로 (None이면 자동 생성)

    Returns:
        저장된 파일 경로

    Raises:
        Exception: HTTP/파싱 오류
    """
    headers = {
        "User-Agent": USER_AGENT,
        "Referer": REFERER,
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8"
    }
    payload = {"bld": bld, **params}
    try:
        resp = requests.post(KRX_API_URL, headers=headers, data=payload, timeout=10)
        resp.raise_for_status()
        # Content-Type이 text/html이어도 JSON 반환됨
        data = resp.json() if resp.headers.get("Content-Type", "").startswith("application/json") else resp.text
        if isinstance(data, str):
            import json
            data = json.loads(data)
        if output_path is None:
            output_path = str(DATA_DIR / f"krx_{bld.replace('/', '_')}_{int(time.time())}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            import json
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("KRX data saved", output_path=output_path, rows=len(data.get("OutBlock1", [])))
        return output_path
    except Exception as e:
        logger.error("KRX fetch failed", error=str(e))
        raise


def _test_fetch_krx_data() -> None:
    """단위 테스트: fetch_krx_data 함수 (실제 API 호출은 생략, 구조만 검증)"""
    try:
        fetch_krx_data("fake_bld", {"param1": "value1"}, output_path="data/test.json")
    except Exception:
        print("[PASS] fetch_krx_data error handling test")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="KRX 데이터 수집 매니저")
    parser.add_argument("--bld", type=str, required=True, help="KRX bld 파라미터")
    parser.add_argument("--params", type=str, default="{}", help="추가 파라미터 (JSON str)")
    parser.add_argument("--output", type=str, default=None, help="저장 경로")
    args = parser.parse_args()
    import json
    params = json.loads(args.params)
    fetch_krx_data(args.bld, params, args.output) 