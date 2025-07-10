#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: dart_client.py
모듈: DART Open API 비동기 클라이언트
목적: DART 공시/기업정보 수집용 HTTP 클라이언트

Author: AI Assistant
Created: 2025-07-10
Version: 1.0.0

Dependencies:
    - aiohttp>=3.9.0
    - pydantic>=2.11.0
    - structlog>=24.1.0

License: MIT
"""

from __future__ import annotations
import asyncio
import aiohttp
import structlog
from typing import Any, Dict, List, Optional
from pydantic import ValidationError
from domain.dart_models import DartDisclosure

logger = structlog.get_logger(__name__)

class DartApiClient:
    """DART Open API 비동기 클라이언트 (공시/기업정보 수집)"""
    BASE_URL: str = "https://opendart.fss.or.kr/api"
    TIMEOUT: int = 30
    MAX_RETRIES: int = 3
    CONCURRENT_LIMIT: int = 10

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.semaphore = asyncio.Semaphore(self.CONCURRENT_LIMIT)

    async def fetch_disclosure_list(
        self,
        session: aiohttp.ClientSession,
        corp_code: str,
        bgn_de: str,
        end_de: str,
        page_no: int = 1,
        page_count: int = 100
    ) -> Optional[List[DartDisclosure]]:
        """특정 기업, 기간, 페이지의 공시 목록을 비동기로 조회합니다.

        Args:
            session: aiohttp 세션
            corp_code: DART 기업코드
            bgn_de: 시작일자(YYYYMMDD)
            end_de: 종료일자(YYYYMMDD)
            page_no: 페이지 번호
            page_count: 페이지당 건수(최대 100)
        Returns:
            DartDisclosure 리스트 또는 None
        """
        url = f"{self.BASE_URL}/list.json"
        params = {
            "crtfc_key": self.api_key,
            "corp_code": corp_code,
            "bgn_de": bgn_de,
            "end_de": end_de,
            "page_no": page_no,
            "page_count": page_count,
        }
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                async with self.semaphore:
                    async with session.get(url, params=params, timeout=self.TIMEOUT) as resp:
                        data = await resp.json()
                        if data.get("status") != "000":
                            logger.warning(
                                "DART API returned error",
                                corp_code=corp_code,
                                bgn_de=bgn_de,
                                end_de=end_de,
                                page_no=page_no,
                                status=data.get("status"),
                                message=data.get("message"),
                            )
                            return None
                        disclosures = []
                        for item in data.get("list", []):
                            try:
                                disclosure = DartDisclosure(**item)
                                disclosures.append(disclosure)
                            except ValidationError as ve:
                                logger.error(
                                    "DartDisclosure validation error",
                                    error=str(ve),
                                    item=item,
                                )
                        return disclosures
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.warning(
                    "DART API request failed, retrying...",
                    attempt=attempt,
                    error=str(e),
                    corp_code=corp_code,
                    bgn_de=bgn_de,
                    end_de=end_de,
                    page_no=page_no,
                )
                await asyncio.sleep(2 ** attempt)
        logger.error(
            "DART API request failed after retries",
            corp_code=corp_code,
            bgn_de=bgn_de,
            end_de=end_de,
            page_no=page_no,
        )
        return None 