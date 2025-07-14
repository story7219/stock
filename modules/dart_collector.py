from datetime import datetime
import timedelta
from __future__ import annotations
from core.dart_client import DartApiClient
from domain.dart_models import DartDisclosure
import DartCorpInfo
from typing import List
import Optional
import Dict, Any
import aiohttp
import asyncio
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

class DartDisclosureCollector:
    START_YEAR: int = 1999
    END_YEAR: int = 2025  # 필요시 동적으로 변경
    BATCH_MONTHS: int = 3  # 3개월 단위 요청
    CSV_PATH: str = "data/historical/dart_all_disclosures.csv"
    CSV_ENCODING: str = "utf-8-sig"

    def __init__(self, api_key: str, corp_list: List[DartCorpInfo]) -> None:
        self.api_key = api_key
        self.corp_list = corp_list
        self.client = DartApiClient(api_key)

    @staticmethod
    def _date_range(start_year: int, end_year: int, batch_months: int) -> List[Dict[str, str]]:
        ranges = []
        for year in range(start_year, end_year + 1):
            for month in range(1, 13, batch_months):
                bgn = datetime(year, month, 1)
                end_month = min(month + batch_months - 1, 12)
                if end_month == 12:
                    end = datetime(year, 12, 31)
                else:
                    end = datetime(year, end_month + 1, 1) - timedelta(days=1)
                ranges.append({
                    "bgn_de": bgn.strftime("%Y%m%d"),
                    "end_de": end.strftime("%Y%m%d")
                })
        return ranges

    async def collect_all(self) -> None:
        all_disclosures: List[Dict[str, Any]] = []
        async with aiohttp.ClientSession() as session:
            tasks = []
            for corp in self.corp_list:
                for dr in self._date_range(self.START_YEAR, self.END_YEAR, self.BATCH_MONTHS):
                    tasks.append(self._collect_corp_period(session, corp, dr["bgn_de"], dr["end_de"]))
            logger.info("Starting DART disclosure collection", total_tasks=len(tasks))
            for future in asyncio.as_completed(tasks):
                result = await future
                if result:
                    all_disclosures.extend(result)
        logger.info("DART disclosure collection complete", total=len(all_disclosures))
        self._save_to_csv(all_disclosures)

    async def _collect_corp_period(self, session: aiohttp.ClientSession, corp: DartCorpInfo, bgn_de: str, end_de: str) -> Optional[List[Dict[str, Any]]]:
        disclosures = []
        page_no = 1
        while True:
            result = await self.client.fetch_disclosure_list(session, corp.corp_code, bgn_de, end_de, page_no=page_no)
            if not result:
                break
            for d in result:
                disclosures.append(d.model_dump())
            if len(result) < 100:
                break
            page_no += 1
        if disclosures:
            logger.info("Collected disclosures", corp_code=corp.corp_code, bgn_de=bgn_de, end_de=end_de, count=len(disclosures))
        return disclosures if disclosures else None

    def _save_to_csv(self, disclosures: List[Dict[str, Any]]) -> None:
        if not disclosures:
            logger.warning("No disclosures to save.")
            return
        df = pd.DataFrame(disclosures)
        df.to_csv(self.CSV_PATH, index=False, encoding=self.CSV_ENCODING)
        logger.info("Saved disclosures to CSV", path=self.CSV_PATH, rows=len(df))
