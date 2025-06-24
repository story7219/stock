#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                      🚀 Smart Data Storage System v1.0                      ║
║                     구글시트 기반 효율적 데이터 저장소                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  • 📊 실시간 데이터 캐싱 및 저장                                             ║
║  • 🔄 스마트 데이터 불러오기 (필요시만)                                      ║
║  • 💾 히스토리 데이터 관리                                                   ║
║  • ⚡ 초고속 조회 시스템                                                     ║
║  • 🎯 AI 분석 최적화 데이터셋                                                ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass, asdict
from dotenv import load_dotenv
import gspread
from google.oauth2.service_account import Credentials
import asyncio
from concurrent.futures import ThreadPoolExecutor
import pickle
import hashlib
import time
from pathlib import Path

# 환경 변수 로드
load_dotenv()


@dataclass
class CacheEntry:
    """캐시 엔트리 클래스"""

    data: Any
    timestamp: datetime
    ttl: int  # seconds
    data_hash: str
    access_count: int = 0
    last_access: datetime = None

    def is_expired(self) -> bool:
        """캐시 만료 여부 확인"""
        return (datetime.now() - self.timestamp).total_seconds() > self.ttl

    def is_valid(self) -> bool:
        """캐시 유효성 확인"""
        return not self.is_expired()


@dataclass
class DataRequest:
    """데이터 요청 클래스"""

    query_type: str
    parameters: Dict[str, Any]
    priority: int = 1  # 1=high, 2=medium, 3=low
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class SmartDataStorage:
    """스마트 데이터 저장소 - 구글시트 기반"""

    def __init__(self):
        """초기화"""
        self.logger = self._setup_logger()

        # 캐시 설정
        self.cache_dir = Path("data_cache")
        self.cache_dir.mkdir(exist_ok=True)

        # 메모리 캐시
        self.memory_cache: Dict[str, CacheEntry] = {}
        self.max_memory_cache_size = 100  # 최대 100개 항목

        # 디스크 캐시
        self.disk_cache_dir = self.cache_dir / "disk_cache"
        self.disk_cache_dir.mkdir(exist_ok=True)

        # 구글 시트 설정
        self.credentials_path = os.getenv(
            "GOOGLE_SHEETS_CREDENTIALS_PATH", "credentials.json"
        )
        self.spreadsheet_id = os.getenv("GOOGLE_SHEETS_SPREADSHEET_ID")

        # 캐시 TTL 설정 (초)
        self.cache_ttl = {
            "stock_data": 300,  # 5분
            "analysis_results": 1800,  # 30분
            "market_data": 600,  # 10분
            "news_data": 900,  # 15분
            "historical_data": 86400,  # 24시간
            "dashboard_data": 60,  # 1분
        }

        # 구글 시트 클라이언트
        self.sheets_client = None
        self.spreadsheet = None
        self.executor = ThreadPoolExecutor(max_workers=5)

        # 통계
        self.stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "sheets_reads": 0,
            "sheets_writes": 0,
            "data_efficiency": 0.0,
        }

        self._initialize_sheets_client()
        self._load_disk_cache_index()

        self.logger.info("🚀 Smart Data Storage 초기화 완료")

    def _setup_logger(self) -> logging.Logger:
        """로거 설정"""
        logger = logging.getLogger("SmartDataStorage")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            # 로그 디렉토리 생성
            os.makedirs("logs", exist_ok=True)

            # 파일 핸들러
            file_handler = logging.FileHandler(
                "logs/smart_storage.log", encoding="utf-8"
            )
            file_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

            # 콘솔 핸들러
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s"
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

        return logger

    def _initialize_sheets_client(self):
        """구글 시트 클라이언트 초기화"""
        try:
            if not self.credentials_path or not os.path.exists(self.credentials_path):
                self.logger.warning("⚠️ 구글 인증 파일이 없습니다")
                return

            if not self.spreadsheet_id:
                self.logger.warning("⚠️ 스프레드시트 ID가 설정되지 않았습니다")
                return

            # 인증 설정
            scopes = [
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive",
            ]

            credentials = Credentials.from_service_account_file(
                self.credentials_path, scopes=scopes
            )

            self.sheets_client = gspread.authorize(credentials)
            self.spreadsheet = self.sheets_client.open_by_key(self.spreadsheet_id)

            self.logger.info("✅ 구글 시트 클라이언트 초기화 완료")

        except Exception as e:
            self.logger.error(f"❌ 구글 시트 초기화 실패: {e}")
            self.sheets_client = None
            self.spreadsheet = None

    def _load_disk_cache_index(self):
        """디스크 캐시 인덱스 로드"""
        try:
            index_file = self.disk_cache_dir / "cache_index.json"
            if index_file.exists():
                with open(index_file, "r", encoding="utf-8") as f:
                    self.disk_cache_index = json.load(f)
            else:
                self.disk_cache_index = {}
        except Exception as e:
            self.logger.warning(f"⚠️ 디스크 캐시 인덱스 로드 실패: {e}")
            self.disk_cache_index = {}

    def _save_disk_cache_index(self):
        """디스크 캐시 인덱스 저장"""
        try:
            index_file = self.disk_cache_dir / "cache_index.json"
            with open(index_file, "w", encoding="utf-8") as f:
                json.dump(self.disk_cache_index, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.warning(f"⚠️ 디스크 캐시 인덱스 저장 실패: {e}")

    def _generate_cache_key(self, data_type: str, parameters: Dict[str, Any]) -> str:
        """캐시 키 생성"""
        key_data = f"{data_type}:{json.dumps(parameters, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _calculate_data_hash(self, data: Any) -> str:
        """데이터 해시 계산"""
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()

    async def get_data(
        self,
        data_type: str,
        parameters: Dict[str, Any] = None,
        force_refresh: bool = False,
    ) -> Optional[Any]:
        """스마트 데이터 조회 - 캐시 우선, 필요시 구글시트에서 조회"""
        if parameters is None:
            parameters = {}

        cache_key = self._generate_cache_key(data_type, parameters)

        # 강제 새로고침이 아닌 경우 캐시 확인
        if not force_refresh:
            # 1. 메모리 캐시 확인
            if cache_key in self.memory_cache:
                entry = self.memory_cache[cache_key]
                if entry.is_valid():
                    entry.access_count += 1
                    entry.last_access = datetime.now()
                    self.stats["cache_hits"] += 1
                    self.logger.debug(f"💾 메모리 캐시 히트: {data_type}")
                    return entry.data
                else:
                    # 만료된 캐시 제거
                    del self.memory_cache[cache_key]

            # 2. 디스크 캐시 확인
            disk_data = await self._load_from_disk_cache(cache_key, data_type)
            if disk_data is not None:
                # 메모리 캐시에도 저장
                self._store_in_memory_cache(cache_key, disk_data, data_type)
                self.stats["cache_hits"] += 1
                self.logger.debug(f"💿 디스크 캐시 히트: {data_type}")
                return disk_data

        # 3. 구글 시트에서 데이터 조회
        self.stats["cache_misses"] += 1
        self.logger.info(f"🔍 구글시트에서 데이터 조회: {data_type}")

        sheet_data = await self._fetch_from_sheets(data_type, parameters)

        if sheet_data is not None:
            # 캐시에 저장
            await self._store_data(cache_key, sheet_data, data_type)
            self.stats["sheets_reads"] += 1
            return sheet_data

        return None

    async def store_data(
        self, data_type: str, data: Any, parameters: Dict[str, Any] = None
    ) -> bool:
        """데이터 저장 - 구글시트 + 캐시"""
        if parameters is None:
            parameters = {}

        try:
            # 1. 구글 시트에 저장
            success = await self._save_to_sheets(data_type, data, parameters)

            if success:
                # 2. 캐시에도 저장
                cache_key = self._generate_cache_key(data_type, parameters)
                await self._store_data(cache_key, data, data_type)
                self.stats["sheets_writes"] += 1

                self.logger.info(f"✅ 데이터 저장 완료: {data_type}")
                return True

            return False

        except Exception as e:
            self.logger.error(f"❌ 데이터 저장 실패: {e}")
            return False

    async def _fetch_from_sheets(
        self, data_type: str, parameters: Dict[str, Any]
    ) -> Optional[Any]:
        """구글 시트에서 데이터 조회"""
        if not self.spreadsheet:
            return None

        try:
            # 데이터 타입별 시트 매핑
            sheet_mapping = {
                "stock_data": "주식데이터",
                "analysis_results": "AI분석결과",
                "korean_market_top5": "한국시장TOP5",
                "us_market_top5": "미국시장TOP5",
                "strategy_summary": "전략요약",
                "master_recommendation": "마스터추천",
                "daily_summary": "일일요약",
                "dashboard_data": "대시보드",
            }

            sheet_name = sheet_mapping.get(data_type)
            if not sheet_name:
                return None

            # 비동기로 시트 데이터 조회
            loop = asyncio.get_event_loop()
            worksheet = await loop.run_in_executor(
                self.executor, self.spreadsheet.worksheet, sheet_name
            )

            # 파라미터에 따른 필터링
            if parameters.get("limit"):
                records = await loop.run_in_executor(
                    self.executor,
                    lambda: worksheet.get_all_records()[: parameters["limit"]],
                )
            elif parameters.get("date_from"):
                # 날짜 범위 필터링
                all_records = await loop.run_in_executor(
                    self.executor, worksheet.get_all_records
                )

                date_from = datetime.strptime(parameters["date_from"], "%Y-%m-%d")
                records = [
                    record
                    for record in all_records
                    if datetime.strptime(record.get("날짜", "1900-01-01"), "%Y-%m-%d")
                    >= date_from
                ]
            else:
                records = await loop.run_in_executor(
                    self.executor, worksheet.get_all_records
                )

            return records

        except Exception as e:
            self.logger.error(f"❌ 구글시트 조회 실패: {e}")
            return None

    async def _save_to_sheets(
        self, data_type: str, data: Any, parameters: Dict[str, Any]
    ) -> bool:
        """구글 시트에 데이터 저장"""
        if not self.spreadsheet:
            return False

        try:
            # 여기서는 기존 GoogleSheetsManager를 활용
            from google_sheets_manager import GoogleSheetsManager

            sheets_manager = GoogleSheetsManager()

            if data_type == "stock_data":
                return await sheets_manager.save_stock_data(data)
            elif data_type == "analysis_results":
                return await sheets_manager.update_analysis_results(data)
            elif data_type == "daily_summary":
                return await sheets_manager.save_daily_summary(data)

            return True

        except Exception as e:
            self.logger.error(f"❌ 구글시트 저장 실패: {e}")
            return False

    def _store_in_memory_cache(self, cache_key: str, data: Any, data_type: str):
        """메모리 캐시에 저장"""
        try:
            # 메모리 캐시 크기 제한
            if len(self.memory_cache) >= self.max_memory_cache_size:
                # LRU 방식으로 오래된 항목 제거
                oldest_key = min(
                    self.memory_cache.keys(),
                    key=lambda k: self.memory_cache[k].last_access or datetime.min,
                )
                del self.memory_cache[oldest_key]

            ttl = self.cache_ttl.get(data_type, 300)
            data_hash = self._calculate_data_hash(data)

            entry = CacheEntry(
                data=data,
                timestamp=datetime.now(),
                ttl=ttl,
                data_hash=data_hash,
                access_count=1,
                last_access=datetime.now(),
            )

            self.memory_cache[cache_key] = entry

        except Exception as e:
            self.logger.warning(f"⚠️ 메모리 캐시 저장 실패: {e}")

    async def _load_from_disk_cache(
        self, cache_key: str, data_type: str
    ) -> Optional[Any]:
        """디스크 캐시에서 로드"""
        try:
            cache_file = self.disk_cache_dir / f"{cache_key}.pkl"

            if not cache_file.exists():
                return None

            # 파일 수정 시간 확인
            file_mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
            ttl = self.cache_ttl.get(data_type, 300)

            if (datetime.now() - file_mtime).total_seconds() > ttl:
                # 만료된 캐시 파일 삭제
                cache_file.unlink()
                return None

            # 데이터 로드
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(
                self.executor, self._load_pickle_file, cache_file
            )

            return data

        except Exception as e:
            self.logger.warning(f"⚠️ 디스크 캐시 로드 실패: {e}")
            return None

    def _load_pickle_file(self, file_path: Path) -> Any:
        """피클 파일 로드 (동기 함수)"""
        with open(file_path, "rb") as f:
            return pickle.load(f)

    async def _store_data(self, cache_key: str, data: Any, data_type: str):
        """캐시에 데이터 저장 (메모리 + 디스크)"""
        # 메모리 캐시
        self._store_in_memory_cache(cache_key, data, data_type)

        # 디스크 캐시
        await self._store_in_disk_cache(cache_key, data, data_type)

    async def _store_in_disk_cache(self, cache_key: str, data: Any, data_type: str):
        """디스크 캐시에 저장"""
        try:
            cache_file = self.disk_cache_dir / f"{cache_key}.pkl"

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.executor, self._save_pickle_file, cache_file, data
            )

            # 인덱스 업데이트
            self.disk_cache_index[cache_key] = {
                "data_type": data_type,
                "timestamp": datetime.now().isoformat(),
                "file_path": str(cache_file),
            }

            self._save_disk_cache_index()

        except Exception as e:
            self.logger.warning(f"⚠️ 디스크 캐시 저장 실패: {e}")

    def _save_pickle_file(self, file_path: Path, data: Any):
        """피클 파일 저장 (동기 함수)"""
        with open(file_path, "wb") as f:
            pickle.dump(data, f)

    async def get_latest_stock_data(
        self, symbols: List[str] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """최신 주식 데이터 조회"""
        parameters = {"limit": limit}
        if symbols:
            parameters["symbols"] = symbols

        return await self.get_data("stock_data", parameters)

    async def get_analysis_results(
        self, date_from: str = None, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """AI 분석 결과 조회"""
        parameters = {"limit": limit}
        if date_from:
            parameters["date_from"] = date_from

        return await self.get_data("analysis_results", parameters)

    async def get_market_top5(self, market: str = "korean") -> List[Dict[str, Any]]:
        """시장별 Top5 조회"""
        data_type = f"{market}_market_top5"
        return await self.get_data(data_type, {"limit": 5})

    async def get_dashboard_data(self) -> Dict[str, Any]:
        """대시보드 데이터 조회"""
        return await self.get_data("dashboard_data", {})

    def get_cache_stats(self) -> Dict[str, Any]:
        """캐시 통계 조회"""
        total_requests = self.stats["cache_hits"] + self.stats["cache_misses"]
        hit_rate = (
            (self.stats["cache_hits"] / total_requests * 100)
            if total_requests > 0
            else 0
        )

        return {
            "cache_hits": self.stats["cache_hits"],
            "cache_misses": self.stats["cache_misses"],
            "hit_rate": f"{hit_rate:.1f}%",
            "sheets_reads": self.stats["sheets_reads"],
            "sheets_writes": self.stats["sheets_writes"],
            "memory_cache_size": len(self.memory_cache),
            "disk_cache_size": len(self.disk_cache_index),
            "efficiency_score": hit_rate,
        }

    async def cleanup_cache(self, max_age_hours: int = 24):
        """캐시 정리"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)

            # 메모리 캐시 정리
            expired_keys = [
                key
                for key, entry in self.memory_cache.items()
                if entry.timestamp < cutoff_time
            ]

            for key in expired_keys:
                del self.memory_cache[key]

            # 디스크 캐시 정리
            expired_files = []
            for cache_key, info in self.disk_cache_index.items():
                timestamp = datetime.fromisoformat(info["timestamp"])
                if timestamp < cutoff_time:
                    file_path = Path(info["file_path"])
                    if file_path.exists():
                        file_path.unlink()
                    expired_files.append(cache_key)

            for key in expired_files:
                del self.disk_cache_index[key]

            self._save_disk_cache_index()

            self.logger.info(
                f"🧹 캐시 정리 완료: 메모리 {len(expired_keys)}개, 디스크 {len(expired_files)}개 삭제"
            )

        except Exception as e:
            self.logger.error(f"❌ 캐시 정리 실패: {e}")

    async def prefetch_data(self, data_requests: List[DataRequest]):
        """데이터 미리 가져오기 (백그라운드)"""
        try:
            # 우선순위별 정렬
            sorted_requests = sorted(data_requests, key=lambda x: x.priority)

            tasks = []
            for request in sorted_requests:
                task = self.get_data(request.query_type, request.parameters)
                tasks.append(task)

            # 병렬 실행
            await asyncio.gather(*tasks, return_exceptions=True)

            self.logger.info(f"🚀 데이터 프리페치 완료: {len(data_requests)}개 요청")

        except Exception as e:
            self.logger.error(f"❌ 데이터 프리페치 실패: {e}")

    async def close(self):
        """리소스 정리"""
        try:
            # 스레드풀 종료
            self.executor.shutdown(wait=True)

            # 캐시 인덱스 저장
            self._save_disk_cache_index()

            self.logger.info("✅ Smart Data Storage 종료 완료")

        except Exception as e:
            self.logger.error(f"❌ 리소스 정리 실패: {e}")


# 사용 예시
async def test_smart_storage():
    """스마트 데이터 저장소 테스트"""
    storage = SmartDataStorage()

    try:
        # 1. 최신 주식 데이터 조회
        print("📊 최신 주식 데이터 조회...")
        stock_data = await storage.get_latest_stock_data(limit=10)
        print(f"조회 결과: {len(stock_data) if stock_data else 0}개")

        # 2. AI 분석 결과 조회
        print("🤖 AI 분석 결과 조회...")
        analysis_data = await storage.get_analysis_results(limit=5)
        print(f"분석 결과: {len(analysis_data) if analysis_data else 0}개")

        # 3. 캐시 통계 확인
        print("📈 캐시 통계:")
        stats = storage.get_cache_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")

        # 4. 데이터 프리페치 테스트
        print("🚀 데이터 프리페치 테스트...")
        prefetch_requests = [
            DataRequest("korean_market_top5", {}, priority=1),
            DataRequest("us_market_top5", {}, priority=1),
            DataRequest("dashboard_data", {}, priority=2),
        ]
        await storage.prefetch_data(prefetch_requests)

        print("✅ 테스트 완료!")

    finally:
        await storage.close()


if __name__ == "__main__":
    asyncio.run(test_smart_storage())
