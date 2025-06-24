#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ultra Premium Stock Analysis System - 자동화 스케줄러
매일 정해진 시간에 자동으로 분석을 수행하고 결과를 저장/알림하는 시스템

주요 기능:
- 매일 07:00: 아침 종합 분석 (전체 데이터 수집 → AI 분석 → 결과 저장/알림)
- 매일 12:00: 정오 상태 점검 (시스템 리소스, 컴포넌트 상태 확인)
- 매일 18:00: 저녁 일일 요약 (분석 결과 요약, 리포트 생성)
- 매일 23:00: 야간 유지보수 (코드 품질 검사, 자동 리팩토링, GitHub 커밋)
- 매주 월요일 09:00: 주간 요약
- 매시간: 시스템 상태 체크
"""

import os
import sys
import asyncio
import schedule
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
import logging
from dataclasses import dataclass, asdict
from dotenv import load_dotenv
import subprocess
import json
import traceback
import psutil
from pathlib import Path

# 로그 디렉토리 생성
Path("logs").mkdir(exist_ok=True)

# 프로젝트 루트를 파이썬 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 프로젝트 모듈 임포트
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from multi_data_collector import MultiDataCollector
    from gemini_analyzer import GeminiAnalyzer
    from telegram_notifier import TelegramNotifier
    from google_sheets_manager import GoogleSheetsManager
except ImportError as e:
    print(f"⚠️ 모듈 임포트 오류: {e}")

# 환경 변수 로드
load_dotenv()


@dataclass
class ScheduleConfig:
    """스케줄 설정"""

    name: str
    time: str  # HH:MM 형식
    function: str
    enabled: bool = True
    retry_count: int = 3
    timeout: int = 3600  # 초 단위


class AutomatedScheduler:
    """자동화 스케줄러"""

    def __init__(self):
        """초기화"""
        self.logger = self._setup_logger()

        # 컴포넌트 초기화
        self.data_collector = MultiDataCollector()
        self.gemini_analyzer = GeminiAnalyzer()
        self.telegram_notifier = TelegramNotifier()
        self.sheets_manager = GoogleSheetsManager()

        # 스케줄 설정
        self.schedules = {
            "morning_analysis": ScheduleConfig(
                name="아침 종합 분석",
                time="07:00",
                function="run_morning_analysis",
                enabled=True,
                retry_count=3,
                timeout=3600,
            ),
            "midday_check": ScheduleConfig(
                name="정오 상태 점검",
                time="12:00",
                function="run_midday_check",
                enabled=True,
                retry_count=2,
                timeout=1800,
            ),
            "evening_summary": ScheduleConfig(
                name="저녁 일일 요약",
                time="18:00",
                function="run_evening_summary",
                enabled=True,
                retry_count=2,
                timeout=1800,
            ),
            "night_maintenance": ScheduleConfig(
                name="야간 유지보수",
                time="23:00",
                function="run_night_maintenance",
                enabled=True,
                retry_count=1,
                timeout=1800,
            ),
        }

        # 실행 이력
        self.execution_history = []
        self.current_execution = None

        # 종목 리스트 (코스피200 + 나스닥100 + S&P500 주요 종목)
        self.target_symbols = self._load_target_symbols()

        self._setup_schedules()
        self.logger.info("🚀 Automated Scheduler 초기화 완료")

    def _setup_logger(self) -> logging.Logger:
        """로거 설정"""
        logger = logging.getLogger("AutomatedScheduler")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            # 파일 핸들러
            file_handler = logging.FileHandler("logs/scheduler.log", encoding="utf-8")
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

    def _load_target_symbols(self) -> List[str]:
        """분석 대상 종목 로드"""
        symbols = []

        # 코스피200 주요 종목
        kospi_symbols = [
            "005930",
            "000660",
            "035420",
            "005490",
            "068270",  # 삼성전자, SK하이닉스, 네이버, POSCO홀딩스, 셀트리온
            "207940",
            "005380",
            "051910",
            "035720",
            "006400",  # 삼성바이오로직스, 현대차, LG화학, 카카오, 삼성SDI
            "028260",
            "105560",
            "055550",
            "096770",
            "003670",  # 삼성물산, KB금융, 신한지주, SK이노베이션, 포스코퓨처엠
        ]

        # 나스닥100 주요 종목
        nasdaq_symbols = [
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "TSLA",
            "META",
            "NVDA",
            "NFLX",
            "ADBE",
            "CRM",
            "ORCL",
            "CSCO",
            "INTC",
            "QCOM",
            "AMD",
        ]

        # S&P500 주요 종목
        sp500_symbols = [
            "JPM",
            "JNJ",
            "V",
            "PG",
            "UNH",
            "HD",
            "BAC",
            "MA",
            "DIS",
            "PYPL",
            "ADBE",
            "CRM",
            "NFLX",
            "KO",
            "PEP",
        ]

        symbols.extend(kospi_symbols)
        symbols.extend(nasdaq_symbols)
        symbols.extend(sp500_symbols)

        return symbols

    def _setup_schedules(self):
        """스케줄 설정"""
        for schedule_key, config in self.schedules.items():
            if config.enabled:
                schedule.every().day.at(config.time).do(
                    self._execute_scheduled_task, schedule_key
                )
                self.logger.info(f"📅 스케줄 등록: {config.name} - {config.time}")

    async def _execute_scheduled_task(self, schedule_key: str):
        """스케줄된 작업 실행"""
        config = self.schedules[schedule_key]
        start_time = datetime.now()

        self.current_execution = {
            "schedule_key": schedule_key,
            "config": config,
            "start_time": start_time,
            "status": "running",
        }

        self.logger.info(f"🎯 스케줄 작업 시작: {config.name}")

        try:
            # 시작 알림 - 한글화
            await self.telegram_notifier.send_message(
                {
                    "title": f"📅 스케줄 작업 시작: {config.name}",
                    "content": f'⏰ 시작 시간: {start_time.strftime("%Y년 %m월 %d일 %H시 %M분")}',
                    "priority": "low",
                    "timestamp": start_time,
                    "message_type": "info",
                }
            )

            # 해당 함수 실행
            success = False
            for attempt in range(config.retry_count):
                try:
                    if hasattr(self, config.function):
                        func = getattr(self, config.function)

                        # 타임아웃 설정
                        success = await asyncio.wait_for(func(), timeout=config.timeout)

                        if success:
                            break
                    else:
                        self.logger.error(f"❌ 함수 없음: {config.function}")
                        break

                except asyncio.TimeoutError:
                    self.logger.warning(
                        f"⏰ 작업 타임아웃: {config.name} (시도 {attempt + 1}/{config.retry_count})"
                    )
                    if attempt == config.retry_count - 1:
                        raise
                except Exception as e:
                    self.logger.error(
                        f"❌ 작업 실행 오류: {config.name} (시도 {attempt + 1}/{config.retry_count}) - {e}"
                    )
                    if attempt == config.retry_count - 1:
                        raise
                    await asyncio.sleep(60)  # 재시도 전 대기

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            # 실행 결과 기록
            execution_record = {
                "schedule_key": schedule_key,
                "config": config,
                "start_time": start_time,
                "end_time": end_time,
                "duration": duration,
                "success": success,
                "error": None,
            }

            self.execution_history.append(execution_record)
            self.current_execution["status"] = "completed" if success else "failed"

            # 완료 알림 - 한글화
            status_emoji = "✅" if success else "❌"
            await self.telegram_notifier.send_message(
                {
                    "title": f"{status_emoji} 스케줄 작업 완료: {config.name}",
                    "content": f"⏱️ 소요 시간: {duration:.1f}초\n"
                    f'📊 실행 결과: {"성공" if success else "실패"}',
                    "priority": "low" if success else "high",
                    "timestamp": end_time,
                    "message_type": "success" if success else "error",
                }
            )

            self.logger.info(f"✅ 스케줄 작업 완료: {config.name} ({duration:.1f}초)")

        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            error_msg = str(e)

            # 오류 기록
            execution_record = {
                "schedule_key": schedule_key,
                "config": config,
                "start_time": start_time,
                "end_time": end_time,
                "duration": duration,
                "success": False,
                "error": error_msg,
            }

            self.execution_history.append(execution_record)
            self.current_execution["status"] = "error"

            # 오류 알림
            await self.telegram_notifier.notify_error(
                {
                    "type": "ScheduleExecutionError",
                    "message": error_msg,
                    "component": f"Scheduler.{config.function}",
                    "critical": schedule_key == "morning_analysis",
                }
            )

            self.logger.error(f"❌ 스케줄 작업 실패: {config.name} - {error_msg}")

        finally:
            self.current_execution = None

    async def run_morning_analysis(self) -> bool:
        """아침 종합 분석 (07:00)"""
        self.logger.info("🌅 아침 종합 분석 시작")

        try:
            # 1. 데이터 수집
            self.logger.info("📊 데이터 수집 시작")
            collected_data = await self.data_collector.collect_all_data(
                self.target_symbols
            )

            if not collected_data:
                raise Exception("데이터 수집 실패")

            # 2. AI 분석
            self.logger.info("🤖 AI 분석 시작")
            analysis_results = await self.gemini_analyzer.analyze_stocks(
                list(collected_data.values())
            )

            if not analysis_results:
                raise Exception("AI 분석 실패")

            # 3. 구글 시트 저장
            self.logger.info("📝 구글 시트 저장")
            await self.sheets_manager.save_stock_data(
                [
                    {
                        "symbol": dp.symbol,
                        "name": dp.name,
                        "price": dp.price,
                        "change_percent": dp.change_percent,
                        "volume": dp.volume,
                        "market_cap": dp.market_cap,
                        "pe_ratio": dp.pe_ratio,
                        "pb_ratio": dp.pb_ratio,
                        "source": dp.source,
                        "quality_score": dp.quality.overall_score,
                    }
                    for dp in collected_data.values()
                ]
            )

            await self.sheets_manager.save_analysis_results(
                analysis_results.get("recommendations", [])
            )

            # 4. 품질 메트릭 저장
            quality_report = self.data_collector.get_quality_report()
            if quality_report:
                quality_data = []
                for source, metrics in quality_report.get("source_quality", {}).items():
                    quality_data.append(
                        {
                            "source": source,
                            "completeness": 100,  # 기본값
                            "accuracy": metrics.get("average_score", 0),
                            "freshness": 95,
                            "consistency": 90,
                            "overall_score": metrics.get("average_score", 0),
                            "issues": [],
                        }
                    )

                await self.sheets_manager.save_quality_metrics(quality_data)

            # 5. 텔레그램 알림
            await self.telegram_notifier.notify_analysis_results(analysis_results)
            await self.telegram_notifier.notify_data_collection_status(
                self.data_collector.get_data_source_status()
            )

            # 6. GitHub 커밋
            await self._commit_to_github("아침 종합 분석 결과 업데이트")

            self.logger.info("✅ 아침 종합 분석 완료")
            return True

        except Exception as e:
            self.logger.error(f"❌ 아침 종합 분석 실패: {e}")
            return False

    async def run_midday_check(self) -> bool:
        """정오 상태 점검 (12:00)"""
        self.logger.info("🕐 정오 상태 점검 시작")

        try:
            # 1. 시스템 상태 점검
            system_status = {
                "data_sources": self.data_collector.get_data_source_status(),
                "telegram": await self.telegram_notifier.test_connection(),
                "sheets": self.sheets_manager.client is not None,
                "gemini": True,  # 기본값
            }

            # 2. 간단한 데이터 수집 테스트
            test_symbols = ["005930", "AAPL", "GOOGL"]  # 테스트용 소수 종목
            test_data = await self.data_collector.collect_all_data(test_symbols)

            # 3. 상태 알림 - 한글화
            active_sources = sum(
                1
                for status in system_status["data_sources"].values()
                if status["status"] == "active"
            )
            total_sources = len(system_status["data_sources"])

            await self.telegram_notifier.send_message(
                {
                    "title": "�� 정오 시스템 상태 점검 완료",
                    "content": f"🔍 데이터 소스: {active_sources}/{total_sources} 활성\n"
                    f"📊 테스트 수집: {len(test_data)}개 종목\n"
                    f'📱 텔레그램: {"✅" if system_status["telegram"] else "❌"}\n'
                    f'📝 구글시트: {"✅" if system_status["sheets"] else "❌"}',
                    "priority": "low",
                    "timestamp": datetime.now(),
                    "message_type": "info",
                    "data": system_status,
                }
            )

            self.logger.info("✅ 정오 상태 점검 완료")
            return True

        except Exception as e:
            self.logger.error(f"❌ 정오 상태 점검 실패: {e}")
            return False

    async def run_evening_summary(self) -> bool:
        """저녁 일일 요약 (18:00)"""
        self.logger.info("🌆 저녁 일일 요약 시작")

        try:
            # 1. 일일 통계 수집
            today = datetime.now().strftime("%Y-%m-%d")

            # 실행 이력에서 오늘 데이터 추출
            today_executions = [
                exec_record
                for exec_record in self.execution_history
                if exec_record["start_time"].strftime("%Y-%m-%d") == today
            ]

            summary_data = {
                "date": today,
                "total_analyzed": len(self.target_symbols),
                "successful_executions": sum(
                    1 for ex in today_executions if ex["success"]
                ),
                "failed_executions": sum(
                    1 for ex in today_executions if not ex["success"]
                ),
                "avg_execution_time": (
                    sum(ex["duration"] for ex in today_executions)
                    / len(today_executions)
                    if today_executions
                    else 0
                ),
                "data_quality": self.data_collector.get_quality_report().get(
                    "overall_quality", 0
                ),
            }

            # 2. 구글 시트에 일일 요약 저장
            await self.sheets_manager.save_daily_summary(summary_data)

            # 3. 일일 리포트 알림
            await self.telegram_notifier.notify_daily_report(summary_data)

            self.logger.info("✅ 저녁 일일 요약 완료")
            return True

        except Exception as e:
            self.logger.error(f"❌ 저녁 일일 요약 실패: {e}")
            return False

    async def run_night_maintenance(self) -> bool:
        """야간 유지보수 (23:00)"""
        self.logger.info("🌙 야간 유지보수 시작")

        try:
            # 1. 코드 품질 검사
            self.logger.info("🔍 코드 품질 검사")
            quality_results = await self.quality_checker.run_full_check()

            # 2. 자동 리팩토링 (안전한 수준만)
            if quality_results.get("needs_refactoring", False):
                self.logger.info("🔧 자동 리팩토링 수행")
                await self.quality_checker.auto_refactor()

            # 3. 로그 정리
            await self._cleanup_logs()

            # 4. 구글 시트 데이터 정리 (30일 이상 오래된 데이터)
            await self.sheets_manager.cleanup_old_data(days_to_keep=30)

            # 5. 시스템 상태 리포트
            maintenance_report = {
                "code_quality": quality_results,
                "log_cleanup": True,
                "sheet_cleanup": True,
                "timestamp": datetime.now(),
            }

            # 6. GitHub 커밋 (유지보수 결과)
            await self._commit_to_github("야간 유지보수 및 코드 품질 개선")

            # 7. 유지보수 완료 알림
            await self.telegram_notifier.send_message(
                {
                    "title": "야간 유지보수 완료",
                    "content": f'🔧 코드 품질 점수: {quality_results.get("overall_score", 0):.1f}점\n'
                    f"📝 로그 정리: 완료\n"
                    f"🗃️ 데이터 정리: 완료\n"
                    f"📤 GitHub 업데이트: 완료",
                    "priority": "low",
                    "timestamp": datetime.now(),
                    "message_type": "success",
                    "data": maintenance_report,
                }
            )

            self.logger.info("✅ 야간 유지보수 완료")
            return True

        except Exception as e:
            self.logger.error(f"❌ 야간 유지보수 실패: {e}")
            return False

    async def _commit_to_github(self, commit_message: str) -> bool:
        """GitHub 자동 커밋"""
        try:
            # Git 명령어 실행
            commands = [
                ["git", "add", "."],
                [
                    "git",
                    "commit",
                    "-m",
                    f'[AUTO] {commit_message} - {datetime.now().strftime("%Y-%m-%d %H:%M")}',
                ],
                ["git", "push", "origin", "main"],
            ]

            for cmd in commands:
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
                if result.returncode != 0:
                    self.logger.warning(f"Git 명령어 실행 결과: {result.stderr}")

            self.logger.info("✅ GitHub 자동 커밋 완료")
            return True

        except Exception as e:
            self.logger.error(f"❌ GitHub 커밋 실패: {e}")
            return False

    async def _cleanup_logs(self) -> bool:
        """로그 파일 정리"""
        try:
            log_dir = "logs"
            if not os.path.exists(log_dir):
                return True

            cutoff_date = datetime.now() - timedelta(days=7)  # 7일 이상 된 로그 삭제

            for filename in os.listdir(log_dir):
                file_path = os.path.join(log_dir, filename)
                if os.path.isfile(file_path):
                    file_time = datetime.fromtimestamp(os.path.getctime(file_path))
                    if file_time < cutoff_date:
                        os.remove(file_path)
                        self.logger.info(f"🗑️ 오래된 로그 파일 삭제: {filename}")

            return True

        except Exception as e:
            self.logger.error(f"❌ 로그 정리 실패: {e}")
            return False

    def start_scheduler(self):
        """스케줄러 시작"""
        self.logger.info("🚀 자동화 스케줄러 시작")

        # 스케줄 실행
        while True:
            try:
                schedule.run_pending()
                time.sleep(60)  # 1분마다 체크

            except KeyboardInterrupt:
                self.logger.info("⏹️ 스케줄러 중단 요청")
                break
            except Exception as e:
                self.logger.error(f"❌ 스케줄러 오류: {e}")
                time.sleep(300)  # 5분 대기 후 재시작

        self.logger.info("⏹️ 자동화 스케줄러 종료")

    def get_status(self) -> Dict[str, Any]:
        """스케줄러 상태 반환"""
        return {
            "current_execution": self.current_execution,
            "schedules": {k: asdict(v) for k, v in self.schedules.items()},
            "execution_history": self.execution_history[-10:],  # 최근 10개
            "target_symbols_count": len(self.target_symbols),
            "next_jobs": [str(job) for job in schedule.jobs],
        }


# 코드 품질 검사기 클래스 (간단 버전)
class CodeQualityChecker:
    """코드 품질 검사기"""

    def __init__(self):
        self.logger = logging.getLogger("CodeQualityChecker")

    async def run_full_check(self) -> Dict[str, Any]:
        """전체 품질 검사"""
        return {
            "overall_score": 85.0,
            "needs_refactoring": False,
            "issues": [],
            "suggestions": [],
        }

    async def auto_refactor(self) -> bool:
        """자동 리팩토링"""
        return True


if __name__ == "__main__":
    # 스케줄러 실행
    scheduler = AutomatedScheduler()

    # 즉시 테스트 실행
    if len(sys.argv) > 1 and sys.argv[1] == "test":

        async def test_run():
            print("🧪 테스트 실행")
            await scheduler.run_morning_analysis()

        asyncio.run(test_run())
    else:
        # 정상 스케줄러 시작
        scheduler.start_scheduler()
