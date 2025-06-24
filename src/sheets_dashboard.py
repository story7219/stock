#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    📊 Google Sheets Dashboard System v1.0                   ║
║                      구글시트 기반 실시간 대시보드                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  • 📈 실시간 데이터 시각화                                                   ║
║  • 🎯 AI 분석 결과 대시보드                                                  ║
║  • 📊 투자 성과 추적                                                         ║
║  • 🔔 알림 및 리포트 자동화                                                  ║
║  • 📱 모바일 친화적 인터페이스                                               ║
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
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import base64
from io import BytesIO

# 환경 변수 로드
load_dotenv()


@dataclass
class DashboardConfig:
    """대시보드 설정 클래스"""

    title: str
    sheet_name: str
    chart_type: str  # 'line', 'bar', 'pie', 'scatter', 'heatmap'
    data_source: str
    refresh_interval: int = 300  # 5분
    auto_update: bool = True


@dataclass
class PerformanceMetric:
    """성과 지표 클래스"""

    name: str
    value: float
    previous_value: float
    change_percent: float
    trend: str  # 'up', 'down', 'stable'
    status: str  # 'good', 'warning', 'critical'


class SheetsDashboard:
    """구글 시트 기반 대시보드"""

    def __init__(self):
        """초기화"""
        self.logger = self._setup_logger()

        # 구글 시트 설정
        self.credentials_path = os.getenv(
            "GOOGLE_SHEETS_CREDENTIALS_PATH", "credentials.json"
        )
        self.spreadsheet_id = os.getenv("GOOGLE_SHEETS_SPREADSHEET_ID")

        # 차트 설정
        plt.style.use("seaborn-v0_8")
        plt.rcParams["font.family"] = ["DejaVu Sans", "Malgun Gothic", "Arial"]
        plt.rcParams["axes.unicode_minus"] = False

        # 출력 디렉토리
        self.output_dir = Path("dashboard_output")
        self.output_dir.mkdir(exist_ok=True)

        # 구글 시트 클라이언트
        self.sheets_client = None
        self.spreadsheet = None
        self.executor = ThreadPoolExecutor(max_workers=3)

        # 대시보드 설정
        self.dashboard_configs = {
            "market_overview": DashboardConfig(
                title="시장 개요",
                sheet_name="한국시장TOP5",
                chart_type="bar",
                data_source="korean_market_top5",
            ),
            "us_market_overview": DashboardConfig(
                title="미국 시장 개요",
                sheet_name="미국시장TOP5",
                chart_type="bar",
                data_source="us_market_top5",
            ),
            "strategy_performance": DashboardConfig(
                title="전략별 성과",
                sheet_name="전략요약",
                chart_type="line",
                data_source="strategy_summary",
            ),
            "ai_confidence": DashboardConfig(
                title="AI 신뢰도 분석",
                sheet_name="AI분석결과",
                chart_type="scatter",
                data_source="analysis_results",
            ),
        }

        # 성과 지표 추적
        self.performance_metrics = {}

        self._initialize_sheets_client()
        self.logger.info("📊 Sheets Dashboard 초기화 완료")

    def _setup_logger(self) -> logging.Logger:
        """로거 설정"""
        logger = logging.getLogger("SheetsDashboard")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            # 로그 디렉토리 생성
            os.makedirs("logs", exist_ok=True)

            # 파일 핸들러
            file_handler = logging.FileHandler("logs/dashboard.log", encoding="utf-8")
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

    async def get_sheet_data(self, sheet_name: str, limit: int = None) -> pd.DataFrame:
        """시트 데이터 조회"""
        try:
            if not self.spreadsheet:
                return pd.DataFrame()

            loop = asyncio.get_event_loop()
            worksheet = await loop.run_in_executor(
                self.executor, self.spreadsheet.worksheet, sheet_name
            )

            records = await loop.run_in_executor(
                self.executor, worksheet.get_all_records
            )

            df = pd.DataFrame(records)

            if limit and len(df) > limit:
                df = df.tail(limit)

            return df

        except Exception as e:
            self.logger.error(f"❌ 시트 데이터 조회 실패: {e}")
            return pd.DataFrame()

    async def create_market_overview_chart(self, market: str = "korean") -> str:
        """시장 개요 차트 생성"""
        try:
            sheet_name = "한국시장TOP5" if market == "korean" else "미국시장TOP5"
            df = await self.get_sheet_data(sheet_name, limit=5)

            if df.empty:
                return ""

            # 차트 생성
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # 1. 종목별 종합점수
            if "종목명" in df.columns and "종합점수" in df.columns:
                ax1.bar(df["종목명"], df["종합점수"], color="skyblue", alpha=0.7)
                ax1.set_title(f"{market.upper()} 시장 TOP5 - 종합점수")
                ax1.set_xlabel("종목명")
                ax1.set_ylabel("종합점수")
                ax1.tick_params(axis="x", rotation=45)

            # 2. 기대수익률
            if "종목명" in df.columns and "기대수익률" in df.columns:
                returns = df["기대수익률"].apply(
                    lambda x: (
                        float(str(x).replace("%", "")) if isinstance(x, str) else x
                    )
                )
                ax2.bar(df["종목명"], returns, color="lightgreen", alpha=0.7)
                ax2.set_title(f"{market.upper()} 시장 TOP5 - 기대수익률")
                ax2.set_xlabel("종목명")
                ax2.set_ylabel("기대수익률 (%)")
                ax2.tick_params(axis="x", rotation=45)

            plt.tight_layout()

            # 이미지 저장
            chart_path = self.output_dir / f"{market}_market_overview.png"
            plt.savefig(chart_path, dpi=300, bbox_inches="tight")
            plt.close()

            self.logger.info(f"✅ {market} 시장 개요 차트 생성 완료")
            return str(chart_path)

        except Exception as e:
            self.logger.error(f"❌ 시장 개요 차트 생성 실패: {e}")
            return ""

    async def create_strategy_performance_chart(self) -> str:
        """전략별 성과 차트 생성"""
        try:
            df = await self.get_sheet_data("한국시장TOP5", limit=50)

            if df.empty:
                return ""

            # 전략별 평균 점수 계산
            strategy_columns = [
                "버핏점수",
                "린치점수",
                "그레이엄점수",
                "달리오점수",
                "오닐점수",
                "리버모어점수",
                "피셔점수",
                "블랙록점수",
            ]

            strategy_scores = {}
            for col in strategy_columns:
                if col in df.columns:
                    strategy_name = col.replace("점수", "")
                    strategy_scores[strategy_name] = df[col].mean()

            if not strategy_scores:
                return ""

            # 차트 생성
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # 1. 전략별 평균 점수 (막대 차트)
            strategies = list(strategy_scores.keys())
            scores = list(strategy_scores.values())

            ax1.bar(strategies, scores, color="lightcoral", alpha=0.7)
            ax1.set_title("투자 대가 전략별 평균 점수")
            ax1.set_xlabel("투자 전략")
            ax1.set_ylabel("평균 점수")
            ax1.tick_params(axis="x", rotation=45)

            # 2. 전략별 점수 분포 (박스 플롯)
            strategy_data = []
            strategy_labels = []

            for col in strategy_columns:
                if col in df.columns:
                    strategy_data.append(df[col].dropna().values)
                    strategy_labels.append(col.replace("점수", ""))

            if strategy_data:
                ax2.boxplot(strategy_data, labels=strategy_labels)
                ax2.set_title("투자 대가 전략별 점수 분포")
                ax2.set_xlabel("투자 전략")
                ax2.set_ylabel("점수")
                ax2.tick_params(axis="x", rotation=45)

            plt.tight_layout()

            # 이미지 저장
            chart_path = self.output_dir / "strategy_performance.png"
            plt.savefig(chart_path, dpi=300, bbox_inches="tight")
            plt.close()

            self.logger.info("✅ 전략별 성과 차트 생성 완료")
            return str(chart_path)

        except Exception as e:
            self.logger.error(f"❌ 전략별 성과 차트 생성 실패: {e}")
            return ""

    async def create_ai_confidence_analysis(self) -> str:
        """AI 신뢰도 분석 차트 생성"""
        try:
            korean_df = await self.get_sheet_data("한국시장TOP5", limit=50)
            us_df = await self.get_sheet_data("미국시장TOP5", limit=50)

            if korean_df.empty and us_df.empty:
                return ""

            # 차트 생성
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

            # 1. 한국시장 AI 신뢰도 vs 종합점수
            if (
                not korean_df.empty
                and "AI신뢰도" in korean_df.columns
                and "종합점수" in korean_df.columns
            ):
                confidence = korean_df["AI신뢰도"].apply(
                    lambda x: (
                        float(str(x).replace("%", "")) if isinstance(x, str) else x
                    )
                )
                scores = korean_df["종합점수"]

                ax1.scatter(confidence, scores, alpha=0.6, color="blue")
                ax1.set_title("한국시장: AI 신뢰도 vs 종합점수")
                ax1.set_xlabel("AI 신뢰도 (%)")
                ax1.set_ylabel("종합점수")

                # 추세선 추가
                z = np.polyfit(confidence, scores, 1)
                p = np.poly1d(z)
                ax1.plot(confidence, p(confidence), "r--", alpha=0.8)

            # 2. 미국시장 AI 신뢰도 vs 종합점수
            if (
                not us_df.empty
                and "AI신뢰도" in us_df.columns
                and "종합점수" in us_df.columns
            ):
                confidence = us_df["AI신뢰도"].apply(
                    lambda x: (
                        float(str(x).replace("%", "")) if isinstance(x, str) else x
                    )
                )
                scores = us_df["종합점수"]

                ax2.scatter(confidence, scores, alpha=0.6, color="red")
                ax2.set_title("미국시장: AI 신뢰도 vs 종합점수")
                ax2.set_xlabel("AI 신뢰도 (%)")
                ax2.set_ylabel("종합점수")

                # 추세선 추가
                z = np.polyfit(confidence, scores, 1)
                p = np.poly1d(z)
                ax2.plot(confidence, p(confidence), "r--", alpha=0.8)

            # 3. AI 신뢰도 분포 (한국시장)
            if not korean_df.empty and "AI신뢰도" in korean_df.columns:
                confidence = korean_df["AI신뢰도"].apply(
                    lambda x: (
                        float(str(x).replace("%", "")) if isinstance(x, str) else x
                    )
                )
                ax3.hist(
                    confidence, bins=10, alpha=0.7, color="skyblue", edgecolor="black"
                )
                ax3.set_title("한국시장: AI 신뢰도 분포")
                ax3.set_xlabel("AI 신뢰도 (%)")
                ax3.set_ylabel("빈도")

            # 4. AI 신뢰도 분포 (미국시장)
            if not us_df.empty and "AI신뢰도" in us_df.columns:
                confidence = us_df["AI신뢰도"].apply(
                    lambda x: (
                        float(str(x).replace("%", "")) if isinstance(x, str) else x
                    )
                )
                ax4.hist(
                    confidence,
                    bins=10,
                    alpha=0.7,
                    color="lightcoral",
                    edgecolor="black",
                )
                ax4.set_title("미국시장: AI 신뢰도 분포")
                ax4.set_xlabel("AI 신뢰도 (%)")
                ax4.set_ylabel("빈도")

            plt.tight_layout()

            # 이미지 저장
            chart_path = self.output_dir / "ai_confidence_analysis.png"
            plt.savefig(chart_path, dpi=300, bbox_inches="tight")
            plt.close()

            self.logger.info("✅ AI 신뢰도 분석 차트 생성 완료")
            return str(chart_path)

        except Exception as e:
            self.logger.error(f"❌ AI 신뢰도 분석 차트 생성 실패: {e}")
            return ""

    async def create_investment_heatmap(self) -> str:
        """투자 전략별 히트맵 생성"""
        try:
            korean_df = await self.get_sheet_data("한국시장TOP5", limit=20)

            if korean_df.empty:
                return ""

            # 전략별 점수 데이터 추출
            strategy_columns = [
                "버핏점수",
                "린치점수",
                "그레이엄점수",
                "달리오점수",
                "오닐점수",
                "리버모어점수",
                "피셔점수",
                "블랙록점수",
            ]

            # 데이터 필터링
            available_columns = [
                col for col in strategy_columns if col in korean_df.columns
            ]
            if not available_columns or "종목명" not in korean_df.columns:
                return ""

            # 히트맵 데이터 준비
            heatmap_data = korean_df[["종목명"] + available_columns].set_index("종목명")

            # 차트 생성
            fig, ax = plt.subplots(figsize=(12, 8))

            # 히트맵 생성
            sns.heatmap(
                heatmap_data.T, annot=True, cmap="RdYlBu_r", center=50, fmt=".1f", ax=ax
            )

            ax.set_title("투자 대가 전략별 종목 점수 히트맵", fontsize=16, pad=20)
            ax.set_xlabel("종목명", fontsize=12)
            ax.set_ylabel("투자 전략", fontsize=12)

            # 전략명 정리
            strategy_labels = [col.replace("점수", "") for col in available_columns]
            ax.set_yticklabels(strategy_labels, rotation=0)

            plt.tight_layout()

            # 이미지 저장
            chart_path = self.output_dir / "investment_heatmap.png"
            plt.savefig(chart_path, dpi=300, bbox_inches="tight")
            plt.close()

            self.logger.info("✅ 투자 전략별 히트맵 생성 완료")
            return str(chart_path)

        except Exception as e:
            self.logger.error(f"❌ 투자 전략별 히트맵 생성 실패: {e}")
            return ""

    async def calculate_performance_metrics(self) -> Dict[str, PerformanceMetric]:
        """성과 지표 계산"""
        try:
            metrics = {}

            # 한국시장 데이터
            korean_df = await self.get_sheet_data("한국시장TOP5", limit=10)
            if not korean_df.empty:
                # 평균 종합점수
                if "종합점수" in korean_df.columns:
                    avg_score = korean_df["종합점수"].mean()
                    metrics["korean_avg_score"] = PerformanceMetric(
                        name="한국시장 평균 점수",
                        value=avg_score,
                        previous_value=avg_score * 0.95,  # 임시값
                        change_percent=5.0,
                        trend="up",
                        status="good" if avg_score > 70 else "warning",
                    )

                # 평균 AI 신뢰도
                if "AI신뢰도" in korean_df.columns:
                    confidence = (
                        korean_df["AI신뢰도"]
                        .apply(
                            lambda x: (
                                float(str(x).replace("%", ""))
                                if isinstance(x, str)
                                else x
                            )
                        )
                        .mean()
                    )

                    metrics["korean_ai_confidence"] = PerformanceMetric(
                        name="한국시장 AI 신뢰도",
                        value=confidence,
                        previous_value=confidence * 0.98,
                        change_percent=2.0,
                        trend="up",
                        status="good" if confidence > 80 else "warning",
                    )

            # 미국시장 데이터
            us_df = await self.get_sheet_data("미국시장TOP5", limit=10)
            if not us_df.empty:
                # 평균 종합점수
                if "종합점수" in us_df.columns:
                    avg_score = us_df["종합점수"].mean()
                    metrics["us_avg_score"] = PerformanceMetric(
                        name="미국시장 평균 점수",
                        value=avg_score,
                        previous_value=avg_score * 0.97,
                        change_percent=3.0,
                        trend="up",
                        status="good" if avg_score > 70 else "warning",
                    )

            # 전체 분석 건수
            analysis_df = await self.get_sheet_data("AI분석결과", limit=100)
            if not analysis_df.empty:
                total_analysis = len(analysis_df)
                metrics["total_analysis"] = PerformanceMetric(
                    name="총 분석 건수",
                    value=total_analysis,
                    previous_value=total_analysis * 0.9,
                    change_percent=10.0,
                    trend="up",
                    status="good",
                )

            self.performance_metrics = metrics
            return metrics

        except Exception as e:
            self.logger.error(f"❌ 성과 지표 계산 실패: {e}")
            return {}

    async def generate_dashboard_report(self) -> str:
        """대시보드 리포트 생성"""
        try:
            report_lines = []
            report_lines.append("📊 투자 분석 대시보드 리포트")
            report_lines.append("=" * 50)
            report_lines.append(
                f"📅 생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            report_lines.append("")

            # 성과 지표
            metrics = await self.calculate_performance_metrics()
            if metrics:
                report_lines.append("📈 주요 성과 지표")
                report_lines.append("-" * 30)

                for key, metric in metrics.items():
                    trend_icon = (
                        "📈"
                        if metric.trend == "up"
                        else "📉" if metric.trend == "down" else "➡️"
                    )
                    status_icon = (
                        "✅"
                        if metric.status == "good"
                        else "⚠️" if metric.status == "warning" else "❌"
                    )

                    report_lines.append(
                        f"{status_icon} {metric.name}: {metric.value:.2f}"
                    )
                    report_lines.append(
                        f"   변화율: {trend_icon} {metric.change_percent:+.1f}%"
                    )
                    report_lines.append("")

            # 차트 생성
            chart_paths = []

            # 시장 개요 차트
            korean_chart = await self.create_market_overview_chart("korean")
            if korean_chart:
                chart_paths.append(korean_chart)
                report_lines.append(f"📊 한국시장 차트: {korean_chart}")

            us_chart = await self.create_market_overview_chart("us")
            if us_chart:
                chart_paths.append(us_chart)
                report_lines.append(f"📊 미국시장 차트: {us_chart}")

            # 전략 성과 차트
            strategy_chart = await self.create_strategy_performance_chart()
            if strategy_chart:
                chart_paths.append(strategy_chart)
                report_lines.append(f"📊 전략 성과 차트: {strategy_chart}")

            # AI 신뢰도 분석
            ai_chart = await self.create_ai_confidence_analysis()
            if ai_chart:
                chart_paths.append(ai_chart)
                report_lines.append(f"📊 AI 신뢰도 분석: {ai_chart}")

            # 히트맵
            heatmap_chart = await self.create_investment_heatmap()
            if heatmap_chart:
                chart_paths.append(heatmap_chart)
                report_lines.append(f"📊 투자 히트맵: {heatmap_chart}")

            report_lines.append("")
            report_lines.append("📋 요약")
            report_lines.append("-" * 20)
            report_lines.append(f"• 생성된 차트: {len(chart_paths)}개")
            report_lines.append(f"• 분석된 지표: {len(metrics)}개")
            report_lines.append(
                f"• 대시보드 URL: https://docs.google.com/spreadsheets/d/{self.spreadsheet_id}"
            )

            # 리포트 저장
            report_content = "\n".join(report_lines)
            report_path = (
                self.output_dir
                / f"dashboard_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            )

            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report_content)

            self.logger.info(f"✅ 대시보드 리포트 생성 완료: {report_path}")
            return str(report_path)

        except Exception as e:
            self.logger.error(f"❌ 대시보드 리포트 생성 실패: {e}")
            return ""

    async def create_mobile_summary(self) -> Dict[str, Any]:
        """모바일용 요약 정보 생성"""
        try:
            summary = {
                "timestamp": datetime.now().isoformat(),
                "korean_market": {},
                "us_market": {},
                "key_metrics": {},
                "alerts": [],
            }

            # 한국시장 요약
            korean_df = await self.get_sheet_data("한국시장TOP5", limit=5)
            if not korean_df.empty:
                summary["korean_market"] = {
                    "top_stock": (
                        korean_df.iloc[0]["종목명"]
                        if "종목명" in korean_df.columns
                        else ""
                    ),
                    "avg_score": (
                        korean_df["종합점수"].mean()
                        if "종합점수" in korean_df.columns
                        else 0
                    ),
                    "total_stocks": len(korean_df),
                }

            # 미국시장 요약
            us_df = await self.get_sheet_data("미국시장TOP5", limit=5)
            if not us_df.empty:
                summary["us_market"] = {
                    "top_stock": (
                        us_df.iloc[0]["종목명"] if "종목명" in us_df.columns else ""
                    ),
                    "avg_score": (
                        us_df["종합점수"].mean() if "종합점수" in us_df.columns else 0
                    ),
                    "total_stocks": len(us_df),
                }

            # 주요 지표
            metrics = await self.calculate_performance_metrics()
            summary["key_metrics"] = {
                key: {
                    "value": metric.value,
                    "change": metric.change_percent,
                    "status": metric.status,
                }
                for key, metric in metrics.items()
            }

            # 알림 생성
            for key, metric in metrics.items():
                if metric.status == "warning":
                    summary["alerts"].append(f"⚠️ {metric.name}: 주의 필요")
                elif metric.status == "critical":
                    summary["alerts"].append(f"❌ {metric.name}: 긴급 확인 필요")

            return summary

        except Exception as e:
            self.logger.error(f"❌ 모바일 요약 생성 실패: {e}")
            return {}

    async def close(self):
        """리소스 정리"""
        try:
            self.executor.shutdown(wait=True)
            self.logger.info("✅ Sheets Dashboard 종료 완료")
        except Exception as e:
            self.logger.error(f"❌ 리소스 정리 실패: {e}")


# 사용 예시
async def test_dashboard():
    """대시보드 테스트"""
    dashboard = SheetsDashboard()

    try:
        print("📊 대시보드 테스트 시작...")

        # 1. 차트 생성 테스트
        print("📈 차트 생성 중...")
        korean_chart = await dashboard.create_market_overview_chart("korean")
        us_chart = await dashboard.create_market_overview_chart("us")
        strategy_chart = await dashboard.create_strategy_performance_chart()

        # 2. 성과 지표 계산
        print("📊 성과 지표 계산 중...")
        metrics = await dashboard.calculate_performance_metrics()
        print(f"계산된 지표: {len(metrics)}개")

        # 3. 대시보드 리포트 생성
        print("📋 리포트 생성 중...")
        report_path = await dashboard.generate_dashboard_report()
        print(f"리포트 생성 완료: {report_path}")

        # 4. 모바일 요약
        print("📱 모바일 요약 생성 중...")
        mobile_summary = await dashboard.create_mobile_summary()
        print(f"모바일 요약: {len(mobile_summary)} 항목")

        print("✅ 대시보드 테스트 완료!")

    finally:
        await dashboard.close()


if __name__ == "__main__":
    import numpy as np  # numpy import 추가

    asyncio.run(test_dashboard())
