#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        Google Sheets Manager v1.0                           ║
║                         구글시트 자동 저장 시스템                             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  • 실시간 데이터 자동 저장                                                   ║
║  • AI 분석 결과 구조화 저장                                                  ║
║  • 히스토리 데이터 관리                                                      ║
║  • 대시보드 자동 업데이트                                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass, asdict
from dotenv import load_dotenv
import gspread
from google.oauth2.service_account import Credentials
import asyncio
from concurrent.futures import ThreadPoolExecutor

# 환경 변수 로드
load_dotenv()


@dataclass
class SheetConfig:
    """시트 설정 클래스"""

    name: str
    headers: List[str]
    data_type: str
    auto_resize: bool = True
    freeze_rows: int = 1


class GoogleSheetsManager:
    """구글 시트 관리자"""
    
    def __init__(self):
        """초기화"""
        self.logger = self._setup_logger()
        
        # 구글 시트 설정 - SERVICE_ACCOUNT_JSON 파일을 직접 사용
        self.credentials_path = "SERVICE_ACCOUNT_JSON"
        self.spreadsheet_id = os.getenv("GOOGLE_SPREADSHEET_ID")

        creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "google_service_account.json")
        if not os.path.exists(creds_path):
            logging.warning("⚠️ 구글 인증 파일이 없습니다. 시트 연동 기능이 제한됩니다. 분석/리포트는 정상적으로 진행됩니다.")
        
        if not os.path.exists(self.credentials_path) or not self.spreadsheet_id:
            logging.warning("⚠️ 구글 시트 설정이 없습니다. 저장 기능이 비활성화됩니다.")
            self.enabled = False
            return
        
        try:
            # 구글 시트 인증
            self.scope = [
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive",
            ]
            
            self.credentials = Credentials.from_service_account_file(
                self.credentials_path, scopes=self.scope
            )
            
            self.gc = gspread.authorize(self.credentials)
            self.spreadsheet = self.gc.open_by_key(self.spreadsheet_id)
            
            self.enabled = True
            
            # 시트 구성
            self.sheet_configs = {
                "stock_data": SheetConfig(
                    name="주식데이터",
                    headers=[
                        "날짜",
                        "시간",
                        "종목코드",
                        "종목명",
                        "현재가",
                        "변동률",
                        "거래량",
                        "시가총액",
                        "PER",
                        "PBR",
                        "데이터소스",
                        "품질점수",
                    ],
                    data_type="realtime",
                ),
                "analysis_results": SheetConfig(
                    name="AI분석결과",
                    headers=[
                        "날짜",
                        "시간",
                        "종목코드",
                        "종목명",
                        "추천점수",
                        "전략점수",
                        "기술점수",
                        "위험점수",
                        "추천사유",
                        "AI모델",
                    ],
                    data_type="analysis",
                ),
                "korean_market_top5": SheetConfig(
                    name="한국시장TOP5",
                    headers=[
                        "날짜",
                        "시간",
                        "순위",
                        "종목코드",
                        "종목명",
                        "시장",
                        "현재가",
                        "진입가",
                        "목표가",
                        "기대수익률",
                        "종합점수",
                        "AI신뢰도",
                        "버핏점수",
                        "린치점수",
                        "그레이엄점수",
                        "달리오점수",
                        "오닐점수",
                        "리버모어점수",
                        "피셔점수",
                        "블랙록점수",
                        "선정사유",
                        "위험평가",
                        "통화",
                    ],
                    data_type="market_analysis",
                ),
                "us_market_top5": SheetConfig(
                    name="미국시장TOP5",
                    headers=[
                        "날짜",
                        "시간",
                        "순위",
                        "종목코드",
                        "종목명",
                        "시장",
                        "현재가",
                        "진입가",
                        "목표가",
                        "기대수익률",
                        "종합점수",
                        "AI신뢰도",
                        "버핏점수",
                        "린치점수",
                        "그레이엄점수",
                        "달리오점수",
                        "오닐점수",
                        "리버모어점수",
                        "피셔점수",
                        "블랙록점수",
                        "선정사유",
                        "위험평가",
                        "통화",
                    ],
                    data_type="market_analysis",
                ),
                "strategy_summary": SheetConfig(
                    name="전략요약",
                    headers=[
                        "날짜",
                        "시간",
                        "시장개요",
                        "한국시장심리",
                        "미국시장심리",
                        "추천전략",
                        "위험수준",
                        "투자기간",
                        "핵심요인",
                        "AI모델",
                    ],
                    data_type="strategy",
                ),
                "master_recommendation": SheetConfig(
                    name="마스터추천",
                    headers=[
                        "날짜",
                        "시간",
                        "마스터추천",
                        "한국종목수",
                        "미국종목수",
                        "총종목수",
                        "AI모델",
                        "분석시간",
                        "전체조언",
                        "한국전략",
                        "미국전략",
                    ],
                    data_type="master",
                ),
                "daily_summary": SheetConfig(
                    name="일일요약",
                    headers=[
                        "날짜",
                        "총분석종목",
                        "추천종목수",
                        "평균품질점수",
                        "상승종목",
                        "하락종목",
                        "시장상황",
                        "주요이슈",
                    ],
                    data_type="summary",
                ),
                "error_log": SheetConfig(
                    name="오류로그",
                    headers=[
                        "날짜",
                        "시간",
                        "오류유형",
                        "구성요소",
                        "오류메시지",
                        "심각도",
                    ],
                    data_type="log",
                ),
                "quality_metrics": SheetConfig(
                    name="데이터품질",
                    headers=[
                        "날짜",
                        "시간",
                        "데이터소스",
                        "완전성",
                        "정확성",
                        "신선도",
                        "일관성",
                        "전체점수",
                        "이슈",
                    ],
                    data_type="quality",
                ),
                "dashboard": SheetConfig(
                    name="대시보드",
                    headers=["Metric", "Value", "Last_Updated", "Trend", "Status"],
                    data_type="dashboard",
                ),
            }
            
            # 클라이언트 초기화
            self.client = None
            self.spreadsheet = None
            self.executor = ThreadPoolExecutor(max_workers=3)
            
            self._initialize_client()
            
        except Exception as e:
            logging.error(f"❌ 구글 시트 초기화 실패: {str(e)}")
            print(f"❌ 구글 시트 초기화 오류 상세: {type(e).__name__}: {str(e)}")
            import traceback

            traceback.print_exc()
            self.enabled = False
        
    def _setup_logger(self) -> logging.Logger:
        """로거 설정"""
        logger = logging.getLogger("GoogleSheetsManager")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _initialize_client(self):
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
            
            self.client = gspread.authorize(credentials)
            self.spreadsheet = self.client.open_by_key(self.spreadsheet_id)
            
            # 필요한 시트들 생성
            self._ensure_sheets_exist()
            
            self.logger.info("✅ 구글 시트 클라이언트 초기화 완료")
            
        except Exception as e:
            self.logger.error(f"❌ 구글 시트 초기화 실패: {e}")
            self.client = None
            self.spreadsheet = None
    
    def _ensure_sheets_exist(self):
        """필요한 시트들이 존재하는지 확인하고 생성"""
        if not self.spreadsheet:
            return
        
        try:
            existing_sheets = [sheet.title for sheet in self.spreadsheet.worksheets()]
            
            for sheet_key, config in self.sheet_configs.items():
                if config.name not in existing_sheets:
                    # 새 시트 생성
                    worksheet = self.spreadsheet.add_worksheet(
                        title=config.name, rows=1000, cols=len(config.headers)
                    )
                    
                    # 헤더 설정
                    worksheet.append_row(config.headers)
                    
                    # 헤더 행 고정
                    if config.freeze_rows > 0:
                        worksheet.freeze(rows=config.freeze_rows)
                    
                    # 헤더 스타일링
                    worksheet.format(
                        "1:1",
                        {
                            "backgroundColor": {"red": 0.2, "green": 0.6, "blue": 0.9},
                            "textFormat": {
                                "bold": True,
                                "foregroundColor": {"red": 1, "green": 1, "blue": 1},
                            },
                        },
                    )
                    
                    self.logger.info(f"📝 시트 생성: {config.name}")
                
        except Exception as e:
            self.logger.error(f"❌ 시트 생성 실패: {e}")
    
    async def save_stock_data(self, stock_data: List[Dict[str, Any]]) -> bool:
        """주식 데이터 저장"""
        if not self.enabled:
            return False
        
        try:
            worksheet = self.spreadsheet.worksheet("주식데이터")
            
            # 데이터 변환
            rows = []
            current_time = datetime.now()
            
            for data in stock_data:
                row = [
                    current_time.strftime("%Y-%m-%d"),
                    current_time.strftime("%H:%M:%S"),
                    data.get("symbol", ""),
                    data.get("name", ""),
                    data.get("price", 0),
                    f"{data.get('change_percent', 0):.2f}%",
                    data.get("volume", 0),
                    data.get("market_cap", ""),
                    data.get("pe_ratio", ""),
                    data.get("pb_ratio", ""),
                    data.get("source", ""),
                    data.get("quality_score", 0),
                ]
                rows.append(row)
            
            # 비동기로 데이터 추가
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.executor, self._append_rows, worksheet, rows
            )
            
            self.logger.info(f"✅ 주식 데이터 {len(rows)}건 저장 완료")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 주식 데이터 저장 실패: {e}")
            return False
    
    async def save_analysis_results(
        self, analysis_results: List[Dict[str, Any]]
    ) -> bool:
        """AI 분석 결과 저장 (기존 호환성 유지)"""
        if not self.enabled:
            return False
        
        try:
            worksheet = self.spreadsheet.worksheet("AI분석결과")
            
            rows = []
            current_time = datetime.now()
            
            for result in analysis_results:
                row = [
                    current_time.strftime("%Y-%m-%d"),
                    current_time.strftime("%H:%M:%S"),
                    result.get("symbol", ""),
                    result.get("name", ""),
                    result.get("total_score", 0),
                    result.get("strategy_score", 0),
                    result.get("technical_score", 0),
                    result.get("risk_score", 0),
                    result.get("reasoning", ""),
                    result.get("ai_model", "Gemini 1.5 Flash"),
                ]
                rows.append(row)
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.executor, self._append_rows, worksheet, rows
            )
            
            self.logger.info(f"✅ 분석 결과 {len(rows)}건 저장 완료")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 분석 결과 저장 실패: {e}")
            return False
    
    async def update_analysis_results(self, analysis_results: Dict[str, Any]) -> bool:
        """확장된 투자 대가 전략 분석 결과 저장"""
        if not self.enabled:
            return False

        try:
            # 한국시장 Top5 저장
            korean_stocks = analysis_results.get("korean_market_top5", [])
            if korean_stocks:
                await self._save_market_analysis("한국시장TOP5", korean_stocks, "KRW")

            # 미국시장 Top5 저장
            us_stocks = analysis_results.get("us_market_top5", [])
            if us_stocks:
                await self._save_market_analysis("미국시장TOP5", us_stocks, "USD")

            # 전략별 요약 저장
            strategy_analysis = analysis_results.get("strategy_analysis", {})
            if strategy_analysis:
                await self._save_strategy_summary(strategy_analysis)

            # 마스터 추천 저장
            master_recommendation = analysis_results.get("master_recommendation", "")
            if master_recommendation:
                await self._save_master_recommendation(analysis_results)

            self.logger.info("✅ 확장된 분석 결과 저장 완료")
            return True

        except Exception as e:
            self.logger.error(f"❌ 확장된 분석 결과 저장 실패: {e}")
            return False

    async def _save_market_analysis(
        self, sheet_name: str, stocks: List[Dict[str, Any]], currency: str
    ) -> bool:
        """시장별 분석 결과 저장"""
        try:
            worksheet = self.spreadsheet.worksheet(sheet_name)

            rows = []
            current_time = datetime.now()

            for i, stock in enumerate(stocks, 1):
                # 전략 점수 처리
                strategy_scores = stock.get("strategy_scores", {})
                warren_buffett = strategy_scores.get("warren_buffett", 0)
                peter_lynch = strategy_scores.get("peter_lynch", 0)
                benjamin_graham = strategy_scores.get("benjamin_graham", 0)
                ray_dalio = strategy_scores.get("ray_dalio", 0)
                william_oneil = strategy_scores.get("william_oneil", 0)
                jesse_livermore = strategy_scores.get("jesse_livermore", 0)
                philip_fisher = strategy_scores.get("philip_fisher", 0)
                blackrock_institutional = strategy_scores.get(
                    "blackrock_institutional", 0
                )

                row = [
                    current_time.strftime("%Y-%m-%d"),
                    current_time.strftime("%H:%M:%S"),
                    i,  # 순위
                    stock.get("symbol", ""),
                    stock.get("name", ""),
                    stock.get("market", ""),
                    stock.get("current_price", 0),
                    stock.get("entry_price", 0),
                    stock.get("target_price", 0),
                    f"{stock.get('expected_return_pct', 0)*100:.2f}%",
                    stock.get("final_score", 0),
                    f"{stock.get('flash_ai_confidence', 0)*100:.1f}%",
                    warren_buffett,
                    peter_lynch,
                    benjamin_graham,
                    ray_dalio,
                    william_oneil,
                    jesse_livermore,
                    philip_fisher,
                    blackrock_institutional,
                    stock.get("selection_reasoning", ""),
                    stock.get("risk_assessment", ""),
                    currency,
                ]
                rows.append(row)

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.executor, self._append_rows, worksheet, rows
            )

            self.logger.info(f"✅ {sheet_name} {len(rows)}건 저장 완료")
            return True

        except Exception as e:
            self.logger.error(f"❌ {sheet_name} 저장 실패: {e}")
            return False

    async def _save_strategy_summary(self, strategy_analysis: Dict[str, Any]) -> bool:
        """전략별 요약 저장"""
        try:
            worksheet = self.spreadsheet.worksheet("전략요약")

            current_time = datetime.now()

            row = [
                current_time.strftime("%Y-%m-%d"),
                current_time.strftime("%H:%M:%S"),
                strategy_analysis.get("market_overview", ""),
                strategy_analysis.get("korean_market_sentiment", ""),
                strategy_analysis.get("us_market_sentiment", ""),
                strategy_analysis.get("recommended_strategy", ""),
                strategy_analysis.get("risk_level", ""),
                strategy_analysis.get("investment_horizon", ""),
                strategy_analysis.get("key_factors", ""),
                "Gemini 1.5 Flash",
            ]

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.executor, self._append_rows, worksheet, [row]
            )

            self.logger.info("✅ 전략 요약 저장 완료")
            return True

        except Exception as e:
            self.logger.error(f"❌ 전략 요약 저장 실패: {e}")
            return False

    async def _save_master_recommendation(
        self, analysis_results: Dict[str, Any]
    ) -> bool:
        """마스터 추천 저장"""
        try:
            worksheet = self.spreadsheet.worksheet("마스터추천")

            current_time = datetime.now()
            korean_count = len(analysis_results.get("korean_market_top5", []))
            us_count = len(analysis_results.get("us_market_top5", []))

            row = [
                current_time.strftime("%Y-%m-%d"),
                current_time.strftime("%H:%M:%S"),
                analysis_results.get("master_recommendation", ""),
                korean_count,
                us_count,
                korean_count + us_count,
                analysis_results.get("model_info", "gemini-1.5-flash-8b"),
                analysis_results.get("analysis_timestamp", ""),
                analysis_results.get("strategy_recommendations", {}).get(
                    "overall_market_advice", ""
                ),
                analysis_results.get("strategy_recommendations", {}).get(
                    "best_strategy_for_korean_market", ""
                ),
                analysis_results.get("strategy_recommendations", {}).get(
                    "best_strategy_for_us_market", ""
                ),
            ]

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.executor, self._append_rows, worksheet, [row]
            )

            self.logger.info("✅ 마스터 추천 저장 완료")
            return True

        except Exception as e:
            self.logger.error(f"❌ 마스터 추천 저장 실패: {e}")
            return False
    
    async def save_daily_summary(self, summary_data: Dict[str, Any]) -> bool:
        """일일 요약 저장"""
        if not self.enabled:
            return False
        
        try:
            worksheet = self.spreadsheet.worksheet("일일요약")
            
            current_date = datetime.now().strftime("%Y-%m-%d")
            
            row = [
                current_date,
                summary_data.get("total_analyzed", 0),
                summary_data.get("recommended_count", 0),
                summary_data.get("avg_quality_score", 0),
                summary_data.get("rising_stocks", 0),
                summary_data.get("falling_stocks", 0),
                summary_data.get("market_condition", ""),
                summary_data.get("key_issues", ""),
            ]
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.executor, self._append_rows, worksheet, [row]
            )
            
            self.logger.info("✅ 일일 요약 저장 완료")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 일일 요약 저장 실패: {e}")
            return False
    
    async def save_error_log(self, error_info: Dict[str, Any]) -> bool:
        """오류 로그 저장"""
        if not self.enabled:
            return False
        
        try:
            worksheet = self.spreadsheet.worksheet("오류로그")
            
            current_time = datetime.now()
            
            row = [
                current_time.strftime("%Y-%m-%d"),
                current_time.strftime("%H:%M:%S"),
                error_info.get("type", ""),
                error_info.get("component", ""),
                error_info.get("message", ""),
                error_info.get("severity", "medium"),
            ]
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.executor, self._append_rows, worksheet, [row]
            )
            
            self.logger.info("✅ 오류 로그 저장 완료")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 오류 로그 저장 실패: {e}")
            return False
    
    async def save_quality_metrics(self, quality_data: List[Dict[str, Any]]) -> bool:
        """데이터 품질 메트릭 저장"""
        if not self.enabled:
            return False
        
        try:
            worksheet = self.spreadsheet.worksheet("데이터품질")
            
            rows = []
            current_time = datetime.now()
            
            for quality in quality_data:
                row = [
                    current_time.strftime("%Y-%m-%d"),
                    current_time.strftime("%H:%M:%S"),
                    quality.get("source", ""),
                    quality.get("completeness", 0),
                    quality.get("accuracy", 0),
                    quality.get("freshness", 0),
                    quality.get("consistency", 0),
                    quality.get("overall_score", 0),
                    ", ".join(quality.get("issues", [])),
                ]
                rows.append(row)
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.executor, self._append_rows, worksheet, rows
            )
            
            self.logger.info(f"✅ 품질 메트릭 {len(rows)}건 저장 완료")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 품질 메트릭 저장 실패: {e}")
            return False
    
    def _append_rows(self, worksheet, rows):
        """행 추가 (동기 함수)"""
        if rows:
            worksheet.append_rows(rows)
    
    async def create_dashboard(self) -> bool:
        """대시보드 시트 생성"""
        if not self.enabled:
            return False
        
        try:
            # 대시보드 시트가 이미 있는지 확인
            existing_sheets = [sheet.title for sheet in self.spreadsheet.worksheets()]
            
            if "대시보드" not in existing_sheets:
                dashboard = self.spreadsheet.add_worksheet(
                    title="대시보드", rows=50, cols=10
                )
                
                # 대시보드 구성
                dashboard_data = [
                    [
                        "📊 주식 분석 시스템 대시보드",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                    ],
                    ["", "", "", "", "", "", "", "", "", ""],
                    ["🕐 마지막 업데이트:", f"=NOW()", "", "", "", "", "", "", "", ""],
                    [
                        "📈 총 분석 종목:",
                        f"=COUNTA(주식데이터!C:C)-1",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                    ],
                    [
                        "🎯 AI 추천 종목:",
                        f"=COUNTA(AI분석결과!C:C)-1",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                    ],
                    [
                        "📊 평균 품질 점수:",
                        f"=AVERAGE(데이터품질!H:H)",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                    ],
                    ["", "", "", "", "", "", "", "", "", ""],
                    ["📋 최근 추천 종목 (TOP 10)", "", "", "", "", "", "", "", "", ""],
                    [
                        "순위",
                        "종목명",
                        "종목코드",
                        "추천점수",
                        "변동률",
                        "추천사유",
                        "",
                        "",
                        "",
                        "",
                    ],
                ]
                
                # 최근 추천 종목 데이터 (수식으로 자동 업데이트)
                for i in range(10):
                    row_num = i + 10
                    dashboard_data.append(
                        [
                            f"=ROW()-9",
                            f"=INDEX(AI분석결과!D:D,ROW()-9)",
                            f"=INDEX(AI분석결과!C:C,ROW()-9)",
                            f"=INDEX(AI분석결과!E:E,ROW()-9)",
                            f"=INDEX(주식데이터!F:F,MATCH(INDEX(AI분석결과!C:C,ROW()-9),주식데이터!C:C,0))",
                            f"=INDEX(AI분석결과!I:I,ROW()-9)",
                            "",
                            "",
                            "",
                            "",
                        ]
                    )
                
                # 데이터 입력
                dashboard.update("A1:J19", dashboard_data)
                
                # 스타일링
                dashboard.format(
                    "A1:J1",
                    {
                        "backgroundColor": {"red": 0.1, "green": 0.5, "blue": 0.8},
                        "textFormat": {
                            "bold": True,
                            "fontSize": 16,
                            "foregroundColor": {"red": 1, "green": 1, "blue": 1},
                        },
                    },
                )

                dashboard.format(
                    "A8:F8",
                    {
                        "backgroundColor": {"red": 0.9, "green": 0.9, "blue": 0.9},
                        "textFormat": {"bold": True},
                    },
                )
                
                self.logger.info("✅ 대시보드 생성 완료")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 대시보드 생성 실패: {e}")
            return False
    
    async def update_dashboard_data(self, update_data: Dict[str, Any]) -> bool:
        """대시보드 데이터 업데이트"""
        if not self.enabled:
            return False
        
        try:
            dashboard = self.spreadsheet.worksheet("대시보드")
            
            # 실시간 통계 업데이트
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # 올바른 셀 업데이트 방식 사용
            dashboard.update("C3", [[current_time]])  # 마지막 업데이트 시간
            
            # 추가 통계가 있으면 업데이트
            if "total_stocks_analyzed" in update_data:
                dashboard.update("B4", [[str(update_data["total_stocks_analyzed"])]])

            if "recommendations_count" in update_data:
                dashboard.update("B5", [[str(update_data["recommendations_count"])]])

            if "average_quality_score" in update_data:
                dashboard.update("B6", [[str(update_data["average_quality_score"])]])

            if "system_status" in update_data:
                dashboard.update("B7", [[str(update_data["system_status"])]])

            # 추가 상태 정보
            if "market_status" in update_data:
                dashboard.update(
                    "A20", [[f"📊 시장 상황: {update_data['market_status']}"]]
                )

            if "top_performer" in update_data:
                dashboard.update(
                    "A21", [[f"🏆 최고 수익률: {update_data['top_performer']}"]]
                )
            
            self.logger.info("✅ 대시보드 업데이트 완료")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 대시보드 업데이트 실패: {e}")
            return False
    
    async def export_to_csv(self, sheet_name: str, file_path: str) -> bool:
        """시트 데이터를 CSV로 내보내기"""
        if not self.enabled:
            return False
        
        try:
            worksheet = self.spreadsheet.worksheet(sheet_name)
            data = worksheet.get_all_records()
            
            df = pd.DataFrame(data)
            df.to_csv(file_path, index=False, encoding="utf-8-sig")
            
            self.logger.info(f"✅ {sheet_name} 데이터 CSV 내보내기 완료: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ CSV 내보내기 실패: {e}")
            return False
    
    def get_sheet_info(self) -> Dict[str, Any]:
        """시트 정보 반환"""
        if not self.spreadsheet:
            return {}
        
        try:
            sheets_info = []
            for sheet in self.spreadsheet.worksheets():
                sheets_info.append(
                    {
                        "name": sheet.title,
                        "rows": sheet.row_count,
                        "cols": sheet.col_count,
                        "url": f"https://docs.google.com/spreadsheets/d/{self.spreadsheet_id}/edit#gid={sheet.id}",
                    }
                )
            
            return {
                "spreadsheet_id": self.spreadsheet_id,
                "spreadsheet_url": f"https://docs.google.com/spreadsheets/d/{self.spreadsheet_id}/edit",
                "sheets": sheets_info,
                "total_sheets": len(sheets_info),
            }
            
        except Exception as e:
            self.logger.error(f"❌ 시트 정보 조회 실패: {e}")
            return {}
    
    async def health_check(self) -> bool:
        """구글 시트 헬스 체크"""
        if not self.enabled:
            self.logger.warning("⚠️ 구글 시트 비활성화 상태")
            return False

        try:
            # 클라이언트가 초기화되었는지 확인
            if not self.client or not self.spreadsheet:
                self._initialize_client()

            if not self.client or not self.spreadsheet:
                self.logger.error("❌ 구글 시트 클라이언트 초기화 실패")
                return False

            # 스프레드시트 메타데이터 접근 테스트
            title = self.spreadsheet.title
            self.logger.info(f"✅ 구글 시트 연결 확인: '{title}'")
            return True
        except gspread.exceptions.APIError as e:
            self.logger.error(f"❌ 구글 시트 API 오류: {e}")
            return False
        except Exception as e:
            self.logger.error(f"❌ 구글 시트 헬스 체크 실패: {e}")
            return False
    
    async def cleanup_old_data(self, days_to_keep: int = 30) -> bool:
        """오래된 데이터 정리"""
        if not self.enabled:
            return False
        
        try:
            cutoff_date = (datetime.now() - timedelta(days=days_to_keep)).strftime(
                "%Y-%m-%d"
            )
            
            # 각 시트에서 오래된 데이터 삭제
            for sheet_name in ["주식데이터", "AI분석결과", "오류로그", "데이터품질"]:
                try:
                    worksheet = self.spreadsheet.worksheet(sheet_name)
                    all_records = worksheet.get_all_records()
                    
                    # 삭제할 행 찾기
                    rows_to_delete = []
                    for i, record in enumerate(all_records, start=2):  # 헤더 제외
                        if record.get("날짜", "") < cutoff_date:
                            rows_to_delete.append(i)
                    
                    # 역순으로 삭제 (인덱스 변화 방지)
                    for row_num in reversed(rows_to_delete):
                        worksheet.delete_rows(row_num)
                    
                    if rows_to_delete:
                        self.logger.info(
                            f"✅ {sheet_name}에서 {len(rows_to_delete)}개 행 정리"
                        )
                
                except Exception as e:
                    self.logger.warning(f"⚠️ {sheet_name} 정리 실패: {e}")
                    continue
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 데이터 정리 실패: {e}")
            return False


if __name__ == "__main__":
    # 테스트 실행
    async def test_sheets_manager():
        manager = GoogleSheetsManager()
        
        if manager.enabled:
            print("✅ 구글 시트 연결 성공")
            
            # 대시보드 생성
            await manager.create_dashboard()
            
            # 테스트 데이터 저장
            test_stock_data = [
                {
                    "symbol": "005930",
                    "name": "삼성전자",
                    "price": 75000,
                    "change_percent": 2.5,
                    "volume": 1000000,
                    "source": "naver",
                    "quality_score": 95.0,
                }
            ]
            
            await manager.save_stock_data(test_stock_data)
            
            # 시트 정보 출력
            info = manager.get_sheet_info()
            print(f"스프레드시트 URL: {info.get('spreadsheet_url')}")
            
        else:
            print("❌ 구글 시트 연결 실패")
    
    # 비동기 실행
    asyncio.run(test_sheets_manager()) 
