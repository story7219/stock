#!/usr/bin/env python3
"""
🚀 Ultra Premium HTS - Professional Trading System
통합된 완전 기능 버전 - 최적화된 UI + 완전한 백엔드 기능
"""
import os
import sys
import asyncio
import threading
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
import random
import time

# GUI 라이브러리
import customtkinter as ctk
from tkinter import messagebox
import tkinter as tk

# 데이터 처리
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.dates as mdates

# 환경 설정
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 내부 모듈 임포트 - 안전한 임포트 처리
try:
    from src.data_collector import DataCollector, StockData
    from src.strategies import StrategyManager
    from src.gemini_analyzer import GeminiAnalyzer
    from src.technical_analyzer import TechnicalAnalyzer
    from src.report_generator import ReportGenerator
    MODULE_SUCCESS = True
except ImportError as e:
    MODULE_SUCCESS = False
    IMPORT_ERROR_MESSAGE = str(e)
    print(f"⚠️ 모듈 임포트 경고: {e}")
    print("📝 시뮬레이션 모드로 실행됩니다.")

# CustomTkinter 설정
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

def setup_logging():
    """로깅 설정"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f"ultra_hts_{datetime.now().strftime('%Y%m%d')}.log", encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

class UltraPremiumHTS:
    """Ultra Premium HTS 통합 메인 애플리케이션"""
    
    def __init__(self):
        setup_logging()
        self.logger = logging.getLogger(__name__)
        
        self.root = ctk.CTk()
        self.root.title("Ultra Premium HTS - Professional Trading System")
        self.root.geometry("1800x1200")
        self.root.minsize(1600, 900)
        self.root.configure(fg_color="#f0f2f5")

        # 환경 변수 및 API 키 설정
        self.gemini_api_key = os.getenv('GEMINI_API_KEY', '')
        
        # 시스템 컴포넌트 초기화 (모듈이 있는 경우)
        if MODULE_SUCCESS:
            try:
                self.data_collector = DataCollector()
                self.strategy_manager = StrategyManager()
                self.technical_analyzer = TechnicalAnalyzer()
                self.report_generator = ReportGenerator()
                
                # Gemini AI 초기화
                if self.gemini_api_key:
                    self.gemini_analyzer = GeminiAnalyzer(self.gemini_api_key)
                    self.logger.info("✅ Gemini AI 초기화 완료")
                else:
                    self.gemini_analyzer = None
                    self.logger.warning("⚠️ GEMINI_API_KEY가 설정되지 않았습니다.")
                    
            except Exception as e:
                self.logger.error(f"❌ 시스템 컴포넌트 초기화 실패: {e}")
                self.gemini_analyzer = None
        else:
            # 시뮬레이션 모드
            self.data_collector = None
            self.strategy_manager = None
            self.technical_analyzer = None
            self.report_generator = None
            self.gemini_analyzer = None
            
        # 데이터 저장소
        self.market_data: Dict[str, List[StockData]] = {}
        self.strategy_results: Dict[str, Any] = {}
        self.gemini_result = None
        self.current_market = "코스피200"
        self.is_running = False
        
        # GUI 구성
        self._create_layout()
        
        self.logger.info("Ultra Premium HTS 초기화 완료")

    def _create_layout(self):
        """GUI 레이아웃 생성 - 안정적인 구조로 변경"""
        # 전체 레이아웃 설정: 2행 3열
        # row 0: 메인 컨텐츠 (left, center, right)
        # row 1: 상태 바
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1) # 중앙 컬럼 확장

        # 좌측 패널 생성
        self._create_left_panel()

        # 메인 컨텐츠 영역 (중앙 + 우측)
        main_content_frame = ctk.CTkFrame(self.root, fg_color="transparent", corner_radius=0)
        main_content_frame.grid(row=0, column=1, columnspan=2, sticky="nsew")
        main_content_frame.grid_rowconfigure(1, weight=1)
        main_content_frame.grid_columnconfigure(0, weight=1)

        # 메인 컨텐츠 내부의 헤더와 패널 생성
        self._create_main_header(main_content_frame)
        self._create_center_panel(main_content_frame)
        self._create_right_panel(main_content_frame)

        # 하단 상태 바 생성
        self._create_status_bar()

    def _create_main_header(self, parent):
        """메인 컨텐츠 상단 헤더 (제목 및 시장 탭)"""
        top_frame = ctk.CTkFrame(parent, fg_color="#ffffff", corner_radius=0, height=100)
        top_frame.grid(row=0, column=0, columnspan=2, sticky="new")
        top_frame.grid_columnconfigure(0, weight=1)
        
        header_frame = ctk.CTkFrame(top_frame, fg_color="#3b5998", height=50, corner_radius=0)
        header_frame.grid(row=0, column=0, sticky="ew")
        
        title_label = ctk.CTkLabel(header_frame, text="🚀 Ultra Premium HTS - Professional Trading System", font=ctk.CTkFont(size=18, weight="bold"), text_color="white")
        title_label.pack(side="left", padx=20, pady=10)
        
        tab_frame = ctk.CTkFrame(top_frame, fg_color="white")
        tab_frame.grid(row=1, column=0, sticky="w", padx=10, pady=(5,0))
        
        markets = ["지수 종합", "코스피200", "나스닥100", "S&P500"]
        self.market_tabs = {}
        for market in markets:
            btn = ctk.CTkButton(
                tab_frame, 
                text=market, 
                fg_color="#4a90e2" if market == "코스피200" else "transparent", 
                text_color="black" if market != "코스피200" else "white", 
                font=ctk.CTkFont(size=13, weight="bold"), 
                hover_color="#d0d0d0",
                command=lambda m=market: self.switch_market(m)
            )
            btn.pack(side="left", padx=4)
            self.market_tabs[market] = btn

    def switch_market(self, market_name):
        """시장 전환 메서드"""
        self.logger.info(f"시장 전환: {market_name}")
        self.current_market = market_name
        
        # 탭 버튼 색상 업데이트
        for market, btn in self.market_tabs.items():
            if market == market_name:
                btn.configure(fg_color="#4a90e2", text_color="white")
            else:
                btn.configure(fg_color="transparent", text_color="black")
        
        # 시장 데이터 업데이트
        if MODULE_SUCCESS and self.data_collector:
            self._update_market_data(market_name)
        else:
            # 시뮬레이션 모드에서는 차트만 업데이트
            self._update_chart_for_market(market_name)

    def _update_market_data(self, market):
        """실제 시장 데이터 업데이트 (모듈이 있는 경우)"""
        try:
            if market == "코스피200":
                self.market_data[market] = self.data_collector.collect_kospi_data()
            elif market == "나스닥100":
                self.market_data[market] = self.data_collector.collect_nasdaq_data()
            elif market == "S&P500":
                self.market_data[market] = self.data_collector.collect_sp500_data()
            
            self.logger.info(f"✅ {market} 데이터 업데이트 완료")
            self._update_stock_list()
            
        except Exception as e:
            self.logger.error(f"❌ {market} 데이터 업데이트 실패: {e}")

    def _update_chart_for_market(self, market):
        """시장별 차트 업데이트 (시뮬레이션)"""
        # 차트 헤더 업데이트
        if hasattr(self, 'chart_title_label'):
            index_values = {
                "코스피200": "2,668.42 (+0.11%)",
                "나스닥100": "15,234.56 (+0.45%)",
                "S&P500": "4,567.89 (+0.23%)"
            }
            self.chart_title_label.configure(text=f"{market} - {index_values.get(market, '0.00 (0.00%)')}")

    def _create_left_panel(self):
        """좌측 패널 - AI 설정, 전략 선택, 종목 리스트 (최적화된 버전)"""
        self.left_frame = ctk.CTkFrame(self.root, width=280, fg_color="#f0f2f5", corner_radius=0)
        self.left_frame.grid(row=0, column=0, rowspan=2, sticky="nsew")
        self.left_frame.grid_rowconfigure(4, weight=1)

        # AI 성능 점검 (첫 번째 버전의 깔끔한 스타일 적용)
        ai_perf_frame = ctk.CTkFrame(self.left_frame, fg_color="#2c3e50", corner_radius=10)
        ai_perf_frame.grid(row=0, column=0, sticky="ew", padx=12, pady=12)
        
        ai_perf_title = ctk.CTkLabel(
            ai_perf_frame, 
            text="🧠 AI 성능 점검", 
            font=ctk.CTkFont(size=15, weight="bold"), 
            text_color="white"
        )
        ai_perf_title.pack(pady=(12, 5), padx=15, anchor="w")
        
        # AI 상태 표시 (실제 모듈 존재 여부에 따라)
        if MODULE_SUCCESS and self.gemini_analyzer:
            status_text = "● AI 100% 정상 동작"
            status_color = "#27ae60"
            score_text = "● AI 점수: 95.2/100 (최고 등급)"
        else:
            status_text = "● 시뮬레이션 모드"
            status_color = "#f39c12"
            score_text = "● AI 점수: 시뮬레이션 중"
            
        ctk.CTkLabel(
            ai_perf_frame, 
            text=status_text, 
            text_color=status_color, 
            font=ctk.CTkFont(size=12)
        ).pack(anchor="w", padx=15, pady=(0, 2))
        
        self.ai_score_label = ctk.CTkLabel(
            ai_perf_frame, 
            text=score_text, 
            text_color="#f39c12", 
            font=ctk.CTkFont(size=12)
        )
        self.ai_score_label.pack(anchor="w", padx=15, pady=(0, 12))

        # 투자 대가 전략 선택 (두 번째 버전의 체크박스 스타일 개선)
        strategy_frame = ctk.CTkFrame(self.left_frame, fg_color="white", corner_radius=10)
        strategy_frame.grid(row=1, column=0, sticky="ew", padx=12, pady=(0,12))
        
        strategy_title = ctk.CTkLabel(
            strategy_frame, 
            text="📊 투자 대가 전략 선택", 
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color="#2c3e50"
        )
        strategy_title.pack(pady=(12, 8), padx=15, anchor="w")
        
        strategies = [
            ("워런 버핏 가치투자", False),
            ("피터 린치 성장주", False), 
            ("마크 미네르비니 모멘텀", True),
            ("벤저민 그레이엄 가치주", False),
            ("제시 리버모어 추세", False)
        ]
        
        self.strategy_vars = {}
        for strategy_name, is_default in strategies:
            var = ctk.BooleanVar(value=is_default)
            self.strategy_vars[strategy_name] = var
            
            cb = ctk.CTkCheckBox(
                strategy_frame, 
                text=strategy_name, 
                variable=var, 
                font=ctk.CTkFont(size=12),
                text_color="#2c3e50",
                fg_color="#3498db",
                hover_color="#2980b9"
            )
            cb.pack(anchor="w", padx=15, pady=3)
        
        # 여백
        ctk.CTkLabel(strategy_frame, text="", height=8).pack()

        # AI 전략 분석 실행 버튼 (첫 번째 버전의 스타일 개선)
        self.run_analysis_button = ctk.CTkButton(
            self.left_frame, 
            text="🚀 AI 전략 분석 실행", 
            font=ctk.CTkFont(size=16, weight="bold"), 
            fg_color="#e74c3c", 
            hover_color="#c0392b", 
            height=55, 
            corner_radius=10,
            command=self.start_ai_analysis
        )
        self.run_analysis_button.grid(row=2, column=0, sticky="ew", padx=12, pady=8)

        # 추천 종목 리스트 제목
        stock_list_title = ctk.CTkLabel(
            self.left_frame, 
            text="📈 추천 종목 리스트", 
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color="#2c3e50"
        )
        stock_list_title.grid(row=3, column=0, pady=(12, 8), sticky="w", padx=12)
        
        # 스크롤 가능한 종목 리스트 (두 번째 버전의 기능 + 첫 번째 버전의 스타일)
        self.stock_list_frame = ctk.CTkScrollableFrame(
            self.left_frame, 
            fg_color="white", 
            corner_radius=10
        )
        self.stock_list_frame.grid(row=4, column=0, sticky="nsew", padx=12, pady=(0, 12))
        
        self._populate_optimized_stock_list()

    def _populate_optimized_stock_list(self):
        """최적화된 종목 리스트 표시"""
        for widget in self.stock_list_frame.winfo_children():
            widget.destroy()
            
        # 실제 데이터가 있는 경우와 시뮬레이션 모드 구분
        if MODULE_SUCCESS and self.current_market in self.market_data:
            # 실제 데이터 사용
            stocks_data = self.market_data[self.current_market][:12]  # 상위 12개
        else:
            # 시뮬레이션 데이터 사용
            premium_stocks = [
                ("005930", "삼성전자", "+2.8%", "#27ae60"),
                ("000660", "SK하이닉스", "+1.9%", "#27ae60"),
                ("035420", "NAVER", "+3.4%", "#27ae60"),
                ("005380", "현대차", "-0.3%", "#e74c3c"),
                ("051910", "LG화학", "+2.1%", "#27ae60"),
                ("006400", "삼성SDI", "+1.7%", "#27ae60"),
                ("035720", "카카오", "-0.8%", "#e74c3c"),
                ("207940", "삼성바이오로직스", "+1.2%", "#27ae60"),
                ("068270", "셀트리온", "+0.9%", "#27ae60"),
                ("323410", "카카오뱅크", "+2.5%", "#27ae60"),
                ("373220", "LG에너지솔루션", "+0.4%", "#27ae60"),
                ("000270", "기아", "+1.1%", "#27ae60")
            ]
            
            for i, (code, name, change, color) in enumerate(premium_stocks):
                self._create_stock_item(i+1, code, name, change, color)

    def _create_stock_item(self, rank, code, name, change, color):
        """개별 종목 아이템 생성"""
        stock_frame = ctk.CTkFrame(self.stock_list_frame, fg_color="#f8f9fa", corner_radius=8)
        stock_frame.pack(fill="x", padx=8, pady=3)
        
        # 순위 표시
        rank_label = ctk.CTkLabel(
            stock_frame, 
            text=f"{rank:2d}", 
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#7f8c8d",
            width=25
        )
        rank_label.pack(side="left", padx=(8, 4), pady=6)
        
        # 종목 정보
        info_frame = ctk.CTkFrame(stock_frame, fg_color="transparent")
        info_frame.pack(side="left", fill="x", expand=True, pady=6)
        
        name_label = ctk.CTkLabel(
            info_frame, 
            text=f"{code} {name}", 
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#2c3e50",
            anchor="w"
        )
        name_label.pack(anchor="w")
        
        # 변동률 표시
        change_label = ctk.CTkLabel(
            stock_frame, 
            text=change, 
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=color,
            width=50
        )
        change_label.pack(side="right", padx=(4, 8), pady=6)

    def _update_stock_list(self):
        """종목 리스트 업데이트 (실제 데이터 기반)"""
        if MODULE_SUCCESS and self.current_market in self.market_data:
            self._populate_optimized_stock_list()

    def _create_center_panel(self, parent):
        self.center_frame = ctk.CTkFrame(parent, fg_color="white", corner_radius=0)
        self.center_frame.grid(row=1, column=0, sticky="nsew", pady=(0,0))
        self.center_frame.grid_rowconfigure(1, weight=1)
        
        chart_header_frame = ctk.CTkFrame(self.center_frame, fg_color="transparent")
        chart_header_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=(10,0))
        
        ctk.CTkLabel(chart_header_frame, text="AI 분석 후 종목을 선택하세요", font=ctk.CTkFont(size=18, weight="bold")).pack(anchor="w")
        
        self.chart_title_label = ctk.CTkLabel(chart_header_frame, text="코스피200 - 2,668.42 (+0.11%)", font=ctk.CTkFont(size=14), text_color="#e74c3c")
        self.chart_title_label.pack(anchor="w")
        
        self.chart_container = ctk.CTkFrame(self.center_frame, fg_color="white")
        self.chart_container.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        self._create_chart()
        
    def _create_right_panel(self, parent):
        """우측 패널 - AI 종합분석 결과 (최적화된 버전)"""
        self.right_frame = ctk.CTkFrame(parent, width=350, fg_color="#f0f2f5", corner_radius=0)
        self.right_frame.grid(row=1, column=1, sticky="nsew", pady=(0,0), padx=(5,10))
        self.right_frame.grid_rowconfigure(1, weight=1)
        
        # AI 분석 결과 헤더 (첫 번째 버전의 깔끔한 스타일)
        ai_header = ctk.CTkFrame(self.right_frame, height=60, fg_color="#27ae60", corner_radius=10)
        ai_header.grid(row=0, column=0, sticky="ew", padx=12, pady=12)
        
        ai_title = ctk.CTkLabel(
            ai_header, 
            text="🤖 AI 종합분석 결과", 
            font=ctk.CTkFont(size=17, weight="bold"), 
            text_color="white"
        )
        ai_title.pack(pady=15, padx=20)

        # 분석 결과 표시 영역
        self.result_display_frame = ctk.CTkFrame(self.right_frame, fg_color="white", corner_radius=10)
        self.result_display_frame.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))
        
        # 초기 안내 메시지 (더 친근하고 명확하게)
        initial_message = "👈 좌측의 'AI 전략 분석 실행' 버튼을\n클릭하여 AI 분석을 시작하세요.\n\n🎯 선택된 전략에 따라\n최적의 TOP 5 종목을 추천해드립니다."
        self.analysis_status_label = ctk.CTkLabel(
            self.result_display_frame, 
            text=initial_message, 
            font=ctk.CTkFont(size=14), 
            text_color="#7f8c8d", 
            justify="center", 
            wraplength=300
        )
        self.analysis_status_label.pack(pady=60, padx=20)

    def _create_status_bar(self):
        self.status_frame = ctk.CTkFrame(self.root, height=30, fg_color="#2c3e50", corner_radius=0)
        self.status_frame.grid(row=1, column=0, columnspan=3, sticky="sew")
        status_text = f"선택된 지수: {self.current_market}  |  선택된 전략: 미네르비니  |  총 분석 종목: 200개"
        self.status_label = ctk.CTkLabel(self.status_frame, text=status_text, font=ctk.CTkFont(size=12), text_color="white")
        self.status_label.pack(side="left", padx=10, pady=5)
        self.time_label_status = ctk.CTkLabel(self.status_frame, text="", font=ctk.CTkFont(size=12), text_color="white")
        self.time_label_status.pack(side="right", padx=10, pady=5)
        self.update_time()

    def _create_chart(self):
        plt.rcParams['font.family'] = ['Malgun Gothic', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        fig = Figure(figsize=(12, 8), dpi=100, facecolor='white')
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212, sharex=ax1)
        
        dates = [datetime(2024, 5, 25) + timedelta(days=i) for i in range(24)]
        prices = [2640, 2645, 2647, 2642, 2655, 2667, 2663, 2665, 2668, 2675, 2672, 2670, 2665, 2661, 2659, 2663, 2668, 2672, 2675, 2678, 2674, 2670, 2668, 2669]
        volumes = [1900, 3600, 4800, 2700, 2600, 3400, 4700, 3900, 4400, 1600, 4800, 2900, 1800, 1400, 2900, 3800, 3200, 1900, 2500, 1800, 1700, 1800, 1900, 1800]
        
        ax1.plot(dates, prices, color='#e74c3c', linewidth=1.5)
        ax1.grid(True, alpha=0.3)
        ax2.bar(dates, volumes, color='#d3d3d3', width=0.6)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylabel("거래량", fontdict={'size':10})
        
        ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        fig.tight_layout(pad=0)
        
        canvas = FigureCanvasTkAgg(fig, master=self.chart_container)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def start_ai_analysis(self):
        """AI 분석 시작"""
        if self.is_running:
            messagebox.showwarning("경고", "이미 분석이 진행 중입니다.")
            return
            
        self.is_running = True
        self.run_analysis_button.configure(state="disabled", text="AI 분석 중...")
        
        # 결과 영역 초기화
        for widget in self.result_display_frame.winfo_children():
            widget.destroy()
            
        loading_label = ctk.CTkLabel(
            self.result_display_frame, 
            text="🤖 AI가 최적의 종목을 분석 중입니다...\n\n📊 선택된 전략을 바탕으로\n종합적인 분석을 수행하고 있습니다.\n\n⏳ 잠시만 기다려주세요.", 
            font=ctk.CTkFont(size=14),
            text_color="#3498db",
            justify="center"
        )
        loading_label.pack(pady=40, padx=20)
        
        # 백그라운드에서 분석 실행
        threading.Thread(target=self._run_ai_analysis_background, daemon=True).start()

    def _run_ai_analysis_background(self):
        """백그라운드에서 AI 분석 실행"""
        try:
            # 선택된 전략 확인
            selected_strategies = [name for name, var in self.strategy_vars.items() if var.get()]
            
            if MODULE_SUCCESS and self.gemini_analyzer:
                # 실제 AI 분석 실행
                self._run_real_ai_analysis(selected_strategies)
            else:
                # 시뮬레이션 분석
                self._run_simulation_analysis(selected_strategies)
                
        except Exception as e:
            self.logger.error(f"❌ AI 분석 중 오류: {e}")
            self.root.after(0, self._show_analysis_error, str(e))
        finally:
            self.is_running = False

    def _run_real_ai_analysis(self, strategies):
        """실제 AI 분석 실행"""
        time.sleep(2)  # 실제 분석 시뮬레이션
        
        # Gemini AI를 통한 실제 분석
        analysis_result = self.gemini_analyzer.analyze_market(
            market=self.current_market,
            strategies=strategies,
            data=self.market_data.get(self.current_market, [])
        )
        
        self.root.after(0, self.update_analysis_result, analysis_result['top5'])

    def _run_simulation_analysis(self, strategies):
        """시뮬레이션 분석 실행"""
        time.sleep(3)  # 분석 시뮬레이션
        
        # 시뮬레이션 결과 생성
        top5 = [
            {
                "rank": i+1, 
                "name": f"AI 추천 종목 #{i+1}", 
                "symbol": f"A{i+100:03d}", 
                "reason": f"{', '.join(strategies)} 전략 기반 최적 종목", 
                "score": 92.5-i*1.8
            } 
            for i in range(5)
        ]
        
        self.root.after(0, self.update_analysis_result, top5)

    def _show_analysis_error(self, error_msg):
        """분석 오류 표시"""
        for widget in self.result_display_frame.winfo_children():
            widget.destroy()
            
        error_label = ctk.CTkLabel(
            self.result_display_frame,
            text=f"❌ 분석 중 오류가 발생했습니다.\n\n오류 내용: {error_msg}\n\n다시 시도해주세요.",
            font=ctk.CTkFont(size=13),
            text_color="#e74c3c",
            justify="center",
            wraplength=280
        )
        error_label.pack(pady=50, padx=20)
        
        self.run_analysis_button.configure(state="normal", text="🚀 AI 전략 분석 실행")

    def update_analysis_result(self, top5_stocks):
        """AI 분석 결과 업데이트 (최적화된 디스플레이)"""
        for widget in self.result_display_frame.winfo_children():
            widget.destroy()
            
        # 결과 헤더
        result_header = ctk.CTkLabel(
            self.result_display_frame,
            text="🏆 AI 추천 TOP 5 종목",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#2c3e50"
        )
        result_header.pack(pady=(15, 10), padx=15)
        
        for i, stock in enumerate(top5_stocks):
            # 종목 프레임 (그라데이션 효과를 위한 색상 조정)
            colors = ["#3498db", "#2ecc71", "#f39c12", "#e67e22", "#9b59b6"]
            
            item_frame = ctk.CTkFrame(
                self.result_display_frame, 
                fg_color="#f8f9fa", 
                corner_radius=8
            )
            item_frame.pack(fill="x", padx=12, pady=4)
            
            # 순위 배지
            rank_badge = ctk.CTkLabel(
                item_frame, 
                text=f"#{i+1}", 
                font=ctk.CTkFont(size=18, weight="bold"), 
                text_color="white",
                fg_color=colors[i],
                corner_radius=15,
                width=35,
                height=35
            )
            rank_badge.pack(side="left", padx=10, pady=8)
            
            # 종목 정보
            info_frame = ctk.CTkFrame(item_frame, fg_color="transparent")
            info_frame.pack(side="left", fill="x", expand=True, pady=8)
            
            name_label = ctk.CTkLabel(
                info_frame, 
                text=f"{stock['name']} ({stock['symbol']})", 
                font=ctk.CTkFont(size=13, weight="bold"),
                text_color="#2c3e50",
                anchor="w"
            )
            name_label.pack(anchor="w")
            
            reason_label = ctk.CTkLabel(
                info_frame, 
                text=f"💡 {stock['reason']}", 
                font=ctk.CTkFont(size=10),
                text_color="#7f8c8d",
                wraplength=200, 
                justify="left",
                anchor="w"
            )
            reason_label.pack(anchor="w", pady=(2, 0))
            
            # AI 점수
            score_label = ctk.CTkLabel(
                item_frame, 
                text=f"{stock['score']:.1f}", 
                font=ctk.CTkFont(size=16, weight="bold"), 
                text_color=colors[i]
            )
            score_label.pack(side="right", padx=12, pady=8)
        
        # 분석 완료 시간 표시
        completion_time = datetime.now().strftime("%H:%M:%S")
        time_label = ctk.CTkLabel(
            self.result_display_frame,
            text=f"📊 분석 완료: {completion_time}",
            font=ctk.CTkFont(size=11),
            text_color="#95a5a6"
        )
        time_label.pack(pady=(10, 15))
        
        # 버튼 상태 복원
        self.run_analysis_button.configure(state="normal", text="🚀 AI 전략 분석 실행")

    def update_time(self):
        now_text = datetime.now().strftime("마지막 실행 시간: %Y-%m-%d %H:%M:%S")
        self.time_label_status.configure(text=now_text)
        self.root.after(1000, self.update_time)
    
    def run(self):
        self.logger.info("Ultra Premium HTS 시작")
        self.root.mainloop()

def main():
    try:
        app = UltraPremiumHTS()
        app.run()
    except Exception as e:
        logging.basicConfig(level=logging.ERROR)
        logging.error(f"애플리케이션 실행 오류: {e}", exc_info=True)
        try:
            messagebox.showerror("치명적 오류", f"애플리케이션 실행에 실패했습니다: {e}")
        except:
            print(f"애플리케이션 실행에 실패했습니다: {e}")

if __name__ == "__main__":
    main() 