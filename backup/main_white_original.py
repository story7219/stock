#!/usr/bin/env python3
"""
🚀 Ultra Premium HTS - 전문적인 HTS 디자인
하얀 바탕의 깔끔한 전문 트레이딩 시스템
"""

import os
import sys
import asyncio
import threading
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import json

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

# 내부 모듈 import
from src.data_collector import DataCollector, StockData
from src.strategies import StrategyManager
from src.gemini_analyzer import GeminiAnalyzer
from src.technical_analyzer import TechnicalAnalyzer
from src.report_generator import ReportGenerator

# CustomTkinter 설정 - 라이트 모드
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

# 로깅 설정
def setup_logging():
    """로깅 설정"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"ultra_hts_{datetime.now().strftime('%Y%m%d')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

class UltraPremiumHTS:
    """Ultra Premium HTS - 전문적인 HTS 시스템"""
    
    def __init__(self):
        # 로깅 초기화
        setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # 메인 윈도우 생성
        self.root = ctk.CTk()
        self.root.title("Ultra Premium HTS - Professional Trading System")
        self.root.geometry("1800x1200")
        self.root.minsize(1400, 900)
        self.root.configure(fg_color="white")
        
        # 환경 변수 로드
        self.gemini_api_key = os.getenv('GEMINI_API_KEY', '')
        
        # 시스템 컴포넌트 초기화
        self.data_collector = DataCollector()
        self.strategy_manager = StrategyManager()
        self.technical_analyzer = TechnicalAnalyzer()
        self.report_generator = ReportGenerator()
        
        # Gemini AI 초기화
        self.gemini_analyzer = None
        if self.gemini_api_key:
            try:
                self.gemini_analyzer = GeminiAnalyzer(self.gemini_api_key)
                self.logger.info("✅ Gemini AI 초기화 완료")
            except Exception as e:
                self.logger.warning(f"⚠️ Gemini AI 초기화 실패: {e}")
        
        # 데이터 저장소
        self.market_data: Dict[str, List[StockData]] = {}
        self.strategy_results: Dict[str, Any] = {}
        self.gemini_result = None
        self.current_market = "코스피200"
        
        # GUI 상태
        self.is_running = False
        
        # GUI 구성
        self._create_layout()
        self._update_time()
        
        self.logger.info("Ultra Premium HTS 초기화 완료")
    
    def _create_layout(self):
        """GUI 레이아웃 생성"""
        # 메인 프레임 설정 (3분할)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(1, weight=1)
        
        # 상단 헤더
        self._create_header()
        
        # 좌측 패널
        self._create_left_panel()
        
        # 중앙 패널 (차트)
        self._create_center_panel()
        
        # 우측 패널 (AI 분석)
        self._create_right_panel()
        
        # 하단 상태바
        self._create_status_bar()
    
    def _create_header(self):
        """상단 헤더 및 탭 메뉴"""
        self.header_frame = ctk.CTkFrame(
            self.root, 
            height=80, 
            fg_color="#f0f0f0", 
            corner_radius=0
        )
        self.header_frame.grid(row=0, column=0, columnspan=3, sticky="ew")
        self.header_frame.grid_columnconfigure(1, weight=1)
        
        # 로고 및 제목
        self.logo_frame = ctk.CTkFrame(self.header_frame, fg_color="transparent")
        self.logo_frame.grid(row=0, column=0, padx=20, pady=15, sticky="w")
        
        self.title_label = ctk.CTkLabel(
            self.logo_frame,
            text="🚀 지수 종합",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color="#2c3e50"
        )
        self.title_label.pack()
        
        # 탭 메뉴
        self.tab_frame = ctk.CTkFrame(self.header_frame, fg_color="transparent")
        self.tab_frame.grid(row=0, column=1, pady=15)
        
        self.market_tabs = {}
        markets = ["코스피200", "나스닥100", "S&P500"]
        colors = ["#4a90e2", "#5cb85c", "#f0ad4e"]
        
        for i, (market, color) in enumerate(zip(markets, colors)):
            tab_btn = ctk.CTkButton(
                self.tab_frame,
                text=market,
                width=120,
                height=40,
                fg_color=color if market == self.current_market else "#ecf0f1",
                hover_color=color,
                text_color="white" if market == self.current_market else "#2c3e50",
                font=ctk.CTkFont(size=14, weight="bold"),
                command=lambda m=market: self.switch_market(m)
            )
            tab_btn.pack(side="left", padx=5)
            self.market_tabs[market] = tab_btn
        
        # 시간 표시
        self.time_frame = ctk.CTkFrame(self.header_frame, fg_color="transparent")
        self.time_frame.grid(row=0, column=2, padx=20, pady=15, sticky="e")
        
        self.time_label = ctk.CTkLabel(
            self.time_frame,
            text="",
            font=ctk.CTkFont(size=12),
            text_color="#7f8c8d"
        )
        self.time_label.pack()
    
    def _create_left_panel(self):
        """좌측 패널 - AI 설정 및 종목 리스트"""
        self.left_frame = ctk.CTkFrame(
            self.root, 
            width=280, 
            fg_color="white", 
            corner_radius=0
        )
        self.left_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 1))
        self.left_frame.grid_rowconfigure(2, weight=1)
        
        # AI 설정 섹션
        self.ai_settings_frame = ctk.CTkFrame(
            self.left_frame, 
            fg_color="#f8f9fa", 
            height=150
        )
        self.ai_settings_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        
        # AI 설정 제목
        ai_title = ctk.CTkLabel(
            self.ai_settings_frame,
            text="🧠 AI 설정 전략",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#2c3e50"
        )
        ai_title.pack(pady=(10, 5))
        
        # 투자 전략 체크박스
        strategies = ["워런 버핏 가치투자", "피터 린치 성장주", "벤저민 그레이엄 안전마진"]
        self.strategy_vars = {}
        
        for strategy in strategies:
            var = ctk.BooleanVar(value=True)
            self.strategy_vars[strategy] = var
            checkbox = ctk.CTkCheckBox(
                self.ai_settings_frame,
                text=strategy,
                variable=var,
                font=ctk.CTkFont(size=11),
                text_color="#34495e"
            )
            checkbox.pack(anchor="w", padx=15, pady=2)
        
        # AI 100% 정확 옵션
        self.ai_accuracy_var = ctk.BooleanVar(value=True)
        ai_accuracy_cb = ctk.CTkCheckBox(
            self.ai_settings_frame,
            text="✅ AI 100% 정확 옵션",
            variable=self.ai_accuracy_var,
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#27ae60"
        )
        ai_accuracy_cb.pack(anchor="w", padx=15, pady=(10, 15))
        
        # 종목 리스트 제목
        list_title = ctk.CTkLabel(
            self.left_frame,
            text="📊 추천 종목 리스트",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#2c3e50"
        )
        list_title.grid(row=1, column=0, pady=(10, 5), sticky="w", padx=10)
        
        # 종목 리스트
        self.stock_list_frame = ctk.CTkScrollableFrame(
            self.left_frame,
            fg_color="white",
            scrollbar_fg_color="#ecf0f1"
        )
        self.stock_list_frame.grid(row=2, column=0, sticky="nsew", padx=5, pady=(0, 5))
        
        # 샘플 종목 데이터 표시
        self._populate_stock_list()
    
    def _populate_stock_list(self):
        """종목 리스트 채우기"""
        sample_stocks = [
            ("005930", "삼성전자", "+2.3%", "#27ae60"),
            ("000660", "SK하이닉스", "+1.8%", "#27ae60"),
            ("035420", "NAVER", "+3.1%", "#27ae60"),
            ("005380", "현대차", "-0.5%", "#e74c3c"),
            ("051910", "LG화학", "+2.7%", "#27ae60"),
            ("006400", "삼성SDI", "+1.2%", "#27ae60"),
            ("035720", "카카오", "+0.8%", "#27ae60"),
            ("207940", "삼성바이오로직스", "+1.5%", "#27ae60"),
            ("068270", "셀트리온", "-0.3%", "#e74c3c"),
            ("323410", "카카오뱅크", "+2.1%", "#27ae60")
        ]
        
        for code, name, change, color in sample_stocks:
            self._create_stock_item(code, name, change, color)
    
    def _create_stock_item(self, code, name, change, color):
        """개별 종목 아이템 생성"""
        item_frame = ctk.CTkFrame(
            self.stock_list_frame, 
            height=40, 
            fg_color="#f8f9fa"
        )
        item_frame.pack(fill="x", padx=2, pady=1)
        item_frame.grid_columnconfigure(1, weight=1)
        
        # 종목코드
        code_label = ctk.CTkLabel(
            item_frame,
            text=code,
            font=ctk.CTkFont(size=10),
            text_color="#7f8c8d",
            width=60
        )
        code_label.grid(row=0, column=0, padx=8, pady=8, sticky="w")
        
        # 종목명
        name_label = ctk.CTkLabel(
            item_frame,
            text=name,
            font=ctk.CTkFont(size=11),
            text_color="#2c3e50"
        )
        name_label.grid(row=0, column=1, padx=5, pady=8, sticky="w")
        
        # 등락률
        change_label = ctk.CTkLabel(
            item_frame,
            text=change,
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=color,
            width=50
        )
        change_label.grid(row=0, column=2, padx=8, pady=8, sticky="e")
    
    def _create_center_panel(self):
        """중앙 패널 - 차트"""
        self.center_frame = ctk.CTkFrame(
            self.root, 
            fg_color="white", 
            corner_radius=0
        )
        self.center_frame.grid(row=1, column=1, sticky="nsew", padx=(0, 1))
        self.center_frame.grid_rowconfigure(1, weight=1)
        
        # 차트 헤더
        self.chart_header = ctk.CTkFrame(
            self.center_frame, 
            height=60, 
            fg_color="#f8f9fa"
        )
        self.chart_header.grid(row=0, column=0, sticky="ew")
        
        # 현재 지수 정보
        info_frame = ctk.CTkFrame(self.chart_header, fg_color="transparent")
        info_frame.pack(side="left", padx=20, pady=10)
        
        self.index_title = ctk.CTkLabel(
            info_frame,
            text="AI 분석 후 종목을 선택하세요",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#2c3e50"
        )
        self.index_title.pack()
        
        self.index_value = ctk.CTkLabel(
            info_frame,
            text="코스피200 - 2,668.42 (+0.11%)",
            font=ctk.CTkFont(size=14),
            text_color="#27ae60"
        )
        self.index_value.pack()
        
        # 차트 컨테이너
        self.chart_container = ctk.CTkFrame(self.center_frame, fg_color="white")
        self.chart_container.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        
        # 차트 생성
        self._create_chart()
    
    def _create_chart(self):
        """전문적인 차트 생성"""
        # matplotlib 설정
        plt.rcParams['font.family'] = ['Malgun Gothic', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig = Figure(figsize=(12, 8), dpi=100, facecolor='white')
        
        # 상단 차트 (가격)
        ax1 = fig.add_subplot(211)
        ax1.set_facecolor('white')
        
        # 샘플 데이터
        import datetime as dt
        dates = [dt.datetime(2024, 5, 25) + dt.timedelta(days=i) for i in range(24)]
        prices = [2640, 2645, 2647, 2642, 2655, 2667, 2663, 2665, 2668, 2675, 2672, 2670, 
                 2665, 2661, 2659, 2663, 2668, 2672, 2675, 2678, 2674, 2670, 2668, 2669]
        
        ax1.plot(dates, prices, color='#e74c3c', linewidth=2.5)
        ax1.fill_between(dates, prices, alpha=0.1, color='#e74c3c')
        ax1.set_title('코스피200 - 2,668.42 (+0.11%)', 
                     fontsize=14, fontweight='bold', pad=20, color='#2c3e50')
        ax1.grid(True, alpha=0.3, color='#bdc3c7')
        ax1.set_ylabel('지수', fontsize=11, color='#2c3e50')
        ax1.tick_params(colors='#2c3e50')
        
        # 하단 차트 (거래량)
        ax2 = fig.add_subplot(212)
        ax2.set_facecolor('white')
        
        volumes = [1900, 3600, 4800, 2700, 2600, 3400, 4700, 3900, 4400, 1600, 4800, 2900,
                  1800, 1400, 2900, 3800, 3200, 1900, 2500, 1800, 1700, 1800, 1900, 1800]
        
        colors = ['#e74c3c' if v > 3000 else '#95a5a6' for v in volumes]
        ax2.bar(dates, volumes, color=colors, alpha=0.7, width=0.8)
        ax2.set_ylabel('거래량', fontsize=11, color='#2c3e50')
        ax2.grid(True, alpha=0.3, color='#bdc3c7')
        ax2.tick_params(colors='#2c3e50')
        
        # 날짜 포맷
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0, ha='center')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0, ha='center')
        
        fig.tight_layout(pad=3.0)
        
        # 차트를 tkinter에 임베드
        self.chart_canvas = FigureCanvasTkAgg(fig, self.chart_container)
        self.chart_canvas.get_tk_widget().pack(fill="both", expand=True)
    
    def _create_right_panel(self):
        """우측 패널 - AI 분석 결과"""
        self.right_frame = ctk.CTkFrame(
            self.root, 
            width=350, 
            fg_color="white", 
            corner_radius=0
        )
        self.right_frame.grid(row=1, column=2, sticky="nsew")
        self.right_frame.grid_rowconfigure(2, weight=1)
        
        # AI 분석 헤더
        ai_header = ctk.CTkFrame(
            self.right_frame, 
            height=60, 
            fg_color="#2ecc71"
        )
        ai_header.grid(row=0, column=0, sticky="ew")
        
        ai_title = ctk.CTkLabel(
            ai_header,
            text="🤖 AI 종합분석 결과",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="white"
        )
        ai_title.pack(pady=15)
        
        # AI 분석 컨트롤
        control_frame = ctk.CTkFrame(
            self.right_frame, 
            fg_color="#f8f9fa", 
            height=120
        )
        control_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        
        # AI 분석 버튼
        self.ai_analyze_btn = ctk.CTkButton(
            control_frame,
            text="🧠 AI 전략 분석 실행",
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#3498db",
            hover_color="#2980b9",
            command=self.start_ai_analysis
        )
        self.ai_analyze_btn.pack(pady=15)
        
        # 분석 상태 메시지
        self.analysis_status = ctk.CTkLabel(
            control_frame,
            text="❌ AI 분석 후 종목을 발견했습니다.\n오류 내용: 'AIManager' object has no attribute 'analyze_market'\n다시 시도해주세요.",
            font=ctk.CTkFont(size=11),
            text_color="#e74c3c",
            justify="left"
        )
        self.analysis_status.pack(pady=(0, 15))
        
        # AI 분석 결과 영역
        self.result_frame = ctk.CTkScrollableFrame(
            self.right_frame,
            fg_color="white",
            scrollbar_fg_color="#ecf0f1"
        )
        self.result_frame.grid(row=2, column=0, sticky="nsew", padx=5, pady=(0, 5))
        
        # 하단 정보
        bottom_info = ctk.CTkFrame(
            self.right_frame, 
            height=40, 
            fg_color="#ecf0f1"
        )
        bottom_info.grid(row=3, column=0, sticky="ew")
        
        self.update_time_label = ctk.CTkLabel(
            bottom_info,
            text="📊 마지막 업데이트 시간: 2025-06-23 19:45:59",
            font=ctk.CTkFont(size=10),
            text_color="#7f8c8d"
        )
        self.update_time_label.pack(pady=10)
    
    def _create_status_bar(self):
        """하단 상태바"""
        self.status_frame = ctk.CTkFrame(
            self.root, 
            height=30, 
            fg_color="#ecf0f1", 
            corner_radius=0
        )
        self.status_frame.grid(row=2, column=0, columnspan=3, sticky="ew")
        
        self.status_label = ctk.CTkLabel(
            self.status_frame,
            text="시스템 준비 완료 | 코스피200·나스닥100·S&P500 전체 종목 분석 시스템",
            font=ctk.CTkFont(size=10),
            text_color="#7f8c8d"
        )
        self.status_label.pack(pady=5)
    
    # ===== 이벤트 핸들러 =====
    
    def switch_market(self, market):
        """시장 탭 전환"""
        self.current_market = market
        
        # 탭 버튼 색상 업데이트
        colors = {"코스피200": "#4a90e2", "나스닥100": "#5cb85c", "S&P500": "#f0ad4e"}
        
        for m, btn in self.market_tabs.items():
            if m == market:
                btn.configure(fg_color=colors[m], text_color="white")
            else:
                btn.configure(fg_color="#ecf0f1", text_color="#2c3e50")
        
        # 차트 및 데이터 업데이트
        self._update_market_data(market)
        self.logger.info(f"시장 전환: {market}")
    
    def _update_market_data(self, market):
        """시장 데이터 업데이트"""
        market_info = {
            "코스피200": ("코스피200 - 2,668.42 (+0.11%)", "#27ae60"),
            "나스닥100": ("나스닥100 - 19,850.33 (+0.85%)", "#27ae60"),
            "S&P500": ("S&P500 - 5,447.87 (+0.23%)", "#27ae60")
        }
        
        if market in market_info:
            info, color = market_info[market]
            self.index_value.configure(text=info, text_color=color)
    
    def start_ai_analysis(self):
        """AI 분석 시작"""
        if self.is_running:
            self.analysis_status.configure(
                text="⚠️ 분석이 이미 진행 중입니다...",
                text_color="#f39c12"
            )
            return
        
        self.is_running = True
        self.ai_analyze_btn.configure(state="disabled", text="🔄 분석 중...")
        self.analysis_status.configure(
            text="🔍 AI가 시장을 분석하고 있습니다...\n선택된 전략을 바탕으로 최적 종목을 찾는 중...",
            text_color="#3498db"
        )
        
        # 백그라운드에서 분석 실행
        threading.Thread(target=self._run_ai_analysis, daemon=True).start()
    
    def _run_ai_analysis(self):
        """AI 분석 실행 (백그라운드)"""
        try:
            # 시뮬레이션 - 실제로는 Gemini AI 호출
            import time
            time.sleep(3)  # 분석 시뮬레이션
            
            # 성공 결과 시뮬레이션
            self.root.after(0, self._show_analysis_success)
            
        except Exception as e:
            error_msg = f"❌ AI 분석 실패: {str(e)}"
            self.root.after(0, lambda: self._show_analysis_error(error_msg))
        finally:
            self.is_running = False
            self.root.after(0, self._reset_analysis_button)
    
    def _show_analysis_success(self):
        """분석 성공 결과 표시"""
        self.analysis_status.configure(
            text="✅ AI 분석 완료!\nTop 5 종목이 선정되었습니다.",
            text_color="#27ae60"
        )
        
        # 결과 영역에 Top 5 종목 표시
        self._display_top5_results()
    
    def _show_analysis_error(self, error_msg):
        """분석 오류 표시"""
        self.analysis_status.configure(
            text=error_msg,
            text_color="#e74c3c"
        )
    
    def _reset_analysis_button(self):
        """분석 버튼 리셋"""
        self.ai_analyze_btn.configure(state="normal", text="🧠 AI 전략 분석 실행")
    
    def _display_top5_results(self):
        """Top 5 결과 표시"""
        # 기존 결과 제거
        for widget in self.result_frame.winfo_children():
            widget.destroy()
        
        # Top 5 종목 (시뮬레이션 데이터)
        top5_stocks = [
            ("1위", "삼성전자", "005930", "95.2점", "#2ecc71"),
            ("2위", "SK하이닉스", "000660", "92.8점", "#27ae60"),
            ("3위", "NAVER", "035420", "90.5점", "#f39c12"),
            ("4위", "LG화학", "051910", "88.9점", "#e67e22"),
            ("5위", "카카오", "035720", "86.3점", "#e74c3c")
        ]
        
        for rank, name, code, score, color in top5_stocks:
            self._create_result_item(rank, name, code, score, color)
    
    def _create_result_item(self, rank, name, code, score, color):
        """결과 아이템 생성"""
        item_frame = ctk.CTkFrame(
            self.result_frame,
            height=60,
            fg_color="#f8f9fa"
        )
        item_frame.pack(fill="x", padx=5, pady=3)
        item_frame.grid_columnconfigure(1, weight=1)
        
        # 순위
        rank_label = ctk.CTkLabel(
            item_frame,
            text=rank,
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=color,
            width=40
        )
        rank_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")
        
        # 종목 정보
        info_frame = ctk.CTkFrame(item_frame, fg_color="transparent")
        info_frame.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        
        name_label = ctk.CTkLabel(
            info_frame,
            text=name,
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#2c3e50"
        )
        name_label.pack(anchor="w")
        
        code_label = ctk.CTkLabel(
            info_frame,
            text=code,
            font=ctk.CTkFont(size=10),
            text_color="#7f8c8d"
        )
        code_label.pack(anchor="w")
        
        # 점수
        score_label = ctk.CTkLabel(
            item_frame,
            text=score,
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=color,
            width=60
        )
        score_label.grid(row=0, column=2, padx=10, pady=10, sticky="e")
    
    def _update_time(self):
        """시간 업데이트"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.time_label.configure(text=current_time)
        self.update_time_label.configure(text=f"📊 마지막 업데이트 시간: {current_time}")
        
        # 1초마다 업데이트
        self.root.after(1000, self._update_time)
    
    def run(self):
        """애플리케이션 실행"""
        self.logger.info("Ultra Premium HTS 시작")
        self.root.mainloop()

def main():
    """메인 함수"""
    try:
        app = UltraPremiumHTS()
        app.run()
    except Exception as e:
        logging.error(f"애플리케이션 실행 오류: {e}")
        messagebox.showerror("오류", f"애플리케이션 실행 실패: {e}")

if __name__ == "__main__":
    main() 