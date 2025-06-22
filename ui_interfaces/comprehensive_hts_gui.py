"""
종합 HTS GUI 시스템
AI 주식 분석 및 차트 시각화 통합 플랫폼
"""
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from datetime import datetime, timedelta
import threading
import time
import os
import sys
import warnings
import asyncio
import json
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = ['Malgun Gothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# --- AI 모듈 로드 ---
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from ai_integration.ultra_ai_analyzer import UltraAIAnalyzer
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    UltraAIAnalyzer = None

# --- 세계 최고 수준 디자인 컬러 팔레트 ---
COLORS = {
    'bg': '#0f0f23',  # 딥 네이비 배경
    'panel': '#1a1a2e',  # 다크 퍼플 패널
    'accent': '#16213e',  # 미드나이트 블루
    'primary': '#0f3460',  # 딥 블루
    'secondary': '#533483',  # 로얄 퍼플
    'success': '#4ade80',  # 모던 그린
    'warning': '#fbbf24',  # 골든 옐로우
    'error': '#ef4444',  # 모던 레드
    'text': '#f8fafc',  # 퓨어 화이트
    'text_secondary': '#cbd5e1',  # 라이트 그레이
    'border': '#374151',  # 다크 그레이
    'chart_bg': '#111827',  # 차트 배경
    'grid': '#374151',  # 그리드 컬러
    'hover': '#1e40af',  # 호버 효과
    'gradient_start': '#667eea',  # 그라디언트 시작
    'gradient_end': '#764ba2',  # 그라디언트 끝
    # 일목균형표 전문 컬러
    'ichimoku_tenkan': '#ff6b6b',  # 전환선 - 코랄 레드
    'ichimoku_kijun': '#4ecdc4',  # 기준선 - 터쿠아즈
    'ichimoku_senkou_a': '#45b7d1',  # 선행스팬A - 스카이 블루
    'ichimoku_senkou_b': '#f9ca24',  # 선행스팬B - 골든 옐로우
    'ichimoku_chikou': '#6c5ce7',  # 후행스팬 - 바이올렛
    'cloud_bullish': '#4ade80',  # 상승 구름 - 그린
    'cloud_bearish': '#ef4444'  # 하락 구름 - 레드
}

# --- 프리미엄 폰트 설정 ---
FONTS = {
    'title': ('Segoe UI', 20, 'bold'),
    'subtitle': ('Segoe UI', 14, 'bold'),
    'body': ('Segoe UI', 11),
    'small': ('Segoe UI', 9),
    'chart_title': ('Segoe UI', 12, 'bold'),
    'button': ('Segoe UI', 10, 'bold'),
    'header': ('Segoe UI', 24, 'bold')
}

# 세계적인 투자 대가들의 투자 철학
GURU_STRATEGIES = {
    'Warren Buffett': '가치투자의 아버지 - 내재가치 중시, 장기투자, 우량기업 선별',
    'Peter Lynch': '성장주 투자의 달인 - PEG 비율 활용, 일상에서 투자 아이디어 발굴',
    'Benjamin Graham': '증권분석의 아버지 - 안전마진 중시, 저평가 주식, 정량적 분석',
    'Philip Fisher': '성장주 투자의 선구자 - 스캐틀버트 방식, 질적 분석 중시',
    'John Templeton': '글로벌 가치투자 - 역발상 투자, 국제적 분산투자',
    'Charlie Munger': '다학제적 사고 - 심리학적 편향 극복, 합리적 의사결정',
    'Joel Greenblatt': '마법공식 투자법 - ROE와 수익률 기반 정량적 선별',
    'David Dreman': '역발상 투자 전략 - 저PER, 저PBR 주식 선호',
    'William O\'Neil': 'CAN SLIM 투자법 - 기술적 분석과 기본적 분석 결합',
    'Ray Dalio': '올웨더 포트폴리오 - 리스크 패리티, 경제 사이클 분석',
    'George Soros': '재귀성 이론 - 시장 심리와 펀더멘털의 상호작용',
    'Carl Icahn': '행동주의 투자 - 기업 지배구조 개선을 통한 가치 창출'
}

# 주요 글로벌 지수 (순서 변경)
MARKET_INDICES = {
    'KOSPI 200': '한국 대형주 200개 기업 지수',
    'KOSDAQ 50': '한국 중소형 성장주 50개 기업 지수',
    'NASDAQ-100': '나스닥 100대 기술주 지수',
    'S&P 500': '미국 대형주 500개 기업 지수'
}

# 재무제표 분석 항목
FINANCIAL_METRICS = {
    'profitability': ['매출액', '영업이익', '당기순이익', '영업이익률', 'ROE', 'ROA', 'ROIC'],
    'stability': ['자기자본비율', '부채비율', '유동비율', '당좌비율', '이자보상배수'],
    'growth': ['매출액증가율', '영업이익증가율', '순이익증가율', 'EPS증가율'],
    'valuation': ['PER', 'PBR', 'PSR', 'PCR', 'EV/EBITDA', 'PEG'],
    'dividend': ['배당수익률', '배당성향', '배당증가율', '연속배당년수']
}

LEFT_PANEL_RATIO = 0.15
RIGHT_PANEL_RATIO = 0.35
MIN_PANEL_WIDTH = 300
NUM_STOCKS_FOR_AI_ANALYSIS = 10


@dataclass
class StockInfo:
    """주식 정보 데이터 클래스 - 한투 증권사 HTS 수준의 상세 정보"""
    name: str
    code: str
    price: float
    change_rate: float
    volume: int
    
    # 기본 재무 정보
    market_cap: int = 0
    per: float = 0.0
    pbr: float = 0.0
    roe: float = 0.0
    debt_ratio: float = 0.0
    dividend_yield: float = 0.0
    
    # 상세 재무 정보
    revenue: int = 0  # 매출액 (억원)
    operating_profit: int = 0  # 영업이익 (억원)
    net_profit: int = 0  # 당기순이익 (억원)
    operating_margin: float = 0.0  # 영업이익률 (%)
    net_margin: float = 0.0  # 순이익률 (%)
    roa: float = 0.0  # 총자산수익률 (%)
    roic: float = 0.0  # 투하자본수익률 (%)
    
    # 안정성 지표
    equity_ratio: float = 0.0  # 자기자본비율 (%)
    current_ratio: float = 0.0  # 유동비율 (%)
    quick_ratio: float = 0.0  # 당좌비율 (%)
    interest_coverage: float = 0.0  # 이자보상배수
    
    # 성장성 지표
    revenue_growth: float = 0.0  # 매출액증가율 (%)
    profit_growth: float = 0.0  # 영업이익증가율 (%)
    net_growth: float = 0.0  # 순이익증가율 (%)
    eps_growth: float = 0.0  # EPS증가율 (%)
    
    # 추가 밸류에이션 지표
    psr: float = 0.0  # 주가매출액비율
    pcr: float = 0.0  # 주가현금흐름비율
    ev_ebitda: float = 0.0  # EV/EBITDA
    peg: float = 0.0  # PEG 비율
    
    # 배당 정보
    dividend_payout: float = 0.0  # 배당성향 (%)
    dividend_growth: float = 0.0  # 배당증가율 (%)
    consecutive_dividend_years: int = 0  # 연속배당년수
    
    # 기술적 분석을 위한 일목균형표 데이터
    tenkan_sen: float = 0.0  # 전환선
    kijun_sen: float = 0.0  # 기준선
    senkou_span_a: float = 0.0  # 선행스팬A
    senkou_span_b: float = 0.0  # 선행스팬B
    chikou_span: float = 0.0  # 후행스팬


class ComprehensiveHTS:
    """종합 HTS 메인 애플리케이션 클래스"""
    
    def __init__(self):
        """애플리케이션을 초기화하고 GUI를 설정합니다."""
        self.root = tk.Tk()
        self.setup_window()
        
        self.selected_index: str = list(MARKET_INDICES.keys())[0]  # 첫 번째 지수 선택
        self.selected_guru: str = list(GURU_STRATEGIES.keys())[0]  # 첫 번째 투자가 선택
        self.current_stock: Optional[StockInfo] = None
        
        self.stock_data: List[StockInfo] = []
        self.stock_tree_map: Dict[str, Any] = {}
        
        if AI_AVAILABLE and UltraAIAnalyzer:
            self.ai_analyzer = UltraAIAnalyzer()
        else:
            self.ai_analyzer = None
        
        self.load_data()
        self.setup_gui()
        
        self.update_time()
        self.root.after(100, self.update_stock_list)

    def setup_window(self):
        """메인 윈도우의 기본 속성을 설정합니다."""
        self.root.title("🚀 Ultra AI 주식 분석 시스템 v5.0 - Premium Edition")
        self.root.geometry("1920x1080")
        self.root.configure(bg=COLORS['bg'])
        self.root.state('zoomed')
        
        # 윈도우 아이콘 설정 (선택사항)
        try:
            self.root.iconbitmap('icon.ico')
        except:
            pass

    def load_data(self):
        """초기 주식 데이터를 로드합니다."""
        self.stock_data = self.generate_sample_data()

    def generate_sample_data(self) -> List[StockInfo]:
        """한투 증권사 HTS 수준의 상세한 샘플 주식 데이터를 생성합니다."""
        stocks_data = [
            # 삼성전자 - 대형주 대표
            StockInfo(
                name="삼성전자", code="005930", price=75000, change_rate=2.5, volume=15000000,
                market_cap=4500000, per=12.5, pbr=1.2, roe=15.8, debt_ratio=25.5, dividend_yield=2.1,
                revenue=2796540, operating_profit=640890, net_profit=556780, operating_margin=22.9, net_margin=19.9,
                roa=8.5, roic=18.2, equity_ratio=74.5, current_ratio=310.2, quick_ratio=245.8, interest_coverage=45.2,
                revenue_growth=8.2, profit_growth=12.5, net_growth=15.8, eps_growth=16.2,
                psr=1.8, pcr=8.5, ev_ebitda=6.8, peg=0.78, dividend_payout=25.8, dividend_growth=5.2, consecutive_dividend_years=12,
                tenkan_sen=74500, kijun_sen=73800, senkou_span_a=75200, senkou_span_b=72900, chikou_span=76200
            ),
            # SK하이닉스 - 메모리 반도체
            StockInfo(
                name="SK하이닉스", code="000660", price=120000, change_rate=-1.8, volume=8500000,
                market_cap=870000, per=18.2, pbr=2.1, roe=22.3, debt_ratio=45.2, dividend_yield=1.5,
                revenue=368920, operating_profit=89420, net_profit=72580, operating_margin=24.3, net_margin=19.7,
                roa=12.8, roic=25.5, equity_ratio=54.8, current_ratio=285.6, quick_ratio=198.4, interest_coverage=28.5,
                revenue_growth=25.8, profit_growth=45.2, net_growth=52.8, eps_growth=48.5,
                psr=2.4, pcr=12.5, ev_ebitda=8.9, peg=0.38, dividend_payout=18.5, dividend_growth=8.5, consecutive_dividend_years=8,
                tenkan_sen=119500, kijun_sen=118200, senkou_span_a=121800, senkou_span_b=116500, chikou_span=122500
            ),
            # NAVER - IT 서비스
            StockInfo(
                name="NAVER", code="035420", price=180000, change_rate=3.2, volume=2100000,
                market_cap=300000, per=25.8, pbr=3.5, roe=18.5, debt_ratio=15.2, dividend_yield=0.8,
                revenue=56280, operating_profit=12580, net_profit=9850, operating_margin=22.4, net_margin=17.5,
                roa=15.2, roic=22.8, equity_ratio=84.8, current_ratio=425.8, quick_ratio=385.2, interest_coverage=85.2,
                revenue_growth=18.5, profit_growth=28.5, net_growth=32.8, eps_growth=35.2,
                psr=5.3, pcr=18.5, ev_ebitda=15.8, peg=0.74, dividend_payout=12.5, dividend_growth=15.2, consecutive_dividend_years=5,
                tenkan_sen=178500, kijun_sen=176800, senkou_span_a=182500, senkou_span_b=174200, chikou_span=185200
            ),
            # 카카오 - 플랫폼
            StockInfo(
                name="카카오", code="035720", price=95000, change_rate=-0.5, volume=1800000,
                market_cap=420000, per=22.1, pbr=2.8, roe=12.5, debt_ratio=35.8, dividend_yield=1.2,
                revenue=68520, operating_profit=8520, net_profit=6850, operating_margin=12.4, net_margin=10.0,
                roa=8.5, roic=15.8, equity_ratio=64.2, current_ratio=185.6, quick_ratio=145.8, interest_coverage=25.8,
                revenue_growth=12.5, profit_growth=8.5, net_growth=5.2, eps_growth=4.8,
                psr=6.1, pcr=22.5, ev_ebitda=18.5, peg=4.6, dividend_payout=28.5, dividend_growth=2.5, consecutive_dividend_years=3,
                tenkan_sen=94200, kijun_sen=93500, senkou_span_a=96800, senkou_span_b=92100, chikou_span=97500
            ),
            # LG에너지솔루션 - 배터리
            StockInfo(
                name="LG에너지솔루션", code="373220", price=485000, change_rate=4.8, volume=950000,
                market_cap=1150000, per=28.5, pbr=4.2, roe=25.8, debt_ratio=52.5, dividend_yield=0.5,
                revenue=258420, operating_profit=28520, net_profit=22850, operating_margin=11.0, net_margin=8.8,
                roa=18.5, roic=28.5, equity_ratio=47.5, current_ratio=125.8, quick_ratio=95.2, interest_coverage=15.8,
                revenue_growth=85.2, profit_growth=125.8, net_growth=145.2, eps_growth=152.8,
                psr=4.4, pcr=28.5, ev_ebitda=22.5, peg=0.19, dividend_payout=8.5, dividend_growth=0.0, consecutive_dividend_years=1,
                tenkan_sen=482000, kijun_sen=478500, senkou_span_a=492500, senkou_span_b=475200, chikou_span=495800
            ),
            # 현대차 - 자동차
            StockInfo(
                name="현대차", code="005380", price=195000, change_rate=1.2, volume=1250000,
                market_cap=415000, per=8.5, pbr=0.8, roe=12.8, debt_ratio=85.2, dividend_yield=3.8,
                revenue=1425680, operating_profit=85420, net_profit=68520, operating_margin=6.0, net_margin=4.8,
                roa=5.8, roic=12.5, equity_ratio=14.8, current_ratio=95.2, quick_ratio=68.5, interest_coverage=8.5,
                revenue_growth=15.8, profit_growth=25.2, net_growth=28.5, eps_growth=32.8,
                psr=0.29, pcr=6.8, ev_ebitda=4.2, peg=0.26, dividend_payout=25.8, dividend_growth=8.5, consecutive_dividend_years=15,
                tenkan_sen=193500, kijun_sen=191800, senkou_span_a=197500, senkou_span_b=189200, chikou_span=198500
            )
        ]
        return stocks_data

    def setup_gui(self):
        """애플리케이션의 메인 GUI 레이아웃을 설정합니다."""
        main_frame = tk.Frame(self.root, bg=COLORS['bg'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.create_header(main_frame)
        
        content_frame = tk.Frame(main_frame, bg=COLORS['bg'])
        content_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        self.root.update_idletasks()
        width = self.root.winfo_width()
        left_width = max(MIN_PANEL_WIDTH, int(width * LEFT_PANEL_RATIO))
        right_width = max(450, int(width * RIGHT_PANEL_RATIO))
        center_width = width - left_width - right_width - 40
        
        self.create_left_panel(content_frame, left_width)
        self.create_center_panel(content_frame, center_width)
        self.create_right_panel(content_frame, right_width)
        self.create_status_bar(main_frame)

    def create_header(self, parent: tk.Frame):
        """상단 헤더 영역을 생성합니다."""
        header_frame = tk.Frame(parent, bg=COLORS['accent'], height=60)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        tk.Label(header_frame, text="Ultra AI 주식 분석 시스템", font=FONTS['title'], bg=COLORS['accent'], fg='white').pack(side=tk.LEFT, padx=20)
        self.time_label = tk.Label(header_frame, font=FONTS['body'], bg=COLORS['accent'], fg='white')
        self.time_label.pack(side=tk.RIGHT, padx=20)

    def create_left_panel(self, parent: tk.Frame, width: int):
        """좌측 패널(지수, 투자가, 종목 리스트)을 생성합니다."""
        left_frame = tk.Frame(parent, bg=COLORS['panel'], width=width)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        left_frame.pack_propagate(False)

        # 글로벌 지수 선택
        index_frame = tk.LabelFrame(left_frame, text="글로벌 지수 선택", font=FONTS['small'], bg=COLORS['panel'], fg=COLORS['text'], bd=1)
        index_frame.pack(fill=tk.X, padx=10, pady=10)
        indices = list(MARKET_INDICES.keys())
        self.index_var = tk.StringVar(value=indices[0])
        for i, idx in enumerate(indices):
            ttk.Radiobutton(index_frame, text=idx, var=self.index_var, value=idx,
                            command=lambda e=idx: self.select_index_with_feedback(e)).grid(row=i//2, column=i%2, sticky='w', padx=5, pady=2)

        # 투자 대가 관점 선택
        guru_frame = tk.LabelFrame(left_frame, text="투자 대가 관점", font=FONTS['small'], bg=COLORS['panel'], fg=COLORS['text'], bd=1)
        guru_frame.pack(fill=tk.X, padx=10, pady=5)
        self.guru_var = tk.StringVar(value=list(GURU_STRATEGIES.keys())[0])
        guru_cb = ttk.Combobox(guru_frame, textvariable=self.guru_var, values=list(GURU_STRATEGIES.keys()), state='readonly', width=25)
        guru_cb.pack(fill=tk.X, padx=5, pady=5)
        guru_cb.bind("<<ComboboxSelected>>", lambda event: self.select_guru(self.guru_var.get()))
        
        # 선택된 투자가 철학 표시
        self.guru_philosophy_label = tk.Label(guru_frame, text="", font=FONTS['small'], bg=COLORS['panel'], fg=COLORS['accent'], wraplength=width-20, justify='left')
        self.guru_philosophy_label.pack(fill=tk.X, padx=5, pady=(0, 5))
        self.update_guru_philosophy()

        # 종목 리스트
        stock_frame = tk.LabelFrame(left_frame, text="종목 리스트", font=FONTS['small'], bg=COLORS['panel'], fg=COLORS['text'], bd=1)
        stock_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        cols = ("종목명", "현재가", "등락률", "PER", "PBR")
        self.stock_tree = ttk.Treeview(stock_frame, columns=cols, show='headings', height=12)
        for col in cols:
            self.stock_tree.heading(col, text=col)
            if col == "종목명":
                self.stock_tree.column(col, width=80, anchor='w')
            elif col in ["현재가"]:
                self.stock_tree.column(col, width=70, anchor='e')
            else:
                self.stock_tree.column(col, width=50, anchor='center')
        
        # 스크롤바 추가
        scrollbar = ttk.Scrollbar(stock_frame, orient="vertical", command=self.stock_tree.yview)
        self.stock_tree.configure(yscrollcommand=scrollbar.set)
        self.stock_tree.pack(side="left", fill='both', expand=True)
        scrollbar.pack(side="right", fill="y")
        
        self.stock_tree.bind('<<TreeviewSelect>>', self.on_stock_select)

    def create_center_panel(self, parent: tk.Frame, width: int):
        """중앙 패널(차트, 분석 탭)을 생성합니다."""
        center_frame = tk.Frame(parent, bg=COLORS['panel'], width=width)
        center_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        center_frame.pack_propagate(False)
        
        # 종목 정보 헤더
        info_frame = tk.Frame(center_frame, bg=COLORS['panel'], height=60)
        info_frame.pack(fill=tk.X, pady=5)
        info_frame.pack_propagate(False)
        
        self.info_label = tk.Label(info_frame, text="종목을 선택하세요", font=FONTS['title'], bg=COLORS['panel'], fg=COLORS['text'])
        self.info_label.pack(side=tk.LEFT, padx=10, pady=10)
        
        # 차트 타입 선택 버튼 (재무제표 옵션으로 변경)
        chart_button_frame = tk.Frame(info_frame, bg=COLORS['panel'])
        chart_button_frame.pack(side=tk.RIGHT, padx=10, pady=10)
        
        self.chart_type_var = tk.StringVar(value="일목균형표")
        chart_types = ["일목균형표", "재무제표", "현금흐름표", "손익계산서"]
        for chart_type in chart_types:
            ttk.Radiobutton(chart_button_frame, text=chart_type, var=self.chart_type_var, value=chart_type,
                            command=self.update_charts).pack(side=tk.LEFT, padx=5)

        # 고해상도 차트 (한투 증권사 수준) - 2개 서브플롯으로 변경
        self.fig = Figure(figsize=(12, 6), dpi=120, facecolor=COLORS['chart_bg'])
        self.fig.patch.set_facecolor(COLORS['chart_bg'])
        
        # 2개 서브플롯: 메인 차트, 거래량
        gs = self.fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.1)
        self.ax_main = self.fig.add_subplot(gs[0])  # 메인 차트 (가격 + 일목균형표)
        self.ax_volume = self.fig.add_subplot(gs[1], sharex=self.ax_main)  # 거래량
        
        # 차트 스타일 설정
        for ax in [self.ax_main, self.ax_volume]:
            ax.set_facecolor(COLORS['chart_bg'])
            ax.grid(True, color=COLORS['grid'], alpha=0.3, linewidth=0.5)
            ax.tick_params(colors='white', labelsize=8)
            for spine in ax.spines.values():
                spine.set_color(COLORS['grid'])
                spine.set_linewidth(0.5)
        
        self.chart_canvas = FigureCanvasTkAgg(self.fig, master=center_frame)
        self.chart_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 분석 탭 (한투 증권사 수준의 상세 분석)
        self.notebook = ttk.Notebook(center_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 재무제표 탭들
        self.financial_summary_text = self.create_tab_with_text("재무요약")
        self.income_statement_text = self.create_tab_with_text("손익계산서")
        self.balance_sheet_text = self.create_tab_with_text("재무상태표")
        self.cashflow_text = self.create_tab_with_text("현금흐름표")
        self.ratio_analysis_text = self.create_tab_with_text("비율분석")
        self.valuation_text = self.create_tab_with_text("밸류에이션")
        
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)
        
    def create_right_panel(self, parent: tk.Frame, width: int):
        """우측 패널(AI 분석 버튼, 결과)을 생성합니다."""
        right_frame = tk.Frame(parent, bg=COLORS['panel'], width=width)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(5, 0))
        right_frame.pack_propagate(False)

        tk.Button(right_frame, text="AI 종합 분석 실행", font=FONTS['subtitle'], bg=COLORS['accent'],
                  fg='white', command=self.run_comprehensive_analysis).pack(fill=tk.X, padx=10, pady=10)

        ai_results_frame = tk.LabelFrame(right_frame, text="AI 투자분석 결과", font=FONTS['small'], bg=COLORS['panel'], fg=COLORS['text'], bd=1)
        self.ai_result_text = self.create_text_widget(ai_results_frame)

    def create_status_bar(self, parent: tk.Frame):
        """하단 상태바를 생성합니다."""
        status_bar = tk.Frame(parent, bg=COLORS['panel'], height=30)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_label = tk.Label(status_bar, text="준비 완료", font=FONTS['small'], bg=COLORS['panel'], fg=COLORS['text'])
        self.status_label.pack(side=tk.LEFT, padx=10)

    # --- 헬퍼 함수 ---
    def create_text_widget(self, parent_frame: tk.LabelFrame) -> tk.Text:
        """LabelFrame 내부에 Text 위젯을 생성하고 반환합니다."""
        parent_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        text_widget = tk.Text(parent_frame, bg=COLORS['panel'], fg=COLORS['text'], wrap=tk.WORD, font=FONTS['body'], relief=tk.FLAT, bd=0)
        text_widget.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        return text_widget

    def create_tab_with_text(self, title: str) -> tk.Text:
        """Notebook에 새로운 탭과 Text 위젯을 생성하고 반환합니다."""
        tab_frame = tk.Frame(self.notebook, bg=COLORS['panel'])
        self.notebook.add(tab_frame, text=title)
        return self.create_text_widget(tab_frame)

    # --- 이벤트 핸들러 및 UI 업데이트 함수 ---
    def update_time(self):
        """상단 헤더의 시간을 1초마다 업데이트합니다."""
        self.time_label.config(text=time.strftime('%Y-%m-%d %H:%M:%S'))
        self.root.after(1000, self.update_time)

    def update_stock_list(self):
        """종목 리스트 Treeview를 한투 증권사 수준으로 업데이트합니다."""
        self.stock_tree.delete(*self.stock_tree.get_children())
        self.stock_tree_map = {}
        
        for stock in self.stock_data:
            color = "green" if stock.change_rate >= 0 else "red"
            values = (stock.name, f"{stock.price:,.0f}", f"{stock.change_rate:+.2f}%", 
                     f"{stock.per:.1f}", f"{stock.pbr:.1f}")
            iid = self.stock_tree.insert('', 'end', values=values, tags=(color,))
            self.stock_tree_map[stock.code] = iid
            
        self.stock_tree.tag_configure('red', foreground=COLORS['error'])
        self.stock_tree.tag_configure('green', foreground=COLORS['success'])

    def select_index_with_feedback(self, index_name: str):
        """지수 선택 라디오 버튼 클릭 시 호출됩니다."""
        self.selected_index = index_name
        self.status_label.config(text=f"{index_name} 선택됨")
        # TODO: 실제 데이터 연동 시 선택된 지수에 따라 종목 리스트 필터링 구현

    def select_guru(self, guru_name: str):
        """투자가를 선택하고 철학을 업데이트합니다."""
        self.selected_guru = guru_name
        self.update_guru_philosophy()

    def update_guru_philosophy(self):
        """선택된 투자가의 철학을 업데이트합니다."""
        selected_guru = self.guru_var.get()
        philosophy = GURU_STRATEGIES.get(selected_guru, "투자 철학 정보가 없습니다.")
        self.guru_philosophy_label.config(text=philosophy)

    def on_stock_select(self, event):
        """종목 리스트에서 종목 선택 시 모든 관련 정보를 업데이트합니다."""
        selected_items = self.stock_tree.selection()
        if not selected_items:
            return
        
        selected_iid = selected_items[0]
        item = self.stock_tree.item(selected_iid)
        stock_name = item['values'][0]
        
        self.current_stock = next((s for s in self.stock_data if s.name == stock_name), None)
        
        if self.current_stock:
            self.update_all_info()

    def update_all_info(self):
        """선택된 종목에 대한 모든 정보를 한투 증권사 수준으로 업데이트합니다."""
        self.update_stock_info_display()
        self.update_charts()
        self.update_financial_summary()  # 기본으로 재무요약 탭 표시

    def update_stock_info_display(self):
        """선택된 종목의 상세 정보를 한투 증권사 수준으로 표시합니다."""
        if not self.current_stock:
            return
            
        stock = self.current_stock
        change_color = COLORS['success'] if stock.change_rate >= 0 else COLORS['error']
        info_text = (f"{stock.name} ({stock.code}) | "
                    f"현재가: {stock.price:,.0f}원 | "
                    f"등락률: {stock.change_rate:+.2f}% | "
                    f"시가총액: {stock.market_cap:,.0f}억원 | "
                    f"거래량: {stock.volume:,.0f}주")
        
        self.info_label.config(text=info_text, fg=change_color)

    def _generate_sample_chart_data(self) -> (pd.DatetimeIndex, np.ndarray, np.ndarray):
        """선택된 종목에 대한 샘플 차트 데이터를 생성합니다."""
        if not self.current_stock:
            return pd.DatetimeIndex([]), np.array([]), np.array([])

        dates = pd.to_datetime([datetime.now() - timedelta(days=i) for i in range(100)])
        prices = self.current_stock.price * (1 + np.random.randn(100).cumsum() * 0.01)
        volume = self.current_stock.volume * (1 + np.random.rand(100) * 0.5)
        return dates, prices, volume

    def update_charts(self):
        """한투 증권사 수준의 전문적인 차트를 업데이트합니다."""
        if not self.current_stock:
            return
            
        dates, prices, volume = self._generate_sample_chart_data()
        if dates.empty:
            return

        # 차트 초기화
        self.ax_main.clear()
        self.ax_volume.clear()

        chart_type = self.chart_type_var.get()
        
        if chart_type == "일목균형표":
            self._draw_ichimoku_chart(dates, prices)
        elif chart_type == "재무제표":
            self._draw_financial_statement_chart(dates, prices)
        elif chart_type == "현금흐름표":
            self._draw_cashflow_chart(dates, prices)
        else:  # 손익계산서
            self._draw_income_statement_chart(dates, prices)
        
        # 거래량 차트
        self._draw_volume_chart(dates, volume)
        
        # 차트 스타일 적용
        self._apply_chart_styling()
        
        self.chart_canvas.draw()

    def _draw_ichimoku_chart(self, dates: pd.DatetimeIndex, prices: np.ndarray):
        """일목균형표 차트를 그립니다."""
        # 가격 라인
        self.ax_main.plot(dates, prices, color='white', linewidth=1.5, label='종가', alpha=0.9)
        
        # 일목균형표 구성요소 계산 (실제로는 더 복잡한 계산이 필요)
        tenkan_sen = np.convolve(prices, np.ones(9)/9, mode='same')  # 전환선 (9일)
        kijun_sen = np.convolve(prices, np.ones(26)/26, mode='same')  # 기준선 (26일)
        
        # 선행스팬 A, B (간소화된 계산)
        senkou_span_a = (tenkan_sen + kijun_sen) / 2
        senkou_span_b = np.convolve(prices, np.ones(52)/52, mode='same')  # 52일 평균
        
        # 후행스팬 (26일 후행)
        chikou_span = np.roll(prices, 26)
        
        # 일목균형표 라인 그리기
        self.ax_main.plot(dates, tenkan_sen, color=COLORS['ichimoku_tenkan'], linewidth=1, label='전환선(9)', alpha=0.8)
        self.ax_main.plot(dates, kijun_sen, color=COLORS['ichimoku_kijun'], linewidth=1, label='기준선(26)', alpha=0.8)
        self.ax_main.plot(dates, senkou_span_a, color=COLORS['ichimoku_senkou_a'], linewidth=1, label='선행스팬A', alpha=0.6)
        self.ax_main.plot(dates, senkou_span_b, color=COLORS['ichimoku_senkou_b'], linewidth=1, label='선행스팬B', alpha=0.6)
        self.ax_main.plot(dates, chikou_span, color=COLORS['ichimoku_chikou'], linewidth=1, label='후행스팬', alpha=0.7)
        
        # 구름대 (선행스팬 A, B 사이 영역)
        self.ax_main.fill_between(dates, senkou_span_a, senkou_span_b, 
                                  where=(senkou_span_a >= senkou_span_b), color='green', alpha=0.1, label='양운(상승)')
        self.ax_main.fill_between(dates, senkou_span_a, senkou_span_b, 
                                  where=(senkou_span_a < senkou_span_b), color='red', alpha=0.1, label='음운(하락)')
        
        self.ax_main.legend(loc='upper left', fontsize=8, facecolor=COLORS['panel'], edgecolor='white')
        self.ax_main.set_title(f"{self.current_stock.name} 일목균형표", color='white', fontsize=12, pad=10)

    def _draw_financial_statement_chart(self, dates: pd.DatetimeIndex, prices: np.ndarray):
        """재무제표 차트를 그립니다."""
        # 가격 라인
        self.ax_main.plot(dates, prices, color='white', linewidth=1.5, label='종가', alpha=0.9)
        
        # 일목균형표 구성요소 계산 (실제로는 더 복잡한 계산이 필요)
        tenkan_sen = np.convolve(prices, np.ones(9)/9, mode='same')  # 전환선 (9일)
        kijun_sen = np.convolve(prices, np.ones(26)/26, mode='same')  # 기준선 (26일)
        
        # 선행스팬 A, B (간소화된 계산)
        senkou_span_a = (tenkan_sen + kijun_sen) / 2
        senkou_span_b = np.convolve(prices, np.ones(52)/52, mode='same')  # 52일 평균
        
        # 후행스팬 (26일 후행)
        chikou_span = np.roll(prices, 26)
        
        # 일목균형표 라인 그리기
        self.ax_main.plot(dates, tenkan_sen, color=COLORS['ichimoku_tenkan'], linewidth=1, label='전환선(9)', alpha=0.8)
        self.ax_main.plot(dates, kijun_sen, color=COLORS['ichimoku_kijun'], linewidth=1, label='기준선(26)', alpha=0.8)
        self.ax_main.plot(dates, senkou_span_a, color=COLORS['ichimoku_senkou_a'], linewidth=1, label='선행스팬A', alpha=0.6)
        self.ax_main.plot(dates, senkou_span_b, color=COLORS['ichimoku_senkou_b'], linewidth=1, label='선행스팬B', alpha=0.6)
        self.ax_main.plot(dates, chikou_span, color=COLORS['ichimoku_chikou'], linewidth=1, label='후행스팬', alpha=0.7)
        
        # 구름대 (선행스팬 A, B 사이 영역)
        self.ax_main.fill_between(dates, senkou_span_a, senkou_span_b, 
                                  where=(senkou_span_a >= senkou_span_b), color='green', alpha=0.1, label='양운(상승)')
        self.ax_main.fill_between(dates, senkou_span_a, senkou_span_b, 
                                  where=(senkou_span_a < senkou_span_b), color='red', alpha=0.1, label='음운(하락)')
        
        self.ax_main.legend(loc='upper left', fontsize=8, facecolor=COLORS['panel'], edgecolor='white')
        self.ax_main.set_title(f"{self.current_stock.name} 일목균형표", color='white', fontsize=12, pad=10)

    def _draw_cashflow_chart(self, dates: pd.DatetimeIndex, prices: np.ndarray):
        """현금흐름표 차트를 그립니다."""
        # 가격 라인
        self.ax_main.plot(dates, prices, color='white', linewidth=1.5, label='종가', alpha=0.9)
        
        # 일목균형표 구성요소 계산 (실제로는 더 복잡한 계산이 필요)
        tenkan_sen = np.convolve(prices, np.ones(9)/9, mode='same')  # 전환선 (9일)
        kijun_sen = np.convolve(prices, np.ones(26)/26, mode='same')  # 기준선 (26일)
        
        # 선행스팬 A, B (간소화된 계산)
        senkou_span_a = (tenkan_sen + kijun_sen) / 2
        senkou_span_b = np.convolve(prices, np.ones(52)/52, mode='same')  # 52일 평균
        
        # 후행스팬 (26일 후행)
        chikou_span = np.roll(prices, 26)
        
        # 일목균형표 라인 그리기
        self.ax_main.plot(dates, tenkan_sen, color=COLORS['ichimoku_tenkan'], linewidth=1, label='전환선(9)', alpha=0.8)
        self.ax_main.plot(dates, kijun_sen, color=COLORS['ichimoku_kijun'], linewidth=1, label='기준선(26)', alpha=0.8)
        self.ax_main.plot(dates, senkou_span_a, color=COLORS['ichimoku_senkou_a'], linewidth=1, label='선행스팬A', alpha=0.6)
        self.ax_main.plot(dates, senkou_span_b, color=COLORS['ichimoku_senkou_b'], linewidth=1, label='선행스팬B', alpha=0.6)
        self.ax_main.plot(dates, chikou_span, color=COLORS['ichimoku_chikou'], linewidth=1, label='후행스팬', alpha=0.7)
        
        # 구름대 (선행스팬 A, B 사이 영역)
        self.ax_main.fill_between(dates, senkou_span_a, senkou_span_b, 
                                  where=(senkou_span_a >= senkou_span_b), color='green', alpha=0.1, label='양운(상승)')
        self.ax_main.fill_between(dates, senkou_span_a, senkou_span_b, 
                                  where=(senkou_span_a < senkou_span_b), color='red', alpha=0.1, label='음운(하락)')
        
        self.ax_main.legend(loc='upper left', fontsize=8, facecolor=COLORS['panel'], edgecolor='white')
        self.ax_main.set_title(f"{self.current_stock.name} 현금흐름표", color='white', fontsize=12, pad=10)

    def _draw_income_statement_chart(self, dates: pd.DatetimeIndex, prices: np.ndarray):
        """손익계산서 차트를 그립니다."""
        # 가격 라인
        self.ax_main.plot(dates, prices, color='white', linewidth=1.5, label='종가', alpha=0.9)
        
        # 일목균형표 구성요소 계산 (실제로는 더 복잡한 계산이 필요)
        tenkan_sen = np.convolve(prices, np.ones(9)/9, mode='same')  # 전환선 (9일)
        kijun_sen = np.convolve(prices, np.ones(26)/26, mode='same')  # 기준선 (26일)
        
        # 선행스팬 A, B (간소화된 계산)
        senkou_span_a = (tenkan_sen + kijun_sen) / 2
        senkou_span_b = np.convolve(prices, np.ones(52)/52, mode='same')  # 52일 평균
        
        # 후행스팬 (26일 후행)
        chikou_span = np.roll(prices, 26)
        
        # 일목균형표 라인 그리기
        self.ax_main.plot(dates, tenkan_sen, color=COLORS['ichimoku_tenkan'], linewidth=1, label='전환선(9)', alpha=0.8)
        self.ax_main.plot(dates, kijun_sen, color=COLORS['ichimoku_kijun'], linewidth=1, label='기준선(26)', alpha=0.8)
        self.ax_main.plot(dates, senkou_span_a, color=COLORS['ichimoku_senkou_a'], linewidth=1, label='선행스팬A', alpha=0.6)
        self.ax_main.plot(dates, senkou_span_b, color=COLORS['ichimoku_senkou_b'], linewidth=1, label='선행스팬B', alpha=0.6)
        self.ax_main.plot(dates, chikou_span, color=COLORS['ichimoku_chikou'], linewidth=1, label='후행스팬', alpha=0.7)
        
        # 구름대 (선행스팬 A, B 사이 영역)
        self.ax_main.fill_between(dates, senkou_span_a, senkou_span_b, 
                                  where=(senkou_span_a >= senkou_span_b), color='green', alpha=0.1, label='양운(상승)')
        self.ax_main.fill_between(dates, senkou_span_a, senkou_span_b, 
                                  where=(senkou_span_a < senkou_span_b), color='red', alpha=0.1, label='음운(하락)')
        
        self.ax_main.legend(loc='upper left', fontsize=8, facecolor=COLORS['panel'], edgecolor='white')
        self.ax_main.set_title(f"{self.current_stock.name} 손익계산서", color='white', fontsize=12, pad=10)

    def _draw_volume_chart(self, dates: pd.DatetimeIndex, volume: np.ndarray):
        """거래량 차트를 그립니다."""
        colors = ['red' if i % 2 == 0 else 'blue' for i in range(len(volume))]
        self.ax_volume.bar(dates, volume, color=colors, alpha=0.6, width=0.8)
        self.ax_volume.set_title("거래량", color='white', fontsize=10)
        
        # 거래량 이동평균
        vol_ma = np.convolve(volume, np.ones(20)/20, mode='same')
        self.ax_volume.plot(dates, vol_ma, color='yellow', linewidth=1, alpha=0.8)

    def _draw_rsi_indicator(self, dates: pd.DatetimeIndex, prices: np.ndarray):
        """RSI 보조지표를 그립니다."""
        # 간단한 RSI 계산 (실제로는 더 정확한 계산 필요)
        price_changes = np.diff(prices, prepend=prices[0])
        gains = np.maximum(price_changes, 0)
        losses = np.maximum(-price_changes, 0)
        
        avg_gains = np.convolve(gains, np.ones(14)/14, mode='same')
        avg_losses = np.convolve(losses, np.ones(14)/14, mode='same')
        
        rs = avg_gains / (avg_losses + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        self.ax_main.plot(dates, rsi, color='purple', linewidth=1)
        self.ax_main.axhline(y=70, color='red', linestyle='--', alpha=0.5)
        self.ax_main.axhline(y=30, color='blue', linestyle='--', alpha=0.5)
        self.ax_main.axhline(y=50, color='gray', linestyle='-', alpha=0.3)
        self.ax_main.set_ylim(0, 100)
        self.ax_main.set_title("RSI (14)", color='white', fontsize=10)

    def _apply_chart_styling(self):
        """차트에 한투 증권사 수준의 전문적인 스타일을 적용합니다."""
        for ax in [self.ax_main, self.ax_volume]:
            ax.set_facecolor(COLORS['chart_bg'])
            ax.grid(True, color=COLORS['grid'], alpha=0.3, linewidth=0.5)
            ax.tick_params(colors='white', labelsize=8)
            
            for spine in ax.spines.values():
                spine.set_color(COLORS['grid'])
                spine.set_linewidth(0.5)
        
        # x축 레이블 회전
        self.ax_main.tick_params(axis='x', rotation=45)
        
        # 여백 조정
        self.fig.tight_layout(pad=1.0)

    def run_comprehensive_analysis(self):
        """'AI 종합 분석 실행' 버튼 클릭 시 AI 분석을 시작합니다."""
        if not self.ai_analyzer:
            messagebox.showwarning("AI 분석 불가", "AI 모듈이 로드되지 않았습니다.")
            return
        
        self.status_label.config(text="AI 종합 분석 진행 중...")
        self.ai_result_text.delete('1.0', tk.END)
        self.ai_result_text.insert('1.0', "AI 분석을 시작합니다. 잠시만 기다려주세요...")

        # 백그라운드 스레드에서 AI 분석 실행
        threading.Thread(target=self.perform_real_ai_analysis, daemon=True).start()

    def perform_real_ai_analysis(self):
        """AI 분석을 실제로 수행하는 로직 (스레드에서 실행)"""
        try:
            stock_codes = [s.code for s in self.stock_data[:NUM_STOCKS_FOR_AI_ANALYSIS]]
            analysis_results = self.ai_analyzer.analyze_stocks(stock_codes)
            self.root.after(0, lambda: self.update_ai_results_display(analysis_results))
        except Exception as e:
            # 에러 발생 시 메인 스레드에서 UI 업데이트
            self.root.after(0, lambda err=e: self.show_ai_error_and_fallback(f"AI 분석 중 오류 발생: {err}"))

    def show_ai_error_and_fallback(self, error_msg: str):
        """AI 분석 오류 메시지를 UI에 표시합니다."""
        self.ai_result_text.delete('1.0', tk.END)
        self.ai_result_text.insert('1.0', f"❌ {error_msg}\n\n")
        self.ai_result_text.insert(tk.END, "AI 분석을 사용할 수 없습니다. 일반 분석 정보를 표시합니다.\n")
        if self.current_stock:
            self.update_financial_summary()

    def update_ai_results_display(self, results: Dict[str, Any]):
        """AI 분석 결과를 UI에 업데이트하고, 첫 번째 종목을 선택합니다."""
        self.ai_result_text.delete('1.0', tk.END)
        self.ai_result_text.insert('1.0', f"✅ AI 분석 완료 ({datetime.now().strftime('%H:%M:%S')})\n\n")
        
        for stock_code, analysis in results.items():
            stock_name = next((s.name for s in self.stock_data if s.code == stock_code), "알수없음")
            self.ai_result_text.insert(tk.END, f"--- {stock_name} ({stock_code}) ---\n")
            self.ai_result_text.insert(tk.END, f"평가: {analysis.get('evaluation', 'N/A')}\n")
            self.ai_result_text.insert(tk.END, f"요약: {analysis.get('summary', 'N/A')}\n")
            self.ai_result_text.insert(tk.END, f"적정주가: {analysis.get('fair_price', 'N/A')}원\n\n")

        # 분석된 첫 종목을 Treeview에서 선택
        if results and self.stock_tree_map:
            first_stock_code = next(iter(results))
            if first_stock_code in self.stock_tree_map:
                iid = self.stock_tree_map[first_stock_code]
                self.stock_tree.selection_set(iid)
                self.stock_tree.focus(iid)
                self.stock_tree.see(iid)
        
        self.status_label.config(text="AI 종합 분석 완료")

    def on_tab_changed(self, event):
        """재무제표 탭 변경 시 해당 내용을 한투 증권사 수준으로 업데이트합니다."""
        if not self.current_stock:
            return
            
        selected_tab_index = self.notebook.index(self.notebook.select())
        tab_functions = [
            self.update_financial_summary,    # 재무요약
            self.update_income_statement,     # 손익계산서  
            self.update_balance_sheet,        # 재무상태표
            self.update_cashflow_statement,   # 현금흐름표
            self.update_ratio_analysis,       # 비율분석
            self.update_valuation_analysis    # 밸류에이션
        ]
        
        if 0 <= selected_tab_index < len(tab_functions):
            tab_functions[selected_tab_index]()

    def update_financial_summary(self):
        """재무요약 탭을 한투 증권사 수준으로 업데이트합니다."""
        if not self.current_stock: 
            return
            
        stock = self.current_stock
        content = f"""
📊 {stock.name} 재무 요약 정보

💰 수익성 지표
├ 매출액: {stock.revenue:,.0f}억원
├ 영업이익: {stock.operating_profit:,.0f}억원  
├ 당기순이익: {stock.net_profit:,.0f}억원
├ 영업이익률: {stock.operating_margin:.1f}%
├ 순이익률: {stock.net_margin:.1f}%
├ ROE: {stock.roe:.1f}%
├ ROA: {stock.roa:.1f}%
└ ROIC: {stock.roic:.1f}%

🛡️ 안정성 지표  
├ 자기자본비율: {stock.equity_ratio:.1f}%
├ 부채비율: {stock.debt_ratio:.1f}%
├ 유동비율: {stock.current_ratio:.1f}%
├ 당좌비율: {stock.quick_ratio:.1f}%
└ 이자보상배수: {stock.interest_coverage:.1f}배

📈 성장성 지표
├ 매출액증가율: {stock.revenue_growth:+.1f}%
├ 영업이익증가율: {stock.profit_growth:+.1f}%  
├ 순이익증가율: {stock.net_growth:+.1f}%
└ EPS증가율: {stock.eps_growth:+.1f}%

💎 밸류에이션
├ PER: {stock.per:.1f}배
├ PBR: {stock.pbr:.1f}배
├ PSR: {stock.psr:.1f}배
├ EV/EBITDA: {stock.ev_ebitda:.1f}배
└ PEG: {stock.peg:.2f}

💰 배당 정보
├ 배당수익률: {stock.dividend_yield:.1f}%
├ 배당성향: {stock.dividend_payout:.1f}%
├ 배당증가율: {stock.dividend_growth:+.1f}%
├ 연속배당년수: {stock.consecutive_dividend_years}년
└ 배당평가: {'우수' if stock.dividend_yield > 3 and stock.consecutive_dividend_years > 10 else '보통'}

🔸 투자 의견
├ 목표주가: {stock.price * 1.2:.0f}원
└ 리스크: {'낮음' if stock.debt_ratio < 30 else '보통' if stock.debt_ratio < 70 else '높음'}
        """
        self._update_text_widget(self.financial_summary_text, "재무 요약", content)

    def update_income_statement(self):
        """손익계산서 탭을 한투 증권사 수준으로 업데이트합니다."""
        if not self.current_stock: 
            return
            
        stock = self.current_stock
        content = f"""
📋 {stock.name} 손익계산서 (단위: 억원)

🔸 매출 관련
├ 매출액: {stock.revenue:,.0f}
├ 매출원가: {stock.revenue * 0.7:,.0f} (추정)
└ 매출총이익: {stock.revenue * 0.3:,.0f} (추정)

🔸 영업 관련  
├ 판매관리비: {stock.revenue * 0.15:,.0f} (추정)
├ 영업이익: {stock.operating_profit:,.0f}
└ 영업이익률: {stock.operating_margin:.1f}%

🔸 영업외 손익
├ 금융수익: {stock.operating_profit * 0.05:,.0f} (추정)
├ 금융비용: {stock.operating_profit * 0.03:,.0f} (추정)
└ 기타손익: {stock.operating_profit * 0.02:,.0f} (추정)

🔸 세전/세후 이익
├ 법인세비용차감전순이익: {stock.net_profit * 1.25:,.0f} (추정)
├ 법인세비용: {stock.net_profit * 0.25:,.0f} (추정)
├ 당기순이익: {stock.net_profit:,.0f}
└ 순이익률: {stock.net_margin:.1f}%

🔸 주당 정보
├ 발행주식수: {stock.market_cap * 100 // stock.price:,.0f}주 (추정)
├ EPS: {stock.price / stock.per:,.0f}원 (추정)
└ EPS 증가율: {stock.eps_growth:+.1f}%
        """
        self._update_text_widget(self.income_statement_text, "손익계산서", content)

    def update_balance_sheet(self):
        """재무상태표 탭을 한투 증권사 수준으로 업데이트합니다."""
        if not self.current_stock: 
            return
            
        stock = self.current_stock
        total_assets = stock.market_cap * 1.5  # 추정
        content = f"""
🏛️ {stock.name} 재무상태표 (단위: 억원)

🔸 자산 (Assets)
├ 유동자산: {total_assets * 0.4:,.0f} (추정)
│  ├ 현금및현금성자산: {total_assets * 0.15:,.0f}
│  ├ 단기금융상품: {total_assets * 0.08:,.0f}
│  ├ 매출채권: {total_assets * 0.12:,.0f}
│  └ 재고자산: {total_assets * 0.05:,.0f}
├ 비유동자산: {total_assets * 0.6:,.0f} (추정)
│  ├ 유형자산: {total_assets * 0.35:,.0f}
│  ├ 무형자산: {total_assets * 0.15:,.0f}
│  └ 기타비유동자산: {total_assets * 0.1:,.0f}
└ 자산총계: {total_assets:,.0f}

🔸 부채 (Liabilities)  
├ 유동부채: {total_assets * stock.debt_ratio/100 * 0.6:,.0f} (추정)
│  ├ 단기차입금: {total_assets * stock.debt_ratio/100 * 0.3:,.0f}
│  └ 매입채무: {total_assets * stock.debt_ratio/100 * 0.3:,.0f}
├ 비유동부채: {total_assets * stock.debt_ratio/100 * 0.4:,.0f} (추정)
│  └ 장기차입금: {total_assets * stock.debt_ratio/100 * 0.4:,.0f}
└ 부채총계: {total_assets * stock.debt_ratio/100:,.0f}

🔸 자본 (Equity)
├ 자본금: {stock.market_cap * 0.1:,.0f} (추정)
├ 자본잉여금: {stock.market_cap * 0.2:,.0f} (추정)  
├ 이익잉여금: {stock.market_cap * 0.7:,.0f} (추정)
└ 자본총계: {total_assets * stock.equity_ratio/100:,.0f}

🔸 주요 비율
├ 자기자본비율: {stock.equity_ratio:.1f}%
├ 부채비율: {stock.debt_ratio:.1f}%
├ 유동비율: {stock.current_ratio:.1f}%
└ 당좌비율: {stock.quick_ratio:.1f}%
        """
        self._update_text_widget(self.balance_sheet_text, "재무상태표", content)

    def update_cashflow_statement(self):
        """현금흐름표 탭을 한투 증권사 수준으로 업데이트합니다."""
        if not self.current_stock: 
            return
            
        stock = self.current_stock
        content = f"""
💰 {stock.name} 현금흐름표 (단위: 억원)

🔸 영업활동 현금흐름
├ 당기순이익: {stock.net_profit:,.0f}
├ 감가상각비: {stock.net_profit * 0.3:,.0f} (추정)
├ 운전자본 변동: {stock.net_profit * -0.1:,.0f} (추정)
├ 기타 영업활동: {stock.net_profit * 0.05:,.0f} (추정)
└ 영업활동 현금흐름: {stock.net_profit * 1.25:,.0f} (추정)

🔸 투자활동 현금흐름
├ 유형자산 취득: {stock.net_profit * -0.4:,.0f} (추정)
├ 무형자산 취득: {stock.net_profit * -0.1:,.0f} (추정)
├ 금융상품 투자: {stock.net_profit * -0.2:,.0f} (추정)
├ 기타 투자활동: {stock.net_profit * 0.05:,.0f} (추정)
└ 투자활동 현금흐름: {stock.net_profit * -0.65:,.0f} (추정)

🔸 재무활동 현금흐름
├ 차입금 증감: {stock.net_profit * 0.1:,.0f} (추정)
├ 배당금 지급: {stock.net_profit * stock.dividend_payout/100:,.0f}
├ 자기주식 거래: {stock.net_profit * -0.05:,.0f} (추정)
└ 재무활동 현금흐름: {stock.net_profit * (0.1 - stock.dividend_payout/100 - 0.05):,.0f} (추정)

🔸 현금 증감
├ 기초 현금: {stock.market_cap * 0.15:,.0f} (추정)
├ 현금 순증감: {stock.net_profit * 0.6:,.0f} (추정)
└ 기말 현금: {stock.market_cap * 0.15 + stock.net_profit * 0.6:,.0f} (추정)

🔸 현금흐름 비율
├ 영업현금흐름/순이익: {1.25:.2f} (추정)
├ 자유현금흐름: {stock.net_profit * 0.6:,.0f} (추정)
└ 현금전환주기: 45일 (추정)
        """
        self._update_text_widget(self.cashflow_text, "현금흐름표", content)

    def update_ratio_analysis(self):
        """비율분석 탭을 한투 증권사 수준으로 업데이트합니다."""
        if not self.current_stock: 
            return
            
        stock = self.current_stock
        content = f"""
📊 {stock.name} 재무비율 종합분석

🔸 수익성 비율 (Profitability Ratios)
├ 매출총이익률: 30.0% (추정)
├ 영업이익률: {stock.operating_margin:.1f}%
├ 순이익률: {stock.net_margin:.1f}%
├ ROE (자기자본수익률): {stock.roe:.1f}%
├ ROA (총자산수익률): {stock.roa:.1f}%
├ ROIC (투하자본수익률): {stock.roic:.1f}%
└ 평가: {'우수' if stock.roe > 15 else '보통' if stock.roe > 10 else '부진'}

🔸 안정성 비율 (Stability Ratios)
├ 자기자본비율: {stock.equity_ratio:.1f}%
├ 부채비율: {stock.debt_ratio:.1f}%
├ 유동비율: {stock.current_ratio:.1f}%
├ 당좌비율: {stock.quick_ratio:.1f}%
├ 이자보상배수: {stock.interest_coverage:.1f}배
└ 평가: {'안전' if stock.debt_ratio < 50 else '보통' if stock.debt_ratio < 100 else '위험'}

🔸 성장성 비율 (Growth Ratios)
├ 매출액증가율: {stock.revenue_growth:+.1f}%
├ 영업이익증가율: {stock.profit_growth:+.1f}%
├ 순이익증가율: {stock.net_growth:+.1f}%
├ EPS증가율: {stock.eps_growth:+.1f}%
├ 배당증가율: {stock.dividend_growth:+.1f}%
└ 평가: {'고성장' if stock.revenue_growth > 20 else '성장' if stock.revenue_growth > 10 else '저성장'}

🔸 활동성 비율 (Activity Ratios)
├ 총자산회전율: 1.2회 (추정)
├ 매출채권회전율: 8.5회 (추정)
├ 재고자산회전율: 12.0회 (추정)
├ 자기자본회전율: 2.1회 (추정)
└ 평가: 보통

🔸 종합 평가
├ 투자 등급: {'A' if stock.roe > 15 and stock.debt_ratio < 50 else 'B' if stock.roe > 10 else 'C'}
├ 리스크 수준: {'낮음' if stock.debt_ratio < 30 else '보통' if stock.debt_ratio < 70 else '높음'}
└ 추천도: {'매수' if stock.roe > 15 and stock.revenue_growth > 10 else '보유' if stock.roe > 10 else '관심'}
        """
        self._update_text_widget(self.ratio_analysis_text, "비율분석", content)

    def update_valuation_analysis(self):
        """밸류에이션 탭을 한투 증권사 수준으로 업데이트합니다."""
        if not self.current_stock:
            return
            
        stock = self.current_stock
        
        # 적정주가 계산 (간단한 모델)
        fair_value_per = stock.price * (15 / stock.per) if stock.per > 0 else stock.price
        fair_value_pbr = stock.price * (1.5 / stock.pbr) if stock.pbr > 0 else stock.price
        fair_value_avg = (fair_value_per + fair_value_pbr) / 2
            
        content = f"""
💎 {stock.name} 밸류에이션 분석

🔸 현재 주가 정보
├ 현재가: {stock.price:,.0f}원
├ 52주 최고가: {stock.price * 1.3:,.0f}원 (추정)
├ 52주 최저가: {stock.price * 0.7:,.0f}원 (추정)
└ 현재 위치: {((stock.price - stock.price * 0.7) / (stock.price * 0.3) * 100):.1f}%

🔸 밸류에이션 지표
├ PER: {stock.per:.1f}배 (업종평균: 15.0배)
├ PBR: {stock.pbr:.1f}배 (업종평균: 1.5배)  
├ PSR: {stock.psr:.1f}배 (업종평균: 2.0배)
├ PCR: {stock.pcr:.1f}배 (업종평균: 10.0배)
├ EV/EBITDA: {stock.ev_ebitda:.1f}배 (업종평균: 8.0배)
└ PEG: {stock.peg:.2f} (1.0 이하 양호)

🔸 적정주가 산출
├ PER 기준 적정가: {fair_value_per:,.0f}원 (15배 기준)
├ PBR 기준 적정가: {fair_value_pbr:,.0f}원 (1.5배 기준)
├ 평균 적정가: {fair_value_avg:,.0f}원
├ 현재가 대비: {((fair_value_avg - stock.price) / stock.price * 100):+.1f}%
└ 투자판단: {'저평가' if fair_value_avg > stock.price * 1.1 else '적정가' if fair_value_avg > stock.price * 0.9 else '고평가'}

🔸 배당 정보
├ 배당수익률: {stock.dividend_yield:.1f}%
├ 배당성향: {stock.dividend_payout:.1f}%
├ 배당증가율: {stock.dividend_growth:+.1f}%
├ 연속배당년수: {stock.consecutive_dividend_years}년
└ 배당평가: {'우수' if stock.dividend_yield > 3 and stock.consecutive_dividend_years > 10 else '보통'}

🔸 투자 의견
├ 목표주가: {fair_value_avg:,.0f}원
├ 상승여력: {((fair_value_avg - stock.price) / stock.price * 100):+.1f}%
├ 투자등급: {'매수' if fair_value_avg > stock.price * 1.2 else '보유' if fair_value_avg > stock.price * 0.8 else '매도'}
└ 리스크: {'낮음' if stock.debt_ratio < 30 else '보통' if stock.debt_ratio < 70 else '높음'}

🔸 동종업계 비교 (추정)
├ 업종 평균 PER: 15.0배
├ 업종 평균 PBR: 1.5배
├ 업종 평균 ROE: 12.0%
├ 상대적 위치: {'상위' if stock.roe > 15 else '중위' if stock.roe > 10 else '하위'}
└ 경쟁력: {'우수' if stock.operating_margin > 20 else '보통' if stock.operating_margin > 10 else '부진'}
        """
        self._update_text_widget(self.valuation_text, "밸류에이션", content)

    def run(self):
        """애플리케이션을 실행합니다."""
        self.root.mainloop()

    def _update_text_widget(self, text_widget: tk.Text, title: str, content: str):
        """Text 위젯의 내용을 업데이트하는 헬퍼 함수입니다."""
        text_widget.delete('1.0', tk.END)
        text_widget.insert('1.0', content.strip())

def main():
    """애플리케이션의 메인 진입점입니다."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    try:
        app = ComprehensiveHTS()
        app.run()
    except Exception as e:
        logging.exception(f"애플리케이션 실행 중 치명적인 오류 발생: {e}")
        messagebox.showerror("실행 오류", f"프로그램 실행 중 예측하지 못한 오류가 발생했습니다.\n\n{e}")

if __name__ == '__main__':
    main()