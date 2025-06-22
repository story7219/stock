"""
종합 HTS GUI 시스템 - AI 주식 분석 및 차트 기능 통합 플랫폼
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

# --- 전문가 최고 수준 투자용 컬러 팔레트 ---
COLORS = {
    'bg': '#0f0f23',  # 다크네이비 배경
    'panel': '#1a1a2e',  # 다크 패널 톤
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
    # 일목균형표용 전문 컬러
    'ichimoku_tenkan': '#ff6b6b',  # 전환선 - 코랄 레드
    'ichimoku_kijun': '#4ecdc4',  # 기준선 - 터키쉬
    'ichimoku_senkou_a': '#45b7d1',  # 선행스팬A - 스카이 블루
    'ichimoku_senkou_b': '#f9ca24',  # 선행스팬B - 골든 옐로우
    'ichimoku_chikou': '#6c5ce7',  # 후행스팬 - 바이올렛
    'cloud_bullish': '#4ade80',  # 양운 구름 - 그린
    'cloud_bearish': '#ef4444'  # 음운 구름 - 레드
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

# 전문가급 투자 거장 각자의 투자 철학
GURU_STRATEGIES = {
    'Warren Buffett': '가치투자의 전설 - 내재가치 중시, 장기투자, 우량기업 선호',
    'Peter Lynch': '성장주 투자의 거인 - PEG 비율 활용, 일상생활서 투자 아이디어 발굴',
    'Benjamin Graham': '증권분석의 전설 - 안전마진 중시, 저평가 주식, 정량적 분석',
    'Philip Fisher': '성장주 투자의 아버지 - 스캐터벗 방식, 질적 분석 중시',
    'John Templeton': '글로벌 가치투자 - 역발상 투자, 국제적 분산투자',
    'Charlie Munger': '철학적 사고 - 심리학적 편향 극복, 합리적 의사결정',
    'Joel Greenblatt': '마법공식 투자법 - ROE와 수익률 기반 정량적 선별',
    'David Dreman': '역발상 투자 전략 - 저PER, 저PBR 주식 선호',
    'William O\'Neil': 'CAN SLIM 투자법 - 기술적 분석과 기본적 분석 결합',
    'Ray Dalio': '올웨더 포트폴리오 - 리스크 패리티, 경제 사이클 분석',
    'George Soros': '재귀성 이론 - 시장 심리와 펀더멘털의 상호작용',
    'Carl Icahn': '행동주의 투자 - 기업 지배구조 개선을 통한 가치창출'
}

# 주요 글로벌 지수 (수정 버전)
MARKET_INDICES = {
    'KOSPI 200': '한국 대형주 200개 기업 지수',
    'NASDAQ-100': '나스닥100대 기술주 지수',
    'S&P 500': '미 대형주 500개 기업 지수'
}

LEFT_PANEL_RATIO = 0.15
RIGHT_PANEL_RATIO = 0.35
MIN_PANEL_WIDTH = 300
NUM_STOCKS_FOR_AI_ANALYSIS = 10


@dataclass
class StockInfo:
    """주식 정보 데이터 클래스 - 기본 정보만"""
    name: str
    code: str
    price: float
    change_rate: float
    volume: int
    sector: str = ""


class ComprehensiveHTS:
    """종합 HTS 메인 애플리케이션 클래스"""
    
    def __init__(self):
        """애플리케이션을 초기화하고 GUI를 설정합니다."""
        self.root = tk.Tk()
        self.setup_window()
        
        self.selected_index: str = list(MARKET_INDICES.keys())[0]  # 첫번째 지수 선택
        self.selected_guru: str = list(GURU_STRATEGIES.keys())[0]  # 첫번째 투자가 선택
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
        """기본 주식 데이터를 생성합니다."""
        stocks_data = [
            StockInfo(name="삼성전자", code="005930", price=75000, change_rate=2.5, volume=15000000, sector="반도체"),
            StockInfo(name="SK하이닉스", code="000660", price=120000, change_rate=-1.8, volume=8500000, sector="반도체"),
            StockInfo(name="NAVER", code="035420", price=180000, change_rate=3.2, volume=2100000, sector="IT서비스"),
            StockInfo(name="카카오", code="035720", price=95000, change_rate=-0.5, volume=1800000, sector="플랫폼"),
            StockInfo(name="LG에너지솔루션", code="373220", price=485000, change_rate=4.8, volume=950000, sector="배터리"),
            StockInfo(name="현대차", code="005380", price=195000, change_rate=1.2, volume=1250000, sector="자동차")
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
        
        tk.Label(header_frame, text="🚀 Ultra AI 주식 분석 시스템", font=FONTS['title'], bg=COLORS['accent'], fg='white').pack(side=tk.LEFT, padx=20)
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

        # 투자 거장 관점 선택
        guru_frame = tk.LabelFrame(left_frame, text="투자 거장 관점", font=FONTS['small'], bg=COLORS['panel'], fg=COLORS['text'], bd=1)
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
        
        # 차트 타입 선택 버튼 (일목균형표 옵션으로 변경)
        chart_button_frame = tk.Frame(info_frame, bg=COLORS['panel'])
        chart_button_frame.pack(side=tk.RIGHT, padx=10, pady=10)
        
        self.chart_type_var = tk.StringVar(value="일목균형표")
        chart_types = ["일목균형표", "재무표", "현금흐름표", "손익계산서"]
        for chart_type in chart_types:
            ttk.Radiobutton(chart_button_frame, text=chart_type, var=self.chart_type_var, value=chart_type,
                            command=self.update_charts).pack(side=tk.LEFT, padx=5)

        # 고해상도 차트 (전문 증권사 수준) - 2개 서브플롯으로 변경
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

        # 분석 탭 (전문 증권사 수준 상세 분석)
        self.notebook = ttk.Notebook(center_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # AI 분석 탭만 유지
        self.ai_analysis_text = self.create_tab_with_text("AI 분석")
        
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)
        
    def create_right_panel(self, parent: tk.Frame, width: int):
        """우측 패널(AI 분석 버튼, 결과)을 생성합니다."""
        right_frame = tk.Frame(parent, bg=COLORS['panel'], width=width)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(5, 0))
        right_frame.pack_propagate(False)

        tk.Button(right_frame, text="🤖 AI 종합 분석 실행", font=FONTS['subtitle'], bg=COLORS['accent'],
                  fg='white', command=self.run_comprehensive_analysis).pack(fill=tk.X, padx=10, pady=10)

        ai_results_frame = tk.LabelFrame(right_frame, text="AI 투자분석 결과", font=FONTS['small'], bg=COLORS['panel'], fg=COLORS['text'], bd=1)
        self.ai_result_text = self.create_text_widget(ai_results_frame)

    def create_status_bar(self, parent: tk.Frame):
        """하단 상태바를 생성합니다."""
        status_bar = tk.Frame(parent, bg=COLORS['panel'], height=30)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_label = tk.Label(status_bar, text="준비 완료", font=FONTS['small'], bg=COLORS['panel'], fg=COLORS['text'])
        self.status_label.pack(side=tk.LEFT, padx=10)

    # --- 헬퍼 메소드 ---
    def create_text_widget(self, parent_frame: tk.LabelFrame) -> tk.Text:
        """LabelFrame 내에 Text 위젯을 생성하고 반환합니다."""
        parent_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        text_widget = tk.Text(parent_frame, bg=COLORS['panel'], fg=COLORS['text'], wrap=tk.WORD, font=FONTS['body'], relief=tk.FLAT, bd=0)
        text_widget.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        return text_widget

    def create_tab_with_text(self, title: str) -> tk.Text:
        """Notebook에 새로운 탭과 Text 위젯을 생성하고 반환합니다."""
        tab_frame = tk.Frame(self.notebook, bg=COLORS['panel'])
        self.notebook.add(tab_frame, text=title)
        text_widget = tk.Text(tab_frame, bg=COLORS['panel'], fg=COLORS['text'], wrap=tk.WORD, font=FONTS['body'], relief=tk.FLAT, bd=0)
        text_widget.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        return text_widget

    # --- 이벤트 핸들러 및 UI 업데이트 메소드 ---
    def update_time(self):
        """상단 헤더의 시간을 1초마다 업데이트합니다."""
        self.time_label.config(text=time.strftime('%Y-%m-%d %H:%M:%S'))
        self.root.after(1000, self.update_time)

    def update_stock_list(self):
        """종목 리스트 Treeview를 전문 증권사 수준으로 업데이트합니다."""
        self.stock_tree.delete(*self.stock_tree.get_children())
        self.stock_tree_map = {}
        
        for stock in self.stock_data:
            color = "green" if stock.change_rate >= 0 else "red"
            values = (stock.name, f"{stock.price:,.0f}", f"{stock.change_rate:+.2f}%", 
                     f"{stock.price:.1f}", f"{stock.price:.1f}")
            iid = self.stock_tree.insert('', 'end', values=values, tags=(color,))
            self.stock_tree_map[stock.code] = iid
            
        self.stock_tree.tag_configure('red', foreground=COLORS['error'])
        self.stock_tree.tag_configure('green', foreground=COLORS['success'])

    def select_index_with_feedback(self, index_name: str):
        """지수 선택 피드백 버튼 클릭 시 실행됩니다."""
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
        """선택된 종목에 대한 모든 정보를 전문 증권사 수준으로 업데이트합니다."""
        self.update_stock_info_display()
            self.update_charts()

    def update_stock_info_display(self):
        """선택된 종목의 상세 정보를 전문 증권사 수준으로 표시합니다."""
        if not self.current_stock:
            return
            
        stock = self.current_stock
        change_color = COLORS['success'] if stock.change_rate >= 0 else COLORS['error']
        info_text = (f"{stock.name} ({stock.code}) | "
                    f"현재가: {stock.price:,.0f}원 | "
                    f"등락률: {stock.change_rate:+.2f}% | "
                    f"거래총액: {stock.volume:,.0f}주")
        
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
        """전문 증권사 수준의 전문가적인 차트를 업데이트합니다."""
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
        else:  # 모든 다른 차트 타입도 일목균형표로 표시
            self._draw_ichimoku_chart(dates, prices)
        
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
        tenkan_sen = np.convolve(prices, np.ones(9)/9, mode='same')  # 전환선(9일)
        kijun_sen = np.convolve(prices, np.ones(26)/26, mode='same')  # 기준선(26일)
        
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

    def _draw_volume_chart(self, dates: pd.DatetimeIndex, volume: np.ndarray):
        """거래량 차트를 그립니다."""
        colors = ['red' if i % 2 == 0 else 'blue' for i in range(len(volume))]
        self.ax_volume.bar(dates, volume, color=colors, alpha=0.6, width=0.8)
        self.ax_volume.set_title("거래량", color='white', fontsize=10)
        
        # 거래량 이동평균
        vol_ma = np.convolve(volume, np.ones(20)/20, mode='same')
        self.ax_volume.plot(dates, vol_ma, color='yellow', linewidth=1, alpha=0.8)

    def _apply_chart_styling(self):
        """차트에 전문 증권사 수준의 전문가적인 스타일을 적용합니다."""
        for ax in [self.ax_main, self.ax_volume]:
            ax.set_facecolor(COLORS['chart_bg'])
            ax.grid(True, color=COLORS['grid'], alpha=0.3, linewidth=0.5)
            ax.tick_params(colors='white', labelsize=8)
            
            for spine in ax.spines.values():
                spine.set_color(COLORS['grid'])
                spine.set_linewidth(0.5)
        
        # x축 라벨 회전
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
        self.ai_result_text.insert('1.0', "AI 분석이 시작됩니다. 잠시만 기다려주세요...")

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
        self.ai_result_text.insert('1.0', f"⚠️ {error_msg}\n\n")
        self.ai_result_text.insert(tk.END, "AI 분석을 사용할 수 없습니다. 일반 분석 정보를 표시합니다.\n")
        if self.current_stock:
            self.show_basic_analysis()

    def show_basic_analysis(self):
        """기본 분석 정보를 표시합니다."""
        if not self.current_stock:
            return
        
            stock = self.current_stock
            content = f"""
📊 {stock.name} 기본 분석 정보

🔸 현재 주가 정보
├ 현재가: {stock.price:,.0f}원
├ 등락률: {stock.change_rate:+.2f}%
├ 거래량: {stock.volume:,.0f}주
└ 섹터: {stock.sector}

🔸 투자 관점
선택된 투자가: {self.selected_guru}
투자 철학: {GURU_STRATEGIES.get(self.selected_guru, 'N/A')}

🔸 시장 지수
선택된 지수: {self.selected_index}
지수 설명: {MARKET_INDICES.get(self.selected_index, 'N/A')}
"""
        self.ai_result_text.delete('1.0', tk.END)
        self.ai_result_text.insert('1.0', content)

    def update_ai_results_display(self, results: Dict[str, Any]):
        """AI 분석 결과를 UI에 업데이트하고, 첫번째 종목을 선택합니다."""
        self.ai_result_text.delete('1.0', tk.END)
        self.ai_result_text.insert('1.0', f"🤖 AI 분석 완료 ({datetime.now().strftime('%H:%M:%S')})\n\n")
        
        for stock_code, analysis in results.items():
            stock_name = next((s.name for s in self.stock_data if s.code == stock_code), "미수신음")
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
        """탭 변경 시 호출되는 이벤트 핸들러"""
        selected_tab = event.widget.select()
        tab_text = event.widget.tab(selected_tab, "text")
        
        if tab_text == "AI 분석":
            self.run_comprehensive_analysis()

    def run(self):
        """애플리케이션을 실행합니다."""
        self.root.mainloop()

    def _update_text_widget(self, text_widget: tk.Text, title: str, content: str):
        """텍스트 위젯을 업데이트합니다."""
        text_widget.delete(1.0, tk.END)
        text_widget.insert(tk.END, content)

def main():
    """메인 함수 - 애플리케이션 시작점"""
    try:
        app = ComprehensiveHTS()
        app.run()
    except Exception as e:
        print(f"애플리케이션 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()