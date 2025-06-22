"""
차트 매니저 - 고성능 비동기 차트 렌더링 및 캐싱
"""
import asyncio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import mplfinance as mpf
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import structlog

from core.cache_manager import cached
from core.performance_monitor import monitor_performance
from core.database_manager import db_manager

logger = structlog.get_logger(__name__)

# 차트 스타일 설정
plt.rcParams['font.family'] = ['Malgun Gothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

CHART_COLORS = {
    'bg': '#111827',
    'grid': '#374151',
    'text': '#f8fafc',
    'up': '#4ade80',
    'down': '#ef4444',
    'volume': '#6366f1',
    'ma5': '#fbbf24',
    'ma20': '#06b6d4',
    'ma60': '#8b5cf6'
}


class ChartManager:
    """차트 매니저 - 비동기 차트 생성 및 관리"""
    
    def __init__(self):
        self._chart_cache: Dict[str, Figure] = {}
        self._canvas_cache: Dict[str, FigureCanvasTkAgg] = {}
        self._initialized = False
    
    async def initialize(self):
        """차트 매니저 초기화"""
        if self._initialized:
            return
        
        # matplotlib 백엔드 설정
        plt.switch_backend('TkAgg')
        
        self._initialized = True
        logger.info("차트 매니저 초기화 완료")
    
    @monitor_performance("create_chart")
    @cached(ttl=300, key_prefix="chart")
    async def create_chart_async(self, stock_code: str, chart_type: str = "candlestick") -> Figure:
        """비동기 차트 생성"""
        try:
            # 주가 데이터 조회
            price_data = await self._get_price_data(stock_code)
            
            if price_data is None or price_data.empty:
                return self._create_empty_chart(f"데이터 없음: {stock_code}")
            
            # 차트 타입별 생성
            if chart_type == "candlestick":
                return await self._create_candlestick_chart(stock_code, price_data)
            elif chart_type == "line":
                return await self._create_line_chart(stock_code, price_data)
            else:
                return await self._create_candlestick_chart(stock_code, price_data)
                
        except Exception as e:
            logger.error(f"차트 생성 실패 {stock_code}: {e}")
            return self._create_error_chart(f"차트 생성 실패: {e}")
    
    async def update_chart_async(self, stock_code: str, parent_frame, chart_type: str = "candlestick"):
        """차트 업데이트 (UI 스레드에서 실행)"""
        try:
            # 차트 생성
            fig = await self.create_chart_async(stock_code, chart_type)
            
            # UI 업데이트는 메인 스레드에서
            def update_ui():
                # 기존 캔버스 제거
                for widget in parent_frame.winfo_children():
                    widget.destroy()
                
                # 새 캔버스 생성
                canvas = FigureCanvasTkAgg(fig, master=parent_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill='both', expand=True)
                
                # 캔버스 캐시에 저장
                self._canvas_cache[stock_code] = canvas
            
            # 메인 스레드에서 UI 업데이트 실행
            parent_frame.after(0, update_ui)
            
        except Exception as e:
            logger.error(f"차트 업데이트 실패 {stock_code}: {e}")
    
    @cached(ttl=60, key_prefix="price_data")
    async def _get_price_data(self, stock_code: str, days: int = 100) -> Optional[pd.DataFrame]:
        """주가 데이터 조회 (캐시됨)"""
        try:
            # 데이터베이스에서 조회
            price_history = await db_manager.get_price_history(stock_code, days)
            
            if not price_history:
                # 샘플 데이터 생성 (실제 환경에서는 API 호출)
                return self._generate_sample_price_data(stock_code, days)
            
            # DataFrame으로 변환
            df = pd.DataFrame(price_history)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # 컬럼명 매핑
            df = df.rename(columns={
                'open_price': 'Open',
                'high_price': 'High',
                'low_price': 'Low',
                'close_price': 'Close',
                'volume': 'Volume'
            })
            
            return df
            
        except Exception as e:
            logger.error(f"주가 데이터 조회 실패 {stock_code}: {e}")
            return None
    
    def _generate_sample_price_data(self, stock_code: str, days: int = 100) -> pd.DataFrame:
        """샘플 주가 데이터 생성"""
        # 기본 가격 설정
        base_prices = {
            "005930": 75000,  # 삼성전자
            "000660": 120000,  # SK하이닉스
            "035420": 180000,  # NAVER
            "035720": 95000,   # 카카오
        }
        
        base_price = base_prices.get(stock_code, 50000)
        
        # 날짜 범위 생성
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        
        # 랜덤 워크로 가격 생성
        np.random.seed(hash(stock_code) % 2**32)  # 종목별 고정 시드
        returns = np.random.normal(0.001, 0.02, days)  # 일일 수익률
        prices = base_price * np.exp(np.cumsum(returns))
        
        # OHLCV 데이터 생성
        data = []
        for i, (date, close) in enumerate(zip(dates, prices)):
            high = close * (1 + np.random.uniform(0, 0.03))
            low = close * (1 - np.random.uniform(0, 0.03))
            open_price = close * (1 + np.random.uniform(-0.01, 0.01))
            volume = int(np.random.uniform(1000000, 10000000))
            
            data.append({
                'Open': open_price,
                'High': high,
                'Low': low,
                'Close': close,
                'Volume': volume
            })
        
        df = pd.DataFrame(data, index=dates)
        return df
    
    async def _create_candlestick_chart(self, stock_code: str, data: pd.DataFrame) -> Figure:
        """캔들스틱 차트 생성"""
        fig = Figure(figsize=(12, 8), dpi=100, facecolor=CHART_COLORS['bg'])
        
        # 서브플롯 생성 (가격 + 거래량)
        gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 0.5], hspace=0.1)
        ax_price = fig.add_subplot(gs[0])
        ax_volume = fig.add_subplot(gs[1], sharex=ax_price)
        ax_indicators = fig.add_subplot(gs[2], sharex=ax_price)
        
        # 캔들스틱 차트
        await self._draw_candlesticks(ax_price, data)
        
        # 이동평균선
        await self._draw_moving_averages(ax_price, data)
        
        # 거래량
        await self._draw_volume(ax_volume, data)
        
        # 기술적 지표
        await self._draw_technical_indicators(ax_indicators, data)
        
        # 스타일 적용
        self._apply_chart_style(fig, [ax_price, ax_volume, ax_indicators], stock_code)
        
        return fig
    
    async def _draw_candlesticks(self, ax, data: pd.DataFrame):
        """캔들스틱 그리기"""
        # 상승/하락 구분
        up = data['Close'] >= data['Open']
        down = ~up
        
        # 캔들 몸체
        ax.bar(data.index[up], data['Close'][up] - data['Open'][up], 
               bottom=data['Open'][up], color=CHART_COLORS['up'], alpha=0.8, width=0.8)
        ax.bar(data.index[down], data['Open'][down] - data['Close'][down], 
               bottom=data['Close'][down], color=CHART_COLORS['down'], alpha=0.8, width=0.8)
        
        # 심지 (High-Low)
        ax.vlines(data.index[up], data['Low'][up], data['High'][up], 
                 colors=CHART_COLORS['up'], linewidth=1, alpha=0.8)
        ax.vlines(data.index[down], data['Low'][down], data['High'][down], 
                 colors=CHART_COLORS['down'], linewidth=1, alpha=0.8)
    
    async def _draw_moving_averages(self, ax, data: pd.DataFrame):
        """이동평균선 그리기"""
        # 5일, 20일, 60일 이동평균
        ma5 = data['Close'].rolling(window=5).mean()
        ma20 = data['Close'].rolling(window=20).mean()
        ma60 = data['Close'].rolling(window=60).mean()
        
        ax.plot(data.index, ma5, color=CHART_COLORS['ma5'], linewidth=1, label='MA5', alpha=0.8)
        ax.plot(data.index, ma20, color=CHART_COLORS['ma20'], linewidth=1, label='MA20', alpha=0.8)
        ax.plot(data.index, ma60, color=CHART_COLORS['ma60'], linewidth=1, label='MA60', alpha=0.8)
        
        ax.legend(loc='upper left', facecolor=CHART_COLORS['bg'], edgecolor='white')
    
    async def _draw_volume(self, ax, data: pd.DataFrame):
        """거래량 차트 그리기"""
        up = data['Close'] >= data['Open']
        
        ax.bar(data.index[up], data['Volume'][up], 
               color=CHART_COLORS['up'], alpha=0.6, width=0.8)
        ax.bar(data.index[~up], data['Volume'][~up], 
               color=CHART_COLORS['down'], alpha=0.6, width=0.8)
        
        # 거래량 이동평균
        vol_ma = data['Volume'].rolling(window=20).mean()
        ax.plot(data.index, vol_ma, color=CHART_COLORS['volume'], linewidth=1, alpha=0.8)
        
        ax.set_ylabel('거래량', color=CHART_COLORS['text'])
    
    async def _draw_technical_indicators(self, ax, data: pd.DataFrame):
        """기술적 지표 그리기 (RSI)"""
        rsi = await self._calculate_rsi(data['Close'])
        
        ax.plot(data.index, rsi, color=CHART_COLORS['volume'], linewidth=1)
        ax.axhline(y=70, color=CHART_COLORS['down'], linestyle='--', alpha=0.5)
        ax.axhline(y=30, color=CHART_COLORS['up'], linestyle='--', alpha=0.5)
        ax.axhline(y=50, color=CHART_COLORS['text'], linestyle='-', alpha=0.3)
        
        ax.set_ylabel('RSI', color=CHART_COLORS['text'])
        ax.set_ylim(0, 100)
    
    async def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    async def _create_line_chart(self, stock_code: str, data: pd.DataFrame) -> Figure:
        """라인 차트 생성"""
        fig = Figure(figsize=(12, 6), dpi=100, facecolor=CHART_COLORS['bg'])
        ax = fig.add_subplot(111)
        
        # 종가 라인
        ax.plot(data.index, data['Close'], color=CHART_COLORS['up'], linewidth=2)
        
        # 이동평균선
        ma20 = data['Close'].rolling(window=20).mean()
        ax.plot(data.index, ma20, color=CHART_COLORS['ma20'], linewidth=1, alpha=0.8, label='MA20')
        
        self._apply_chart_style(fig, [ax], stock_code)
        ax.legend()
        
        return fig
    
    def _create_empty_chart(self, message: str) -> Figure:
        """빈 차트 생성"""
        fig = Figure(figsize=(12, 6), dpi=100, facecolor=CHART_COLORS['bg'])
        ax = fig.add_subplot(111)
        
        ax.text(0.5, 0.5, message, transform=ax.transAxes, 
                ha='center', va='center', color=CHART_COLORS['text'], fontsize=16)
        
        self._apply_chart_style(fig, [ax], "")
        return fig
    
    def _create_error_chart(self, error_message: str) -> Figure:
        """에러 차트 생성"""
        fig = Figure(figsize=(12, 6), dpi=100, facecolor=CHART_COLORS['bg'])
        ax = fig.add_subplot(111)
        
        ax.text(0.5, 0.5, f"⚠️ {error_message}", transform=ax.transAxes,
                ha='center', va='center', color=CHART_COLORS['down'], fontsize=14)
        
        self._apply_chart_style(fig, [ax], "")
        return fig
    
    def _apply_chart_style(self, fig: Figure, axes: List, title: str):
        """차트 스타일 적용"""
        fig.patch.set_facecolor(CHART_COLORS['bg'])
        
        for ax in axes:
            ax.set_facecolor(CHART_COLORS['bg'])
            ax.grid(True, color=CHART_COLORS['grid'], alpha=0.3, linewidth=0.5)
            ax.tick_params(colors=CHART_COLORS['text'], labelsize=8)
            
            for spine in ax.spines.values():
                spine.set_color(CHART_COLORS['grid'])
                spine.set_linewidth(0.5)
        
        if title and axes:
            axes[0].set_title(f"{title} 차트", color=CHART_COLORS['text'], fontsize=12, pad=10)
        
        fig.tight_layout(pad=1.0)
    
    def clear_cache(self):
        """차트 캐시 정리"""
        self._chart_cache.clear()
        
        # 캔버스 정리
        for canvas in self._canvas_cache.values():
            try:
                canvas.get_tk_widget().destroy()
            except:
                pass
        self._canvas_cache.clear()
        
        logger.info("차트 캐시 정리 완료")
    
    async def cleanup(self):
        """차트 매니저 정리"""
        self.clear_cache()
        logger.info("차트 매니저 정리 완료") 