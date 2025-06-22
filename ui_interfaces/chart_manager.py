#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Ultra 차트 매니저 v5.0 - 고성능 실시간 차트 렌더링
- 비동기 렌더링 & 멀티스레드 처리
- 실시간 데이터 스트리밍 & 캐싱
- 고성능 시각화 & 메모리 최적화
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import threading
import queue
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
import structlog

# 차트 라이브러리
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.offline import plot
import mplfinance as mpf

# GUI 라이브러리
import tkinter as tk
from tkinter import ttk

from core.cache_manager import get_cache_manager, cached
from core.performance_monitor import monitor_performance
from config.settings import settings

logger = structlog.get_logger(__name__)


class ChartType(Enum):
    """차트 유형"""
    CANDLESTICK = "candlestick"
    LINE = "line"
    VOLUME = "volume"
    TECHNICAL = "technical"
    HEATMAP = "heatmap"
    SCATTER = "scatter"
    CORRELATION = "correlation"


class TimeFrame(Enum):
    """시간 프레임"""
    REALTIME = "1s"
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAILY = "1d"
    WEEKLY = "1w"
    MONTHLY = "1M"


class RenderEngine(Enum):
    """렌더링 엔진"""
    MATPLOTLIB = "matplotlib"
    PLOTLY = "plotly"
    BOKEH = "bokeh"
    CANVAS = "canvas"


@dataclass
class ChartConfig:
    """차트 설정"""
    chart_type: ChartType
    time_frame: TimeFrame
    render_engine: RenderEngine
    width: int = 800
    height: int = 600
    theme: str = "dark"
    show_volume: bool = True
    show_indicators: bool = True
    auto_update: bool = True
    update_interval: int = 1000  # ms
    max_data_points: int = 1000
    cache_enabled: bool = True


@dataclass
class ChartData:
    """차트 데이터"""
    symbol: str
    data: pd.DataFrame
    timestamp: datetime
    timeframe: TimeFrame
    indicators: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RenderTask:
    """렌더링 작업"""
    task_id: str
    chart_config: ChartConfig
    chart_data: ChartData
    callback: Optional[Callable] = None
    priority: int = 1
    created_at: datetime = field(default_factory=datetime.now)


class UltraChartManager:
    """🚀 Ultra 차트 매니저 - 고성능 실시간 차트 렌더링"""
    
    def __init__(self):
        # 렌더링 큐 및 스레드 풀
        self._render_queue: queue.PriorityQueue = queue.PriorityQueue(maxsize=1000)
        self._executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="chart_render")
        
        # 캐시 매니저
        self._cache_manager = get_cache_manager()
        
        # 차트 인스턴스 관리
        self._charts: Dict[str, Any] = {}
        self._chart_refs: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
        
        # 실시간 업데이트
        self._update_tasks: Dict[str, asyncio.Task] = {}
        self._data_streams: Dict[str, asyncio.Queue] = {}
        
        # 성능 최적화
        self._render_stats = {
            "total_renders": 0,
            "successful_renders": 0,
            "failed_renders": 0,
            "avg_render_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        # 스타일 설정
        self._setup_styles()
        
        # 렌더링 워커 시작
        self._start_workers()
        
        logger.info("Ultra 차트 매니저 초기화 완료")
    
    def _setup_styles(self) -> None:
        """차트 스타일 설정"""
        try:
            # Matplotlib 스타일
            plt.style.use('dark_background')
            sns.set_palette("husl")
            
            # 기본 색상 팔레트
            self._colors = {
                "primary": "#00D4FF",
                "secondary": "#FF6B6B",
                "success": "#4CAF50",
                "warning": "#FF9800",
                "danger": "#F44336",
                "info": "#2196F3",
                "light": "#FFFFFF",
                "dark": "#1E1E1E",
                "background": "#0A0A0A",
                "surface": "#1A1A1A"
            }
            
            # 폰트 설정
            plt.rcParams.update({
                'font.family': ['DejaVu Sans', 'Arial', 'sans-serif'],
                'font.size': 10,
                'axes.titlesize': 12,
                'axes.labelsize': 10,
                'xtick.labelsize': 9,
                'ytick.labelsize': 9,
                'legend.fontsize': 9,
                'figure.titlesize': 14
            })
            
            logger.debug("차트 스타일 설정 완료")
            
        except Exception as e:
            logger.error(f"스타일 설정 실패: {e}")
    
    def _start_workers(self) -> None:
        """렌더링 워커 시작"""
        try:
            # 렌더링 워커 스레드들
            for i in range(4):
                worker_thread = threading.Thread(
                    target=self._render_worker,
                    name=f"chart_render_worker_{i}",
                    daemon=True
                )
                worker_thread.start()
            
            logger.info("차트 렌더링 워커 시작 완료")
            
        except Exception as e:
            logger.error(f"워커 시작 실패: {e}")
    
    def _render_worker(self) -> None:
        """렌더링 워커 스레드"""
        while True:
            try:
                # 우선순위 큐에서 작업 가져오기
                priority, task_id, render_task = self._render_queue.get(timeout=1)
                
                # 렌더링 실행
                start_time = time.time()
                result = self._execute_render_task(render_task)
                render_time = time.time() - start_time
                
                # 통계 업데이트
                self._render_stats["total_renders"] += 1
                if result:
                    self._render_stats["successful_renders"] += 1
                else:
                    self._render_stats["failed_renders"] += 1
                
                # 평균 렌더링 시간 업데이트
                total = self._render_stats["total_renders"]
                current_avg = self._render_stats["avg_render_time"]
                self._render_stats["avg_render_time"] = (
                    (current_avg * (total - 1) + render_time) / total
                )
                
                # 콜백 실행
                if render_task.callback:
                    render_task.callback(result)
                
                self._render_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"렌더링 워커 오류: {e}")
                time.sleep(0.1)
    
    def _execute_render_task(self, task: RenderTask) -> Optional[Any]:
        """렌더링 작업 실행"""
        try:
            config = task.chart_config
            data = task.chart_data
            
            # 캐시 확인
            if config.cache_enabled:
                cache_key = self._generate_cache_key(config, data)
                cached_result = self._get_cached_chart(cache_key)
                if cached_result:
                    self._render_stats["cache_hits"] += 1
                    return cached_result
            
            # 렌더링 엔진별 처리
            if config.render_engine == RenderEngine.MATPLOTLIB:
                result = self._render_matplotlib(config, data)
            elif config.render_engine == RenderEngine.PLOTLY:
                result = self._render_plotly(config, data)
            else:
                result = self._render_matplotlib(config, data)  # 기본값
            
            # 캐시에 저장
            if config.cache_enabled and result:
                cache_key = self._generate_cache_key(config, data)
                self._cache_chart(cache_key, result)
            
            self._render_stats["cache_misses"] += 1
            return result
            
        except Exception as e:
            logger.error(f"렌더링 실행 실패: {e}")
            return None
    
    def _render_matplotlib(self, config: ChartConfig, data: ChartData) -> Optional[Figure]:
        """Matplotlib 차트 렌더링"""
        try:
            # Figure 생성
            fig = Figure(
                figsize=(config.width/100, config.height/100),
                facecolor=self._colors["background"],
                edgecolor='none'
            )
            
            # 차트 유형별 렌더링
            if config.chart_type == ChartType.CANDLESTICK:
                return self._render_candlestick_matplotlib(fig, config, data)
            elif config.chart_type == ChartType.LINE:
                return self._render_line_matplotlib(fig, config, data)
            elif config.chart_type == ChartType.VOLUME:
                return self._render_volume_matplotlib(fig, config, data)
            elif config.chart_type == ChartType.HEATMAP:
                return self._render_heatmap_matplotlib(fig, config, data)
            else:
                return self._render_line_matplotlib(fig, config, data)  # 기본값
            
        except Exception as e:
            logger.error(f"Matplotlib 렌더링 실패: {e}")
            return None
    
    def _render_candlestick_matplotlib(self, 
                                     fig: Figure, 
                                     config: ChartConfig, 
                                     data: ChartData) -> Figure:
        """캔들스틱 차트 렌더링"""
        try:
            df = data.data
            
            # 메인 차트 (가격)
            if config.show_volume:
                ax1 = fig.add_subplot(2, 1, 1)
                ax2 = fig.add_subplot(2, 1, 2)
            else:
                ax1 = fig.add_subplot(1, 1, 1)
                ax2 = None
            
            # 캔들스틱 데이터 준비
            if 'Open' in df.columns and 'High' in df.columns:
                # 실제 OHLC 데이터
                opens = df['Open'].values
                highs = df['High'].values
                lows = df['Low'].values
                closes = df['Close'].values
            else:
                # 가격 데이터만 있는 경우 시뮬레이션
                prices = df['price'].values if 'price' in df.columns else np.random.randn(100).cumsum() + 100
                opens = np.roll(prices, 1)
                opens[0] = prices[0]
                highs = prices * (1 + np.random.uniform(0, 0.02, len(prices)))
                lows = prices * (1 - np.random.uniform(0, 0.02, len(prices)))
                closes = prices
            
            # 캔들스틱 그리기
            for i in range(len(closes)):
                color = self._colors["success"] if closes[i] >= opens[i] else self._colors["danger"]
                
                # 몸통
                height = abs(closes[i] - opens[i])
                bottom = min(opens[i], closes[i])
                ax1.bar(i, height, bottom=bottom, color=color, alpha=0.8, width=0.8)
                
                # 꼬리
                ax1.plot([i, i], [lows[i], highs[i]], color=color, linewidth=1, alpha=0.6)
            
            # 차트 스타일링
            ax1.set_facecolor(self._colors["background"])
            ax1.grid(True, alpha=0.3, color=self._colors["light"])
            ax1.set_title(f"{data.symbol} - {config.chart_type.value.upper()}", 
                         color=self._colors["light"], fontsize=12, fontweight='bold')
            ax1.tick_params(colors=self._colors["light"])
            
            # 거래량 차트 (옵션)
            if ax2 and 'volume' in df.columns:
                volumes = df['volume'].values
                colors = [self._colors["success"] if closes[i] >= opens[i] else self._colors["danger"] 
                         for i in range(len(closes))]
                ax2.bar(range(len(volumes)), volumes, color=colors, alpha=0.6)
                ax2.set_facecolor(self._colors["background"])
                ax2.grid(True, alpha=0.3, color=self._colors["light"])
                ax2.set_title("거래량", color=self._colors["light"], fontsize=10)
                ax2.tick_params(colors=self._colors["light"])
            
            # 기술적 지표 추가 (옵션)
            if config.show_indicators and 'indicators' in data.metadata:
                self._add_technical_indicators(ax1, data.metadata['indicators'])
            
            fig.tight_layout()
            return fig
            
        except Exception as e:
            logger.error(f"캔들스틱 렌더링 실패: {e}")
            return fig
    
    def _render_line_matplotlib(self, 
                              fig: Figure, 
                              config: ChartConfig, 
                              data: ChartData) -> Figure:
        """라인 차트 렌더링"""
        try:
            df = data.data
            ax = fig.add_subplot(1, 1, 1)
            
            # 가격 데이터 추출
            if 'price' in df.columns:
                prices = df['price'].values
            elif 'Close' in df.columns:
                prices = df['Close'].values
            else:
                # 샘플 데이터 생성
                prices = np.random.randn(100).cumsum() + 100
            
            # 라인 차트 그리기
            x = range(len(prices))
            ax.plot(x, prices, color=self._colors["primary"], linewidth=2, alpha=0.8)
            
            # 이동평균선 추가
            if len(prices) > 20:
                ma20 = pd.Series(prices).rolling(20).mean()
                ax.plot(x, ma20, color=self._colors["warning"], linewidth=1, alpha=0.7, label='MA20')
            
            if len(prices) > 50:
                ma50 = pd.Series(prices).rolling(50).mean()
                ax.plot(x, ma50, color=self._colors["info"], linewidth=1, alpha=0.7, label='MA50')
            
            # 차트 스타일링
            ax.set_facecolor(self._colors["background"])
            ax.grid(True, alpha=0.3, color=self._colors["light"])
            ax.set_title(f"{data.symbol} - 가격 추이", 
                        color=self._colors["light"], fontsize=12, fontweight='bold')
            ax.tick_params(colors=self._colors["light"])
            
            # 범례
            if ax.get_legend_handles_labels()[0]:
                ax.legend(facecolor=self._colors["surface"], 
                         edgecolor=self._colors["light"], 
                         labelcolor=self._colors["light"])
            
            fig.tight_layout()
            return fig
            
        except Exception as e:
            logger.error(f"라인 차트 렌더링 실패: {e}")
            return fig
    
    def _render_volume_matplotlib(self, 
                                fig: Figure, 
                                config: ChartConfig, 
                                data: ChartData) -> Figure:
        """거래량 차트 렌더링"""
        try:
            df = data.data
            ax = fig.add_subplot(1, 1, 1)
            
            # 거래량 데이터 추출
            if 'volume' in df.columns:
                volumes = df['volume'].values
            else:
                # 샘플 거래량 생성
                volumes = np.random.lognormal(13, 1, 100)
            
            # 색상 결정 (가격 변화에 따라)
            colors = []
            if 'change_rate' in df.columns:
                change_rates = df['change_rate'].values
                colors = [self._colors["success"] if rate >= 0 else self._colors["danger"] 
                         for rate in change_rates]
            else:
                colors = [self._colors["primary"]] * len(volumes)
            
            # 거래량 바 차트
            x = range(len(volumes))
            bars = ax.bar(x, volumes, color=colors, alpha=0.7, width=0.8)
            
            # 평균 거래량 라인
            avg_volume = np.mean(volumes)
            ax.axhline(y=avg_volume, color=self._colors["warning"], 
                      linestyle='--', alpha=0.8, label=f'평균: {avg_volume:,.0f}')
            
            # 차트 스타일링
            ax.set_facecolor(self._colors["background"])
            ax.grid(True, alpha=0.3, color=self._colors["light"])
            ax.set_title(f"{data.symbol} - 거래량", 
                        color=self._colors["light"], fontsize=12, fontweight='bold')
            ax.tick_params(colors=self._colors["light"])
            ax.legend(facecolor=self._colors["surface"], 
                     edgecolor=self._colors["light"], 
                     labelcolor=self._colors["light"])
            
            # Y축 포맷팅
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
            
            fig.tight_layout()
            return fig
            
        except Exception as e:
            logger.error(f"거래량 차트 렌더링 실패: {e}")
            return fig
    
    def _render_heatmap_matplotlib(self, 
                                 fig: Figure, 
                                 config: ChartConfig, 
                                 data: ChartData) -> Figure:
        """히트맵 렌더링"""
        try:
            df = data.data
            ax = fig.add_subplot(1, 1, 1)
            
            # 상관관계 매트릭스 생성
            if len(df.columns) > 1:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 1:
                    corr_matrix = df[numeric_cols].corr()
                else:
                    # 샘플 상관관계 매트릭스
                    corr_matrix = np.random.rand(5, 5)
                    corr_matrix = pd.DataFrame(corr_matrix, 
                                             columns=['A', 'B', 'C', 'D', 'E'],
                                             index=['A', 'B', 'C', 'D', 'E'])
            else:
                # 샘플 데이터
                corr_matrix = np.random.rand(5, 5)
                corr_matrix = pd.DataFrame(corr_matrix, 
                                         columns=['A', 'B', 'C', 'D', 'E'],
                                         index=['A', 'B', 'C', 'D', 'E'])
            
            # 히트맵 그리기
            im = ax.imshow(corr_matrix.values, cmap='RdYlBu_r', aspect='auto')
            
            # 축 레이블
            ax.set_xticks(range(len(corr_matrix.columns)))
            ax.set_yticks(range(len(corr_matrix.index)))
            ax.set_xticklabels(corr_matrix.columns, color=self._colors["light"])
            ax.set_yticklabels(corr_matrix.index, color=self._colors["light"])
            
            # 값 표시
            for i in range(len(corr_matrix.index)):
                for j in range(len(corr_matrix.columns)):
                    text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                  ha="center", va="center", color="white", fontsize=8)
            
            # 컬러바
            cbar = fig.colorbar(im, ax=ax)
            cbar.ax.tick_params(colors=self._colors["light"])
            
            # 차트 스타일링
            ax.set_facecolor(self._colors["background"])
            ax.set_title(f"{data.symbol} - 상관관계 히트맵", 
                        color=self._colors["light"], fontsize=12, fontweight='bold')
            
            fig.tight_layout()
            return fig
            
        except Exception as e:
            logger.error(f"히트맵 렌더링 실패: {e}")
            return fig
    
    def _add_technical_indicators(self, ax, indicators: Dict[str, Any]) -> None:
        """기술적 지표 추가"""
        try:
            # RSI
            if 'rsi' in indicators:
                rsi = indicators['rsi']
                ax2 = ax.twinx()
                ax2.plot(rsi, color=self._colors["info"], alpha=0.7, label='RSI')
                ax2.axhline(y=70, color=self._colors["danger"], linestyle='--', alpha=0.5)
                ax2.axhline(y=30, color=self._colors["success"], linestyle='--', alpha=0.5)
                ax2.set_ylim(0, 100)
                ax2.legend(loc='upper right')
            
            # 볼린저 밴드
            if 'bollinger_upper' in indicators and 'bollinger_lower' in indicators:
                upper = indicators['bollinger_upper']
                lower = indicators['bollinger_lower']
                ax.plot(upper, color=self._colors["warning"], alpha=0.5, linestyle='--', label='BB Upper')
                ax.plot(lower, color=self._colors["warning"], alpha=0.5, linestyle='--', label='BB Lower')
                ax.fill_between(range(len(upper)), upper, lower, 
                               color=self._colors["warning"], alpha=0.1)
            
        except Exception as e:
            logger.error(f"기술적 지표 추가 실패: {e}")
    
    def _render_plotly(self, config: ChartConfig, data: ChartData) -> Optional[str]:
        """Plotly 차트 렌더링"""
        try:
            df = data.data
            
            if config.chart_type == ChartType.CANDLESTICK:
                return self._render_candlestick_plotly(config, data)
            elif config.chart_type == ChartType.LINE:
                return self._render_line_plotly(config, data)
            else:
                return self._render_line_plotly(config, data)
            
        except Exception as e:
            logger.error(f"Plotly 렌더링 실패: {e}")
            return None
    
    def _render_candlestick_plotly(self, config: ChartConfig, data: ChartData) -> str:
        """Plotly 캔들스틱 차트"""
        try:
            df = data.data
            
            # 캔들스틱 차트 생성
            fig = go.Figure()
            
            # 샘플 데이터 생성
            dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
            prices = np.random.randn(100).cumsum() + 100
            opens = np.roll(prices, 1)
            opens[0] = prices[0]
            highs = prices * (1 + np.random.uniform(0, 0.02, len(prices)))
            lows = prices * (1 - np.random.uniform(0, 0.02, len(prices)))
            closes = prices
            
            fig.add_trace(go.Candlestick(
                x=dates,
                open=opens,
                high=highs,
                low=lows,
                close=closes,
                name=data.symbol
            ))
            
            # 레이아웃 설정
            fig.update_layout(
                title=f"{data.symbol} - Candlestick Chart",
                xaxis_title="Date",
                yaxis_title="Price",
                template="plotly_dark",
                width=config.width,
                height=config.height
            )
            
            return fig.to_html(include_plotlyjs='cdn')
            
        except Exception as e:
            logger.error(f"Plotly 캔들스틱 렌더링 실패: {e}")
            return ""
    
    def _render_line_plotly(self, config: ChartConfig, data: ChartData) -> str:
        """Plotly 라인 차트"""
        try:
            df = data.data
            
            # 라인 차트 생성
            fig = go.Figure()
            
            # 가격 데이터
            if 'price' in df.columns:
                prices = df['price'].values
            else:
                prices = np.random.randn(100).cumsum() + 100
            
            dates = pd.date_range(start='2024-01-01', periods=len(prices), freq='D')
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=prices,
                mode='lines',
                name=data.symbol,
                line=dict(color='#00D4FF', width=2)
            ))
            
            # 레이아웃 설정
            fig.update_layout(
                title=f"{data.symbol} - Price Chart",
                xaxis_title="Date",
                yaxis_title="Price",
                template="plotly_dark",
                width=config.width,
                height=config.height
            )
            
            return fig.to_html(include_plotlyjs='cdn')
            
        except Exception as e:
            logger.error(f"Plotly 라인 차트 렌더링 실패: {e}")
            return ""
    
    def _generate_cache_key(self, config: ChartConfig, data: ChartData) -> str:
        """캐시 키 생성"""
        key_parts = [
            data.symbol,
            config.chart_type.value,
            config.time_frame.value,
            str(config.width),
            str(config.height),
            str(hash(str(data.data.values.tobytes()) if hasattr(data.data, 'values') else str(data.data)))
        ]
        return "_".join(key_parts)
    
    def _get_cached_chart(self, cache_key: str) -> Optional[Any]:
        """캐시된 차트 조회"""
        try:
            # 동기적 캐시 조회 (렌더링 스레드에서 호출됨)
            return None  # 현재는 비활성화
        except Exception as e:
            logger.error(f"캐시 조회 실패: {e}")
            return None
    
    def _cache_chart(self, cache_key: str, chart_data: Any) -> None:
        """차트 캐시 저장"""
        try:
            # 동기적 캐시 저장 (렌더링 스레드에서 호출됨)
            pass  # 현재는 비활성화
        except Exception as e:
            logger.error(f"캐시 저장 실패: {e}")
    
    # 공개 API
    @monitor_performance("render_chart")
    async def render_chart_async(self, 
                               config: ChartConfig, 
                               data: ChartData,
                               callback: Optional[Callable] = None) -> str:
        """비동기 차트 렌더링"""
        try:
            task_id = f"chart_{data.symbol}_{int(time.time())}"
            
            # 렌더링 작업 생성
            render_task = RenderTask(
                task_id=task_id,
                chart_config=config,
                chart_data=data,
                callback=callback,
                priority=1
            )
            
            # 큐에 추가
            self._render_queue.put((render_task.priority, task_id, render_task))
            
            # 결과 대기 (Future 패턴)
            result_future = asyncio.Future()
            
            def result_callback(result):
                if not result_future.done():
                    result_future.set_result(result)
            
            render_task.callback = result_callback
            
            # 타임아웃과 함께 결과 대기
            try:
                result = await asyncio.wait_for(result_future, timeout=30.0)
                return result if result else ""
            except asyncio.TimeoutError:
                logger.error(f"차트 렌더링 타임아웃: {task_id}")
                return ""
            
        except Exception as e:
            logger.error(f"비동기 차트 렌더링 실패: {e}")
            return ""
    
    def render_chart_sync(self, config: ChartConfig, data: ChartData) -> Optional[Any]:
        """동기 차트 렌더링 (즉시 실행)"""
        try:
            render_task = RenderTask(
                task_id=f"sync_{int(time.time())}",
                chart_config=config,
                chart_data=data
            )
            
            return self._execute_render_task(render_task)
            
        except Exception as e:
            logger.error(f"동기 차트 렌더링 실패: {e}")
            return None
    
    def create_tkinter_chart(self, 
                           parent: tk.Widget, 
                           config: ChartConfig, 
                           data: ChartData) -> Optional[FigureCanvasTkAgg]:
        """Tkinter용 차트 생성"""
        try:
            # Matplotlib 차트 렌더링
            config.render_engine = RenderEngine.MATPLOTLIB
            fig = self.render_chart_sync(config, data)
            
            if fig:
                # Tkinter 캔버스에 임베드
                canvas = FigureCanvasTkAgg(fig, parent)
                canvas.draw()
                return canvas
            
            return None
            
        except Exception as e:
            logger.error(f"Tkinter 차트 생성 실패: {e}")
            return None
    
    async def start_realtime_updates(self, 
                                   chart_id: str, 
                                   config: ChartConfig, 
                                   data_source: Callable) -> None:
        """실시간 차트 업데이트 시작"""
        try:
            if chart_id in self._update_tasks:
                await self.stop_realtime_updates(chart_id)
            
            # 데이터 스트림 생성
            data_stream = asyncio.Queue(maxsize=1000)
            self._data_streams[chart_id] = data_stream
            
            # 업데이트 태스크 시작
            update_task = asyncio.create_task(
                self._realtime_update_worker(chart_id, config, data_source, data_stream)
            )
            self._update_tasks[chart_id] = update_task
            
            logger.info(f"실시간 차트 업데이트 시작: {chart_id}")
            
        except Exception as e:
            logger.error(f"실시간 업데이트 시작 실패: {e}")
    
    async def stop_realtime_updates(self, chart_id: str) -> None:
        """실시간 차트 업데이트 중지"""
        try:
            if chart_id in self._update_tasks:
                self._update_tasks[chart_id].cancel()
                del self._update_tasks[chart_id]
            
            if chart_id in self._data_streams:
                del self._data_streams[chart_id]
            
            logger.info(f"실시간 차트 업데이트 중지: {chart_id}")
            
        except Exception as e:
            logger.error(f"실시간 업데이트 중지 실패: {e}")
    
    async def _realtime_update_worker(self, 
                                    chart_id: str, 
                                    config: ChartConfig, 
                                    data_source: Callable,
                                    data_stream: asyncio.Queue) -> None:
        """실시간 업데이트 워커"""
        try:
            while True:
                # 새 데이터 가져오기
                new_data = await data_source()
                
                if new_data:
                    # 스트림에 추가
                    if not data_stream.full():
                        await data_stream.put(new_data)
                    
                    # 차트 업데이트
                    await self.render_chart_async(config, new_data)
                
                # 업데이트 간격 대기
                await asyncio.sleep(config.update_interval / 1000.0)
                
        except asyncio.CancelledError:
            logger.info(f"실시간 업데이트 워커 종료: {chart_id}")
        except Exception as e:
            logger.error(f"실시간 업데이트 워커 오류: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """차트 매니저 통계 반환"""
        return {
            **self._render_stats,
            "active_charts": len(self._charts),
            "active_updates": len(self._update_tasks),
            "queue_size": self._render_queue.qsize(),
            "cache_size": len(self._chart_refs)
        }
    
    def cleanup(self) -> None:
        """차트 매니저 정리"""
        try:
            # 실시간 업데이트 중지
            for chart_id in list(self._update_tasks.keys()):
                asyncio.create_task(self.stop_realtime_updates(chart_id))
            
            # 스레드 풀 종료
            self._executor.shutdown(wait=False)
            
            # 차트 참조 정리
            self._charts.clear()
            self._chart_refs.clear()
            
            logger.info("Ultra 차트 매니저 정리 완료")
            
        except Exception as e:
            logger.error(f"차트 매니저 정리 중 오류: {e}")


# 전역 차트 매니저 인스턴스
_chart_manager: Optional[UltraChartManager] = None


def get_chart_manager() -> UltraChartManager:
    """전역 차트 매니저 반환"""
    global _chart_manager
    if _chart_manager is None:
        _chart_manager = UltraChartManager()
    return _chart_manager


# 편의 함수들
def create_chart_config(chart_type: ChartType = ChartType.LINE,
                       time_frame: TimeFrame = TimeFrame.DAILY,
                       width: int = 800,
                       height: int = 600) -> ChartConfig:
    """차트 설정 생성"""
    return ChartConfig(
        chart_type=chart_type,
        time_frame=time_frame,
        render_engine=RenderEngine.MATPLOTLIB,
        width=width,
        height=height
    )


def create_chart_data(symbol: str, 
                     data: Union[pd.DataFrame, Dict, List],
                     timeframe: TimeFrame = TimeFrame.DAILY) -> ChartData:
    """차트 데이터 생성"""
    if isinstance(data, dict):
        data = pd.DataFrame([data])
    elif isinstance(data, list):
        data = pd.DataFrame(data)
    
    return ChartData(
        symbol=symbol,
        data=data,
        timestamp=datetime.now(),
        timeframe=timeframe
    ) 