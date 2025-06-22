"""
차트 매니저 - 주식 차트 생성 및 관리
"""
import tkinter as tk
from tkinter import ttk, Frame, Canvas
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import structlog

logger = structlog.get_logger(__name__)


class ChartManager:
    """차트 생성 및 관리 클래스"""
    
    def __init__(self):
        """차트 매니저 초기화"""
        self.figure = None
        self.canvas = None
        self.ax_price = None
        self.ax_volume = None
        self._initialized = False
        
        # 차트 스타일 설정
        plt.style.use('dark_background')
        
        logger.info("차트 매니저 초기화 완료")
    
    async def initialize(self):
        """비동기 초기화"""
        if self._initialized:
            return
        
        try:
            # 차트 설정 초기화
            self._setup_chart_style()
            self._initialized = True
            logger.info("차트 매니저 비동기 초기화 완료")
        except Exception as e:
            logger.error(f"차트 매니저 초기화 실패: {e}")
            raise
    
    def _setup_chart_style(self):
        """차트 스타일 설정"""
        # matplotlib 한글 폰트 설정
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False
        
        # 차트 색상 설정
        plt.rcParams['figure.facecolor'] = '#1e1e1e'
        plt.rcParams['axes.facecolor'] = '#2d2d2d'
        plt.rcParams['axes.edgecolor'] = '#666666'
        plt.rcParams['text.color'] = 'white'
        plt.rcParams['axes.labelcolor'] = 'white'
        plt.rcParams['xtick.color'] = 'white'
        plt.rcParams['ytick.color'] = 'white'
    
    def create_chart_widget(self, parent_frame: Frame) -> Frame:
        """차트 위젯 생성"""
        try:
            # 차트 프레임 생성
            chart_frame = Frame(parent_frame, bg='#1e1e1e')
            
            # Figure 생성
            self.figure = Figure(figsize=(12, 8), facecolor='#1e1e1e')
            
            # 서브플롯 생성 (가격 + 거래량)
            self.ax_price = self.figure.add_subplot(211, facecolor='#2d2d2d')
            self.ax_volume = self.figure.add_subplot(212, facecolor='#2d2d2d', sharex=self.ax_price)
            
            # Canvas 생성
            self.canvas = FigureCanvasTkAgg(self.figure, chart_frame)
            self.canvas.draw()
            
            # Canvas 위젯을 프레임에 추가
            canvas_widget = self.canvas.get_tk_widget()
            canvas_widget.pack(fill=tk.BOTH, expand=True)
            
            # 초기 차트 그리기
            self._draw_sample_chart()
            
            logger.info("차트 위젯 생성 완료")
            return chart_frame
            
        except Exception as e:
            logger.error(f"차트 위젯 생성 실패: {e}")
            # 에러 발생 시 빈 프레임 반환
            return Frame(parent_frame, bg='#1e1e1e')
    
    def _draw_sample_chart(self):
        """샘플 차트 그리기"""
        try:
            # 샘플 데이터 생성
            dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
            np.random.seed(42)
            
            # 주가 데이터 생성 (랜덤 워크)
            returns = np.random.normal(0.001, 0.02, len(dates))
            prices = 100 * np.exp(np.cumsum(returns))
            
            # 거래량 데이터 생성
            volumes = np.random.lognormal(10, 0.5, len(dates))
            
            # 가격 차트 그리기
            self.ax_price.clear()
            self.ax_price.plot(dates, prices, color='#00d4aa', linewidth=1.5, label='주가')
            self.ax_price.set_title('샘플 주식 차트', color='white', fontsize=14, pad=20)
            self.ax_price.set_ylabel('가격 (원)', color='white')
            self.ax_price.grid(True, alpha=0.3, color='#666666')
            self.ax_price.legend(loc='upper left')
            
            # 거래량 차트 그리기
            self.ax_volume.clear()
            self.ax_volume.bar(dates, volumes, color='#4285f4', alpha=0.7, width=1)
            self.ax_volume.set_ylabel('거래량', color='white')
            self.ax_volume.set_xlabel('날짜', color='white')
            self.ax_volume.grid(True, alpha=0.3, color='#666666')
            
            # x축 레이블 회전
            self.figure.autofmt_xdate()
            
            # 레이아웃 조정
            self.figure.tight_layout()
            
            # 차트 업데이트
            if self.canvas:
                self.canvas.draw()
                
            logger.info("샘플 차트 그리기 완료")
            
        except Exception as e:
            logger.error(f"샘플 차트 그리기 실패: {e}")
    
    def update_chart(self, stock_data: Dict[str, Any]):
        """차트 업데이트"""
        try:
            if not self._initialized or not self.ax_price:
                logger.warning("차트 매니저가 초기화되지 않음")
                return
            
            # 실제 주식 데이터로 차트 업데이트
            # 현재는 샘플 데이터 사용
            self._draw_sample_chart()
            
            logger.info(f"차트 업데이트 완료: {stock_data.get('name', 'Unknown')}")
            
        except Exception as e:
            logger.error(f"차트 업데이트 실패: {e}")
    
    def draw_candlestick_chart(self, df: pd.DataFrame, title: str = "주식 차트"):
        """캔들스틱 차트 그리기"""
        try:
            if not self._initialized:
                logger.warning("차트 매니저가 초기화되지 않음")
                return
            
            # 캔들스틱 차트 구현
            # 현재는 라인 차트로 대체
            self.ax_price.clear()
            
            if 'close' in df.columns:
                self.ax_price.plot(df.index, df['close'], color='#00d4aa', linewidth=1.5)
            
            self.ax_price.set_title(title, color='white', fontsize=14, pad=20)
            self.ax_price.set_ylabel('가격 (원)', color='white')
            self.ax_price.grid(True, alpha=0.3, color='#666666')
            
            if 'volume' in df.columns:
                self.ax_volume.clear()
                self.ax_volume.bar(df.index, df['volume'], color='#4285f4', alpha=0.7)
                self.ax_volume.set_ylabel('거래량', color='white')
                self.ax_volume.set_xlabel('날짜', color='white')
                self.ax_volume.grid(True, alpha=0.3, color='#666666')
            
            self.figure.tight_layout()
            
            if self.canvas:
                self.canvas.draw()
                
            logger.info(f"캔들스틱 차트 그리기 완료: {title}")
            
        except Exception as e:
            logger.error(f"캔들스틱 차트 그리기 실패: {e}")
    
    def add_technical_indicators(self, df: pd.DataFrame):
        """기술적 지표 추가"""
        try:
            if not self._initialized or df.empty:
                return
            
            # 이동평균선 추가
            if 'close' in df.columns:
                ma5 = df['close'].rolling(window=5).mean()
                ma20 = df['close'].rolling(window=20).mean()
                
                self.ax_price.plot(df.index, ma5, color='#ff6b6b', linewidth=1, alpha=0.8, label='MA5')
                self.ax_price.plot(df.index, ma20, color='#4ecdc4', linewidth=1, alpha=0.8, label='MA20')
                
                self.ax_price.legend(loc='upper left')
            
            if self.canvas:
                self.canvas.draw()
                
            logger.info("기술적 지표 추가 완료")
            
        except Exception as e:
            logger.error(f"기술적 지표 추가 실패: {e}")
    
    def clear_chart(self):
        """차트 초기화"""
        try:
            if self.ax_price:
                self.ax_price.clear()
            if self.ax_volume:
                self.ax_volume.clear()
            if self.canvas:
                self.canvas.draw()
                
            logger.info("차트 초기화 완료")
            
        except Exception as e:
            logger.error(f"차트 초기화 실패: {e}")
    
    async def cleanup(self):
        """차트 매니저 정리"""
        try:
            if self.figure:
                plt.close(self.figure)
            
            self.figure = None
            self.canvas = None
            self.ax_price = None
            self.ax_volume = None
            self._initialized = False
            
            logger.info("차트 매니저 정리 완료")
            
        except Exception as e:
            logger.error(f"차트 매니저 정리 실패: {e}") 