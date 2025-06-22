#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Ultra HTS GUI v5.0 - 고성능 실시간 주식 거래 시스템
- 비동기 GUI & 멀티스레드 처리
- 실시간 데이터 스트리밍 & 차트
- AI 투자 전략 분석 & 추천
"""

import asyncio
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
import structlog

# 내부 모듈
from ui_interfaces.data_manager import get_data_manager, initialize_data_manager
from ui_interfaces.ai_manager import get_ai_manager, initialize_ai_manager
from ui_interfaces.chart_manager import get_chart_manager, ChartType, TimeFrame, create_chart_config, create_chart_data
from core.cache_manager import get_cache_manager
from core.performance_monitor import monitor_performance
from config.settings import settings

logger = structlog.get_logger(__name__)


class UltraHTSGUI:
    """🚀 Ultra HTS GUI - 고성능 실시간 주식 거래 시스템"""
    
    def __init__(self):
        # 메인 윈도우
        self.root = tk.Tk()
        self.root.title("Ultra HTS v5.0 - 고성능 실시간 주식 거래 시스템")
        self.root.geometry("1400x900")
        self.root.configure(bg='#0A0A0A')
        
        # 매니저들
        self.data_manager = None
        self.ai_manager = None
        self.chart_manager = None
        self.cache_manager = None
        
        # GUI 컴포넌트
        self.widgets = {}
        self.charts = {}
        
        # 상태 관리
        self.is_running = False
        self.update_tasks = {}
        
        # 스타일 설정
        self._setup_styles()
        
        # GUI 초기화
        self._create_widgets()
        
        # 비동기 초기화
        self._initialize_async()
        
        logger.info("Ultra HTS GUI 초기화 완료")
    
    def _setup_styles(self):
        """GUI 스타일 설정"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # 다크 테마 색상
        colors = {
            'bg': '#0A0A0A',
            'fg': '#FFFFFF',
            'select_bg': '#00D4FF',
            'select_fg': '#000000',
            'button_bg': '#1A1A1A',
            'entry_bg': '#2A2A2A'
        }
        
        # 스타일 구성
        style.configure('Dark.TFrame', background=colors['bg'])
        style.configure('Dark.TLabel', background=colors['bg'], foreground=colors['fg'])
        style.configure('Dark.TButton', background=colors['button_bg'], foreground=colors['fg'])
        style.configure('Dark.TEntry', fieldbackground=colors['entry_bg'], foreground=colors['fg'])
        style.configure('Dark.Treeview', background=colors['entry_bg'], foreground=colors['fg'])
    
    def _create_widgets(self):
        """GUI 위젯 생성"""
        # 메인 프레임
        main_frame = ttk.Frame(self.root, style='Dark.TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 상단 프레임 (컨트롤)
        top_frame = ttk.Frame(main_frame, style='Dark.TFrame')
        top_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 지수 선택
        ttk.Label(top_frame, text="지수 선택:", style='Dark.TLabel').pack(side=tk.LEFT, padx=(0, 5))
        
        self.index_combo = ttk.Combobox(top_frame, values=["KOSPI 200", "NASDAQ-100", "S&P 500"])
        self.index_combo.set("KOSPI 200")
        self.index_combo.pack(side=tk.LEFT, padx=(0, 10))
        
        # 버튼들
        ttk.Button(top_frame, text="데이터 조회", command=self.load_data, style='Dark.TButton').pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(top_frame, text="AI 분석", command=self.analyze_stocks, style='Dark.TButton').pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(top_frame, text="차트 보기", command=self.show_chart, style='Dark.TButton').pack(side=tk.LEFT, padx=(0, 5))
        
        # 중앙 프레임 (데이터 테이블과 차트)
        center_frame = ttk.Frame(main_frame, style='Dark.TFrame')
        center_frame.pack(fill=tk.BOTH, expand=True)
        
        # 좌측 프레임 (데이터 테이블)
        left_frame = ttk.Frame(center_frame, style='Dark.TFrame')
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # 데이터 테이블
        columns = ("종목명", "종목코드", "현재가", "등락률", "거래량", "시가총액")
        self.tree = ttk.Treeview(left_frame, columns=columns, show='headings', style='Dark.Treeview')
        
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100, anchor=tk.CENTER)
        
        # 스크롤바
        scrollbar = ttk.Scrollbar(left_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 우측 프레임 (차트)
        right_frame = ttk.Frame(center_frame, style='Dark.TFrame')
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # 차트 프레임
        self.chart_frame = ttk.Frame(right_frame, style='Dark.TFrame')
        self.chart_frame.pack(fill=tk.BOTH, expand=True)
        
        # 하단 프레임 (AI 분석 결과)
        bottom_frame = ttk.Frame(main_frame, style='Dark.TFrame')
        bottom_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Label(bottom_frame, text="AI 분석 결과:", style='Dark.TLabel').pack(anchor=tk.W)
        
        # AI 분석 결과 텍스트
        self.analysis_text = tk.Text(bottom_frame, height=8, bg='#2A2A2A', fg='#FFFFFF', 
                                   wrap=tk.WORD, font=('Consolas', 10))
        self.analysis_text.pack(fill=tk.X, pady=(5, 0))
        
        # 상태바
        self.status_var = tk.StringVar(value="준비")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, style='Dark.TLabel')
        status_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 0))
    
    def _initialize_async(self):
        """비동기 초기화"""
        def init_async():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._async_init())
            loop.close()
        
        init_thread = threading.Thread(target=init_async, daemon=True)
        init_thread.start()
    
    async def _async_init(self):
        """비동기 초기화 실행"""
        try:
            self.status_var.set("시스템 초기화 중...")
            
            # 매니저들 초기화
            await initialize_data_manager()
            self.data_manager = get_data_manager()
            
            await initialize_ai_manager()
            self.ai_manager = get_ai_manager()
            
            self.chart_manager = get_chart_manager()
            self.cache_manager = get_cache_manager()
            
            self.status_var.set("초기화 완료 - 준비")
            self.is_running = True
            
            # 초기 데이터 로드
            await self._load_initial_data()
            
        except Exception as e:
            logger.error(f"비동기 초기화 실패: {e}")
            self.status_var.set(f"초기화 실패: {e}")
    
    async def _load_initial_data(self):
        """초기 데이터 로드"""
        try:
            # KOSPI 200 데이터 로드
            stocks = await self.data_manager.get_stocks_by_index("KOSPI 200")
            
            # GUI 업데이트 (메인 스레드에서)
            self.root.after(0, lambda: self._update_stock_table(stocks))
            
        except Exception as e:
            logger.error(f"초기 데이터 로드 실패: {e}")
    
    def _update_stock_table(self, stocks: List[Dict[str, Any]]):
        """주식 테이블 업데이트"""
        try:
            # 기존 데이터 삭제
            for item in self.tree.get_children():
                self.tree.delete(item)
            
            # 새 데이터 추가
            for stock in stocks[:20]:  # 상위 20개만 표시
                values = (
                    stock.get('name', ''),
                    stock.get('code', ''),
                    f"{stock.get('price', 0):,.0f}",
                    f"{stock.get('change_rate', 0):+.2f}%",
                    f"{stock.get('volume', 0):,}",
                    f"{stock.get('market_cap', 0):,.0f}억"
                )
                
                # 색상 설정 (등락률에 따라)
                change_rate = stock.get('change_rate', 0)
                if change_rate > 0:
                    tags = ('positive',)
                elif change_rate < 0:
                    tags = ('negative',)
                else:
                    tags = ('neutral',)
                
                self.tree.insert('', tk.END, values=values, tags=tags)
            
            # 태그 색상 설정
            self.tree.tag_configure('positive', foreground='#4CAF50')
            self.tree.tag_configure('negative', foreground='#F44336')
            self.tree.tag_configure('neutral', foreground='#FFFFFF')
            
        except Exception as e:
            logger.error(f"테이블 업데이트 실패: {e}")
    
    def load_data(self):
        """데이터 로드 (버튼 클릭)"""
        def load_async():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._load_data_async())
            loop.close()
        
        self.status_var.set("데이터 로드 중...")
        threading.Thread(target=load_async, daemon=True).start()
    
    async def _load_data_async(self):
        """비동기 데이터 로드"""
        try:
            index_name = self.index_combo.get()
            stocks = await self.data_manager.get_stocks_by_index(index_name)
            
            # GUI 업데이트
            self.root.after(0, lambda: self._update_stock_table(stocks))
            self.root.after(0, lambda: self.status_var.set(f"{index_name} 데이터 로드 완료"))
            
        except Exception as e:
            logger.error(f"데이터 로드 실패: {e}")
            self.root.after(0, lambda: self.status_var.set(f"데이터 로드 실패: {e}"))
    
    def analyze_stocks(self):
        """AI 주식 분석 (버튼 클릭)"""
        def analyze_async():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._analyze_stocks_async())
            loop.close()
        
        self.status_var.set("AI 분석 중...")
        threading.Thread(target=analyze_async, daemon=True).start()
    
    async def _analyze_stocks_async(self):
        """비동기 AI 분석"""
        try:
            index_name = self.index_combo.get()
            
            # AI 분석 실행
            analysis_result = await self.ai_manager.analyze_market_ultra(
                market_type=index_name,
                analysis_type="comprehensive"
            )
            
            # 결과 포맷팅
            formatted_result = self._format_analysis_result(analysis_result)
            
            # GUI 업데이트
            self.root.after(0, lambda: self._update_analysis_text(formatted_result))
            self.root.after(0, lambda: self.status_var.set("AI 분석 완료"))
            
        except Exception as e:
            logger.error(f"AI 분석 실패: {e}")
            self.root.after(0, lambda: self.status_var.set(f"AI 분석 실패: {e}"))
    
    def _format_analysis_result(self, result: Dict[str, Any]) -> str:
        """분석 결과 포맷팅"""
        try:
            formatted = "🚀 Ultra AI 분석 결과\n"
            formatted += "=" * 50 + "\n\n"
            
            # TOP 5 추천 종목
            if 'recommendations' in result:
                formatted += "📈 TOP 5 추천 종목:\n"
                for i, stock in enumerate(result['recommendations'][:5], 1):
                    formatted += f"{i}. {stock.get('name', 'Unknown')} ({stock.get('code', 'N/A')})\n"
                    formatted += f"   점수: {stock.get('score', 0):.2f} | 신뢰도: {stock.get('confidence', 0):.1f}%\n"
                    formatted += f"   추천 이유: {stock.get('reason', '분석 중...')}\n\n"
            
            # 시장 요약
            if 'market_summary' in result:
                summary = result['market_summary']
                formatted += "📊 시장 요약:\n"
                formatted += f"• 전체 종목 수: {summary.get('total_stocks', 0)}개\n"
                formatted += f"• 평균 등락률: {summary.get('avg_change_rate', 0):+.2f}%\n"
                formatted += f"• 상승 종목: {summary.get('gainers', 0)}개\n"
                formatted += f"• 하락 종목: {summary.get('losers', 0)}개\n\n"
            
            # 투자 전략별 분석
            if 'strategy_analysis' in result:
                formatted += "🎯 투자 전략별 분석:\n"
                for strategy, analysis in result['strategy_analysis'].items():
                    formatted += f"• {strategy}: {analysis.get('summary', '분석 중...')}\n"
            
            formatted += "\n" + "=" * 50
            formatted += f"\n분석 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            return formatted
            
        except Exception as e:
            logger.error(f"결과 포맷팅 실패: {e}")
            return f"분석 결과 포맷팅 오류: {e}"
    
    def _update_analysis_text(self, text: str):
        """분석 결과 텍스트 업데이트"""
        try:
            self.analysis_text.delete(1.0, tk.END)
            self.analysis_text.insert(1.0, text)
        except Exception as e:
            logger.error(f"분석 텍스트 업데이트 실패: {e}")
    
    def show_chart(self):
        """차트 표시"""
        def show_chart_async():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._show_chart_async())
            loop.close()
        
        threading.Thread(target=show_chart_async, daemon=True).start()
    
    async def _show_chart_async(self):
        """비동기 차트 표시"""
        try:
            # 선택된 종목 가져오기
            selection = self.tree.selection()
            if not selection:
                self.root.after(0, lambda: messagebox.showwarning("경고", "종목을 선택해주세요."))
                return
            
            # 선택된 종목 정보
            item = self.tree.item(selection[0])
            stock_name = item['values'][0]
            stock_code = item['values'][1]
            
            # 차트 데이터 준비
            index_name = self.index_combo.get()
            stocks = await self.data_manager.get_stocks_by_index(index_name)
            
            # 해당 종목 찾기
            target_stock = None
            for stock in stocks:
                if stock.get('code') == stock_code:
                    target_stock = stock
                    break
            
            if target_stock:
                # 차트 설정
                config = create_chart_config(
                    chart_type=ChartType.LINE,
                    time_frame=TimeFrame.DAILY,
                    width=600,
                    height=400
                )
                
                # 차트 데이터
                chart_data = create_chart_data(
                    symbol=f"{stock_name}({stock_code})",
                    data=[target_stock]
                )
                
                # 차트 생성
                canvas = self.chart_manager.create_tkinter_chart(
                    self.chart_frame, config, chart_data
                )
                
                if canvas:
                    # 기존 차트 제거
                    for widget in self.chart_frame.winfo_children():
                        widget.destroy()
                    
                    # 새 차트 표시
                    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                    self.root.after(0, lambda: self.status_var.set(f"{stock_name} 차트 표시 완료"))
                else:
                    self.root.after(0, lambda: self.status_var.set("차트 생성 실패"))
            
        except Exception as e:
            logger.error(f"차트 표시 실패: {e}")
            self.root.after(0, lambda: self.status_var.set(f"차트 표시 실패: {e}"))
    
    def run(self):
        """GUI 실행"""
        try:
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
            self.root.mainloop()
        except Exception as e:
            logger.error(f"GUI 실행 실패: {e}")
    
    def on_closing(self):
        """종료 처리"""
        try:
            self.is_running = False
            
            # 정리 작업
            if self.chart_manager:
                self.chart_manager.cleanup()
            
            self.root.destroy()
            
        except Exception as e:
            logger.error(f"종료 처리 실패: {e}")


def main():
    """메인 함수"""
    try:
        # 로거 설정
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        # GUI 실행
        app = UltraHTSGUI()
        app.run()
        
    except Exception as e:
        print(f"애플리케이션 실행 실패: {e}")


if __name__ == "__main__":
    main() 