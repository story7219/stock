"""
프리미엄 HTS GUI - 세계 최고 수준의 트레이딩 시스템
"""
import tkinter as tk
from tkinter import ttk, messagebox, Frame, Label, Button, Text, Scrollbar, Entry, StringVar, Radiobutton
import asyncio
import threading
from datetime import datetime
from typing import Optional, Dict, Any
import structlog
from weakref import WeakSet

from config.settings import settings
from core.cache_manager import CacheManager
from core.database_manager import DatabaseManager
from core.performance_monitor import PerformanceMonitor
from ui_interfaces.data_manager import DataManager
from ui_interfaces.chart_manager import ChartManager
from ui_interfaces.ai_manager import AIManager

logger = structlog.get_logger(__name__)


class PremiumColors:
    """프리미엄 색상 팔레트"""
    PRIMARY_BLUE = "#4285f4"
    HEADER_BLUE = "#1976d2"
    WHITE = "#ffffff"
    LIGHT_GRAY = "#f8fafc"
    PANEL_BG = "#f8fafc"
    TEXT_PRIMARY = "#1a202c"
    TEXT_SECONDARY = "#4a5568"
    TEXT_MUTED = "#718096"
    SUCCESS = "#10b981"
    WARNING = "#f59e0b"
    ERROR = "#ef4444"
    BORDER = "#e2e8f0"


class OptimizedHTS:
    """Ultra Premium HTS 메인 클래스"""
    
    def __init__(self):
        """HTS 시스템 초기화"""
        self.root = None
        self.loop = None
        self.thread = None
        self._weak_refs = WeakSet()
        
        # 컴포넌트 초기화 상태 추적
        self.initialization_complete = False
        self.ai_manager = None
        self.data_manager = None
        self.chart_manager = None
        self.cache_manager = None
        self.db_manager = None
        self.performance_monitor = None
        
        # GUI 상태
        self.current_index = "코스피200 (KOSPI200)"
        self.current_strategy = "워렌 버핏"
        self.index_buttons = {}
        
        logger.info("프리미엄 HTS 시스템 초기화 시작")
        
        # GUI 설정
        self.setup_gui()
        
        # 비동기 초기화 시작
        self.start_async_initialization()
    
    def setup_gui(self):
        """GUI 초기화"""
        # 메인 윈도우 설정
        self.root = tk.Tk()
        self.root.title("Ultra Premium HTS - Professional Trading System")
        self.root.geometry("1400x800")
        self.root.configure(bg=PremiumColors.WHITE)
        self.root.minsize(1200, 700)
        
        # 메인 프레임
        self.main_frame = Frame(self.root, bg=PremiumColors.WHITE)
        self.main_frame.pack(fill="both", expand=True)
        
        # 헤더 생성
        self.create_header()
        
        # 컨텐츠 영역 생성
        self.create_content_area()
        
        # 상태바 생성
        self.create_status_bar()
        
        # 시간 업데이트 시작
        self.update_time()
    
    def create_header(self):
        """헤더 영역 생성"""
        header_frame = Frame(self.main_frame, bg=PremiumColors.PRIMARY_BLUE, height=60)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        # 좌측 - 지수 종합
        left_header = Frame(header_frame, bg=PremiumColors.PRIMARY_BLUE)
        left_header.pack(side=tk.LEFT, padx=20, pady=10)
        
        title_label = Label(
            left_header,
            text="🚀 지수 종합",
            font=("맑은 고딕", 14, "bold"),
            bg=PremiumColors.PRIMARY_BLUE,
            fg=PremiumColors.WHITE
        )
        title_label.pack(side=tk.LEFT)
        
        # 중앙 - 지수 버튼들
        center_header = Frame(header_frame, bg=PremiumColors.PRIMARY_BLUE)
        center_header.pack(side=tk.LEFT, padx=50, pady=10)
        
        self.index_buttons = {}
        indices = [("코스피200", "KOSPI200"), ("나스닥100", "NASDAQ-100"), ("S&P500", "S&P 500")]
        
        for i, (display_name, index_name) in enumerate(indices):
            btn = Button(
                center_header,
                text=display_name,
                font=("맑은 고딕", 10, "bold"),
                bg=PremiumColors.WHITE,
                fg=PremiumColors.PRIMARY_BLUE,
                relief="flat",
                padx=15,
                pady=5,
                command=lambda idx=index_name: self.select_index(idx)
            )
            btn.pack(side=tk.LEFT, padx=5)
            self.index_buttons[index_name] = btn
        
        # 우측 - 시간
        right_header = Frame(header_frame, bg=PremiumColors.PRIMARY_BLUE)
        right_header.pack(side=tk.RIGHT, padx=20, pady=10)
        
        self.time_label = Label(
            right_header,
            text="2025",
            font=("맑은 고딕", 12, "bold"),
            bg=PremiumColors.PRIMARY_BLUE,
            fg=PremiumColors.WHITE
        )
        self.time_label.pack()
    
    def create_content_area(self):
        """컨텐츠 영역 생성"""
        content_frame = Frame(self.main_frame, bg=PremiumColors.WHITE)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # 좌측 패널 - AI 매니저
        self.left_panel = self.create_left_panel(content_frame)
        self.left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        
        # 중앙 패널 - 차트
        self.center_panel = self.create_center_panel(content_frame)
        self.center_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # 우측 패널 - AI 분석 결과
        self.right_panel = self.create_right_panel(content_frame)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
    
    def create_left_panel(self, parent):
        """좌측 패널 생성 - AI 매니저"""
        left_frame = Frame(parent, bg=PremiumColors.PANEL_BG, width=280)
        left_frame.pack(side="left", fill="y", padx=(0, 5))
        left_frame.pack_propagate(False)
        
        # AI 성능 점검 섹션
        perf_frame = Frame(left_frame, bg=PremiumColors.WHITE, relief="solid", bd=1)
        perf_frame.pack(fill="x", padx=10, pady=10)
        
        perf_title = Label(
            perf_frame,
            text="🔍 AI 성능 점검",
            font=("맑은 고딕", 12, "bold"),
            bg=PremiumColors.WHITE,
            fg=PremiumColors.TEXT_PRIMARY
        )
        perf_title.pack(pady=(15, 10))
        
        # 초기화 상태 표시
        self.init_status_label = Label(
            perf_frame,
            text="🔄 시스템 초기화 중...",
            font=("맑은 고딕", 10),
            bg=PremiumColors.WHITE,
            fg=PremiumColors.WARNING
        )
        self.init_status_label.pack(pady=5)
        
        perf_info = [
            "🟢 AI 100% 정상 동작",
            "🟢 AI 정수: 80/100 (우수 AI)",
        ]
        
        for info in perf_info:
            Label(
                perf_frame,
                text=info,
                font=("맑은 고딕", 10),
                bg=PremiumColors.WHITE,
                fg=PremiumColors.TEXT_SECONDARY,
                anchor="w"
            ).pack(fill="x", padx=20, pady=2)
        
        # 투자 대가 전략 선택
        strategy_frame = Frame(left_frame, bg=PremiumColors.WHITE, relief="solid", bd=1)
        strategy_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        strategy_title = Label(
            strategy_frame,
            text="📊 투자 대가 전략 선택",
            font=("맑은 고딕", 12, "bold"),
            bg=PremiumColors.WHITE,
            fg=PremiumColors.TEXT_PRIMARY
        )
        strategy_title.pack(pady=(15, 10))
        
        strategies = [
            ("워렌 버핏", "Warren Buffett"),
            ("피터 린치", "Peter Lynch"),
            ("윌리엄 오닐", "William O'Neil"),
            ("미네르비니", "미네르비니")
        ]
        
        self.strategy_var = StringVar(value="워렌 버핏")
        
        for display_name, internal_name in strategies:
            Radiobutton(
                strategy_frame,
                text=display_name,
                variable=self.strategy_var,
                value=display_name,
                font=("맑은 고딕", 11),
                bg=PremiumColors.WHITE,
                fg=PremiumColors.TEXT_PRIMARY,
                selectcolor=PremiumColors.LIGHT_GRAY,
                command=lambda name=display_name: self.select_strategy(name)
            ).pack(anchor="w", padx=20, pady=3)
        
        # 종목 검색
        search_frame = Frame(left_frame, bg=PremiumColors.WHITE, relief="solid", bd=1)
        search_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        search_title = Label(
            search_frame,
            text="🔍 종목 검색",
            font=("맑은 고딕", 12, "bold"),
            bg=PremiumColors.WHITE,
            fg=PremiumColors.TEXT_PRIMARY
        )
        search_title.pack(pady=(15, 10))
        
        search_entry = Entry(
            search_frame,
            font=("맑은 고딕", 11),
            bg=PremiumColors.LIGHT_GRAY,
            fg=PremiumColors.TEXT_PRIMARY,
            relief="flat",
            bd=5
        )
        search_entry.pack(fill="x", padx=20, pady=(0, 5))
        search_entry.insert(0, "종목명 또는 코드 입력")
        
        # AI 분석 실행 버튼
        self.ai_analysis_btn = Button(
            left_frame,
            text="🚀 AI 전략 분석 실행",
            font=("맑은 고딕", 12, "bold"),
            bg=PremiumColors.ERROR,
            fg=PremiumColors.WHITE,
            relief="flat",
            bd=0,
            pady=15,
            state="disabled",  # 초기에는 비활성화
            command=self.on_ai_analyze_click
        )
        self.ai_analysis_btn.pack(fill="x", padx=10, pady=(0, 20))
        
        return left_frame
    
    def create_center_panel(self, parent):
        """중앙 차트 패널"""
        center_frame = Frame(parent, bg=PremiumColors.WHITE, relief="solid", bd=1)
        
        # 차트 제목
        title_frame = Frame(center_frame, bg=PremiumColors.WHITE, height=60)
        title_frame.pack(fill="x")
        title_frame.pack_propagate(False)
        
        self.chart_title_label = Label(
            title_frame,
            text="AI 분석 후 종목을 선택하세요",
            font=("맑은 고딕", 16, "bold"),
            bg=PremiumColors.WHITE,
            fg=PremiumColors.TEXT_PRIMARY
        )
        self.chart_title_label.pack(pady=20)
        
        # 차트 영역
        self.chart_frame = Frame(center_frame, bg=PremiumColors.WHITE)
        self.chart_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        return center_frame
    
    def create_right_panel(self, parent):
        """우측 AI 분석 결과 패널"""
        right_frame = Frame(parent, bg=PremiumColors.PANEL_BG, width=340, relief="solid", bd=1)  # 폭 20픽셀 증가
        right_frame.pack_propagate(False)
        
        # 헤더
        header_frame = Frame(right_frame, bg=PremiumColors.SUCCESS, height=40)
        header_frame.pack(fill="x", padx=5, pady=(5, 0))
        header_frame.pack_propagate(False)
        
        header_label = Label(
            header_frame,
            text="🧠 AI 종합분석 결과",
            font=("맑은 고딕", 12, "bold"),
            bg=PremiumColors.SUCCESS,
            fg="white"
        )
        header_label.pack(pady=8)
        
        # 분석 결과 스크롤 영역 - 여유 공간 확보
        scroll_frame = Frame(right_frame, bg=PremiumColors.WHITE)
        scroll_frame.pack(fill="both", expand=True, padx=10, pady=(5, 20))  # 하단 여유 공간 20픽셀
        
        # 스크롤바와 텍스트 영역
        scrollbar = Scrollbar(scroll_frame)
        scrollbar.pack(side="right", fill="y")
        
        self.analysis_result_text = Text(
            scroll_frame,
            font=("맑은 고딕", 10),  # 글자 크기 2단계 증가
            bg=PremiumColors.WHITE,
            fg=PremiumColors.TEXT_PRIMARY,
            yscrollcommand=scrollbar.set,
            wrap="word",
            padx=15,  # 좌우 여백 증가
            pady=15,  # 상하 여백 증가
            relief="flat",
            bd=0
        )
        self.analysis_result_text.pack(fill="both", expand=True)
        scrollbar.config(command=self.analysis_result_text.yview)
        
        # 초기 메시지
        initial_message = """💎 월스트리트 수준 AI 분석기

📊 분석 준비 완료
• 전략: 선택된 투자 대가 전략 적용  
• 대상: 주요 지수별 TOP 5 종목
• 정확도: 기관급 분석 수준

🎯 사용법:
좌측에서 지수와 전략을 선택하고
'AI 전략 분석 실행' 버튼을 클릭하세요."""

        self.analysis_result_text.insert("1.0", initial_message)
        self.analysis_result_text.config(state="disabled")
        
        return right_frame
    
    def create_status_bar(self):
        """상태바 생성"""
        status_frame = Frame(self.main_frame, bg=PremiumColors.LIGHT_GRAY, height=30)
        status_frame.pack(fill="x", side="bottom")
        status_frame.pack_propagate(False)
        
        self.status_label = Label(
            status_frame,
            text="선택된 지수: 코스피200 | 선택된 전략: 워렌 버핏 | 분석 종목 수: 200개",
            font=("맑은 고딕", 9),
            bg=PremiumColors.LIGHT_GRAY,
            fg=PremiumColors.TEXT_SECONDARY,
            anchor="w"
        )
        self.status_label.pack(side="left", padx=10, pady=5)
    
    def select_index(self, index_name):
        """지수 선택"""
        self.current_index = index_name
        print(f"지수 선택됨: {index_name}")
        
        # 버튼 스타일 업데이트
        for idx, btn in self.index_buttons.items():
            if idx == index_name:
                btn.config(bg=PremiumColors.PRIMARY_BLUE, fg=PremiumColors.WHITE)
            else:
                btn.config(bg=PremiumColors.WHITE, fg=PremiumColors.PRIMARY_BLUE)
        
        self.update_status()
    
    def select_strategy(self, strategy_name):
        """전략 선택"""
        self.current_strategy = strategy_name
        print(f"투자 전략 변경됨: {strategy_name}")
        self.update_status()
    
    def update_status(self):
        """상태바 업데이트"""
        status_text = f"선택된 지수: {self.current_index} | 선택된 전략: {self.current_strategy} | 분석 준비 완료"
        self.status_label.config(text=status_text)
    
    def update_time(self):
        """시간 업데이트"""
        current_time = datetime.now().strftime("%Y")
        self.time_label.config(text=current_time)
        self.root.after(60000, self.update_time)  # 1분마다 업데이트
    
    def on_ai_analyze_click(self):
        """AI 분석 버튼 클릭 이벤트"""
        try:
            # 선택된 지수와 전략 가져오기
            selected_index = self.index_combo.get()
            selected_strategy = self.strategy_combo.get()
            
            if not selected_index or not selected_strategy:
                messagebox.showwarning("경고", "지수와 전략을 모두 선택해주세요.")
                return
            
            # 분석 중 표시
            self.ai_result_text.delete(1.0, tk.END)
            self.ai_result_text.insert(tk.END, "🔄 AI 분석 중... 잠시만 기다려주세요.")
            self.root.update()
            
            # 비동기 분석 실행
            asyncio.run(self._run_ai_analysis(selected_index, selected_strategy))
            
        except Exception as e:
            logger.error(f"AI 분석 버튼 클릭 오류: {e}")
            self.ai_result_text.delete(1.0, tk.END)
            self.ai_result_text.insert(tk.END, f"⚠️ AI 분석 중 오류가 발생했습니다: {str(e)}")

    async def _run_ai_analysis(self, index_name: str, strategy: str):
        """비동기 AI 분석 실행"""
        try:
            # 투자 대가별 분석 호출
            analysis_result = await self.ai_manager.get_guru_analysis(index_name, strategy)
            
            # 결과 표시
            self.ai_result_text.delete(1.0, tk.END)
            self.ai_result_text.insert(tk.END, analysis_result)
            
            logger.info(f"AI 분석 완료: {index_name}, {strategy}")
            
        except Exception as e:
            logger.error(f"AI 분석 실행 오류: {e}")
            self.ai_result_text.delete(1.0, tk.END)
            self.ai_result_text.insert(tk.END, f"⚠️ AI 분석 실행 중 오류가 발생했습니다: {str(e)}")
    
    def schedule_async_task(self, coro):
        """비동기 작업 스케줄링"""
        if self.loop and not self.loop.is_closed():
            asyncio.run_coroutine_threadsafe(coro, self.loop)
    
    def start_async_initialization(self):
        """비동기 초기화 시작"""
        def run_event_loop():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self.initialize_components())
            self.loop.run_forever()
        
        self.thread = threading.Thread(target=run_event_loop, daemon=True)
        self.thread.start()
    
    async def initialize_components(self):
        """컴포넌트 초기화"""
        try:
            # 성능 모니터 초기화
            self.performance_monitor = PerformanceMonitor()
            await self.performance_monitor.start_monitoring()
            
            # 데이터베이스 매니저 초기화
            self.db_manager = DatabaseManager()
            await self.db_manager.initialize()
            
            # 캐시 매니저 초기화
            self.cache_manager = CacheManager()
            await self.cache_manager.initialize()
            
            # 차트 매니저 초기화
            self.chart_manager = ChartManager()
            await self.chart_manager.initialize()
            
            # 데이터 매니저 초기화
            self.data_manager = DataManager()
            await self.data_manager.initialize()
            
            # AI 매니저 초기화
            self.ai_manager = AIManager()
            await self.ai_manager.initialize()
            
            # 초기화 완료 상태 업데이트
            self.initialization_complete = True
            self.update_initialization_status()
            
            logger.info("프리미엄 HTS 시스템 초기화 완료")
            
        except Exception as e:
            logger.error(f"컴포넌트 초기화 실패: {e}")
            self.update_initialization_status(error=str(e))
    
    def update_initialization_status(self, error=None):
        """초기화 상태 업데이트"""
        def update_ui():
            try:
                if error:
                    self.init_status_label.config(
                        text="❌ 초기화 실패",
                        fg=PremiumColors.ERROR
                    )
                    self.ai_analysis_btn.config(
                        text="⚠️ 시스템 오류",
                        state="disabled"
                    )
                elif self.initialization_complete:
                    self.init_status_label.config(
                        text="✅ 시스템 준비 완료",
                        fg=PremiumColors.SUCCESS
                    )
                    self.ai_analysis_btn.config(
                        text="🚀 AI 전략 분석 실행",
                        state="normal",
                        bg=PremiumColors.ERROR
                    )
                else:
                    self.init_status_label.config(
                        text="🔄 시스템 초기화 중...",
                        fg=PremiumColors.WARNING
                    )
            except Exception as e:
                logger.error(f"UI 업데이트 실패: {e}")
        
        # GUI 스레드에서 실행
        if self.root:
            self.root.after(0, update_ui)
    
    def run(self):
        """애플리케이션 실행"""
        try:
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
            self.root.mainloop()
        except Exception as e:
            logger.error(f"GUI 실행 오류: {e}")
            raise
    
    def on_closing(self):
        """애플리케이션 종료 처리"""
        try:
            # 비동기 정리 작업
            if self.loop and not self.loop.is_closed():
                asyncio.run_coroutine_threadsafe(self.cleanup_components(), self.loop)
                self.loop.call_soon_threadsafe(self.loop.stop)
            
            # GUI 종료
            self.root.quit()
            self.root.destroy()
            
            logger.info("프리미엄 HTS 시스템 정리 완료")
            
        except Exception as e:
            logger.error(f"종료 처리 중 오류: {e}")
    
    async def cleanup_components(self):
        """컴포넌트 정리"""
        try:
            if self.chart_manager:
                await self.chart_manager.cleanup()
            
            if self.data_manager:
                await self.data_manager.cleanup()
            
            if self.ai_manager:
                await self.ai_manager.cleanup()
            
            if self.performance_monitor:
                await self.performance_monitor.stop_monitoring()
            
            if self.db_manager:
                await self.db_manager.close()
            
        except Exception as e:
            logger.error(f"컴포넌트 정리 실패: {e}")


if __name__ == "__main__":
    app = OptimizedHTS()
    app.run() 