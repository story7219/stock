"""
HTS 스타일 주식 분석 GUI
증권사 HTS처럼 전문적인 인터페이스 구현
"""

import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
from datetime import datetime
import threading
from mid_term import MidTermInvestmentAnalyzer

class HTSStyleGUI:
    """HTS 스타일 주식 분석 GUI"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.analyzer = MidTermInvestmentAnalyzer()
        self.data = None
        self.recommendations = None
        
        # HTS 스타일 색상 설정
        self.colors = {
            'bg': '#1e1e1e',           # 어두운 배경
            'panel': '#2d2d2d',        # 패널 배경
            'text': '#ffffff',         # 텍스트
            'red': '#ff4444',          # 상승/매수
            'blue': '#4488ff',         # 하락/매도
            'green': '#44ff88',        # 중립
            'yellow': '#ffff44',       # 강조
            'border': '#404040'        # 테두리
        }
        
        self.setup_gui()
        
    def setup_gui(self):
        """GUI 초기 설정"""
        self.root.title("📊 중기투자 퀀트 분석 시스템 (HTS Style)")
        self.root.geometry("1600x1000")
        self.root.configure(bg=self.colors['bg'])
        
        # 메인 프레임 구성
        self.create_header()
        self.create_main_panels()
        self.create_status_bar()
        
    def create_header(self):
        """헤더 영역 생성"""
        header_frame = tk.Frame(self.root, bg=self.colors['panel'], height=60)
        header_frame.pack(fill='x', padx=5, pady=5)
        header_frame.pack_propagate(False)
        
        # 타이틀
        title_label = tk.Label(header_frame, text="📊 중기투자 퀀트 분석 시스템", 
                              font=('맑은 고딕', 16, 'bold'),
                              fg=self.colors['yellow'], bg=self.colors['panel'])
        title_label.pack(side='left', padx=20, pady=15)
        
        # 시간 표시
        time_label = tk.Label(header_frame, text=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                             font=('맑은 고딕', 10),
                             fg=self.colors['text'], bg=self.colors['panel'])
        time_label.pack(side='right', padx=20, pady=20)
        
        # 분석 버튼
        analyze_btn = tk.Button(header_frame, text="🚀 분석 시작", 
                               font=('맑은 고딕', 12, 'bold'),
                               bg=self.colors['red'], fg='white',
                               command=self.start_analysis,
                               width=12, height=1)
        analyze_btn.pack(side='right', padx=10, pady=15)
        
    def create_main_panels(self):
        """메인 패널들 생성"""
        main_frame = tk.Frame(self.root, bg=self.colors['bg'])
        main_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # 좌측 패널 (종목 리스트 + 차트)
        left_frame = tk.Frame(main_frame, bg=self.colors['bg'])
        left_frame.pack(side='left', fill='both', expand=True, padx=(0, 5))
        
        # 우측 패널 (상세 정보)
        right_frame = tk.Frame(main_frame, bg=self.colors['bg'], width=400)
        right_frame.pack(side='right', fill='y', padx=(5, 0))
        right_frame.pack_propagate(False)
        
        self.create_stock_list_panel(left_frame)
        self.create_chart_panel(left_frame)
        self.create_detail_panel(right_frame)
        
    def create_stock_list_panel(self, parent):
        """종목 리스트 패널 (HTS 호가창 스타일)"""
        list_frame = tk.LabelFrame(parent, text="📈 추천 종목 현황", 
                                  font=('맑은 고딕', 12, 'bold'),
                                  fg=self.colors['yellow'], bg=self.colors['panel'],
                                  height=300)
        list_frame.pack(fill='x', pady=(0, 5))
        list_frame.pack_propagate(False)
        
        # 트리뷰 생성 (HTS 스타일)
        columns = ('순위', '종목코드', '시장', '현재가', 'PER', 'ROE', '6M수익률', '점수')
        self.stock_tree = ttk.Treeview(list_frame, columns=columns, show='headings', height=12)
        
        # 컬럼 설정
        for col in columns:
            self.stock_tree.heading(col, text=col)
            if col == '종목코드':
                self.stock_tree.column(col, width=100, anchor='center')
            elif col in ['PER', 'ROE', '6M수익률', '점수']:
                self.stock_tree.column(col, width=80, anchor='center')
            else:
                self.stock_tree.column(col, width=60, anchor='center')
        
        # 스크롤바
        scrollbar = ttk.Scrollbar(list_frame, orient='vertical', command=self.stock_tree.yview)
        self.stock_tree.configure(yscrollcommand=scrollbar.set)
        
        # 패킹
        self.stock_tree.pack(side='left', fill='both', expand=True, padx=10, pady=10)
        scrollbar.pack(side='right', fill='y', pady=10)
        
        # 선택 이벤트 바인딩
        self.stock_tree.bind('<<TreeviewSelect>>', self.on_stock_select)
        
    def create_chart_panel(self, parent):
        """차트 패널"""
        chart_frame = tk.LabelFrame(parent, text="📊 분석 차트", 
                                   font=('맑은 고딕', 12, 'bold'),
                                   fg=self.colors['yellow'], bg=self.colors['panel'])
        chart_frame.pack(fill='both', expand=True)
        
        # matplotlib 차트 영역
        self.fig = Figure(figsize=(12, 6), facecolor=self.colors['bg'])
        self.canvas = FigureCanvasTkAgg(self.fig, chart_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)
        
        # 초기 차트 생성
        self.create_initial_chart()
        
    def create_detail_panel(self, parent):
        """상세 정보 패널"""
        detail_frame = tk.LabelFrame(parent, text="🔍 종목 상세 정보", 
                                    font=('맑은 고딕', 12, 'bold'),
                                    fg=self.colors['yellow'], bg=self.colors['panel'])
        detail_frame.pack(fill='both', expand=True)
        
        # 스크롤 가능한 텍스트 영역
        text_frame = tk.Frame(detail_frame, bg=self.colors['panel'])
        text_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.detail_text = tk.Text(text_frame, 
                                  font=('맑은 고딕', 10),
                                  bg=self.colors['bg'], fg=self.colors['text'],
                                  wrap='word', state='disabled')
        
        detail_scrollbar = ttk.Scrollbar(text_frame, orient='vertical', 
                                        command=self.detail_text.yview)
        self.detail_text.configure(yscrollcommand=detail_scrollbar.set)
        
        self.detail_text.pack(side='left', fill='both', expand=True)
        detail_scrollbar.pack(side='right', fill='y')
        
    def create_status_bar(self):
        """상태바 생성"""
        self.status_bar = tk.Label(self.root, text="준비", 
                                  font=('맑은 고딕', 9),
                                  fg=self.colors['text'], bg=self.colors['panel'],
                                  anchor='w', relief='sunken')
        self.status_bar.pack(side='bottom', fill='x')
        
    def create_initial_chart(self):
        """초기 차트 생성"""
        self.fig.clear()
        ax = self.fig.add_subplot(111, facecolor=self.colors['bg'])
        
        # 샘플 데이터로 차트 생성
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        ax.plot(x, y, color=self.colors['yellow'], linewidth=2)
        ax.set_title('분석을 시작하려면 "분석 시작" 버튼을 클릭하세요', 
                    color=self.colors['text'], fontsize=14)
        ax.grid(True, color=self.colors['border'], alpha=0.3)
        ax.tick_params(colors=self.colors['text'])
        
        self.canvas.draw()
        
    def start_analysis(self):
        """분석 시작 (별도 스레드에서 실행)"""
        self.update_status("🔄 분석 진행 중... 잠시만 기다려주세요")
        
        # 분석 버튼 비활성화
        for widget in self.root.winfo_children():
            if isinstance(widget, tk.Frame):
                for child in widget.winfo_children():
                    if isinstance(child, tk.Button) and "분석 시작" in child.cget('text'):
                        child.config(state='disabled', text="🔄 분석 중...")
        
        # 분석을 별도 스레드에서 실행
        analysis_thread = threading.Thread(target=self.run_analysis)
        analysis_thread.daemon = True
        analysis_thread.start()
        
    def run_analysis(self):
        """실제 분석 실행"""
        try:
            # 분석 실행
            self.recommendations = self.analyzer.run_analysis('data/stock_data.csv')
            
            if self.recommendations is not None and len(self.recommendations) > 0:
                # GUI 업데이트는 메인 스레드에서 실행
                self.root.after(0, self.update_gui_with_results)
            else:
                self.root.after(0, lambda: self.update_status("❌ 분석 결과가 없습니다."))
                
        except Exception as e:
            self.root.after(0, lambda: self.update_status(f"❌ 분석 오류: {str(e)}"))
            
    def update_gui_with_results(self):
        """분석 결과로 GUI 업데이트"""
        self.update_stock_list()
        self.update_main_chart()
        self.update_status("✅ 분석 완료! 종목을 선택하여 상세 정보를 확인하세요")
        
        # 분석 버튼 다시 활성화
        for widget in self.root.winfo_children():
            if isinstance(widget, tk.Frame):
                for child in widget.winfo_children():
                    if isinstance(child, tk.Button) and "분석 중" in child.cget('text'):
                        child.config(state='normal', text="🚀 분석 시작")
        
    def update_stock_list(self):
        """종목 리스트 업데이트"""
        # 기존 데이터 삭제
        for item in self.stock_tree.get_children():
            self.stock_tree.delete(item)
            
        # 새 데이터 추가
        for idx, (_, row) in enumerate(self.recommendations.head(20).iterrows()):
            values = (
                f"{idx+1:2d}",
                row['Ticker'],
                '🇰🇷' if row['Market'] == 'KR' else '🇺🇸',
                f"{row['Close']:,.0f}" if row['Market'] == 'KR' else f"${row['Close']:.2f}",
                f"{row['PER']:.1f}",
                f"{row['ROE']:.1f}%",
                f"{row['6M_Return_pct']:+.1f}%",
                f"{row['final_score']:.0f}"
            )
            
            # 수익률에 따른 색상 태그
            tag = 'positive' if row['6M_Return_pct'] > 0 else 'negative'
            self.stock_tree.insert('', 'end', values=values, tags=(tag,))
            
        # 태그 색상 설정
        self.stock_tree.tag_configure('positive', foreground=self.colors['red'])
        self.stock_tree.tag_configure('negative', foreground=self.colors['blue'])
        
    def update_main_chart(self):
        """메인 차트 업데이트"""
        self.fig.clear()
        
        # 2x2 서브플롯 생성
        gs = self.fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # 1. 점수 분포
        ax1 = self.fig.add_subplot(gs[0, 0], facecolor=self.colors['bg'])
        scores = self.recommendations['final_score'].head(15)
        bars = ax1.bar(range(len(scores)), scores, 
                      color=[self.colors['red'] if s > 70 else self.colors['blue'] if s > 50 else self.colors['green'] 
                            for s in scores], alpha=0.8)
        ax1.set_title('투자 점수 분포', color=self.colors['text'], fontsize=12)
        ax1.tick_params(colors=self.colors['text'])
        ax1.grid(True, color=self.colors['border'], alpha=0.3)
        
        # 2. PER vs ROE
        ax2 = self.fig.add_subplot(gs[0, 1], facecolor=self.colors['bg'])
        kr_data = self.recommendations[self.recommendations['Market'] == 'KR'].head(10)
        us_data = self.recommendations[self.recommendations['Market'] == 'US'].head(10)
        
        if len(kr_data) > 0:
            ax2.scatter(kr_data['PER'], kr_data['ROE'], c=self.colors['red'], 
                       s=100, alpha=0.7, label='🇰🇷 한국', edgecolors='white')
        if len(us_data) > 0:
            ax2.scatter(us_data['PER'], us_data['ROE'], c=self.colors['blue'], 
                       s=100, alpha=0.7, label='🇺🇸 미국', marker='s', edgecolors='white')
        
        ax2.set_title('PER vs ROE', color=self.colors['text'], fontsize=12)
        ax2.set_xlabel('PER', color=self.colors['text'])
        ax2.set_ylabel('ROE (%)', color=self.colors['text'])
        ax2.tick_params(colors=self.colors['text'])
        ax2.grid(True, color=self.colors['border'], alpha=0.3)
        ax2.legend()
        
        # 3. 수익률 분포
        ax3 = self.fig.add_subplot(gs[1, 0], facecolor=self.colors['bg'])
        returns_3m = self.recommendations['3M_Return_pct'].head(15)
        returns_6m = self.recommendations['6M_Return_pct'].head(15)
        
        x = np.arange(len(returns_3m))
        width = 0.35
        
        ax3.bar(x - width/2, returns_3m, width, label='3개월', 
               color=self.colors['yellow'], alpha=0.7)
        ax3.bar(x + width/2, returns_6m, width, label='6개월', 
               color=self.colors['green'], alpha=0.7)
        
        ax3.set_title('수익률 비교', color=self.colors['text'], fontsize=12)
        ax3.set_ylabel('수익률 (%)', color=self.colors['text'])
        ax3.tick_params(colors=self.colors['text'])
        ax3.grid(True, color=self.colors['border'], alpha=0.3)
        ax3.legend()
        ax3.axhline(y=0, color='white', linestyle='-', alpha=0.5)
        
        # 4. 시장별 비교
        ax4 = self.fig.add_subplot(gs[1, 1], facecolor=self.colors['bg'])
        
        kr_count = len(self.recommendations[self.recommendations['Market'] == 'KR'])
        us_count = len(self.recommendations[self.recommendations['Market'] == 'US'])
        
        if kr_count > 0 or us_count > 0:
            labels = []
            sizes = []
            colors_pie = []
            
            if kr_count > 0:
                labels.append(f'한국 ({kr_count}개)')
                sizes.append(kr_count)
                colors_pie.append(self.colors['red'])
                
            if us_count > 0:
                labels.append(f'미국 ({us_count}개)')
                sizes.append(us_count)
                colors_pie.append(self.colors['blue'])
            
            ax4.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%',
                   startangle=90, textprops={'color': self.colors['text']})
            ax4.set_title('시장별 분포', color=self.colors['text'], fontsize=12)
        
        self.canvas.draw()
        
    def on_stock_select(self, event):
        """종목 선택 시 상세 정보 표시"""
        selection = self.stock_tree.selection()
        if not selection:
            return
            
        item = selection[0]
        values = self.stock_tree.item(item, 'values')
        
        if not values or len(values) < 2:
            return
            
        ticker = values[1]  # 종목코드
        
        # 해당 종목 데이터 찾기
        stock_data = self.recommendations[self.recommendations['Ticker'] == ticker]
        if len(stock_data) == 0:
            return
            
        stock = stock_data.iloc[0]
        
        # 상세 정보 텍스트 생성
        detail_info = f"""
📊 {stock['Ticker']} 종목 상세 분석

🏢 기본 정보
├ 시장: {'한국 (KOSPI/KOSDAQ)' if stock['Market'] == 'KR' else '미국 (NASDAQ/NYSE)'}
├ 현재가: {stock['Close']:,.2f} {'원' if stock['Market'] == 'KR' else '달러'}
└ 시가총액: {stock['MarketCap_display']:,} {stock['MarketCap_unit']}

📈 밸류에이션 지표
├ PER: {stock['PER']:.1f} 배
├ ROE: {stock['ROE']:.1f} %
└ 가치 점수: {stock['value_score']*100:.1f} / 100

💰 수익률 현황
├ 3개월: {stock['3M_Return_pct']:+.1f} %
├ 6개월: {stock['6M_Return_pct']:+.1f} %
├ 변동성: {stock['Volatility_pct']:.1f} %
└ 모멘텀 점수: {stock['momentum_score']*100:.1f} / 100

🎯 종합 평가
├ 품질 점수: {stock['quality_score']*100:.1f} / 100
├ 안정성 점수: {stock['stability_score']*100:.1f} / 100
└ 최종 점수: {stock['final_score']:.1f} / 100

💡 투자 의견
{'🟢 매수 추천' if stock['final_score'] > 70 else '🟡 관심 종목' if stock['final_score'] > 50 else '🔴 관망'}

📝 분석 요약
- PER {stock['PER']:.1f}배로 {'저평가' if stock['PER'] < 15 else '적정' if stock['PER'] < 25 else '고평가'} 상태
- ROE {stock['ROE']:.1f}%로 {'우수한' if stock['ROE'] > 15 else '양호한' if stock['ROE'] > 10 else '보통'} 수익성
- 6개월 수익률 {stock['6M_Return_pct']:+.1f}%로 {'상승' if stock['6M_Return_pct'] > 0 else '하락'} 추세
- 변동성 {stock['Volatility_pct']:.1f}%로 {'안정적' if stock['Volatility_pct'] < 30 else '보통' if stock['Volatility_pct'] < 50 else '높은'} 리스크
"""
        
        # 텍스트 업데이트
        self.detail_text.config(state='normal')
        self.detail_text.delete(1.0, tk.END)
        self.detail_text.insert(tk.END, detail_info)
        self.detail_text.config(state='disabled')
        
    def update_status(self, message):
        """상태바 업데이트"""
        self.status_bar.config(text=f"{datetime.now().strftime('%H:%M:%S')} - {message}")
        
    def run(self):
        """GUI 실행"""
        self.root.mainloop()

# 실행 코드
if __name__ == "__main__":
    app = HTSStyleGUI()
    app.run() 