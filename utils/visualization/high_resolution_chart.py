#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
고해상도 캔들스틱 차트 생성기
삼성전자 주식 데이터를 활용한 전문적인 차트 시스템
"""

import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class HighResolutionCandlestickChart:
    """고해상도 캔들스틱 차트 생성 클래스"""
    
    def __init__(self):
        """초기화"""
        # 한글 폰트 설정
        plt.rcParams['font.family'] = ['Malgun Gothic', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 고해상도 설정
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['figure.figsize'] = [19.2, 10.8]  # 1920x1080 비율
        
        # HTS 스타일 색상
        self.colors = {
            'bg': '#0a0a0a',
            'panel': '#1a1a1a',
            'text': '#ffffff',
            'green': '#00ff88',    # 상승
            'red': '#ff4444',      # 하락
            'blue': '#4488ff',
            'yellow': '#ffdd44',
            'gray': '#666666',
            'grid': '#333333'
        }
        
    def fetch_stock_data(self, symbol="005930.KS", start_date="2024-01-01", end_date="2025-06-21"):
        """주식 데이터 가져오기"""
        try:
            print(f"📊 {symbol} 데이터 다운로드 중...")
            
            # yfinance로 데이터 가져오기
            stock = yf.Ticker(symbol)
            data = stock.history(start=start_date, end=end_date)
            
            if data.empty:
                raise ValueError("데이터를 가져올 수 없습니다")
                
            print(f"✅ 데이터 다운로드 완료: {len(data)}일 데이터")
            return data
            
        except Exception as e:
            print(f"❌ 데이터 다운로드 오류: {e}")
            # 샘플 데이터 생성
            return self.generate_sample_data(start_date, end_date)
    
    def generate_sample_data(self, start_date, end_date):
        """샘플 데이터 생성 (yfinance 실패 시)"""
        print("📈 샘플 데이터 생성 중...")
        
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        dates = dates[dates.weekday < 5]  # 주말 제외
        
        # 삼성전자 주가 시뮬레이션 (70,000원 기준)
        np.random.seed(42)
        base_price = 70000
        
        data = []
        current_price = base_price
        
        for date in dates:
            # 일일 변동률 (-3% ~ +3%)
            change_rate = np.random.normal(0, 0.015)
            
            open_price = current_price
            high_price = open_price * (1 + abs(change_rate) + np.random.uniform(0, 0.01))
            low_price = open_price * (1 - abs(change_rate) - np.random.uniform(0, 0.01))
            close_price = open_price * (1 + change_rate)
            volume = np.random.randint(10000000, 50000000)  # 1천만~5천만주
            
            current_price = close_price
            
            data.append({
                'Open': open_price,
                'High': high_price,
                'Low': low_price,
                'Close': close_price,
                'Volume': volume
            })
        
        df = pd.DataFrame(data, index=dates)
        print(f"✅ 샘플 데이터 생성 완료: {len(df)}일 데이터")
        return df
    
    def create_candlestick_chart(self, data, symbol="삼성전자", save_format="both"):
        """고해상도 캔들스틱 차트 생성"""
        print("🎨 고해상도 캔들스틱 차트 생성 중...")
        
        # 피규어 생성 (1920x1080 해상도)
        fig = plt.figure(figsize=(19.2, 10.8), facecolor=self.colors['bg'])
        
        # 2x1 서브플롯 (차트 3:1 거래량)
        gs = fig.add_gridspec(4, 1, height_ratios=[3, 1, 0, 0], hspace=0.1)
        
        # 가격 차트
        ax1 = fig.add_subplot(gs[0], facecolor=self.colors['panel'])
        
        # 거래량 차트
        ax2 = fig.add_subplot(gs[1], facecolor=self.colors['panel'], sharex=ax1)
        
        # 캔들스틱 그리기
        self.draw_candlesticks(ax1, data)
        
        # 거래량 차트 그리기
        self.draw_volume_chart(ax2, data)
        
        # 차트 스타일링
        self.style_price_chart(ax1, symbol)
        self.style_volume_chart(ax2)
        
        # 전체 레이아웃 조정
        plt.tight_layout()
        
        # 저장
        self.save_chart(fig, symbol, save_format)
        
        return fig
    
    def draw_candlesticks(self, ax, data):
        """캔들스틱 그리기"""
        dates = data.index
        opens = data['Open'].values
        highs = data['High'].values
        lows = data['Low'].values
        closes = data['Close'].values
        
        # 캔들 너비 계산
        if len(dates) > 1:
            width = (dates[1] - dates[0]).days * 0.6
        else:
            width = 0.6
            
        for i, date in enumerate(dates):
            open_price = opens[i]
            high_price = highs[i]
            low_price = lows[i]
            close_price = closes[i]
            
            # 상승/하락 색상 결정
            if close_price >= open_price:
                color = self.colors['green']
                body_color = self.colors['green']
            else:
                color = self.colors['red']
                body_color = self.colors['red']
            
            # 고저선 (심지) 그리기
            ax.plot([date, date], [low_price, high_price], 
                   color=color, linewidth=1, alpha=0.8)
            
            # 캔들 몸통 그리기
            body_height = abs(close_price - open_price)
            body_bottom = min(open_price, close_price)
            
            if body_height > 0:
                rect = Rectangle((date - timedelta(days=width/2), body_bottom),
                               timedelta(days=width), body_height,
                               facecolor=body_color, edgecolor=color,
                               alpha=0.8, linewidth=0.5)
                ax.add_patch(rect)
            else:
                # 도지 (시가 == 종가)
                ax.plot([date - timedelta(days=width/2), date + timedelta(days=width/2)],
                       [close_price, close_price], color=color, linewidth=2)
    
    def draw_volume_chart(self, ax, data):
        """거래량 차트 그리기"""
        dates = data.index
        volumes = data['Volume'].values
        opens = data['Open'].values
        closes = data['Close'].values
        
        # 거래량 바 차트
        colors = [self.colors['green'] if close >= open else self.colors['red'] 
                 for open, close in zip(opens, closes)]
        
        bars = ax.bar(dates, volumes, color=colors, alpha=0.7, width=0.8)
        
        # 거래량 단위 변환 (백만주)
        max_volume = max(volumes)
        if max_volume > 1000000:
            ax.set_ylabel('거래량 (백만주)', color=self.colors['text'], fontsize=12, fontweight='bold')
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000000:.0f}M'))
        else:
            ax.set_ylabel('거래량', color=self.colors['text'], fontsize=12, fontweight='bold')
    
    def style_price_chart(self, ax, symbol):
        """가격 차트 스타일링"""
        # 제목
        ax.set_title(f'📊 {symbol} 고해상도 캔들스틱 차트 (1920x1080)\n'
                    f'📅 기간: {ax.get_xlim()[0]:.0f} - {ax.get_xlim()[1]:.0f}',
                    color=self.colors['text'], fontsize=16, fontweight='bold', pad=20)
        
        # Y축 레이블
        ax.set_ylabel('주가 (원)', color=self.colors['text'], fontsize=12, fontweight='bold')
        
        # 격자선
        ax.grid(True, color=self.colors['grid'], alpha=0.3, linestyle='-', linewidth=0.5)
        
        # 축 색상
        ax.tick_params(colors=self.colors['text'])
        ax.spines['bottom'].set_color(self.colors['text'])
        ax.spines['top'].set_color(self.colors['text'])
        ax.spines['left'].set_color(self.colors['text'])
        ax.spines['right'].set_color(self.colors['text'])
        
        # X축 날짜 포맷
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        
        # 가격 포맷 (천 단위 구분)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
        
        # X축 레이블 숨기기 (거래량 차트와 공유)
        plt.setp(ax.get_xticklabels(), visible=False)
    
    def style_volume_chart(self, ax):
        """거래량 차트 스타일링"""
        # X축 레이블
        ax.set_xlabel('날짜', color=self.colors['text'], fontsize=12, fontweight='bold')
        
        # 격자선
        ax.grid(True, color=self.colors['grid'], alpha=0.3, linestyle='-', linewidth=0.5)
        
        # 축 색상
        ax.tick_params(colors=self.colors['text'])
        ax.spines['bottom'].set_color(self.colors['text'])
        ax.spines['top'].set_color(self.colors['text'])
        ax.spines['left'].set_color(self.colors['text'])
        ax.spines['right'].set_color(self.colors['text'])
        
        # X축 날짜 포맷
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        
        # X축 레이블 회전
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    def save_chart(self, fig, symbol, save_format):
        """차트 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if save_format in ["png", "both"]:
            png_filename = f"{symbol}_candlestick_{timestamp}.png"
            fig.savefig(png_filename, 
                       facecolor=self.colors['bg'],
                       edgecolor='none',
                       dpi=300,
                       bbox_inches='tight',
                       pad_inches=0.2)
            print(f"✅ PNG 저장 완료: {png_filename}")
        
        if save_format in ["svg", "both"]:
            svg_filename = f"{symbol}_candlestick_{timestamp}.svg"
            fig.savefig(svg_filename,
                       facecolor=self.colors['bg'],
                       edgecolor='none',
                       format='svg',
                       bbox_inches='tight',
                       pad_inches=0.2)
            print(f"✅ SVG 저장 완료: {svg_filename}")
    
    def create_comprehensive_chart(self, symbol="005930.KS", korean_name="삼성전자", 
                                 start_date="2024-01-01", end_date="2025-06-21", 
                                 save_format="both"):
        """종합 차트 생성 메인 함수"""
        print("🚀 고해상도 캔들스틱 차트 생성 시작")
        print(f"📊 종목: {korean_name} ({symbol})")
        print(f"📅 기간: {start_date} ~ {end_date}")
        print(f"💾 저장 형식: {save_format}")
        print("=" * 50)
        
        try:
            # 데이터 가져오기
            data = self.fetch_stock_data(symbol, start_date, end_date)
            
            # 차트 생성
            fig = self.create_candlestick_chart(data, korean_name, save_format)
            
            # 통계 정보 출력
            self.print_statistics(data, korean_name)
            
            print("🎉 고해상도 캔들스틱 차트 생성 완료!")
            return fig
            
        except Exception as e:
            print(f"❌ 차트 생성 오류: {e}")
            return None
    
    def print_statistics(self, data, symbol):
        """주식 통계 정보 출력"""
        print("\n📈 주식 통계 정보")
        print("=" * 30)
        print(f"종목명: {symbol}")
        print(f"데이터 기간: {data.index[0].strftime('%Y-%m-%d')} ~ {data.index[-1].strftime('%Y-%m-%d')}")
        print(f"총 거래일: {len(data)}일")
        print(f"시작가: {data['Open'].iloc[0]:,.0f}원")
        print(f"종료가: {data['Close'].iloc[-1]:,.0f}원")
        print(f"최고가: {data['High'].max():,.0f}원")
        print(f"최저가: {data['Low'].min():,.0f}원")
        print(f"평균 거래량: {data['Volume'].mean():,.0f}주")
        
        # 수익률 계산
        total_return = ((data['Close'].iloc[-1] / data['Open'].iloc[0]) - 1) * 100
        print(f"총 수익률: {total_return:+.2f}%")


def main():
    """메인 실행 함수"""
    # 차트 생성기 초기화
    chart_generator = HighResolutionCandlestickChart()
    
    # 삼성전자 차트 생성
    chart_generator.create_comprehensive_chart(
        symbol="005930.KS",
        korean_name="삼성전자",
        start_date="2024-01-01",
        end_date="2025-06-21",
        save_format="both"  # PNG, SVG 둘 다 저장
    )
    
    # 차트 표시
    plt.show()


if __name__ == "__main__":
    main() 