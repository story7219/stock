# utils/chart_generator.py
# mplfinance를 사용하여 주식 차트 이미지를 생성하는 모듈

import pandas as pd
import mplfinance as mpf
from utils.logger import log_event
import os
from data.fetcher import fetch_daily_data_for_chart
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import requests
from typing import Optional

CHART_DIR = 'charts'
os.makedirs(CHART_DIR, exist_ok=True)

def generate_stock_chart(ticker: str, period: str = "3mo") -> str | None:
    """
    주어진 종목의 시세 차트 이미지를 생성하고 파일 경로를 반환합니다.

    Args:
        ticker (str): KIS API와 호환되는 종목 코드 (예: '005930')
        period (str): yfinance에서 데이터를 가져올 기간 (예: "3mo")

    Returns:
        str | None: 생성된 차트 이미지 파일의 경로 또는 실패 시 None
    """
    try:
        # yfinance는 '.KS' 접미사가 필요합니다.
        df = fetch_daily_data_for_chart(ticker, period=period)
        
        if df is None or df.empty:
            log_event("WARNING", f"[{ticker}] 차트 생성을 위한 데이터를 가져올 수 없습니다.")
            return None

        # 차트 스타일 및 설정
        mc = mpf.make_marketcolors(
            up='red', down='blue',
            edge={'up':'red', 'down':'blue'},
            wick={'up':'red', 'down':'blue'},
            volume='inherit'
        )
        style = mpf.make_mpf_style(marketcolors=mc, gridstyle=':')

        # 차트 파일 경로
        chart_filename = os.path.join(CHART_DIR, f"{ticker}_chart_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png")

        # 차트 생성 및 저장
        mpf.plot(
            df,
            type='candle',
            title=f"{ticker} Chart ({period})",
            ylabel='Price (KRW)',
            volume=True,
            mav=(5, 20, 60), # 5, 20, 60일 이동평균선
            style=style,
            savefig=chart_filename
        )
        log_event("INFO", f"[{ticker}] 차트 이미지 생성 완료: {chart_filename}")
        return chart_filename

    except Exception as e:
        log_event("ERROR", f"[{ticker}] 차트 생성 중 오류 발생: {e}")
        return None

def generate_market_heatmap(market_data: dict) -> Optional[str]:
    """시장 히트맵 생성"""
    try:
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # 업종별 등락률 히트맵 (더미 데이터)
        sectors = ['기술주', '금융', '화학', '자동차', '건설', '통신', '유통', '에너지']
        changes = [2.5, -1.2, 1.8, 0.5, -0.8, 1.1, -2.1, 3.2]  # 더미 데이터
        
        colors = ['red' if x > 0 else 'blue' for x in changes]
        bars = ax.barh(sectors, changes, color=colors, alpha=0.7)
        
        ax.set_title('업종별 등락률', fontsize=16, fontweight='bold')
        ax.set_xlabel('등락률 (%)', fontsize=12)
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        
        # 수치 표시
        for i, (bar, change) in enumerate(zip(bars, changes)):
            ax.text(change + 0.1 if change > 0 else change - 0.1, 
                   i, f'{change:+.1f}%', 
                   va='center', ha='left' if change > 0 else 'right')
        
        chart_path = 'charts/market_heatmap.png'
        plt.tight_layout()
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return chart_path
        
    except Exception as e:
        log_event("ERROR", f"히트맵 생성 실패: {e}")
        return None

if __name__ == '__main__':
    # 테스트
    chart_path = generate_stock_chart("005930") # 삼성전자
    if chart_path:
        print(f"차트가 성공적으로 생성되었습니다: {chart_path}")
    
    chart_path_fail = generate_stock_chart("999999") # 없는 종목
    if not chart_path_fail:
        print("예상대로 없는 종목의 차트 생성에 실패했습니다.") 