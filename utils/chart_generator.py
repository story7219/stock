# utils/chart_generator.py
# mplfinance를 사용하여 주식 차트 이미지를 생성하는 모듈

import pandas as pd
import mplfinance as mpf
from utils.logger import log_event
import os
from data.fetcher import fetch_daily_data

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
        yf_ticker = f"{ticker}.KS"
        df = fetch_daily_data(yf_ticker, period=period)
        
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

if __name__ == '__main__':
    # 테스트
    chart_path = generate_stock_chart("005930") # 삼성전자
    if chart_path:
        print(f"차트가 성공적으로 생성되었습니다: {chart_path}")
    
    chart_path_fail = generate_stock_chart("999999") # 없는 종목
    if not chart_path_fail:
        print("예상대로 없는 종목의 차트 생성에 실패했습니다.") 