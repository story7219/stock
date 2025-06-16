# strategy/short_term_scanner.py
# 단기 투자에 적합한 종목을 전통적인 기술적 분석으로 스크리닝하는 모듈

import pandas as pd
from data.fetcher import fetch_daily_data
from utils.logger import log_event

def find_short_term_candidates(tickers: list) -> list:
    """
    단기 투자에 적합한 후보 종목을 기술적 지표로 스크리닝합니다.
    - 조건 1: 현재가가 20일 이동평균선 위에 있을 것 (상승 추세)
    - 조건 2: 최근 거래량이 5일 평균 거래량보다 150% 이상일 것 (시장 관심)

    Args:
        tickers (list): KOSPI 200 등 스캔 대상 티커 리스트 (예: ['005930', '000660'])

    Returns:
        list: 1차 스크리닝 조건을 통과한 종목 코드 리스트
    """
    candidates = []
    log_event("INFO", f"단기 투자 1차 스크리닝 시작 (대상: {len(tickers)}개 종목)")

    for ticker in tickers:
        try:
            # yfinance는 티커에 .KS 접미사가 필요합니다.
            yf_ticker = f"{ticker}.KS"
            
            # 20일 이평선 및 5일 거래량 평균을 계산하기 위해 약 2개월치 데이터 로드
            df = fetch_daily_data(yf_ticker, period="2mo") 
            
            # 데이터가 없거나 분석에 필요한 최소 기간(21일)을 만족하지 못하면 건너뜁니다.
            if df is None or len(df) < 21:
                continue

            # 기술적 지표 계산
            df['MA20'] = df['Close'].rolling(window=20).mean()
            # 어제까지의 5일 평균 거래량을 계산하기 위해 shift(1) 사용
            df['AvgVolume5'] = df['Volume'].rolling(window=5).mean().shift(1)

            # 가장 최근 데이터 추출
            latest = df.iloc[-1]
            
            # 조건 검사
            is_in_uptrend = latest['Close'] > latest['MA20']
            is_volume_spike = latest['Volume'] > latest['AvgVolume5'] * 1.5

            if is_in_uptrend and is_volume_spike:
                log_event("INFO", f"✅ [단기 후보 발견] {ticker}: 상승 추세 및 거래량 급증 포착")
                candidates.append(ticker)

        except Exception as e:
            log_event("WARNING", f"[{ticker}] 단기 후보 스크리닝 중 오류 발생: {e}")
            continue
            
    log_event("INFO", f"단기 투자 1차 스크리닝 완료. 총 {len(candidates)}개 후보 발견.")
    return candidates 