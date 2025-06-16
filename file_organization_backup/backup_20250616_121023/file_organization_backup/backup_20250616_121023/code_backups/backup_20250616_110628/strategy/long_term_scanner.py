# strategy/long_term_scanner.py
# '1년 신저가 후 상승 추세' 종목을 스캔하는 모듈 (조건 완화)

import yfinance as yf
import pandas as pd
from utils.logger import log_event

def find_long_term_targets(tickers: list) -> list:
    """
    주어진 티커 리스트에서 장기 투자에 유망한 종목을 찾아 리스트로 반환합니다.
    (조건을 현실적으로 완화하여 필터링 성공률을 높임)
    """
    promising_stocks = []
    log_event("INFO", f"[장기 스캐너] 총 {len(tickers)}개 종목에 대한 분석 시작...")

    for ticker in tickers:
        try:
            # 1. 1년치 일봉 데이터 수집
            df = yf.download(ticker, period="1y", interval="1d", auto_adjust=True, progress=False)
            if df.empty or len(df) < 200: # 거래일 기준을 200일로 완화
                log_event("DEBUG", f"[{ticker}] 데이터 부족 (200일 미만)으로 건너뜁니다.")
                continue

            # 2. 1년 신저가 및 날짜 찾기
            low_price = df['Low'].min()
            low_date = df['Low'].idxmin()

            # 신저가 기록 후 최소 42일 이상 경과 조건 (사용자 요청으로 42일 복원)
            days_since_low = (df.index[-1] - low_date).days
            if days_since_low < 42:
                log_event("DEBUG", f"[{ticker}] 신저가 후 42일 미경과({days_since_low}일)로 건너뜁니다.")
                continue

            # 3. 신저가 이후 42일간 박스권 확인
            sideways_period = df.loc[low_date : low_date + pd.Timedelta(days=42)]
            sideways_high = sideways_period['High'].max()
            
            # 박스권 상단이 신저가 대비 40% 이상 벌어지지 않아야 함 (사용자 요청: 40%로 완화)
            if (sideways_high / low_price) > 1.40:
                log_event("DEBUG", f"[{ticker}] 신저가 대비 박스권 상단이 40% 초과 상승({(sideways_high/low_price-1):.1%})하여 건너뜁니다.")
                continue

            # 4. 현재가가 박스권 상단을 돌파했는지 확인 (상승 추세)
            current_price = df['Close'].iloc[-1]
            # 기존: 상단 근처(-2% ~ +10%) -> 변경: 상단보다 높기만 하면 OK (단, 너무 많이 오른 건 제외)
            if not (current_price > sideways_high and current_price < sideways_high * 1.5):
                log_event("DEBUG", f"[{ticker}] 현재가({current_price:,.0f})가 박스권 상단({sideways_high:,.0f}) 돌파 조건 미충족. 건너뜁니다.")
                continue

            # 모든 조건을 통과한 유망주
            log_event("SUCCESS", f"[장기 유망주 발견] {ticker}: 1년 신저가 후 상승 추세 조건 만족!")
            promising_stocks.append(ticker.replace('.KS', '')) # .KS 제거 후 반환

        except Exception as e:
            log_event("ERROR", f"[{ticker}] 장기 분석 중 오류 발생: {e}")
            continue
            
    log_event("INFO", f"[장기 스캐너] 분석 완료. 총 {len(promising_stocks)}개의 유망 종목 발견.")
    return promising_stocks 