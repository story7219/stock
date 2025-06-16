# run_analysis.py
# 사용자가 요청한 8단계 상세 분석 전략을 구현한 독립 실행 스크립트

import yfinance as yf
import pandas as pd
import requests
from io import StringIO

def fetch_kospi200_tickers():
    """
    네이버 금융에서 KOSPI 200 종목 리스트를 스크래핑하여 반환합니다.
    yfinance에서 사용 가능하도록 종목코드 뒤에 '.KS'를 붙여줍니다.
    """
    try:
        # 네이버 금융 KOSPI 200 URL
        url = 'https://finance.naver.com/sise/sise_market_sum.naver?sosok=0&page=1'
        headers = {'User-Agent': 'Mozilla/5.0'}
        
        # 네이버 금융은 여러 페이지에 걸쳐 KOSPI 200 종목을 보여줍니다. (보통 1~4페이지)
        df_list = []
        for page in range(1, 5):
            page_url = f'https://finance.naver.com/sise/sise_market_sum.naver?sosok=0&page={page}'
            response = requests.get(page_url, headers=headers)
            # 네이버 금융의 인코딩은 'euc-kr' 입니다.
            response.encoding = 'euc-kr' 
            # read_html은 페이지의 모든 테이블을 리스트로 반환합니다.
            all_tables = pd.read_html(StringIO(response.text))
            df = all_tables[1] # 종목 정보가 있는 테이블은 보통 두 번째입니다.
            df_list.append(df)

        # 4개 페이지의 데이터프레임을 하나로 합칩니다.
        full_df = pd.concat(df_list, ignore_index=True)
        
        # '종목명'이 없는 행(구분선 등)을 제거하고, '종목코드'를 추출합니다.
        full_df.dropna(subset=['종목명'], inplace=True)
        # 종목코드를 6자리 문자열로 포맷팅하고, yfinance 형식에 맞게 '.KS'를 추가합니다.
        tickers = [f"{str(int(code)).zfill(6)}.KS" for code in full_df['N']]
        
        print(f"✅ 코스피200 최신 종목 리스트 {len(tickers)}개를 성공적으로 불러왔습니다.")
        return tickers
    except Exception as e:
        print(f"🔥 코스피200 종목 리스트를 불러오는 데 실패했습니다: {e}")
        # 실패 시 예시 종목 반환
        return ['005930.KS', '000660.KS', '035420.KS']

def analyze_complete_strategy(ticker, verbose=True):
    """
    입력된 종목코드에 대해 8단계 분석 전략을 수행하고 결과를 반환합니다.
    """
    if verbose:
        print(f"\n--- {ticker} 상세 분석 시작 ---")
        
    try:
        # 1단계: 350일치 OHLCV 데이터 수집
        df = yf.download(ticker, period='350d', interval='1d', progress=False)
        if df.empty or len(df) < 300:
            if verbose: print(f"[1단계-데이터] 데이터 부족 (300일 미만). 분석 중단.")
            return None

        # 2단계: 300일 최저가 및 날짜 찾기
        df_300 = df.iloc[-300:]
        absolute_low = df_300['Low'].min()
        low_date = df_300['Low'].idxmin()
        if verbose:
            print(f"[2단계-최저가] 300일 최저가: {absolute_low:,.0f}원 (날짜: {low_date.date()})")

        # 3단계: 횡보 판단
        df_after_low = df.loc[low_date:]
        high_since_low = df_after_low['High'].max()
        current_close = df['Close'].iloc[-1]
        
        rise_ratio = high_since_low / absolute_low
        current_position = current_close / absolute_low
        
        is_sideways = (rise_ratio < 1.2) and (0.9 <= current_position <= 1.15)
        if verbose:
            print(f"[3단계-횡보판단] 최저가 이후 상승률: {rise_ratio-1:.2%}, 현재 위치: {current_position-1:.2%}")
            print(f"  -> 조건 만족 여부: {'횡보중' if is_sideways else '횡보 아님'}")

        # 4단계: MA30, MA60, 골든크로스 계산
        df['MA30'] = df['Close'].rolling(window=30).mean()
        df['MA60'] = df['Close'].rolling(window=60).mean()
        ma_cross = False
        if len(df) >= 61 and pd.notna(df['MA30'].iloc[-1]) and pd.notna(df['MA60'].iloc[-1]):
            ma_cross = (df['MA30'].iloc[-2] < df['MA60'].iloc[-2]) and (df['MA30'].iloc[-1] > df['MA60'].iloc[-1])
        if verbose:
            print(f"[4단계-MA] MA30: {df['MA30'].iloc[-1]:,.0f}, MA60: {df['MA60'].iloc[-1]:,.0f}")
            print(f"  -> 골든크로스 발생: {'✅' if ma_cross else '❌'}")

        # 5단계: 일목균형표 계산
        df['전환선'] = (df['High'].rolling(9).max() + df['Low'].rolling(9).min()) / 2
        df['기준선'] = (df['High'].rolling(26).max() + df['Low'].rolling(26).min()) / 2
        ichimoku_1 = df['전환선'].iloc[-1] > df['기준선'].iloc[-1]
        ichimoku_2 = df['Close'].iloc[-1] > df['Close'].shift(25).iloc[-1]
        if verbose:
            print(f"[5단계-일목] 전환선 > 기준선: {'✅' if ichimoku_1 else '❌'}, 현재가 > 26일전 종가: {'✅' if ichimoku_2 else '❌'}")

        # 6단계: 최종 조건 확인
        all_conditions_met = is_sideways and ma_cross and ichimoku_2
        if verbose:
            print(f"[6단계-최종] 모든 조건 만족: {'🔥 통과! 🔥' if all_conditions_met else '탈락'}")
            
        return {
            'ticker': ticker,
            '최종통과': all_conditions_met,
            '횡보': is_sideways,
            '골든크로스': ma_cross,
            '2역호전': ichimoku_2
        }
    except Exception as e:
        if verbose:
            print(f"🔥 [{ticker}] 분석 중 오류 발생: {e}")
        return None

if __name__ == "__main__":
    # 7단계: 삼성전자로 테스트
    analyze_complete_strategy('005930.KS', verbose=True)

    # 8단계: 코스피200 전체 적용
    print("\n\n--- 코스피200 전체 종목 분석 시작 ---")
    tickers = fetch_kospi200_tickers()
    results = []
    
    # 멀티프로세싱을 사용하면 훨씬 빠르지만, 간단한 구현을 위해 순차 실행
    for ticker in tickers:
        res = analyze_complete_strategy(ticker, verbose=False) # 전체 분석 시에는 상세 로그 생략
        if res and res['최종통과']:
            results.append(res)
            
    if results:
        df_result = pd.DataFrame(results)
        print("\n\n🎉 최종 조건을 모두 만족하는 유망 종목 리스트 🎉")
        print(df_result)
    else:
        print("\n\nℹ️ 최종 조건을 모두 만족하는 종목을 찾지 못했습니다.") 