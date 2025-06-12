# fetcher.py
# yfinance를 이용한 주가 데이터 수집 및 분석 함수 모음 (일목균형표 2역호전 포함)

import yfinance as yf
import pandas as pd
from datetime import datetime
from trading.kis_api import KIS_API # KIS_API 클래스를 직접 import
from utils.logger import log_event # 누락된 log_event import 추가
import config # IS_MOCK_TRADING 값을 사용하기 위해 import
import requests
from collections import Counter

# 1단계: 350일치 OHLCV 데이터 수집

def fetch_ohlcv_350(ticker):
    """
    yfinance로 350일치 OHLCV 데이터프레임 반환
    """
    df = yf.download(ticker, period='350d', interval='1d')
    return df

def fetch_daily_data(ticker: str, period: str = "2mo") -> pd.DataFrame:
    """
    단기 스캐너를 위해 yfinance에서 일봉 데이터를 가져옵니다.
    Args:
        ticker (str): 종목 티커 (예: '005930.KS')
        period (str): 데이터 기간 (기본값: "2mo" - 2개월)
    Returns:
        pd.DataFrame: OHLCV 데이터프레임
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval="1d")
        if df.empty:
            return None
        return df
    except Exception as e:
        print(f"[{ticker}] 일봉 데이터 수집 오류: {e}")
        return None

# 실시간(또는 최근) 데이터 수집 함수 (자동매매 실전용)
def fetch_realtime_ohlcv(ticker, period="5d", interval="1m"):
    """
    yfinance로 실시간(또는 최근) OHLCV 데이터 수집
    """
    df = yf.download(ticker, period=period, interval=interval)
    return df

# 한국투자증권 API용 실시간 데이터 수집 함수 (예시)
def fetch_realtime_ohlcv_kis(ticker, api_key):
    """
    한국투자증권 API를 활용한 실시간 데이터 수집 예시
    """
    # 실제 구현 필요 (API 문서 참고)
    pass

# 2~6단계: 전략 통합 분석 함수

def analyze_complete_strategy(ticker, verbose=True):
    """
    입력: 종목코드
    출력: 각 단계별 결과와 최종 조건 만족 여부
    """
    try:
        df = fetch_ohlcv_350(ticker)
        if df.empty or len(df) < 300:
            if verbose:
                print(f"[데이터 부족] {ticker}")
            return None
        # 2단계: 300일 최저가 및 날짜
        df_300 = df[-300:]
        absolute_low = df_300['Low'].min()
        low_date = df_300['Low'].idxmin()
        if verbose:
            print(f"[절대최저가] {absolute_low}")
            print(f"[최저가 날짜] {low_date.date()}")
        # 3단계: 횡보 판단
        df_after_low = df.loc[low_date:]
        high_since_low = df_after_low['High'].max()
        current_close = df['Close'][-1]
        상승률 = high_since_low / absolute_low
        현재위치 = current_close / absolute_low
        is_sideways = (상승률 < 1.2) and (0.9 <= 현재위치 <= 1.15)
        if verbose:
            print(f"[최저가 이후 최고가] {high_since_low}")
            print(f"[상승률] {상승률:.2f}")
            print(f"[현재위치] {현재위치:.2f}")
            print(f"[횡보여부] {'횡보중' if is_sideways else '횡보아님'}")
        # 4단계: MA30, MA60, 골든크로스
        df['MA30'] = df['Close'].rolling(window=30).mean()
        df['MA60'] = df['Close'].rolling(window=60).mean()
        ma_cross = False
        if len(df) >= 61:
            ma30_yesterday = df['MA30'].iloc[-2]
            ma60_yesterday = df['MA60'].iloc[-2]
            ma30_today = df['MA30'].iloc[-1]
            ma60_today = df['MA60'].iloc[-1]
            if pd.notna(ma30_yesterday) and pd.notna(ma60_yesterday) and pd.notna(ma30_today) and pd.notna(ma60_today):
                ma_cross = (ma30_yesterday < ma60_yesterday) and (ma30_today > ma60_today)
        if verbose:
            print(f"[MA30] {df['MA30'].iloc[-1]:.2f}")
            print(f"[MA60] {df['MA60'].iloc[-1]:.2f}")
            print(f"[골든크로스] {'발생' if ma_cross else '없음'}")
        # 5단계: 일목균형표 계산
        df['전환선'] = (df['High'].rolling(9).max() + df['Low'].rolling(9).min()) / 2
        df['기준선'] = (df['High'].rolling(26).max() + df['Low'].rolling(26).min()) / 2
        ichimoku_1 = df['전환선'].iloc[-1] > df['기준선'].iloc[-1]
        ichimoku_2 = df['Close'].iloc[-1] > df['Close'].shift(25).iloc[-1]
        if verbose:
            print(f"[전환선] {df['전환선'].iloc[-1]:.2f}")
            print(f"[기준선] {df['기준선'].iloc[-1]:.2f}")
            print(f"[1역호전] {'O' if ichimoku_1 else 'X'}")
            print(f"[2역호전] {'O' if ichimoku_2 else 'X'}")
        # 6단계: 최종 조건
        all_ok = is_sideways and ma_cross and ichimoku_2
        if verbose:
            print(f"[최종 조건] {'만족' if all_ok else '불만족'}")
        return {
            'ticker': ticker,
            'absolute_low': absolute_low,
            'low_date': str(low_date.date()),
            'is_sideways': is_sideways,
            'ma_cross': ma_cross,
            'ichimoku_2': ichimoku_2,
            'all_ok': all_ok
        }
    except Exception as e:
        if verbose:
            print(f"[에러] {ticker}: {e}")
        return None

# 8단계: 코스피200 전체 적용 예시
def fetch_kospi200_tickers(use_web_scraping=True):
    """
    코스피200 종목코드 리스트를 웹에서 스크래핑하여 최신 상태로 반환합니다.
    yfinance에서 사용 가능하도록 종목코드 뒤에 '.KS'를 붙여줍니다.
    스크래핑 실패에 대비하여 여러 URL을 시도하고, 최종적으로 내장 리스트를 사용합니다.
    """
    if not use_web_scraping:
        return _get_fallback_kospi200_tickers()

    # 시도할 URL 리스트 (첫 번째: 네이버 금융, 두 번째: 연합인포맥스)
    urls = [
        'https://finance.naver.com/sise/sise_market_sum.naver?sosok=0&page=1',
        'https://www.infomax.co.kr/web/kospi200/component'
    ]
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
    
    for i, url in enumerate(urls):
        try:
            tables = pd.read_html(requests.get(url, headers=headers).text)
            
            if "naver.com" in url:
                # 네이버 금융: 여러 페이지에 걸쳐 있으므로 4페이지까지 읽어옴
                df_list = []
                for page in range(1, 5): # 1~4 페이지
                    page_url = f'https://finance.naver.com/sise/sise_market_sum.naver?sosok=0&page={page}'
                    df_list.append(pd.read_html(requests.get(page_url, headers=headers).text)[1])
                df = pd.concat(df_list)
                df = df[df['종목명'].notna()]
                # 네이버는 종목코드가 없으므로, 종목명을 기반으로 yfinance 티커를 찾아야 함 (여기서는 생략하고 다른 URL 우선)
                # 간소화를 위해 네이버는 건너뛰고 연합인포맥스 우선 시도
                continue # 네이버 로직이 복잡하므로 다음 URL로 넘어감

            elif "infomax.co.kr" in url:
                 # 연합인포맥스는 두 번째 테이블에 정보가 있음
                df = tables[1]
                df['단축코드'] = df['단축코드'].astype(str).str.zfill(6)
                tickers = [f"{code}.KS" for code in df['단축코드'] if len(code) == 6 and code.isdigit()]

            else: # 일반적인 경우 (예: KRX)
                df = tables[0]
                # '종목코드' 열 이름을 실제 테이블에 맞게 수정해야 할 수 있음
                if '종목코드' not in df.columns:
                    # '티커' 등 다른 이름으로 되어있을 경우를 대비
                    code_col = [col for col in df.columns if '코드' in col or '티커' in col][0]
                else:
                    code_col = '종목코드'
                
                df[code_col] = df[code_col].astype(str).str.zfill(6)
                tickers = [f"{code}.KS" for code in df[code_col] if len(code) == 6 and code.isdigit()]

            if tickers:
                log_event("INFO", f"웹 스크래핑으로 KOSPI200 최신 종목 {len(tickers)}개를 성공적으로 불러왔습니다. (URL: {url})")
                # KIS API와 호환을 위해 .KS 접미사 제거
                return [ticker.replace('.KS', '') for ticker in tickers]

        except Exception as e:
            log_event("WARNING", f"URL {url} 에서 KOSPI200 종목 스크래핑 실패: {e}")
            continue # 다음 URL 시도

    # 모든 웹 스크래핑 실패 시
    log_event("ERROR", "모든 웹 스크래핑 시도에 실패했습니다. 내장된 대체 종목 리스트를 사용합니다.")
    return _get_fallback_kospi200_tickers()

def _get_fallback_kospi200_tickers():
    """스크래핑 실패 시 사용할 내장 KOSPI200 리스트를 반환합니다."""
    # 2024년 5월 기준 KOSPI 200 일부 종목 (전체 포함 시 너무 길어짐)
    # 실제 운영 시에는 전체 200개 종목을 넣어두는 것이 좋습니다.
    log_event("WARNING", "내장된 KOSPI200 대체 예시 종목 리스트를 사용합니다. 최신 정보가 아닐 수 있습니다.")
    tickers = [
        '005930.KS', '000660.KS', '035420.KS', '035720.KS', '051910.KS', '005380.KS', '005490.KS', '068270.KS', '105560.KS', '003670.KS',
        '012330.KS', '000270.KS', '066570.KS', '096770.KS', '034730.KS', '028260.KS', '015760.KS', '032830.KS', '006400.KS', '017670.KS'
    ]
    # KIS API와 호환을 위해 .KS 접미사 제거
    return [ticker.replace('.KS', '') for ticker in tickers]

def fetch_market_ranking(kis_api: KIS_API, ranking_type: str, top_n: int = 20) -> list:
    """
    KIS API를 사용하여 지정된 유형의 시장 랭킹 정보를 가져옵니다.
    
    Args:
        kis_api (KIS_API): 초기화된 KIS_API 인스턴스.
        ranking_type (str): 'volume'(거래량), 'gainer'(상승률) 등 순위 유형.
        top_n (int): 가져올 상위 종목의 수.
        
    Returns:
        list: 상위 종목의 티커 리스트 (예: ['005930', '000660', ...]). 실패 시 빈 리스트.
    """
    ranking_data = kis_api.fetch_ranking_data(ranking_type)
    
    if not ranking_data:
        return []
    
    # KIS API는 종목코드에 'KS' 접미사를 붙이지 않으므로, 그대로 사용합니다.
    # API 응답에서 종목 코드(h_kor_iscd)만 추출합니다.
    tickers = [item.get('h_kor_iscd', '').strip() for item in ranking_data]
    
    # None이나 빈 문자열이 포함될 수 있으므로 필터링합니다.
    tickers = [ticker for ticker in tickers if ticker]
    
    log_event("INFO", f"[{ranking_type}] 랭킹 상위 {len(tickers)}개 종목 조회 성공.")
    return tickers[:top_n]

def fetch_short_term_candidates_hybrid(kis_api: KIS_API, top_n: int = 50) -> list:
    """
    '마켓 프리즘' 전략으로 단기 투자 후보군을 선정합니다.
    1. 주도 업종 포착: 상승률 상위 종목으로 오늘의 주도 업종을 결정.
    2. 다중 랭킹 조회: 거래량, 거래대금 등 주요 랭킹 정보를 모두 조회.
    3. 교차 필터링: 다중 랭킹 종목 중 '주도 업종'에 속하는 것만 1차 필터링.
    4. 최종 압축: 필터링된 종목 중 가장 많이 중복된 순서로 최종 후보 선정.
    """
    if config.IS_MOCK_TRADING:
        log_event("WARNING", "[마켓 프리즘] 모의투자 환경에서는 순위 조회를 지원하지 않습니다.")
        return _get_fallback_kospi200_tickers()

    # 1. 주도 업종 포착
    log_event("INFO", "[마켓 프리즘] 1. 주도 업종 포착 시작...")
    gainer_rankers = fetch_market_ranking(kis_api, 'gainer', top_n=top_n)
    if not gainer_rankers:
        log_event("ERROR", "[마켓 프리즘] 주도 업종 포착 실패(상승률 랭킹 조회 불가). 빈 리스트를 반환합니다.")
        return []

    sector_scores = Counter()
    for ticker in gainer_rankers:
        industry = kis_api.get_stock_industry(ticker)
        if industry and '증권' not in industry:
            sector_scores[industry] += 1
    
    if not sector_scores:
        log_event("WARNING", "[마켓 프리즘] 유효한 주도 업종을 찾지 못했습니다.")
        return []
        
    leading_sector = sector_scores.most_common(1)[0][0]
    log_event("SUCCESS", f"[마켓 프리즘] 오늘의 주도 업종: '{leading_sector}'")

    # 2. 다중 랭킹 조회
    log_event("INFO", "[마켓 프리즘] 2. 다중 랭킹(거래량,대금,외국인,기관) 조회...")
    ranking_types = ['volume', 'value', 'foreign_buy', '기관_buy']
    all_rankers = []
    for r_type in ranking_types:
        rankers = fetch_market_ranking(kis_api, r_type, top_n=top_n)
        if rankers:
            all_rankers.extend(rankers)

    if not all_rankers:
        log_event("ERROR", "[마켓 프리즘] 다중 랭킹 정보 조회 실패. 빈 리스트를 반환합니다.")
        return []

    # 3. 교차 필터링: 다중 랭킹 종목들을 주도 업종으로 필터링
    log_event("INFO", f"[마켓 프리즘] 3. 다중 랭킹 종목들을 '{leading_sector}' 업종으로 필터링...")
    filtered_by_sector = []
    # 중복 조회를 피하기 위해 set으로 변환
    unique_rankers = set(all_rankers) 
    for ticker in unique_rankers:
        industry = kis_api.get_stock_industry(ticker)
        if industry == leading_sector:
            # 주도 업종에 속하는 종목들을 원래의 all_rankers 리스트에서 그대로 가져와서 추가
            # (중복 횟수 유지를 위해)
            filtered_by_sector.extend([t for t in all_rankers if t == ticker])
            
    if not filtered_by_sector:
        log_event("WARNING", f"[마켓 프리즘] 주도 업종('{leading_sector}')에 속하는 랭킹 상위 종목이 없습니다.")
        return []

    # 4. 최종 압축
    log_event("INFO", "[마켓 프리즘] 4. 최종 후보군 압축...")
    ticker_counts = Counter(filtered_by_sector)
    sorted_candidates = sorted(ticker_counts.items(), key=lambda item: (-item[1], item[0]))
    final_candidates = [ticker for ticker, count in sorted_candidates]
    
    log_event("SUCCESS", f"[마켓 프리즘] 최종 후보군 {len(final_candidates)}개 선정 완료: {final_candidates}")
    return final_candidates

def _fetch_candidates_by_multi_rank(kis_api: KIS_API, top_n: int = 30) -> list:
    """
    (예비 계획) 다중 랭킹을 종합하여 후보군을 선정합니다.
    """
    log_event("INFO", "[대체 로직] 다중 랭킹(거래량,대금,외국인,기관) 종합 분석을 시작합니다...")
    ranking_types = ['volume', 'value', 'foreign_buy', '기관_buy']
    all_rankers = []
    
    for r_type in ranking_types:
        rankers = fetch_market_ranking(kis_api, r_type, top_n=top_n)
        if rankers:
            all_rankers.extend(rankers)

    if not all_rankers:
        log_event("ERROR", "[대체 로직] 모든 랭킹 조회 실패. 코스피200으로 대체합니다.")
        return fetch_kospi200_tickers()

    from collections import Counter
    ticker_counts = Counter(all_rankers)
    sorted_candidates = sorted(ticker_counts.items(), key=lambda item: (-item[1], item[0]))
    final_candidates = [ticker for ticker, count in sorted_candidates]
    log_event("INFO", f"[대체 로직] 후보군 {len(final_candidates)}개 선정 완료.")
    return final_candidates

class MarketPrism:
    """
    '마켓 프리즘' 전략으로 시장의 핵심 주도주 후보군을 탐색하는 클래스.
    다양한 랭킹 정보를 종합하여 가장 주목받는 종목을 찾아냅니다.
    """
    def __init__(self, kis_api: KIS_API):
        self.kis_api = kis_api
        # 실시간 오디션 전략에서 사용할 랭킹 종류
        self.ranking_types = [
            'trading_value',  # 거래대금
            'volume',         # 거래량
            'gainer',         # 상승률
            'foreign_buy',    # 외국인 순매수
            'institution_buy' # 기관 순매수
        ]

    def _fetch_ranking(self, ranking_type: str, top_n: int) -> list[str]:
        """KIS API를 사용하여 지정된 유형의 시장 랭킹 정보를 가져옵니다."""
        ranking_data = self.kis_api.fetch_ranking_data(ranking_type)
        if not ranking_data:
            return []
        
        tickers = [item.get('h_kor_iscd', '').strip() for item in ranking_data if item.get('h_kor_iscd')]
        log_event("INFO", f"[{ranking_type.upper()}] 랭킹 상위 {len(tickers)}개 종목 조회 성공.")
        return tickers[:top_n]

    def find_market_prism_candidates(self, top_n: int = 30) -> list[str]:
        """
        '마켓 프리즘' 전략의 핵심. 여러 랭킹에서 중복 등장하는 상위 종목을 찾습니다.
        """
        if config.IS_MOCK_TRADING:
            log_event("WARNING", "[MarketPrism] 모의투자 환경에서는 순위 조회를 지원하지 않으므로, 대체 종목 리스트를 반환합니다.")
            return ['005930', '000660', '035420'] # 예시 티커 반환

        log_event("INFO", "[MarketPrism] 다중 랭킹 정보 수집 시작...")
        
        all_ranked_tickers = []
        for rank_type in self.ranking_types:
            ranked_list = self._fetch_ranking(rank_type, top_n)
            all_ranked_tickers.extend(ranked_list)
            
        if not all_ranked_tickers:
            log_event("WARNING", "[MarketPrism] 모든 랭킹에서 종목을 가져오지 못했습니다.")
            return []

        # 가장 많이 중복된 종목 순서대로 정렬
        ticker_counts = Counter(all_ranked_tickers)
        # 2번 이상 중복된 종목만 후보로 인정 (최소한의 신뢰도 확보)
        candidates = [ticker for ticker, count in ticker_counts.items() if count >= 2]
        
        # 중복 횟수가 높은 순으로 정렬
        candidates.sort(key=lambda x: ticker_counts[x], reverse=True)
        
        log_event("INFO", f"[MarketPrism] 최종 후보 {len(candidates)}개 선정: {candidates}")
        
        return candidates

if __name__ == "__main__":
    # 삼성전자 테스트
    print("\n[삼성전자 전략 분석 결과]")
    analyze_complete_strategy('005930.KS', verbose=True)
    # 전체 종목 분석
    print("\n[코스피200 전략 필터링 결과]")
    tickers = fetch_kospi200_tickers()
    results = []
    for ticker in tickers:
        res = analyze_complete_strategy(ticker, verbose=False)
        if res and res['all_ok']:
            results.append(res)
    df_result = pd.DataFrame(results)
    print(df_result) 