import requests
import pandas as pd
from datetime import datetime, timedelta
import time

# 1. KIS OpenAPI 인증 정보 입력
KIS_APP_KEY = "여기에_모의투자_APP_KEY"
KIS_APP_SECRET = "여기에_모의투자_APP_SECRET"

def get_access_token(app_key, app_secret):
    url = "https://openapi.koreainvestment.com:9443/oauth2/tokenP"
    headers = {"content-type": "application/json"}
    body = {
        "grant_type": "client_credentials",
        "appkey": app_key,
        "appsecret": app_secret
    }
    res = requests.post(url, headers=headers, json=body)
    data = res.json()
    if data.get('access_token'):
        return data['access_token']
    else:
        raise Exception(f"토큰 발급 실패: {data}")

ACCESS_TOKEN = get_access_token(KIS_APP_KEY, KIS_APP_SECRET)
BASE_URL = "https://openapi.koreainvestment.com:9000"
HEADERS = {
    "content-type": "application/json",
    "authorization": f"Bearer {ACCESS_TOKEN}",
    "appkey": KIS_APP_KEY,
    "appsecret": KIS_APP_SECRET,
    "tr_id": "FHKST01010100"
}

def fetch_daily_price(ticker, start_date, end_date):
    url = f"{BASE_URL}/uapi/domestic-stock/v1/quotations/inquire-daily-price"
    params = {
        "fid_cond_mrkt_div_code": "J",
        "fid_input_iscd": ticker,
        "fid_org_adj_prc": "0",
        "fid_period_div_code": "D",
        "fid_vol_cond_code": "0",
        "fid_input_date_1": start_date,
        "fid_input_date_2": end_date,
        "fid_output_cnt": "800"
    }
    res = requests.get(url, headers=HEADERS, params=params)
    data = res.json()
    if data.get('rt_cd') == '0':
        df = pd.DataFrame(data['output'])
        df = df.rename(columns={"stck_bsop_date": "date", "stck_clpr": "close"})
        df["date"] = pd.to_datetime(df["date"])
        df["close"] = df["close"].astype(float)
        df = df.sort_values("date")
        return df[["date", "close"]]
    else:
        print(f"{ticker} 데이터 조회 실패:", data.get('msg1'))
        return None

# 2. 종목/기간 지정
long_term_ticker = "005930"
short_term_candidates = ["000660", "035420", "035720", "051910"]
all_tickers = [long_term_ticker] + short_term_candidates

# 기간 지정 (예: 최근 3년)
end_date = datetime.now().strftime("%Y%m%d")
start_date = (datetime.now() - timedelta(days=3*365)).strftime("%Y%m%d")

# 3. 데이터 수집 및 CSV 저장
for ticker in all_tickers:
    df = fetch_daily_price(ticker, start_date, end_date)
    if df is not None:
        df.to_csv(f"{ticker}.csv", index=False)
        print(f"{ticker} 저장 완료 ({len(df)} rows)")
    time.sleep(0.2)  # API Rate Limit 보호

# 4. 백테스트 실행 (backtest_mixed.py의 주요 로직을 함수로 가져와서 실행)
def run_backtest_mixed(long_term_ticker, short_term_candidates, total_capital=500_000_000):
    """
    장기 1종목 + 단기 2종목 혼합 전략 백테스트
    """
    long_term_df = pd.read_csv(f"{long_term_ticker}.csv", parse_dates=["date"])
    short_term_dfs = {t: pd.read_csv(f"{t}.csv", parse_dates=["date"]) for t in short_term_candidates}
    long_term_ratio = 0.25
    short_term_ratio = 0.75
    min_cash_ratio = 0.25

    long_term_capital = total_capital * long_term_ratio
    short_term_capital = total_capital * short_term_ratio

    # 장기투자: 3년간 buy&hold
    long_term_buy_price = long_term_df.iloc[0]["close"]
    long_term_quantity = int(long_term_capital // long_term_buy_price)
    long_term_invested = long_term_quantity * long_term_buy_price
    long_term_final_price = long_term_df.iloc[-1]["close"]
    long_term_final_value = long_term_quantity * long_term_final_price
    long_term_profit = long_term_final_value - long_term_invested

    # 단기투자: (아주 단순화된 오디션/본대/현금 유지 예시)
    short_term_cash = short_term_capital
    short_term_holdings = {}
    short_term_trade_log = []

    for day in range(0, len(long_term_df), 20):  # 20일마다 오디션(척후병) 실시
        selected = short_term_candidates[:2]
        invest_per_stock = (short_term_capital * 0.5) / 2  # 2종목, 50%만 투자(현금 25% 유지)
        for ticker in selected:
            price = short_term_dfs[ticker].iloc[day]["close"]
            qty = int(invest_per_stock // price)
            cost = qty * price
            if short_term_cash - cost < total_capital * min_cash_ratio:
                continue  # 현금 25% 유지
            short_term_cash -= cost
            short_term_holdings[ticker] = (qty, price)
            short_term_trade_log.append((long_term_df.iloc[day]["date"], ticker, "BUY", qty, price))
        # 20일 후 매도(단순화)
        for ticker in list(short_term_holdings.keys()):
            qty, buy_price = short_term_holdings[ticker]
            sell_price = short_term_dfs[ticker].iloc[min(day+20, len(short_term_df)-1)]["close"]
            short_term_cash += qty * sell_price
            short_term_trade_log.append((long_term_df.iloc[min(day+20, len(short_term_df)-1)]["date"], ticker, "SELL", qty, sell_price))
            del short_term_holdings[ticker]

    # 결과 리포트
    total_final = long_term_final_value + short_term_cash
    total_profit = total_final + sum([qty*long_term_final_price for qty, _ in short_term_holdings.values()]) - total_capital

    print("===== [혼합전략] 백테스트 결과 리포트 =====")
    print(f"장기투자({long_term_ticker}) 3년 수익: {long_term_profit:,.0f}원")
    print(f"단기투자(2종목 오디션/본대) 3년 후 현금: {short_term_cash:,.0f}원")
    print(f"총 누적 수익: {total_profit:,.0f}원")
    print("단기 매매 내역(일부):", short_term_trade_log[:5])
    print("================================")

def run_backtest_shortterm_only(short_term_candidates, total_capital=500_000_000):
    """
    단기 2종목 전략만 백테스트 (장기투자 없음, 100% 단기 운용)
    """
    short_term_df_len = None
    short_term_dfs = {t: pd.read_csv(f"{t}.csv", parse_dates=["date"]) for t in short_term_candidates}
    for df in short_term_dfs.values():
        if short_term_df_len is None or len(df) < short_term_df_len:
            short_term_df_len = len(df)
    short_term_cash = total_capital
    short_term_holdings = {}
    short_term_trade_log = []
    min_cash_ratio = 0.25

    for day in range(0, short_term_df_len, 20):
        selected = short_term_candidates[:2]
        invest_per_stock = (total_capital * 0.5) / 2  # 2종목, 50%만 투자(현금 25% 유지)
        for ticker in selected:
            price = short_term_dfs[ticker].iloc[day]["close"]
            qty = int(invest_per_stock // price)
            cost = qty * price
            if short_term_cash - cost < total_capital * min_cash_ratio:
                continue
            short_term_cash -= cost
            short_term_holdings[ticker] = (qty, price)
            short_term_trade_log.append((short_term_dfs[ticker].iloc[day]["date"], ticker, "BUY", qty, price))
        # 20일 후 매도(단순화)
        for ticker in list(short_term_holdings.keys()):
            qty, buy_price = short_term_holdings[ticker]
            sell_price = short_term_dfs[ticker].iloc[min(day+20, short_term_df_len-1)]["close"]
            short_term_cash += qty * sell_price
            short_term_trade_log.append((short_term_dfs[ticker].iloc[min(day+20, short_term_df_len-1)]["date"], ticker, "SELL", qty, sell_price))
            del short_term_holdings[ticker]

    total_profit = short_term_cash - total_capital
    print("===== [단기전략] 백테스트 결과 리포트 =====")
    print(f"단기투자(2종목 오디션/본대) 3년 후 현금: {short_term_cash:,.0f}원")
    print(f"총 누적 수익: {total_profit:,.0f}원")
    print("단기 매매 내역(일부):", short_term_trade_log[:5])
    print("================================")

# 혼합 전략 백테스트
run_backtest_mixed(long_term_ticker, short_term_candidates)

# 단기 2종목 전략만 백테스트
run_backtest_shortterm_only(short_term_candidates) 