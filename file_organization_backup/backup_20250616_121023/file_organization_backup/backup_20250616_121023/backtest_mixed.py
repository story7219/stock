import pandas as pd
from datetime import datetime, timedelta

# 1. 데이터 준비 (예: 삼성전자, 단기 후보 4종목)
# 실제로는 KIS OpenAPI에서 fetch_daily_price로 데이터프레임을 받아와야 함
long_term_ticker = "005930"
short_term_candidates = ["000660", "035420", "035720", "051910"]  # 예시: 하이닉스, NAVER, 카카오, LG화학

# 2. 데이터 로딩 (여기서는 CSV로 가정, 실제로는 fetch_daily_price로 대체)
def load_price_df(ticker):
    # 예시: '005930.csv' 등으로 저장된 일봉 데이터
    return pd.read_csv(f"{ticker}.csv", parse_dates=["date"])

long_term_df = load_price_df(long_term_ticker)
short_term_dfs = {t: load_price_df(t) for t in short_term_candidates}

# 3. 초기 자본 및 비중
total_capital = 500_000_000
long_term_ratio = 0.25
short_term_ratio = 0.75
min_cash_ratio = 0.25

long_term_capital = total_capital * long_term_ratio
short_term_capital = total_capital * short_term_ratio

# 4. 장기투자: 3년간 buy&hold
long_term_buy_price = long_term_df.iloc[0]["close"]
long_term_quantity = int(long_term_capital // long_term_buy_price)
long_term_invested = long_term_quantity * long_term_buy_price
long_term_final_price = long_term_df.iloc[-1]["close"]
long_term_final_value = long_term_quantity * long_term_final_price
long_term_profit = long_term_final_value - long_term_invested

# 5. 단기투자: (아주 단순화된 오디션/본대/현금 유지 예시)
short_term_cash = short_term_capital
short_term_holdings = {}
short_term_trade_log = []

for day in range(0, len(long_term_df), 20):  # 20일마다 오디션(척후병) 실시
    # 후보 4종목 중 2종목 랜덤 선정(여기선 단순히 앞 2개)
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

# 6. 결과 요약
total_final = long_term_final_value + short_term_cash
total_profit = total_final + sum([qty*long_term_final_price for qty, _ in short_term_holdings.values()]) - total_capital

print(f"장기투자({long_term_ticker}) 3년 수익: {long_term_profit:,.0f}원")
print(f"단기투자(2종목 오디션/본대) 3년 후 현금: {short_term_cash:,.0f}원")
print(f"총 누적 수익: {total_profit:,.0f}원")
print("단기 매매 내역(일부):", short_term_trade_log[:5]) 