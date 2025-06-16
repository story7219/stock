import requests
import pandas as pd
from datetime import datetime, timedelta
import time

# 1. KIS OpenAPI 인증 정보 입력 (실제 값으로 교체)
KIS_APP_KEY = "PSJHToqNQYzVvVH1DfkndIodXaCsEgAHBHPr"
KIS_APP_SECRET = "W5ts9iDYGxjNGaPdKqDcjAQz2FdLwakr/2sC3K44zs9dtljT2P8UbB/zOo2hsWZpkP/kraOmF9P1vqqcHxbz/YiVwKcR6FCmj/WZdoAdnCfQi/KMntP9V1b6dn7RLoOiTZtgwLaoVfWKJPP+hcmxNI/st+oCp3iDv/ZdKoQg4Hu9OG4myW0="
ACCESS_TOKEN = "50128558-01"  # 또는 토큰 발급 코드 추가
BASE_URL = "https://openapi.koreainvestment.com:9000"
HEADERS = {
    "content-type": "application/json",
    "authorization": f"Bearer {ACCESS_TOKEN}",
    "appkey": KIS_APP_KEY,
    "appsecret": KIS_APP_SECRET,
    "tr_id": "FHKST01010100"
}

def fetch_daily_price(ticker, count=800):
    url = f"{BASE_URL}/uapi/domestic-stock/v1/quotations/inquire-daily-price"
    params = {
        "fid_cond_mrkt_div_code": "J",
        "fid_input_iscd": ticker,
        "fid_org_adj_prc": "0",
        "fid_period_div_code": "D",
        "fid_vol_cond_code": "0",
        "fid_input_date_1": (datetime.now() - timedelta(days=count*1.5)).strftime("%Y%m%d"),
        "fid_input_date_2": datetime.now().strftime("%Y%m%d"),
        "fid_output_cnt": str(count)
    }
    res = requests.get(url, headers=HEADERS, params=params)
    data = res.json()
    if data.get('rt_cd') == '0':
        df = pd.DataFrame(data['output'])
        # 컬럼명 통일 및 정렬
        df = df.rename(columns={"stck_bsop_date": "date", "stck_clpr": "close"})
        df["date"] = pd.to_datetime(df["date"])
        df["close"] = df["close"].astype(float)
        df = df.sort_values("date")
        return df[["date", "close"]]
    else:
        print(f"{ticker} 데이터 조회 실패:", data.get('msg1'))
        return None

# 2. 종목 리스트
long_term_ticker = "005930"
short_term_candidates = ["000660", "035420", "035720", "051910"]
all_tickers = [long_term_ticker] + short_term_candidates

# 3. 데이터 수집 및 CSV 저장
for ticker in all_tickers:
    df = fetch_daily_price(ticker, count=800)
    if df is not None:
        df.to_csv(f"{ticker}.csv", index=False)
        print(f"{ticker} 저장 완료 ({len(df)} rows)")
    time.sleep(0.2)  # API Rate Limit 보호

# 4. 백테스트 실행 (앞서 제공한 backtest_mixed.py 코드 복사/붙여넣기)
import backtest_mixed 