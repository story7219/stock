"""
종합 주식 추천 시스템 (All-in-One)
- 재무, 수급, 차트 분석을 통한 상대평가 기반 주식 추천
"""
import os
import pandas as pd
import yfinance as yf
from pykrx import stock
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import warnings

# 경고 메시지 무시
warnings.filterwarnings('ignore', category=UserWarning)

# --- 설정 ---
DATA_PATH = "data"
STOCK_LIST_FILE = os.path.join(DATA_PATH, "stock_list.csv")
# DART_API_KEY = os.getenv("DART_API_KEY", "YOUR_API_KEY") # 환경변수 사용 권장

# 분석 가중치
WEIGHTS = {
    'financial': 0.40,
    'supply_demand': 0.35,
    'chart': 0.25
}

def fetch_stock_data(ticker, name):
    """개별 종목의 모든 데이터를 한번에 가져오는 함수"""
    try:
        # 1. 재무 데이터 (pykrx)
        today_str = datetime.now().strftime("%Y%m%d")
        f_data = stock.get_market_fundamental(today_str, ticker, "y")
        if f_data.empty: return None
        
        per = f_data['PER'].iloc[-1]
        pbr = f_data['PBR'].iloc[-1]
        roe = f_data['ROE'].iloc[-1]
        
        # 2. 수급 데이터 (pykrx, 최근 1개월)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        s_data = stock.get_market_trading_value_by_date(start_date.strftime('%Y%m%d'), end_date.strftime('%Y%m%d'), ticker)
        foreign_net = s_data['외국인합계'].sum()
        inst_net = s_data['기관합계'].sum()
        
        # 3. 차트 데이터 (yfinance, 최근 1년)
        c_data = yf.download(f"{ticker}.KS", period="1y", progress=False, show_errors=False)
        if c_data.empty or len(c_data) < 60: return None
        
        # 차트 분석
        c_data['MA20'] = c_data['Close'].rolling(window=20).mean()
        c_data['MA60'] = c_data['Close'].rolling(window=60).mean()
        
        price = c_data['Close'].iloc[-1]
        is_uptrend = price > c_data['MA20'].iloc[-1] and price > c_data['MA60'].iloc[-1]
        
        return {
            'ticker': ticker,
            'name': name,
            'price': price,
            'per': per if per > 0 else float('inf'), # PER이 0이하면 최하위
            'pbr': pbr if pbr > 0 else float('inf'), # PBR이 0이하면 최하위
            'roe': roe,
            'foreign_net': foreign_net,
            'inst_net': inst_net,
            'is_uptrend': is_uptrend
        }
    except Exception:
        return None

def analyze_and_rank(stocks):
    """종합 분석 및 랭킹 산출"""
    df = pd.DataFrame(stocks)
    
    # 각 지표별 순위 매기기 (값이 작을수록 순위가 높음)
    df['per_rank'] = df['per'].rank(ascending=True)
    df['pbr_rank'] = df['pbr'].rank(ascending=True)
    df['roe_rank'] = df['roe'].rank(ascending=False) # ROE는 높을수록 좋음
    
    df['foreign_rank'] = df['foreign_net'].rank(ascending=False)
    df['inst_rank'] = df['inst_net'].rank(ascending=False)
    
    df['chart_rank'] = df['is_uptrend'].rank(ascending=False)

    # 가중치 적용하여 최종 점수 계산 (점수가 낮을수록 순위가 높음)
    df['financial_score'] = df['per_rank'] + df['pbr_rank'] + df['roe_rank']
    df['supply_score'] = df['foreign_rank'] + df['inst_rank']
    
    df['total_score'] = (df['financial_score'].rank() * WEIGHTS['financial'] +
                         df['supply_score'].rank() * WEIGHTS['supply_demand'] +
                         df['chart_rank'].rank() * WEIGHTS['chart'])

    return df.sort_values(by='total_score', ascending=True)

def display_results(df):
    """결과 출력"""
    print("\n" + "="*80)
    print("📈 종합 주식 추천 Top 10 (재무, 수급, 차트 기반 상대평가)")
    print("="*80)
    
    top_10 = df.head(10)
    
    for i, row in enumerate(top_10.itertuples(), 1):
        print(f"\n{i}위: {row.name} ({row.ticker})")
        print(f"  - 현재가: {row.price:,.0f}원")
        print(f"  - 재무: PER {row.per:.2f} (상위 {row.per_rank/len(df)*100:.0f}%) | PBR {row.pbr:.2f} (상위 {row.pbr_rank/len(df)*100:.0f}%) | ROE {row.roe:.2f}% (상위 {row.roe_rank/len(df)*100:.0f}%)")
        print(f"  - 수급(1개월): 외국인 {row.foreign_net/1e8:,.0f}억 | 기관 {row.inst_net/1e8:,.0f}억")
        print(f"  - 차트: {'상승추세' if row.is_uptrend else '하락/횡보'}")

def main():
    """메인 실행 함수"""
    # 1. 분석 대상 종목 로드
    if not os.path.exists(STOCK_LIST_FILE):
        print(f"'{STOCK_LIST_FILE}' 파일이 없습니다. 먼저 데이터 수집을 실행하세요.")
        return
        
    stock_list_df = pd.read_csv(STOCK_LIST_FILE)
    
    # 2. 모든 종목 데이터 수집 (병렬 처리)
    all_data = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_stock_data, row['ticker'], row['name']): row['name'] for _, row in stock_list_df.iterrows()}
        
        with tqdm(total=len(futures), desc="[1/2] 모든 종목 데이터 수집 중") as progress:
            for future in as_completed(futures):
                result = future.result()
                if result:
                    all_data.append(result)
                progress.update(1)

    if not all_data:
        print("분석 가능한 종목 데이터를 수집하지 못했습니다.")
        return
        
    # 3. 랭킹 및 분석
    print("\n[2/2] 데이터 분석 및 순위 산정 중...")
    ranked_df = analyze_and_rank(all_data)
    
    # 4. 결과 출력
    display_results(ranked_df)

if __name__ == "__main__":
    main() 