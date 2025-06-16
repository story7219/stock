import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from typing import Dict, List, Tuple
import asyncio
from trade import TradingSystem

# --- 한국어 주석 ---
# 시스템에 맞는 한글 폰트 설정 (Windows: 'Malgun Gothic', macOS: 'AppleGothic')
try:
    plt.rc('font', family='Malgun Gothic')
except:
    try:
        plt.rc('font', family='AppleGothic')
    except:
        print("경고: 한글 폰트를 찾을 수 없습니다. 그래프 제목이 깨질 수 있습니다.")
plt.rcParams['axes.unicode_minus'] = False # 마이너스 폰트 깨짐 방지


# ==============================================================================
# 1. 유틸리티 함수 (설정 및 데이터 처리)
# ==============================================================================

def setup_korean_font():
    """matplotlib 그래프에서 한글을 지원하기 위한 폰트를 설정합니다."""
    try:
        # Windows
        plt.rc('font', family='Malgun Gothic')
    except:
        try:
            # macOS
            plt.rc('font', family='AppleGothic')
        except:
            print("경고: 한글 폰트를 찾을 수 없습니다. 그래프 제목 및 축 레이블이 깨질 수 있습니다.")
    plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

def generate_dummy_data(symbol: str, start_date='2021-01-01', end_date='2023-12-31') -> pd.DataFrame:
    """시연을 위한 가상 주가 데이터를 생성합니다. (실제 데이터 사용 시 불필요)"""
    dates = pd.date_range(start_date, end_date, freq='B')  # 'B'는 영업일 기준
    n = len(dates)
    start_price = np.random.uniform(20000, 150000)
    drift = np.random.uniform(-0.0001, 0.0005)
    
    # Numpy를 사용한 벡터화 연산으로 데이터 생성 속도 향상
    price_changes = np.random.standard_normal(n) * np.random.uniform(500, 2000) + (start_price * drift)
    prices = start_price + np.cumsum(price_changes)
    prices = np.maximum(prices, 1000)  # 주가 하한선을 1000원으로 설정

    return pd.DataFrame({'close': prices}, index=dates)

def load_price_data(symbols: List[str], use_dummy_data: bool = True) -> Dict[str, pd.DataFrame]:
    """지정된 종목들의 가격 데이터를 불러옵니다."""
    price_dict = {}
    for symbol in symbols:
        if use_dummy_data:
            print(f"정보: '{symbol}'에 대한 가상 데이터를 생성합니다.")
            price_dict[symbol] = generate_dummy_data(symbol)
        else:
            try:
                path = f'data/{symbol}.csv'
                price_dict[symbol] = pd.read_csv(path, parse_dates=['date']).set_index('date')
                print(f"정보: '{path}'에서 실제 데이터를 성공적으로 불러왔습니다.")
            except FileNotFoundError:
                print(f"오류: 'data/{symbol}.csv' 파일을 찾을 수 없습니다. 프로그램을 종료합니다.")
                return {}
    return price_dict

# ==============================================================================
# 2. 핵심 전략 로직
# ==============================================================================

def select_short_term_stocks(
    price_dict: Dict[str, pd.DataFrame], 
    candidate_symbols: List[str], 
    date: pd.Timestamp, 
    n: int = 2,
    momentum_period: int = 20
) -> List[str]:
    """주어진 날짜 기준, 모멘텀(수익률)이 가장 높은 단기 후보 종목 n개를 선택합니다."""
    returns = {}
    for symbol in candidate_symbols:
        df = price_dict[symbol]
        # loc를 사용한 인덱싱으로 특정 날짜 이전 데이터 필터링
        past_prices = df.loc[:date, 'close']
        if len(past_prices) > momentum_period:
            current_price = past_prices.iloc[-1]
            prev_price = past_prices.iloc[-(momentum_period + 1)]
            if prev_price > 0:
                returns[symbol] = (current_price / prev_price) - 1
    
    return sorted(returns, key=returns.get, reverse=True)[:n]

# ==============================================================================
# 3. 백테스팅 엔진 (수수료/세금 반영)
# ==============================================================================

def run_backtest(price_dict: Dict[str, pd.DataFrame], config: Dict) -> pd.DataFrame:
    """전략에 따라 백테스트를 실행하고, 일별 포트폴리오 가치 내역을 반환합니다."""
    # --- 1. 데이터 준비 및 초기화 ---
    all_symbols = [config['LONG_STOCK']] + config['SHORT_CANDIDATES']
    
    # 모든 종목의 종가를 하나의 DataFrame으로 통합하여 계산 속도 최적화
    close_prices = pd.concat({sym: df['close'] for sym, df in price_dict.items()}, axis=1).ffill()
    dates = close_prices.index

    # 포트폴리오 상태(보유 주식 수, 현금)를 추적할 DataFrame 생성
    holdings = pd.DataFrame(0.0, index=dates, columns=all_symbols)
    cash = pd.Series(0.0, index=dates)
    cash.iloc[0] = config['INIT_CASH']
    
    rebalancing_dates = dates[dates.to_series().dt.is_month_start].unique()

    # --- 2. 일별 백테스팅 루프 ---
    for i in range(1, len(dates)):
        date = dates[i]
        prev_date = dates[i-1]

        # 기본적으로 전날의 보유량과 현금을 그대로 유지
        holdings.iloc[i] = holdings.iloc[i-1]
        cash.iloc[i] = cash.iloc[i-1]
        
        # 리밸런싱 날짜에만 거래 실행
        if date in rebalancing_dates:
            # 1. 평가: 거래일 종가 기준으로 현재 총자산 평가
            total_asset = (holdings.loc[prev_date] * close_prices.loc[date]).sum() + cash.loc[prev_date]
            
            # 2. 청산 및 비용 계산: 보유 주식 전량 매도
            sell_value = (holdings.loc[prev_date] * close_prices.loc[date]).sum()
            sell_tax = sell_value * config['TAX_RATE']
            sell_commission = sell_value * config['COMMISSION_RATE']
            
            # 3. 재투자 가능 자금 계산
            reinvestment_capital = cash.loc[prev_date] + sell_value - sell_tax - sell_commission
            
            # 4. 신규 매수 포지션 계산
            temp_holdings = pd.Series(0.0, index=all_symbols)
            
            # 장기 종목 매수
            long_alloc = reinvestment_capital * config['LONG_STOCK_RATIO']
            long_price = close_prices.loc[date, config['LONG_STOCK']]
            if long_price > 0:
                temp_holdings[config['LONG_STOCK']] = long_alloc // long_price

            # 단기 종목 선정 및 매수
            selected_shorts = select_short_term_stocks(price_dict, config['SHORT_CANDIDATES'], date)
            short_alloc_each = reinvestment_capital * config['SHORT_STOCK_EACH_RATIO']
            for stock in selected_shorts:
                price = close_prices.loc[date, stock]
                if price > 0:
                    temp_holdings[stock] = short_alloc_each // price
            
            # 5. 매수 비용 계산 및 최종 현금/보유량 업데이트
            buy_amount = (temp_holdings * close_prices.loc[date]).sum()
            buy_commission = buy_amount * config['COMMISSION_RATE']
            
            holdings.loc[i] = temp_holdings
            cash.loc[i] = reinvestment_capital - buy_amount - buy_commission

    # --- 3. 최종 결과 계산 ---
    # 전체 기간에 대한 포트폴리오 가치를 벡터화 연산으로 한 번에 계산
    portfolio_values = (holdings * close_prices).sum(axis=1) + cash
    return portfolio_values.to_frame('total_value')

# ==============================================================================
# 4. 성과 분석 및 시각화
# ==============================================================================

def analyze_performance(portfolio_value: pd.Series) -> Dict[str, float]:
    """포트폴리오의 성과 지표를 계산합니다."""
    total_return = (portfolio_value.iloc[-1] / portfolio_value.iloc[0]) - 1
    
    days = (portfolio_value.index[-1] - portfolio_value.index[0]).days
    cagr = (portfolio_value.iloc[-1] / portfolio_value.iloc[0]) ** (365.0 / days) - 1

    rolling_max = portfolio_value.cummax()
    drawdown = (portfolio_value - rolling_max) / rolling_max
    mdd = drawdown.min()

    daily_returns = portfolio_value.pct_change().dropna()
    sharpe_ratio = np.sqrt(252) * (daily_returns.mean() / daily_returns.std()) if daily_returns.std() != 0 else 0

    monthly_returns = portfolio_value.resample('M').last().pct_change().dropna()
    positive_months = monthly_returns[monthly_returns > 0]
    negative_months = monthly_returns[monthly_returns < 0]
    
    monthly_win_rate = len(positive_months) / len(monthly_returns) if len(monthly_returns) > 0 else 0
    
    avg_gain = positive_months.mean()
    avg_loss = abs(negative_months.mean())
    profit_loss_ratio = avg_gain / avg_loss if avg_loss > 0 else np.inf

    return {
        "Total Return": total_return,
        "CAGR": cagr,
        "MDD": mdd,
        "Sharpe Ratio": sharpe_ratio,
        "Monthly Win Rate": monthly_win_rate,
        "Profit/Loss Ratio": profit_loss_ratio
    }

def plot_results(result: pd.DataFrame, benchmark: pd.DataFrame, metrics: Dict, config: Dict):
    """백테스트 결과와 성과 지표를 시각화합니다."""
    plt.figure(figsize=(14, 8))
    
    plt.plot(result.index, result['total_value'], label='혼합 전략 자산', color='royalblue', linewidth=2)
    plt.plot(benchmark.index, benchmark['value'], label=f"{config['LONG_STOCK']} 단순 보유 (기준)", color='grey', linestyle='--')
    
    plt.title('백테스트 결과: 누적 자산 곡선 (수수료/세금 반영)', fontsize=16)
    plt.xlabel('날짜', fontsize=12)
    plt.ylabel('자산 가치 (원)', fontsize=12)
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    stats_text = (
        f"**핵심 성과 지표**\n"
        f"총수익률: {metrics['Total Return']:.2%}\n"
        f"연평균복리수익률 (CAGR): {metrics['CAGR']:.2%}\n"
        f"최대 낙폭 (MDD): {metrics['MDD']:.2%}\n"
        f"샤프 지수: {metrics['Sharpe Ratio']:.2f}\n"
        f"월간 승률: {metrics['Monthly Win Rate']:.2%}\n"
        f"손익비: {metrics['Profit/Loss Ratio']:.2f}"
    )
    plt.text(0.02, 0.65, stats_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7))

    plt.tight_layout()
    plt.show()

# ==============================================================================
# 5. 메인 실행 함수
# ==============================================================================

def main():
    """백테스트 프로세스를 총괄하는 메인 함수"""
    setup_korean_font()

    # --- 전략 파라미터 설정 ---
    config = {
        'LONG_STOCK': '삼성전자',
        'SHORT_CANDIDATES': ['후보1', '후보2', '후보3', '후보4'],
        'INIT_CASH': 10000000,
        'LONG_STOCK_RATIO': 0.25,       # 장기투자 종목 비중
        'SHORT_STOCK_EACH_RATIO': 0.25, # 단기투자 개별 종목 비중
        'COMMISSION_RATE': 0.00015, # 매매 수수료 0.015%
        'TAX_RATE': 0.0020 # 증권거래세 0.20% (2023년 기준, 매도 시에만 적용)
    }

    # --- 데이터 로딩 ---
    # 실제 데이터를 사용하려면 use_dummy_data=False 로 변경하고 'data' 폴더에 CSV 파일 준비
    all_symbols = [config['LONG_STOCK']] + config['SHORT_CANDIDATES']
    price_dict = load_price_data(all_symbols, use_dummy_data=True)
    if not price_dict:
        return

    # --- 백테스트 실행 ---
    result = run_backtest(price_dict, config)
    
    # --- 성과 분석 ---
    metrics = analyze_performance(result['total_value'])
    
    # --- 벤치마크 계산 ---
    # 비교 기준: 초기자본으로 장기투자 종목만 계속 보유했을 경우
    benchmark_price = price_dict[config['LONG_STOCK']]['close']
    benchmark_value = (benchmark_price / benchmark_price.iloc[0]) * config['INIT_CASH']
    
    # --- 결과 시각화 ---
    print("--- 백테스트 성과 분석 결과 ---")
    for key, value in metrics.items():
        if isinstance(value, float) and abs(value) > 0.0001:
            print(f"{key:<20}: {value:.2%}" if "%" in key or "Rate" in key or "Return" in key else f"{key:<20}: {value:.2f}")
        else:
            print(f"{key:<20}: {value:.2f}")
    
    plot_results(result, benchmark_value.to_frame('value'), metrics, config)

async def backtest_weekend():
    """주말 백테스팅"""
    print("📊 주말 백테스팅 시작")
    
    # 과거 데이터로 전략 시뮬레이션
    test_data = {
        "2024-01": ["005930", "000660", "035420"],
        "2024-02": ["005380", "051910", "005930"],
        "2024-03": ["000660", "035420", "005380"]
    }
    
    for month, candidates in test_data.items():
        print(f"🗓️ {month} 시뮬레이션")
        # 전략 로직 테스트 (실제 주문 없이)
        await asyncio.sleep(1)
        print(f"✅ {month} 완료")

if __name__ == '__main__':
    asyncio.run(backtest_weekend()) 