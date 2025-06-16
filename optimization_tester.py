"""
파라미터 최적화 백테스터 (v3.1 - 효율성 개선)
- '상한가 포착' 전략의 최적 파라미터(손절, 익절, 트레일링 스탑)를 과학적으로 찾아냅니다.
- 반복적인 API 호출을 제거하여 'Quota Exceeded' 오류를 해결하고 테스트 속도를 향상시킵니다.
"""
import logging
import pandas as pd
from itertools import product
from datetime import datetime, timedelta
from core_trader import CoreTrader

# --- 로깅 설정 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OptimizationBacktester:
    """
    최적의 매매 파라미터를 찾기 위한 백테스팅 클래스
    - 데이터 로딩은 처음에 한 번만 수행하여 최적화 속도를 높입니다.
    """

    def __init__(self, stock_code, start_date, end_date, initial_cash=10_000_000):
        """
        백테스터를 초기화하고, 가장 무거운 작업인 데이터 로딩을 이 단계에서 한 번만 실행합니다.
        """
        self.trader = CoreTrader()
        self.stock_code = stock_code
        self.initial_cash = initial_cash
        
        logger.info(f"'{stock_code}' 종목의 과거 데이터를 로드합니다... (이 작업은 한 번만 수행됩니다)")
        self.historical_data = self._fetch_and_prepare_data(start_date, end_date)

    def _fetch_and_prepare_data(self, start_date, end_date):
        """지정된 기간의 과거 데이터를 한번만 불러와 가공 후 멤버 변수로 저장합니다."""
        try:
            # KIS API가 가져올 수 있는 최대 기간으로 데이터를 한번에 요청
            days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 5 
            # 'days'가 1000을 넘어도 API는 자동으로 최대 1000일치만 반환합니다.
            df = self.trader.get_daily_data(self.stock_code, days=days)
            
            if df is None or df.empty:
                logger.warning(f"'{self.stock_code}' 데이터를 가져오지 못했습니다.")
                return None

            df['stck_bsop_date'] = pd.to_datetime(df['stck_bsop_date'])
            df.set_index('stck_bsop_date', inplace=True)
            
            for col in ['stck_oprc', 'stck_hgpr', 'stck_lwpr', 'stck_clpr', 'acml_vol']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df.dropna(inplace=True)
            df = df.sort_index()
            df = df.loc[start_date:end_date]
            
            if len(df) < 20:
                logger.warning("백테스트를 위한 데이터가 부족합니다 (최소 20일 필요).")
                return None
                
            df['avg_vol_20'] = df['acml_vol'].rolling(window=20).mean()
            logger.info("✅ 데이터 로드 및 전처리 완료.")
            return df
        except Exception as e:
            logger.error(f"데이터 수집/처리 중 에러 발생: {e}")
            return None

    def _run_simulation(self, params):
        """
        미리 로드된 데이터를 사용하여, 주어진 파라미터로 시뮬레이션을 실행합니다.
        이 함수는 API 호출을 하지 않아 매우 빠릅니다.
        """
        cash = self.initial_cash
        holdings = {}
        df = self.historical_data.copy()

        for date, row in df.iterrows():
            if self.stock_code in holdings:
                purchase_price = holdings[self.stock_code]['purchase_price']
                current_high = holdings[self.stock_code]['high_price']
                if row['stck_hgpr'] > current_high:
                    holdings[self.stock_code]['high_price'] = row['stck_hgpr']
                    current_high = row['stck_hgpr']

                stop_loss_price = purchase_price * (1 - params['initial_stop_loss'] / 100)
                trailing_activation_price = purchase_price * (1 + params['trailing_activation'] / 100)
                trailing_stop_price = current_high * (1 - params['trailing_stop'] / 100)
                
                is_activated = current_high >= trailing_activation_price
                
                sell_price = 0
                if is_activated and row['stck_lwpr'] <= trailing_stop_price:
                    sell_price = trailing_stop_price
                elif not is_activated and row['stck_lwpr'] <= stop_loss_price:
                    sell_price = stop_loss_price

                if sell_price > 0:
                    quantity = holdings[self.stock_code]['qty']
                    proceeds = sell_price * quantity * (1 - 0.00015 - 0.002) # 수수료, 세금
                    cash += proceeds
                    del holdings[self.stock_code]
                    continue

            if self.stock_code not in holdings:
                is_volume_spike = row['acml_vol'] > (row.get('avg_vol_20', 0) * 5)
                is_strong_candle = row['stck_clpr'] > (row['stck_oprc'] * 1.15)
                
                if is_volume_spike and is_strong_candle:
                    buy_price = row['stck_clpr']
                    quantity = int((cash * 0.5) // buy_price)
                    if quantity > 0:
                        cost = buy_price * quantity * (1 + 0.00015)
                        cash -= cost
                        holdings[self.stock_code] = {'qty': quantity, 'purchase_price': buy_price, 'high_price': buy_price}
        
        final_value = cash
        if self.stock_code in holdings:
            last_price = df.iloc[-1]['stck_clpr']
            final_value += holdings[self.stock_code]['qty'] * last_price
        
        total_return = (final_value / self.initial_cash - 1) * 100
        return {"params": params, "total_return": total_return}

    def run_optimization(self, param_grid):
        """모든 파라미터 조합으로 백테스트를 실행하고 최적의 결과를 찾습니다."""
        if self.historical_data is None:
            logger.error("데이터가 없어 최적화를 실행할 수 없습니다.")
            return

        keys, values = zip(*param_grid.items())
        param_combinations = [dict(zip(keys, v)) for v in product(*values)]
        
        all_results = []
        logger.info(f"총 {len(param_combinations)}개의 파라미터 조합으로 최적화 시뮬레이션을 시작합니다...")

        for i, params in enumerate(param_combinations):
            result = self._run_simulation(params)
            if result:
                all_results.append(result)
        
        if not all_results:
            logger.warning("유효한 백테스트 결과가 없습니다.")
            return

        results_df = pd.DataFrame(all_results)
        results_df = results_df.sort_values(by="total_return", ascending=False).reset_index(drop=True)
        
        print("\n" + "="*80)
        print("🏆 파라미터 최적화 결과 (Top 5)")
        print(results_df.head(5).to_string())
        print("="*80)
        
        best_params = results_df.iloc[0]['params']
        best_return = results_df.iloc[0]['total_return']
        print(f"\n✅ 최적의 파라미터 조합: {best_params}")
        print(f"✅ 예상 최대 수익률: {best_return:.2f}%")


if __name__ == "__main__":
    # --- 설정 ---
    TEST_STOCK_CODE = "038460"
    
    # --- 기간 설정: 오늘을 기준으로 최대 1000일 전 데이터 사용 ---
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1000)

    TEST_START_DATE = start_date.strftime('%Y-%m-%d')
    TEST_END_DATE = end_date.strftime('%Y-%m-%d')
    
    logger.info(f"백테스트 기간을 최대로 설정합니다: {TEST_START_DATE} ~ {TEST_END_DATE} (약 1000일)")


    # 테스트할 파라미터 범위 정의
    PARAM_GRID = {
        'initial_stop_loss': [2, 3, 4],       # 초기 손절: -2% ~ -4%
        'trailing_activation': [3, 4, 5, 6],  # 감시 시작: +3% ~ +6%
        'trailing_stop': [2, 3, 4]            # 트레일링 스탑: -2% ~ -4%
    }
    
    # 최적화 실행
    optimizer = OptimizationBacktester(
        stock_code=TEST_STOCK_CODE,
        start_date=TEST_START_DATE,
        end_date=TEST_END_DATE
    )
    optimizer.run_optimization(param_grid=PARAM_GRID) 