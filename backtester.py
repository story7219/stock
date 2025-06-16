"""
상한가 포착 전략 백테스터 (v1.0)
- 과거 일봉 데이터를 기반으로 '상한가 포착' 전략의 수익성을 검증합니다.
"""
import logging
import pandas as pd
from core_trader import CoreTrader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Backtester:
    def __init__(self, stock_code, start_date, end_date):
        self.trader = CoreTrader()
        self.stock_code = stock_code
        self.start_date = start_date
        self.end_date = end_date
        
        # 가상 포트폴리오 설정
        self.initial_cash = 10_000_000  # 초기 자본금: 1천만원
        self.cash = self.initial_cash
        self.holdings = {} # 보유 주식: { '종목코드': {'qty': 수량, 'purchase_price': 매수가} }
        self.trade_history = []
        self.commission_rate = 0.00015 # 수수료

    def fetch_historical_data(self):
        """과거 일봉 데이터 수집"""
        logger.info(f"💾 {self.stock_code}의 과거 데이터 수집 중 ({self.start_date} ~ {self.end_date})...")
        # get_daily_data는 최대 100일치만 가져오므로, 여러 번 호출해야 함
        # 여기서는 설명을 위해 단순화된 형태로 가정합니다.
        # 실제로는 KIS '기간별 시세' API를 사용해야 합니다.
        # 이 예제에서는 CoreTrader에 유사한 함수가 있다고 가정합니다.
        try:
            # 설명을 위해 임시로 get_daily_data를 사용. 실제로는 기간지정 API 필요
            df = self.trader.get_daily_data(self.stock_code, days=730) # 약 2년치
            df['stck_bsop_date'] = pd.to_datetime(df['stck_bsop_date'])
            # 날짜 컬럼을 인덱스로 설정
            df.set_index('stck_bsop_date', inplace=True)
            # 숫자형으로 변환
            for col in ['stck_oprc', 'stck_hgpr', 'stck_lwpr', 'stck_clpr', 'acml_vol']:
                df[col] = pd.to_numeric(df[col])
            logger.info(f"✅ 데이터 수집 완료. 총 {len(df)}일")
            return df.sort_index()
        except Exception as e:
            logger.error(f"데이터 수집 실패: {e}")
            return None

    def run_backtest(self):
        """백테스트 실행"""
        df = self.fetch_historical_data()
        if df is None:
            return

        df['avg_vol_20'] = df['acml_vol'].rolling(window=20).mean()
        
        logger.info("🚀 백테스트 시작...")

        for date, row in df.iterrows():
            # 1. 매도 조건 확인 (익절/손절)
            if self.stock_code in self.holdings:
                purchase_price = self.holdings[self.stock_code]['purchase_price']
                take_profit_price = purchase_price * 1.05 # +5% 익절
                stop_loss_price = purchase_price * 0.98  # -2% 손절

                if row['stck_hgpr'] >= take_profit_price:
                    self.simulate_order('SELL', take_profit_price, date)
                    continue
                elif row['stck_lwpr'] <= stop_loss_price:
                    self.simulate_order('SELL', stop_loss_price, date)
                    continue

            # 2. 매수 조건 확인
            if self.stock_code not in self.holdings:
                # 조건1: 거래량 폭증 (20일 평균의 5배 이상)
                is_volume_spike = row['acml_vol'] > (row['avg_vol_20'] * 5)
                # 조건2: 장대 양봉 (시가 대비 15% 이상 상승)
                is_strong_candle = row['stck_clpr'] > (row['stck_oprc'] * 1.15)
                
                if is_volume_spike and is_strong_candle:
                    self.simulate_order('BUY', row['stck_clpr'], date)

        self.print_results()

    def simulate_order(self, order_type, price, date):
        """가상 주문 처리 및 포트폴리오 업데이트"""
        if order_type == 'BUY':
            quantity = int((self.cash * 0.5) // price) # 현금의 50% 매수
            if quantity > 0:
                cost = price * quantity * (1 + self.commission_rate)
                self.cash -= cost
                self.holdings[self.stock_code] = {'qty': quantity, 'purchase_price': price}
                self.trade_history.append({'date': date, 'type': 'BUY', 'price': price, 'qty': quantity})
                logger.info(f"  -> [매수] 날짜: {date.strftime('%Y-%m-%d')}, 가격: {price:,.0f}, 수량: {quantity}")

        elif order_type == 'SELL':
            if self.stock_code in self.holdings:
                quantity = self.holdings[self.stock_code]['qty']
                proceeds = price * quantity * (1 - self.commission_rate)
                self.cash += proceeds
                
                purchase_price = self.holdings[self.stock_code]['purchase_price']
                profit = (price - purchase_price) * quantity
                self.trade_history.append({'date': date, 'type': 'SELL', 'price': price, 'qty': quantity, 'profit': profit})
                logger.info(f"  -> [매도] 날짜: {date.strftime('%Y-%m-%d')}, 가격: {price:,.0f}, 수익: {profit:,.0f}원")
                del self.holdings[self.stock_code]

    def print_results(self):
        """백테스트 결과 출력"""
        logger.info("\n" + "="*50)
        logger.info("📈 백테스트 결과")
        
        final_value = self.cash
        if self.stock_code in self.holdings:
            # 백테스트 종료 시점에 보유 중인 주식은 마지막 날 종가로 평가
            last_price = self.fetch_historical_data().iloc[-1]['stck_clpr']
            final_value += self.holdings[self.stock_code]['qty'] * last_price

        total_return = (final_value / self.initial_cash - 1) * 100
        
        buys = [t for t in self.trade_history if t['type'] == 'BUY']
        sells = [t for t in self.trade_history if t['type'] == 'SELL']
        wins = [t for t in sells if t['profit'] > 0]
        
        win_rate = (len(wins) / len(sells)) * 100 if sells else 0
        total_profit = sum(t['profit'] for t in sells)

        print(f" - 테스트 기간: {self.start_date} ~ {self.end_date}")
        print(f" - 초기 자본: {self.initial_cash:,.0f}원")
        print(f" - 최종 자산: {final_value:,.0f}원")
        print(f" - 총 수익률: {total_return:.2f}%")
        print(f" - 총 손익: {total_profit:,.0f}원")
        print(f" - 총 거래 횟수 (매수 기준): {len(buys)}회")
        print(f" - 승률: {win_rate:.2f}% ({len(wins)}승 / {len(sells) - len(wins)}패)")
        logger.info("="*50 + "\n")


if __name__ == "__main__":
    # 백테스트할 종목 코드와 기간 설정
    test_stock_code = "038460"  # 예시: 바이오스마트
    test_start_date = "2022-01-01"
    test_end_date = "2023-12-31"

    backtester = Backtester(stock_code=test_stock_code, start_date=test_start_date, end_date=test_end_date)
    backtester.run_backtest() 