"""
퀀트 분석 엔진 (All-in-One)
- 재무, 수급, 차트 팩터 기반의 상대평가 랭킹 시스템
"""
import pandas as pd
import yfinance as yf
from pykrx import stock
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

class QuantAnalyzer:
    """다중 팩터 기반의 종목 분석 및 랭킹을 담당하는 클래스"""

    def __init__(self, stock_list_df):
        self.stock_list = stock_list_df
        self.all_data = []
        
        # 팩터별 가중치
        self.weights = {
            'value': 0.30,      # 가치 (PER, PBR)
            'quality': 0.20,    # 퀄리티 (ROE)
            'supply': 0.30,     # 수급 (외국인, 기관)
            'momentum': 0.20    # 모멘텀 (차트 추세)
        }

    def run_analysis(self):
        """분석 실행 및 최종 랭킹 반환"""
        self._fetch_all_data_parallel()
        if not self.all_data:
            print("분석 가능한 데이터가 없습니다.")
            return None
        
        ranked_df = self._calculate_rank()
        return ranked_df

    def _fetch_all_data_parallel(self):
        """모든 종목의 데이터를 병렬로 수집"""
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = {
                executor.submit(self._fetch_single_stock_data, row['ticker'], row['name']): row['ticker']
                for _, row in self.stock_list.iterrows()
            }
            with tqdm(total=len(futures), desc="[1/2] 퀀트 데이터 수집 중") as progress:
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        self.all_data.append(result)
                    progress.update(1)

    def _fetch_single_stock_data(self, ticker, name):
        """개별 종목의 모든 팩터 데이터를 수집"""
        try:
            today_str = datetime.now().strftime("%Y%m%d")
            
            # 재무/가치 팩터 (PER, PBR, ROE)
            f_data = stock.get_market_fundamental(today_str, ticker, "y")
            if f_data.empty: return None
            per = f_data['PER'].iloc[-1]
            pbr = f_data['PBR'].iloc[-1]
            roe = f_data['ROE'].iloc[-1]

            # 수급 팩터 (최근 1개월 누적)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            s_data = stock.get_market_trading_value_by_date(start_date.strftime('%Y%m%d'), end_date.strftime('%Y%m%d'), ticker)
            foreign_net = s_data['외국인합계'].sum()
            inst_net = s_data['기관합계'].sum()

            # 모멘텀 팩터 (차트, 1년 데이터)
            c_data = yf.download(f"{ticker}.KS", period="1y", progress=False, show_errors=False)
            if c_data.empty or len(c_data) < 120: return None
            price = c_data['Close'].iloc[-1]
            ma60 = c_data['Close'].rolling(window=60).mean().iloc[-1]
            ma120 = c_data['Close'].rolling(window=120).mean().iloc[-1]
            is_uptrend = price > ma60 > ma120

            return {
                'ticker': ticker, 'name': name, 'price': price,
                'per': per if per > 0 else float('inf'),
                'pbr': pbr if pbr > 0 else float('inf'),
                'roe': roe,
                'foreign_net': foreign_net,
                'inst_net': inst_net,
                'momentum': 1 if is_uptrend else 0
            }
        except Exception:
            return None

    def _calculate_rank(self):
        """수집된 데이터를 바탕으로 팩터별 순위를 매기고 종합 순위를 계산"""
        df = pd.DataFrame(self.all_data)
        
        # 팩터별 순위 계산 (낮을수록 좋음: PER, PBR / 높을수록 좋음: 나머지)
        value_rank = df['per'].rank(ascending=True) + df['pbr'].rank(ascending=True)
        quality_rank = df['roe'].rank(ascending=False)
        supply_rank = df['foreign_net'].rank(ascending=False) + df['inst_net'].rank(ascending=False)
        momentum_rank = df['momentum'].rank(ascending=False)

        # 가중치를 적용하여 종합 점수 계산 (점수가 낮을수록 순위가 높음)
        df['total_score'] = (
            value_rank.rank() * self.weights['value'] +
            quality_rank.rank() * self.weights['quality'] +
            supply_rank.rank() * self.weights['supply'] +
            momentum_rank.rank() * self.weights['momentum']
        )
        
        # 최종 순위에 따라 정렬
        return df.sort_values(by='total_score', ascending=True) 