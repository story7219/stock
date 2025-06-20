import pandas as pd

class CanSlimScreener:
    """
    윌리엄 오닐의 CAN SLIM 투자 전략에 따라 종목을 필터링합니다.
    """
    def __init__(self, data_manager):
        """
        스크리너를 초기화합니다.
        
        Args:
            data_manager (DataManager): 데이터 조회를 위한 DataManager 인스턴스.
        """
        self.data_manager = data_manager

    def screen_stocks(self, stock_list: list) -> list:
        """
        주어진 종목 리스트에 대해 CAN SLIM 기준을 적용하여 유망 종목을 필터링합니다.
        
        Args:
            stock_list (list): 필터링할 전체 종목 코드 리스트.
            
        Returns:
            list: CAN SLIM 기준을 통과한 종목 코드 리스트.
        """
        screened_stocks = []
        
        # 1. 시장 방향성 확인 (M)
        if not self.check_market_direction():
            print("[CAN SLIM] 시장이 하락 추세이므로 스크리닝을 중단합니다.")
            return []

        for stock_code in stock_list:
            print(f"[CAN SLIM] '{stock_code}' 분석 중...")
            
            # 2. 실적 기준 확인 (C & A)
            if not self.check_earnings(stock_code):
                continue
            
            # 3. 주도주 및 신고가 확인 (N & L)
            if not self.check_leader_and_new_high(stock_code):
                continue
                
            # 4. 수급 확인 (S & I)
            if not self.check_supply_demand(stock_code):
                continue

            print(f"✅ '{stock_code}' CAN SLIM 기준 통과!")
            screened_stocks.append(stock_code)
            
        return screened_stocks

    def check_market_direction(self) -> bool:
        """ M: Market Direction - 시장이 상승 추세인지 확인합니다. """
        # 코스피 지수의 20일 이동평균선이 60일 이동평균선 위에 있는지 확인
        try:
            kospi_df = self.data_manager.fetch_market_data(['KOSPI'], 100)['KOSPI']['price_df']
            kospi_df['ma20'] = kospi_df['종가'].rolling(window=20).mean()
            kospi_df['ma60'] = kospi_df['종가'].rolling(window=60).mean()
            if kospi_df['ma20'].iloc[-1] > kospi_df['ma60'].iloc[-1]:
                return True
            return False
        except Exception as e:
            print(f"[오류] 시장 방향성 확인 실패: {e}")
            return True # 오류 발생 시 보수적으로 진행

    def check_earnings(self, stock_code: str) -> bool:
        """ C & A: Current & Annual Earnings - 분기 및 연간 실적 성장률을 확인합니다. """
        try:
            annual_df, quarterly_df = self.data_manager.get_financial_info_naver(stock_code)
            if annual_df is None or quarterly_df is None:
                return False

            # (A) 최근 3년간 연간 EPS가 꾸준히 증가했는지 확인
            annual_eps = annual_df.loc['EPS(원)'].dropna().astype(float)
            if len(annual_eps) < 3: return False
            if not (annual_eps.iloc[-1] > annual_eps.iloc[-2] > annual_eps.iloc[-3]):
                return False

            # (C) 최근 분기 EPS가 전년 동분기 대비 25% 이상 증가했는지 확인
            quarterly_eps = quarterly_df.loc['EPS(원)'].dropna().astype(float)
            if len(quarterly_eps) < 5: return False # 전년 동분기 비교를 위해 최소 5개 분기 데이터 필요
            
            latest_q_eps = quarterly_eps.iloc[-1]
            prev_year_q_eps = quarterly_eps.iloc[-5]
            
            if prev_year_q_eps <= 0: return latest_q_eps > 0 # 흑자 전환은 긍정적 신호

            eps_growth_rate = ((latest_q_eps - prev_year_q_eps) / abs(prev_year_q_eps)) * 100
            if eps_growth_rate < 25:
                return False
                
            return True
        except (KeyError, IndexError, ValueError) as e:
            # print(f"[{stock_code}] 실적 데이터 분석 오류: {e}")
            return False

    def check_leader_and_new_high(self, stock_code: str) -> bool:
        """ N & L: New Highs & Leader - 신고가 및 업종 주도주 여부를 확인합니다. """
        try:
            # (N) 52주 신고가 근처(15% 이내)에 있는지 확인
            price_df = self.data_manager.fetch_market_data([stock_code], 365)[stock_code]['price_df']
            if price_df.empty: return False
            
            high_52_week = price_df['고가'].max()
            current_price = price_df['종가'].iloc[-1]
            
            if not (current_price >= high_52_week * 0.85):
                return False

            # (L) 업종 내 상대강도(RS)가 상위권인지 확인 (현재는 단순 업종 정보만 확인)
            # TODO: 추후 동일 업종 타 종목과 비교하여 RS 점수 계산 로직 추가 필요
            industry = self.data_manager.get_stock_industry_info(stock_code)
            if not industry: # 업종 정보가 없으면 통과
                return True 
                
            return True # 현재는 업종 정보 존재 시 무조건 통과
        except Exception as e:
            # print(f"[{stock_code}] 주도주/신고가 분석 오류: {e}")
            return False
            
    def check_supply_demand(self, stock_code: str) -> bool:
        """ S & I: Supply/Demand & Institutional Sponsorship - 수급을 확인합니다. """
        try:
            # (S) 시가총액이 너무 크지 않은지 확인 (예: 20조원 이하)
            market_cap = self.data_manager.market_cap_df.loc[stock_code, '시가총액']
            if market_cap > 20_0000_0000_0000: # 20조
                return False
            
            # (I) 최근 한달간 기관 순매수가 양수인지 확인
            inst_buy = self.data_manager.get_institutional_buying_info(stock_code)
            if inst_buy is None or inst_buy < 0:
                return False

            return True
        except Exception as e:
            # print(f"[{stock_code}] 수급 분석 오류: {e}")
            return False 