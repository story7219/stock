from typing import Dict, Optional
import pandas as pd
from pypfopt import EfficientFrontier, risk_models, expected_returns

class StrategyGenerator:
    """
    AI 분석 결과를 바탕으로, 다양한 투자 전략에 따라
    최적의 포트폴리오(종목별 비중)를 생성합니다.
    """
    def __init__(self):
        pass

    def generate_portfolio(
        self, 
        prices: pd.DataFrame, 
        strategy_name: str = 'max_sharpe'
    ) -> Optional[Dict[str, float]]:
        """
        주어진 가격 데이터와 전략에 따라 포트폴리오를 생성합니다.

        Args:
            prices (pd.DataFrame): 종목별 과거 가격 데이터. 각 열이 종목 코드.
            strategy_name (str): 사용할 전략 이름. ('max_sharpe', 'equal_weight_with_cash')

        Returns:
            Optional[Dict[str, float]]: 종목 코드와 비중을 담은 딕셔너리.
        """
        if prices.empty:
            print("오류: 포트폴리오 생성을 위한 가격 데이터가 없습니다.")
            return None

        if strategy_name == "equal_weight_with_cash":
            return self._equal_weight_with_cash_strategy(prices)
        
        elif strategy_name == "max_sharpe":
            return self._max_sharpe_strategy(prices)
        
        else:
            print(f"오류: '{strategy_name}'은(는) 지원하지 않는 전략입니다.")
            return None

    def _equal_weight_with_cash_strategy(self, prices: pd.DataFrame) -> Dict[str, float]:
        """
        사용자 정의 규칙에 따라 종목별 가중치를 할당합니다.
        - 4종목: 각 20%
        - 3종목: 각 25%
        - 기타: 동일 비중으로 할당 (현금 제외)
        """
        num_stocks = len(prices.columns)
        stock_codes = prices.columns.tolist()
        
        print(f"'{num_stocks}개 종목'에 대한 사용자 정의 동일 가중치 전략을 적용합니다.")

        if num_stocks == 4:
            weight = 0.20
            cash_weight = 0.20
        elif num_stocks == 3:
            weight = 0.25
            cash_weight = 0.25
        else:
            print(f"경고: 3, 4종목 외의 경우({num_stocks}개) 현금 비중 없이 동일 가중치를 적용합니다.")
            weight = 1.0 / num_stocks
            cash_weight = 0.0

        weights = {code: weight for code in stock_codes}
        
        print(f"  - 종목별 할당 비중: {weight*100:.1f}%")
        print(f"  - 현금 비중: {cash_weight*100:.1f}%")
        
        return weights

    def _max_sharpe_strategy(self, prices: pd.DataFrame) -> Optional[Dict[str, float]]:
        """
        최대 샤프 지수(Max Sharpe Ratio)를 목표로 포트폴리오를 최적화합니다.
        """
        print("최대 샤프 지수(Max Sharpe) 최적화 전략을 적용합니다.")
        try:
            # 기대 수익률과 공분산 행렬 계산
            mu = expected_returns.mean_historical_return(prices)
            S = risk_models.sample_cov(prices)

            # 최적화 수행
            ef = EfficientFrontier(mu, S)
            raw_weights = ef.max_sharpe()
            cleaned_weights = ef.clean_weights()

            return cleaned_weights
        except Exception as e:
            print(f"최대 샤프 지수 최적화 중 오류 발생: {e}")
            return None 