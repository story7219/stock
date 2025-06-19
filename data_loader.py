"""
주식 데이터 로딩을 담당하는 모듈
- yfinance API를 사용하여 주식 데이터를 가져옵니다.
"""
import yfinance as yf
import pandas as pd
from typing import Optional

class DataLoader:
    """주식 데이터를 로드하는 클래스"""

    def __init__(self, period: str = '3mo'):
        self.default_period = period
        self._cache = {}

    def fetch_stock_data(self, symbol: str, period: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        yfinance를 통해 주식 데이터를 가져오고 캐시합니다.

        :param symbol: 주식 티커
        :param period: 데이터 기간 (e.g., '1y', '3mo'). 지정하지 않으면 기본값을 사용합니다.
        :return: 주식 데이터프레임 또는 None
        """
        fetch_period = period or self.default_period
        cache_key = f"{symbol}_{fetch_period}"

        if cache_key in self._cache:
            return self._cache[cache_key]
            
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=fetch_period)
            if data.empty:
                print(f"Warning: No data found for symbol {symbol} with period {fetch_period}")
                return None
            self._cache[cache_key] = data
            return data
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None 