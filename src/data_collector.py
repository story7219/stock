"""
데이터 수집 모듈
코스피200·나스닥100·S&P500 전체 종목 데이터 자동 수집
"""

import asyncio
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import aiohttp
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

logger = logging.getLogger(__name__)

@dataclass
class StockData:
    """주식 데이터 클래스"""
    symbol: str
    name: str
    price: float
    volume: int
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    moving_avg_20: Optional[float] = None
    moving_avg_60: Optional[float] = None
    rsi: Optional[float] = None
    bollinger_upper: Optional[float] = None
    bollinger_lower: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class KospiCollector:
    """코스피200 종목 데이터 수집기"""
    
    def __init__(self):
        self.base_url = "https://finance.naver.com"
        self.kospi200_url = "https://finance.naver.com/sise/sise_index_detail.naver?code=KPI200"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
    async def get_kospi200_symbols(self) -> List[str]:
        """코스피200 종목 리스트 가져오기"""
        try:
            # KRX 공식 API 사용 (실제 구현 시 API 키 필요)
            # 여기서는 네이버 금융을 통해 수집
            response = self.session.get(self.kospi200_url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 네이버 금융에서 코스피200 구성 종목 파싱
            symbols = []
            # 실제 구현에서는 정확한 셀렉터 사용
            stock_links = soup.select('a[href*="/item/main.nhn?code="]')
            
            for link in stock_links:
                href = link.get('href')
                if href and 'code=' in href:
                    symbol = href.split('code=')[1].split('&')[0]
                    if len(symbol) == 6 and symbol.isdigit():
                        symbols.append(f"{symbol}.KS")  # Yahoo Finance 형식
            
            # 최대 200개로 제한
            symbols = list(set(symbols))[:200]
            logger.info(f"코스피200 종목 {len(symbols)}개 수집 완료")
            return symbols
            
        except Exception as e:
            logger.error(f"코스피200 종목 리스트 수집 실패: {e}")
            # 백업용 하드코딩된 주요 종목들
            return [
                "005930.KS",  # 삼성전자
                "000660.KS",  # SK하이닉스
                "373220.KS",  # LG에너지솔루션
                "207940.KS",  # 삼성바이오로직스
                "005380.KS",  # 현대차
                "051910.KS",  # LG화학
                "035420.KS",  # NAVER
                "012330.KS",  # 현대모비스
                "028260.KS",  # 삼성물산
                "006400.KS",  # 삼성SDI
            ]
    
    async def collect_stock_data(self, symbol: str) -> Optional[StockData]:
        """개별 종목 데이터 수집"""
        try:
            # Yahoo Finance API 사용
            ticker = yf.Ticker(symbol)
            
            # 기본 정보
            info = ticker.info
            hist = ticker.history(period="3mo")  # 3개월 데이터
            
            if hist.empty:
                logger.warning(f"주식 데이터가 없습니다: {symbol}")
                return None
            
            current_price = hist['Close'].iloc[-1]
            volume = int(hist['Volume'].iloc[-1])
            
            # 기술적 지표 계산
            moving_avg_20 = hist['Close'].rolling(window=20).mean().iloc[-1]
            moving_avg_60 = hist['Close'].rolling(window=60).mean().iloc[-1]
            
            # RSI 계산
            rsi = self._calculate_rsi(hist['Close'])
            
            # 볼린저 밴드 계산
            bollinger_upper, bollinger_lower = self._calculate_bollinger_bands(hist['Close'])
            
            # MACD 계산
            macd, macd_signal = self._calculate_macd(hist['Close'])
            
            stock_data = StockData(
                symbol=symbol,
                name=info.get('shortName', symbol),
                price=float(current_price),
                volume=volume,
                market_cap=info.get('marketCap'),
                pe_ratio=info.get('trailingPE'),
                pb_ratio=info.get('priceToBook'),
                dividend_yield=info.get('dividendYield'),
                moving_avg_20=float(moving_avg_20) if pd.notna(moving_avg_20) else None,
                moving_avg_60=float(moving_avg_60) if pd.notna(moving_avg_60) else None,
                rsi=float(rsi) if pd.notna(rsi) else None,
                bollinger_upper=float(bollinger_upper) if pd.notna(bollinger_upper) else None,
                bollinger_lower=float(bollinger_lower) if pd.notna(bollinger_lower) else None,
                macd=float(macd) if pd.notna(macd) else None,
                macd_signal=float(macd_signal) if pd.notna(macd_signal) else None
            )
            
            return stock_data
            
        except Exception as e:
            logger.error(f"종목 데이터 수집 실패 {symbol}: {e}")
            return None
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """RSI 계산"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1]
        except:
            return np.nan
    
    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, num_std: int = 2) -> Tuple[float, float]:
        """볼린저 밴드 계산"""
        try:
            rolling_mean = prices.rolling(window=window).mean()
            rolling_std = prices.rolling(window=window).std()
            upper_band = rolling_mean + (rolling_std * num_std)
            lower_band = rolling_mean - (rolling_std * num_std)
            return upper_band.iloc[-1], lower_band.iloc[-1]
        except:
            return np.nan, np.nan
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float]:
        """MACD 계산"""
        try:
            exp1 = prices.ewm(span=fast).mean()
            exp2 = prices.ewm(span=slow).mean()
            macd = exp1 - exp2
            signal_line = macd.ewm(span=signal).mean()
            return macd.iloc[-1], signal_line.iloc[-1]
        except:
            return np.nan, np.nan
    
    async def collect_all_data(self) -> List[StockData]:
        """모든 코스피200 종목 데이터 수집"""
        symbols = await self.get_kospi200_symbols()
        
        tasks = []
        for symbol in symbols:
            task = self.collect_stock_data(symbol)
            tasks.append(task)
            
        # 병렬 처리로 데이터 수집
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 성공한 데이터만 필터링
        stock_data_list = []
        for result in results:
            if isinstance(result, StockData):
                stock_data_list.append(result)
        
        logger.info(f"코스피200 종목 데이터 {len(stock_data_list)}개 수집 완료")
        return stock_data_list

class NasdaqCollector:
    """나스닥100 종목 데이터 수집기"""
    
    def __init__(self):
        self.nasdaq100_symbols = [
            "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "META", "TSLA", "NVDA",
            "PYPL", "ADBE", "NFLX", "INTC", "CMCSA", "PEP", "CSCO", "TMUS",
            "COST", "AVGO", "TXN", "QCOM", "CHTR", "SBUX", "INTU", "AMAT",
            "ISRG", "BKNG", "MDLZ", "GILD", "AMD", "MU", "ADP", "ADI",
            "LRCX", "REGN", "FISV", "CSX", "ATVI", "MELI", "KLAC", "SNPS",
            "NXPI", "ORLY", "CDNS", "MCHP", "WDAY", "CTAS", "MNST", "PAYX"
        ]
        
    async def get_nasdaq100_symbols(self) -> List[str]:
        """나스닥100 종목 리스트 가져오기"""
        try:
            # 실제로는 나스닥 공식 API를 사용해야 함
            # 여기서는 하드코딩된 주요 종목 사용
            logger.info(f"나스닥100 종목 {len(self.nasdaq100_symbols)}개 준비 완료")
            return self.nasdaq100_symbols
        except Exception as e:
            logger.error(f"나스닥100 종목 리스트 가져오기 실패: {e}")
            return self.nasdaq100_symbols[:50]  # 백업으로 상위 50개만
    
    async def collect_stock_data(self, symbol: str) -> Optional[StockData]:
        """개별 종목 데이터 수집 (코스피와 동일한 로직)"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="3mo")
            
            if hist.empty:
                logger.warning(f"주식 데이터가 없습니다: {symbol}")
                return None
            
            current_price = hist['Close'].iloc[-1]
            volume = int(hist['Volume'].iloc[-1])
            
            # 기술적 지표 계산 (코스피와 동일)
            kospi_collector = KospiCollector()
            moving_avg_20 = hist['Close'].rolling(window=20).mean().iloc[-1]
            moving_avg_60 = hist['Close'].rolling(window=60).mean().iloc[-1]
            rsi = kospi_collector._calculate_rsi(hist['Close'])
            bollinger_upper, bollinger_lower = kospi_collector._calculate_bollinger_bands(hist['Close'])
            macd, macd_signal = kospi_collector._calculate_macd(hist['Close'])
            
            stock_data = StockData(
                symbol=symbol,
                name=info.get('shortName', symbol),
                price=float(current_price),
                volume=volume,
                market_cap=info.get('marketCap'),
                pe_ratio=info.get('trailingPE'),
                pb_ratio=info.get('priceToBook'),
                dividend_yield=info.get('dividendYield'),
                moving_avg_20=float(moving_avg_20) if pd.notna(moving_avg_20) else None,
                moving_avg_60=float(moving_avg_60) if pd.notna(moving_avg_60) else None,
                rsi=float(rsi) if pd.notna(rsi) else None,
                bollinger_upper=float(bollinger_upper) if pd.notna(bollinger_upper) else None,
                bollinger_lower=float(bollinger_lower) if pd.notna(bollinger_lower) else None,
                macd=float(macd) if pd.notna(macd) else None,
                macd_signal=float(macd_signal) if pd.notna(macd_signal) else None
            )
            
            return stock_data
            
        except Exception as e:
            logger.error(f"종목 데이터 수집 실패 {symbol}: {e}")
            return None
    
    async def collect_all_data(self) -> List[StockData]:
        """모든 나스닥100 종목 데이터 수집"""
        symbols = await self.get_nasdaq100_symbols()
        
        tasks = []
        for symbol in symbols:
            task = self.collect_stock_data(symbol)
            tasks.append(task)
            
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        stock_data_list = []
        for result in results:
            if isinstance(result, StockData):
                stock_data_list.append(result)
        
        logger.info(f"나스닥100 종목 데이터 {len(stock_data_list)}개 수집 완료")
        return stock_data_list

class SP500Collector:
    """S&P500 종목 데이터 수집기"""
    
    def __init__(self):
        # S&P500 주요 종목들 (실제로는 API에서 가져와야 함)
        self.sp500_symbols = [
            "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "TSLA", "BRK-B", "UNH", 
            "JNJ", "XOM", "JPM", "V", "PG", "MA", "CVX", "HD", "PFE", "ABBV",
            "BAC", "KO", "AVGO", "PEP", "TMO", "COST", "WMT", "DIS", "ABT",
            "DHR", "VZ", "ADBE", "CRM", "ACN", "MRK", "TXN", "LIN", "NKE",
            "WFC", "NEE", "RTX", "QCOM", "UPS", "T", "ORCL", "PM", "MS",
            "COP", "LOW", "HON", "INTU", "UNP", "IBM", "AMGN", "SPGI", "GS",
            "CAT", "AXP", "BLK", "DE", "LMT", "AMD", "MDT", "BKNG", "TGT",
            "ISRG", "SCHW", "PLD", "ADP", "SYK", "TJX", "CVS", "MDLZ", "CI",
            "GILD", "MO", "SO", "ZTS", "CB", "DUK", "BSX", "MMC", "CME",
            "TMUS", "EL", "PYPL", "ITW", "ICE", "EQIX", "PNC", "AON", "CL",
            "APD", "GM", "SHW", "USB", "GD", "NFLX", "EMR", "NSC", "HUM",
            "MU", "INTC", "F", "MCO", "FCX", "TFC", "ATVI", "COF", "PSA"
        ]
        
    async def get_sp500_symbols(self) -> List[str]:
        """S&P500 종목 리스트 가져오기"""
        try:
            # 실제로는 S&P Global API나 Wikipedia에서 파싱해야 함
            # 여기서는 하드코딩된 주요 종목 사용
            logger.info(f"S&P500 종목 {len(self.sp500_symbols)}개 준비 완료")
            return self.sp500_symbols
        except Exception as e:
            logger.error(f"S&P500 종목 리스트 가져오기 실패: {e}")
            return self.sp500_symbols[:100]  # 백업으로 상위 100개만
    
    async def collect_stock_data(self, symbol: str) -> Optional[StockData]:
        """개별 종목 데이터 수집"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="3mo")
            
            if hist.empty:
                logger.warning(f"주식 데이터가 없습니다: {symbol}")
                return None
            
            current_price = hist['Close'].iloc[-1]
            volume = int(hist['Volume'].iloc[-1])
            
            # 기술적 지표 계산
            kospi_collector = KospiCollector()
            moving_avg_20 = hist['Close'].rolling(window=20).mean().iloc[-1]
            moving_avg_60 = hist['Close'].rolling(window=60).mean().iloc[-1]
            rsi = kospi_collector._calculate_rsi(hist['Close'])
            bollinger_upper, bollinger_lower = kospi_collector._calculate_bollinger_bands(hist['Close'])
            macd, macd_signal = kospi_collector._calculate_macd(hist['Close'])
            
            stock_data = StockData(
                symbol=symbol,
                name=info.get('shortName', symbol),
                price=float(current_price),
                volume=volume,
                market_cap=info.get('marketCap'),
                pe_ratio=info.get('trailingPE'),
                pb_ratio=info.get('priceToBook'),
                dividend_yield=info.get('dividendYield'),
                moving_avg_20=float(moving_avg_20) if pd.notna(moving_avg_20) else None,
                moving_avg_60=float(moving_avg_60) if pd.notna(moving_avg_60) else None,
                rsi=float(rsi) if pd.notna(rsi) else None,
                bollinger_upper=float(bollinger_upper) if pd.notna(bollinger_upper) else None,
                bollinger_lower=float(bollinger_lower) if pd.notna(bollinger_lower) else None,
                macd=float(macd) if pd.notna(macd) else None,
                macd_signal=float(macd_signal) if pd.notna(macd_signal) else None
            )
            
            return stock_data
            
        except Exception as e:
            logger.error(f"종목 데이터 수집 실패 {symbol}: {e}")
            return None
    
    async def collect_all_data(self) -> List[StockData]:
        """모든 S&P500 종목 데이터 수집"""
        symbols = await self.get_sp500_symbols()
        
        tasks = []
        for symbol in symbols:
            task = self.collect_stock_data(symbol)
            tasks.append(task)
            
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        stock_data_list = []
        for result in results:
            if isinstance(result, StockData):
                stock_data_list.append(result)
        
        logger.info(f"S&P500 종목 데이터 {len(stock_data_list)}개 수집 완료")
        return stock_data_list

class DataCollector:
    """전체 데이터 수집 관리자"""
    
    def __init__(self):
        self.kospi_collector = KospiCollector()
        self.nasdaq_collector = NasdaqCollector()
        self.sp500_collector = SP500Collector()
        
    async def collect_all_market_data(self) -> Dict[str, List[StockData]]:
        """코스피200 + 나스닥100 + S&P500 전체 데이터 수집"""
        logger.info("전체 시장 데이터 수집 시작 (코스피200, 나스닥100, S&P500)")
        
        # 병렬로 세 시장 데이터 수집
        kospi_task = self.kospi_collector.collect_all_data()
        nasdaq_task = self.nasdaq_collector.collect_all_data()
        sp500_task = self.sp500_collector.collect_all_data()
        
        kospi_data, nasdaq_data, sp500_data = await asyncio.gather(
            kospi_task, nasdaq_task, sp500_task
        )
        
        result = {
            'kospi200': kospi_data,
            'nasdaq100': nasdaq_data,
            'sp500': sp500_data
        }
        
        total_stocks = len(kospi_data) + len(nasdaq_data) + len(sp500_data)
        logger.info(f"전체 시장 데이터 수집 완료: {total_stocks}개 종목 (코스피200: {len(kospi_data)}, 나스닥100: {len(nasdaq_data)}, S&P500: {len(sp500_data)})")
        
        return result
    
    def to_dataframe(self, market_data: Dict[str, List[StockData]]) -> pd.DataFrame:
        """수집된 데이터를 DataFrame으로 변환"""
        all_data = []
        
        for market, stocks in market_data.items():
            for stock in stocks:
                stock_dict = {
                    'market': market,
                    'symbol': stock.symbol,
                    'name': stock.name,
                    'price': stock.price,
                    'volume': stock.volume,
                    'market_cap': stock.market_cap,
                    'pe_ratio': stock.pe_ratio,
                    'pb_ratio': stock.pb_ratio,
                    'dividend_yield': stock.dividend_yield,
                    'moving_avg_20': stock.moving_avg_20,
                    'moving_avg_60': stock.moving_avg_60,
                    'rsi': stock.rsi,
                    'bollinger_upper': stock.bollinger_upper,
                    'bollinger_lower': stock.bollinger_lower,
                    'macd': stock.macd,
                    'macd_signal': stock.macd_signal,
                    'timestamp': stock.timestamp
                }
                all_data.append(stock_dict)
        
        df = pd.DataFrame(all_data)
        
        # 데이터 타입 최적화
        numeric_columns = ['price', 'volume', 'market_cap', 'pe_ratio', 'pb_ratio', 
                          'dividend_yield', 'moving_avg_20', 'moving_avg_60', 'rsi',
                          'bollinger_upper', 'bollinger_lower', 'macd', 'macd_signal']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 결측치 처리
        df = self._handle_missing_values(df)
        
        logger.info(f"DataFrame 생성 완료: {len(df)}행 x {len(df.columns)}열")
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """결측치 자동 보정"""
        try:
            # 가격 관련 결측치는 전체 평균으로 대체
            price_columns = ['pe_ratio', 'pb_ratio', 'dividend_yield']
            for col in price_columns:
                if col in df.columns:
                    df[col] = df[col].fillna(df[col].median())
            
            # 기술적 지표 결측치는 0으로 대체
            technical_columns = ['rsi', 'macd', 'macd_signal']
            for col in technical_columns:
                if col in df.columns:
                    df[col] = df[col].fillna(0)
            
            # 이동평균은 현재가격으로 대체
            if 'moving_avg_20' in df.columns:
                df['moving_avg_20'] = df['moving_avg_20'].fillna(df['price'])
            if 'moving_avg_60' in df.columns:
                df['moving_avg_60'] = df['moving_avg_60'].fillna(df['price'])
            
            # 볼린저 밴드는 현재가격으로 대체
            if 'bollinger_upper' in df.columns:
                df['bollinger_upper'] = df['bollinger_upper'].fillna(df['price'] * 1.02)
            if 'bollinger_lower' in df.columns:
                df['bollinger_lower'] = df['bollinger_lower'].fillna(df['price'] * 0.98)
            
            logger.info("결측치 자동 보정 완료")
            return df
            
        except Exception as e:
            logger.error(f"결측치 처리 실패: {e}")
            return df 