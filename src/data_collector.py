"""
데이터 수집 모듈
코스피200·나스닥100·S&P500 전체 종목 데이터 자동 수집
🚀 Gemini AI 최적화를 위한 고품질 데이터 가공 시스템
"""

import asyncio
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
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
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class DataQualityMetrics:
    """데이터 품질 지표"""
    completeness_score: float = 0.0  # 완성도 점수 (0-100)
    accuracy_score: float = 0.0      # 정확도 점수 (0-100)
    consistency_score: float = 0.0   # 일관성 점수 (0-100)
    timeliness_score: float = 0.0    # 시의성 점수 (0-100)
    overall_quality: float = 0.0     # 전체 품질 점수 (0-100)
    missing_data_ratio: float = 0.0  # 결측치 비율
    outlier_ratio: float = 0.0       # 이상치 비율
    data_freshness_hours: float = 0.0 # 데이터 신선도 (시간)

@dataclass
class StockData:
    """주식 데이터 클래스 - Gemini AI 최적화"""
    # 기본 정보
    symbol: str
    name: str
    price: float
    volume: int
    market_cap: Optional[float] = None

    # 가치 지표
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    roe: Optional[float] = None
    debt_ratio: Optional[float] = None

    # 기술적 지표
    moving_avg_20: Optional[float] = None
    moving_avg_60: Optional[float] = None
    rsi: Optional[float] = None
    bollinger_upper: Optional[float] = None
    bollinger_lower: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None

    # 고급 기술적 지표 (Gemini AI 최적화)
    stochastic_k: Optional[float] = None
    stochastic_d: Optional[float] = None
    williams_r: Optional[float] = None
    atr: Optional[float] = None  # Average True Range
    adx: Optional[float] = None  # Average Directional Index

    # 가격 동향 분석
    price_change_1d: Optional[float] = None
    price_change_5d: Optional[float] = None
    price_change_20d: Optional[float] = None
    volume_ratio_20d: Optional[float] = None
    volatility_20d: Optional[float] = None

    # 시장 상대 성과
    market_beta: Optional[float] = None
    relative_strength: Optional[float] = None

    # 품질 및 메타데이터
    data_quality: DataQualityMetrics = field(default_factory=DataQualityMetrics)
    timestamp: datetime = None
    data_source: str = "yfinance"

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def calculate_quality_score(self) -> float:
        """데이터 품질 점수 계산"""
        try:
            # 필수 필드 완성도 확인
            required_fields = [self.price, self.volume, self.symbol, self.name]
            completeness = sum(1 for field in required_fields if field is not None) / len(required_fields)

            # 기술적 지표 완성도
            technical_fields = [self.rsi, self.macd, self.moving_avg_20, self.bollinger_upper]
            technical_completeness = sum(1 for field in technical_fields if field is not None) / len(technical_fields)

            # 전체 품질 점수
            overall_quality = (completeness * 0.6 + technical_completeness * 0.4) * 100

            self.data_quality.completeness_score = completeness * 100
            self.data_quality.overall_quality = overall_quality

            return overall_quality

        except Exception as e:
            logger.warning(f"품질 점수 계산 실패 {self.symbol}: {e}")
            return 0.0

class KospiCollector:
    """코스피200 데이터 수집기 - Gemini AI 최적화"""

    def __init__(self):
        self.base_url = "https://finance.naver.com"
        self.kospi200_url = "https://finance.naver.com/sise/sise_index_detail.naver?code=KPI200"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

    async def get_kospi200_symbols(self) -> List[str]:
        """코스피200 종목 리스트 가져오기"""
        try:
            # 백업용 주요 코스피200 종목들 (실제 운영에서는 KRX API 사용)
            major_kospi200_symbols = [
                "005930",  # 삼성전자
                "000660",  # SK하이닉스
                "373220",  # LG에너지솔루션
                "207940",  # 삼성바이오로직스
                "005380",  # 현대차
                "051910",  # LG화학
                "035420",  # NAVER
                "012330",  # 현대모비스
                "028260",  # 삼성물산
                "006400",  # 삼성SDI
                "003670",  # 포스코홀딩스
                "017670",  # SK텔레콤
                "096770",  # SK이노베이션
                "034730",  # SK
                "018260",  # 삼성에스디에스
                "009150",  # 삼성전기
                "010950",  # S-Oil
                "032830",  # 삼성생명
                "066570",  # LG전자
                "323410",  # 카카오뱅크
                "035720",  # 카카오
                "068270",  # 셀트리온
                "091990",  # 셀트리온헬스케어
                "196170",  # 알테오젠
                "302440",  # SK바이오사이언스
                "086790",  # 하나금융지주
                "105560",  # KB금융
                "055550",  # 신한지주
                "024110",  # 기업은행
                "316140",  # 우리금융지주
            ]
            
            # Yahoo Finance 형식으로 변환
            symbols = [f"{symbol}.KS" for symbol in major_kospi200_symbols]
            logger.info(f"코스피200 주요 종목 {len(symbols)}개 로드 완료")
            return symbols

        except Exception as e:
            logger.error(f"코스피200 종목 리스트 수집 실패: {e}")
            # 최소한의 백업 종목들
            return [
                "005930.KS",  # 삼성전자
                "000660.KS",  # SK하이닉스
                "373220.KS",  # LG에너지솔루션
                "207940.KS",  # 삼성바이오로직스
                "005380.KS",  # 현대차
            ]

    async def collect_stock_data(self, symbol: str) -> Optional[StockData]:
        """개별 종목 데이터 수집 - Gemini AI 최적화"""
        try:
            # Yahoo Finance에서 기본 데이터 수집
            # 심볼이 이미 .KS로 끝나면 그대로 사용, 아니면 .KS 추가
            if symbol.endswith('.KS'):
                ticker_symbol = symbol
            else:
                ticker_symbol = f"{symbol}.KS"
                
            ticker = yf.Ticker(ticker_symbol)
            info = ticker.info
            hist = ticker.history(period="3mo")

            if hist.empty:
                logger.warning(f"주식 데이터가 없습니다: {symbol}")
                return None

            current_price = hist['Close'].iloc[-1]
            volume = int(hist['Volume'].iloc[-1])

            # 기본 기술적 지표 계산
            moving_avg_20 = hist['Close'].rolling(window=20).mean().iloc[-1]
            moving_avg_60 = hist['Close'].rolling(window=60).mean().iloc[-1]
            rsi = self._calculate_rsi(hist['Close'])
            bollinger_upper, bollinger_lower = self._calculate_bollinger_bands(hist['Close'])
            macd, macd_signal = self._calculate_macd(hist['Close'])

            # 고급 기술적 지표 계산 (Gemini AI 최적화)
            advanced_indicators = self._calculate_advanced_indicators(hist)

            # 가치 지표 추가 (ROE, 부채비율 등)
            roe = info.get('returnOnEquity')
            debt_to_equity = info.get('debtToEquity')
            debt_ratio = debt_to_equity / (1 + debt_to_equity) if debt_to_equity else None

            # 시장 베타 계산
            market_beta = info.get('beta')

            stock_data = StockData(
                symbol=symbol,
                name=info.get('shortName', symbol),
                price=float(current_price),
                volume=volume,
                market_cap=info.get('marketCap'),
                
                # 가치 지표
                pe_ratio=info.get('trailingPE'),
                pb_ratio=info.get('priceToBook'),
                dividend_yield=info.get('dividendYield'),
                roe=roe,
                debt_ratio=debt_ratio,
                
                # 기본 기술적 지표
                moving_avg_20=float(moving_avg_20) if pd.notna(moving_avg_20) else None,
                moving_avg_60=float(moving_avg_60) if pd.notna(moving_avg_60) else None,
                rsi=float(rsi) if pd.notna(rsi) else None,
                bollinger_upper=float(bollinger_upper) if pd.notna(bollinger_upper) else None,
                bollinger_lower=float(bollinger_lower) if pd.notna(bollinger_lower) else None,
                macd=float(macd) if pd.notna(macd) else None,
                macd_signal=float(macd_signal) if pd.notna(macd_signal) else None,
                
                # 고급 기술적 지표 (Gemini AI 최적화)
                stochastic_k=advanced_indicators.get('stochastic_k'),
                stochastic_d=advanced_indicators.get('stochastic_d'),
                williams_r=advanced_indicators.get('williams_r'),
                atr=advanced_indicators.get('atr'),
                
                # 가격 동향 분석
                price_change_1d=advanced_indicators.get('price_change_1d'),
                price_change_5d=advanced_indicators.get('price_change_5d'),
                price_change_20d=advanced_indicators.get('price_change_20d'),
                volume_ratio_20d=advanced_indicators.get('volume_ratio_20d'),
                volatility_20d=advanced_indicators.get('volatility_20d'),
                
                # 시장 상대 성과
                market_beta=market_beta,
                relative_strength=advanced_indicators.get('relative_strength'),
                
                # 메타데이터
                data_source="yfinance_kospi",
                timestamp=datetime.now()
            )

            # 데이터 검증 및 정제
            stock_data = self._validate_and_clean_data(stock_data)

            return stock_data

        except Exception as e:
            logger.error(f"코스피 종목 데이터 수집 실패 {symbol}: {e}")
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

    def _calculate_advanced_indicators(self, hist_data: pd.DataFrame) -> Dict[str, float]:
        """고급 기술적 지표 계산 (Gemini AI 최적화)"""
        try:
            indicators = {}

            # Stochastic Oscillator
            low_14 = hist_data['Low'].rolling(window=14).min()
            high_14 = hist_data['High'].rolling(window=14).max()
            k_percent = 100 * ((hist_data['Close'] - low_14) / (high_14 - low_14))
            indicators['stochastic_k'] = k_percent.rolling(window=3).mean().iloc[-1]
            indicators['stochastic_d'] = k_percent.rolling(window=3).mean().rolling(window=3).mean().iloc[-1]

            # Williams %R
            williams_r = -100 * ((high_14 - hist_data['Close']) / (high_14 - low_14))
            indicators['williams_r'] = williams_r.iloc[-1]

            # Average True Range (ATR)
            high_low = hist_data['High'] - hist_data['Low']
            high_close = np.abs(hist_data['High'] - hist_data['Close'].shift())
            low_close = np.abs(hist_data['Low'] - hist_data['Close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            indicators['atr'] = true_range.rolling(window=14).mean().iloc[-1]

            # 가격 변화율
            indicators['price_change_1d'] = ((hist_data['Close'].iloc[-1] - hist_data['Close'].iloc[-2]) / hist_data['Close'].iloc[-2]) * 100
            indicators['price_change_5d'] = ((hist_data['Close'].iloc[-1] - hist_data['Close'].iloc[-6]) / hist_data['Close'].iloc[-6]) * 100
            indicators['price_change_20d'] = ((hist_data['Close'].iloc[-1] - hist_data['Close'].iloc[-21]) / hist_data['Close'].iloc[-21]) * 100

            # 거래량 비율
            avg_volume_20d = hist_data['Volume'].rolling(window=20).mean().iloc[-1]
            indicators['volume_ratio_20d'] = hist_data['Volume'].iloc[-1] / avg_volume_20d if avg_volume_20d > 0 else 1.0

            # 변동성 (20일 표준편차)
            indicators['volatility_20d'] = hist_data['Close'].pct_change().rolling(window=20).std().iloc[-1] * 100

            # 상대 강도 (시장 대비)
            # 여기서는 간단히 20일 수익률로 계산
            indicators['relative_strength'] = indicators['price_change_20d']

            return indicators

        except Exception as e:
            logger.warning(f"고급 지표 계산 실패: {e}")
            return {}

    def _validate_and_clean_data(self, stock_data: StockData) -> StockData:
        """데이터 검증 및 정제 (Gemini AI 최적화)"""
        try:
            # 가격 데이터 검증
            if stock_data.price is not None and stock_data.price <= 0:
                stock_data.price = None

            # 거래량 검증
            if stock_data.volume is not None and stock_data.volume < 0:
                    stock_data.volume = 0

            # PE 비율 이상치 제거 (음수 또는 1000 초과)
            if stock_data.pe_ratio is not None and (stock_data.pe_ratio < 0 or stock_data.pe_ratio > 1000):
                stock_data.pe_ratio = None

            # PB 비율 이상치 제거 (음수 또는 100 초과)
            if stock_data.pb_ratio is not None and (stock_data.pb_ratio < 0 or stock_data.pb_ratio > 100):
                stock_data.pb_ratio = None

            # RSI 범위 검증 (0-100)
            if stock_data.rsi is not None and (stock_data.rsi < 0 or stock_data.rsi > 100):
                stock_data.rsi = None

            # 데이터 품질 점수 계산
            quality_score = stock_data.calculate_quality_score()

            # 데이터 신선도 계산
            if stock_data.timestamp:
                hours_old = (datetime.now() - stock_data.timestamp).total_seconds() / 3600
                stock_data.data_quality.data_freshness_hours = hours_old
                stock_data.data_quality.timeliness_score = max(0, 100 - (hours_old * 2))  # 2시간마다 2점 감점

            logger.debug(f"데이터 품질 점수 {stock_data.symbol}: {quality_score:.1f}")
            return stock_data

        except Exception as e:
            logger.error(f"데이터 검증 실패 {stock_data.symbol}: {e}")
            return stock_data

    def _calculate_technical_indicators(self, stock_data: StockData) -> Dict[str, float]:
        """기술적 지표 계산 (강화된 결측치 처리)"""
        try:
            indicators = {}
            
            # 안전한 숫자 변환
            def safe_float(value, default=0.0):
                """안전한 float 변환"""
                if value is None or value == '' or str(value).lower() in ['nan', 'null', 'none']:
                    return default
                try:
                    return float(str(value).replace(',', '').replace('%', '').replace('$', ''))
                except (ValueError, TypeError):
                    return default
            
            # 기본값 설정 (결측치 대응)
            price = safe_float(stock_data.price, 100.0)  # 기본 가격 100달러
            volume = safe_float(stock_data.volume, 1000000)  # 기본 거래량 100만주
            
            # 가격 기반 지표들
            if hasattr(stock_data, 'high') and hasattr(stock_data, 'low'):
                high = safe_float(getattr(stock_data, 'high', None), price * 1.05)
                low = safe_float(getattr(stock_data, 'low', None), price * 0.95)
                
                # RSI (14일) - 결측치 시 중립값 50
                indicators['rsi'] = safe_float(getattr(stock_data, 'rsi', None), 50.0)
                
                # 볼린저 밴드 - 결측치 시 현재가 기준으로 계산
                bb_upper = safe_float(getattr(stock_data, 'bb_upper', None), price * 1.1)
                bb_lower = safe_float(getattr(stock_data, 'bb_lower', None), price * 0.9)
                
                indicators['bollinger_position'] = (price - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
                
                # MACD - 결측치 시 0 (중립)
                indicators['macd'] = safe_float(getattr(stock_data, 'macd', None), 0.0)
                indicators['macd_signal'] = safe_float(getattr(stock_data, 'macd_signal', None), 0.0)
                indicators['macd_histogram'] = indicators['macd'] - indicators['macd_signal']
                
                # 이동평균선들 - 결측치 시 현재가 기준으로 추정
                ma5 = safe_float(getattr(stock_data, 'ma5', None), price * 0.98)
                ma20 = safe_float(getattr(stock_data, 'ma20', None), price * 0.95)
                ma60 = safe_float(getattr(stock_data, 'ma60', None), price * 0.92)
                
                indicators['ma5_distance'] = (price - ma5) / ma5 * 100 if ma5 > 0 else 0
                indicators['ma20_distance'] = (price - ma20) / ma20 * 100 if ma20 > 0 else 0
                indicators['ma60_distance'] = (price - ma60) / ma60 * 100 if ma60 > 0 else 0
                
                # 거래량 지표 - 결측치 시 평균 거래량 기준
                avg_volume = safe_float(getattr(stock_data, 'avg_volume', None), volume)
                indicators['volume_ratio'] = volume / avg_volume if avg_volume > 0 else 1.0
                
                # 변동성 지표 - 결측치 시 표준 변동성 2% 가정
                indicators['volatility'] = safe_float(getattr(stock_data, 'volatility', None), 0.02)
                
                # 추가 기술적 지표들
                indicators['price_momentum'] = safe_float(getattr(stock_data, 'momentum', None), 0.0)
                indicators['stochastic_k'] = safe_float(getattr(stock_data, 'stoch_k', None), 50.0)
                indicators['stochastic_d'] = safe_float(getattr(stock_data, 'stoch_d', None), 50.0)
                
                # Williams %R - 결측치 시 중립값 -50
                indicators['williams_r'] = safe_float(getattr(stock_data, 'williams_r', None), -50.0)
                
                # CCI (Commodity Channel Index) - 결측치 시 0
                indicators['cci'] = safe_float(getattr(stock_data, 'cci', None), 0.0)
                
                # ADX (Average Directional Index) - 결측치 시 25 (중립 트렌드)
                indicators['adx'] = safe_float(getattr(stock_data, 'adx', None), 25.0)
                
            else:
                # 최소한의 기본 지표만 설정
                indicators['rsi'] = 50.0
                indicators['bollinger_position'] = 0.5
                indicators['macd'] = 0.0
                indicators['macd_signal'] = 0.0
                indicators['macd_histogram'] = 0.0
                indicators['ma5_distance'] = 0.0
                indicators['ma20_distance'] = 0.0
                indicators['ma60_distance'] = 0.0
                indicators['volume_ratio'] = 1.0
                indicators['volatility'] = 0.02
                indicators['price_momentum'] = 0.0
                indicators['stochastic_k'] = 50.0
                indicators['stochastic_d'] = 50.0
                indicators['williams_r'] = -50.0
                indicators['cci'] = 0.0
                indicators['adx'] = 25.0
            
            # 기술적 지표를 StockData 객체에 추가
            for key, value in indicators.items():
                if not hasattr(stock_data, key) or getattr(stock_data, key) is None:
                    setattr(stock_data, key, value)
            
            return indicators
            
        except Exception as e:
            logger.warning(f"기술적 지표 계산 실패 ({stock_data.symbol}): {e}")
            # 실패 시 모든 지표를 중립값으로 설정
            default_indicators = {
                'rsi': 50.0, 'bollinger_position': 0.5, 'macd': 0.0, 'macd_signal': 0.0,
                'macd_histogram': 0.0, 'ma5_distance': 0.0, 'ma20_distance': 0.0, 
                'ma60_distance': 0.0, 'volume_ratio': 1.0, 'volatility': 0.02,
                'price_momentum': 0.0, 'stochastic_k': 50.0, 'stochastic_d': 50.0,
                'williams_r': -50.0, 'cci': 0.0, 'adx': 25.0
            }
            
            # 기본값을 StockData 객체에 설정
            for key, value in default_indicators.items():
                setattr(stock_data, key, value)
                
            return default_indicators

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
    """전체 데이터 수집 관리자 - Gemini AI 최적화"""

    def __init__(self):
        self.kospi_collector = KospiCollector()
        self.nasdaq_collector = NasdaqCollector()
        self.sp500_collector = SP500Collector()

    async def collect_all_market_data(self) -> Dict[str, List[StockData]]:
        """코스피200 + 나스닥100 + S&P500 전체 데이터 수집"""
        logger.info("🚀 전체 시장 데이터 수집 시작 (Gemini AI 최적화)")

        # 병렬로 세 시장 데이터 수집
        kospi_task = self.kospi_collector.collect_all_data()
        nasdaq_task = self.nasdaq_collector.collect_all_data()
        sp500_task = self.sp500_collector.collect_all_data()

        kospi_data, nasdaq_data, sp500_data = await asyncio.gather(
            kospi_task, nasdaq_task, sp500_task
        )

        # 데이터 품질 필터링
        kospi_data = self._filter_high_quality_data(kospi_data, "KOSPI200")
        nasdaq_data = self._filter_high_quality_data(nasdaq_data, "NASDAQ100")
        sp500_data = self._filter_high_quality_data(sp500_data, "S&P500")

        result = {
            'kospi200': kospi_data,
            'nasdaq100': nasdaq_data,
            'sp500': sp500_data
        }

        total_stocks = len(kospi_data) + len(nasdaq_data) + len(sp500_data)
        logger.info(f"✅ 고품질 데이터 수집 완료: {total_stocks}개 종목 (코스피200: {len(kospi_data)}, 나스닥100: {len(nasdaq_data)}, S&P500: {len(sp500_data)})")

        return result

    def _filter_high_quality_data(self, stock_list: List[StockData], market_name: str) -> List[StockData]:
        """고품질 데이터만 필터링 (Gemini AI 최적화)"""
        try:
            if not stock_list:
                return stock_list

            # 품질 점수 계산
            for stock in stock_list:
                stock.calculate_quality_score()

            # 품질 점수 70점 이상만 유지
            high_quality_stocks = [stock for stock in stock_list if stock.data_quality.overall_quality >= 70.0]

            # 필수 데이터가 있는 종목만 유지
            filtered_stocks = []
            for stock in high_quality_stocks:
                if (stock.price is not None and stock.price > 0 and 
                    stock.volume is not None and stock.volume > 0 and
                    stock.symbol and stock.name):
                    filtered_stocks.append(stock)

            logger.info(f"{market_name} 고품질 데이터 필터링: {len(stock_list)} → {len(filtered_stocks)}개 종목")
            return filtered_stocks

        except Exception as e:
            logger.error(f"데이터 품질 필터링 실패 {market_name}: {e}")
            return stock_list

    def prepare_gemini_dataset(self, market_data: Dict[str, List[StockData]]) -> Dict[str, any]:
        """Gemini AI 분석을 위한 최적화된 데이터셋 준비"""
        try:
            logger.info("🧠 Gemini AI 분석용 데이터셋 준비 중...")

            # 전체 데이터를 하나의 리스트로 통합
            all_stocks = []
            for market, stocks in market_data.items():
                for stock in stocks:
                    stock_dict = self._stock_to_gemini_format(stock, market)
                    all_stocks.append(stock_dict)

            # 시장별 통계 계산
            market_stats = self._calculate_market_statistics(market_data)

            # 상위 성과 종목 식별
            top_performers = self._identify_top_performers(all_stocks)

            # 기술적 패턴 분석
            technical_patterns = self._analyze_technical_patterns(all_stocks)

            # 섹터별 분석 (간단한 분류)
            sector_analysis = self._analyze_by_sectors(all_stocks)

            gemini_dataset = {
                "timestamp": datetime.now().isoformat(),
                "total_stocks": len(all_stocks),
                "markets": list(market_data.keys()),
                "market_statistics": market_stats,
                "all_stocks": all_stocks,
                "top_performers": top_performers,
                "technical_patterns": technical_patterns,
                "sector_analysis": sector_analysis,
                "data_quality_summary": self._generate_quality_summary(market_data),
                "analysis_instructions": {
                    "focus_areas": [
                        "기술적 분석 기반 종목 선정",
                        "투자 대가 전략 적용 (워런 버핏, 피터 린치, 벤저민 그레이엄)",
                        "리스크 대비 수익률 최적화",
                        "시장 상황 고려한 포트폴리오 구성"
                    ],
                    "selection_criteria": {
                        "technical_strength": "RSI, MACD, 볼린저밴드 등 기술적 지표 우수",
                        "momentum": "가격 모멘텀 및 거래량 증가 패턴",
                        "risk_management": "변동성 대비 안정적 수익률",
                        "market_position": "시장 대비 상대적 강세"
                    }
                }
            }
            
            logger.info(f"✅ Gemini AI 데이터셋 준비 완료: {len(all_stocks)}개 종목, 품질 점수 평균 {self._calculate_avg_quality(all_stocks):.1f}")
            return gemini_dataset

        except Exception as e:
            logger.error(f"Gemini 데이터셋 준비 실패: {e}")
            return {"error": str(e), "stocks": []}

    def _stock_to_gemini_format(self, stock: StockData, market: str) -> Dict[str, any]:
        """개별 종목을 Gemini AI 분석용 포맷으로 변환"""
        return {
            "symbol": stock.symbol,
            "name": stock.name,
            "market": market,
            "basic_info": {
                "price": stock.price,
                "volume": stock.volume,
                "market_cap": stock.market_cap,
            },
            "valuation_metrics": {
                "pe_ratio": stock.pe_ratio,
                "pb_ratio": stock.pb_ratio,
                "dividend_yield": stock.dividend_yield,
                "roe": stock.roe,
                "debt_ratio": stock.debt_ratio,
            },
            "technical_indicators": {
                "moving_averages": {
                    "ma_20": stock.moving_avg_20,
                    "ma_60": stock.moving_avg_60,
                },
                "momentum": {
                    "rsi": stock.rsi,
                    "stochastic_k": stock.stochastic_k,
                    "stochastic_d": stock.stochastic_d,
                    "williams_r": stock.williams_r,
                },
                "trend": {
                    "macd": stock.macd,
                    "macd_signal": stock.macd_signal,
                    "bollinger_upper": stock.bollinger_upper,
                    "bollinger_lower": stock.bollinger_lower,
                },
                "volatility": {
                    "atr": stock.atr,
                    "volatility_20d": stock.volatility_20d,
                }
            },
            "price_performance": {
                "change_1d": stock.price_change_1d,
                "change_5d": stock.price_change_5d,
                "change_20d": stock.price_change_20d,
                "volume_ratio": stock.volume_ratio_20d,
                "relative_strength": stock.relative_strength,
            },
            "risk_metrics": {
                "beta": stock.market_beta,
                "volatility": stock.volatility_20d,
            },
            "data_quality": {
                "overall_score": stock.data_quality.overall_quality,
                "completeness": stock.data_quality.completeness_score,
                "freshness_hours": stock.data_quality.data_freshness_hours,
            }
        }

    def _calculate_market_statistics(self, market_data: Dict[str, List[StockData]]) -> Dict[str, any]:
        """시장별 통계 계산"""
        stats = {}

        for market, stocks in market_data.items():
            if not stocks:
                continue

            prices = [s.price for s in stocks if s.price is not None]
            volumes = [s.volume for s in stocks if s.volume is not None]
            rsi_values = [s.rsi for s in stocks if s.rsi is not None]

            stats[market] = {
                "total_stocks": len(stocks),
                "avg_price": np.mean(prices) if prices else 0,
                "avg_volume": np.mean(volumes) if volumes else 0,
                "avg_rsi": np.mean(rsi_values) if rsi_values else 50,
                "high_rsi_count": len([r for r in rsi_values if r and r > 70]),
                "low_rsi_count": len([r for r in rsi_values if r and r < 30]),
            }

        return stats

    def _identify_top_performers(self, all_stocks: List[Dict]) -> List[Dict]:
        """상위 성과 종목 식별"""
        try:
            # 20일 수익률 기준 상위 20개
            stocks_with_returns = [s for s in all_stocks if s.get("price_performance", {}).get("change_20d") is not None]
            top_20_returns = sorted(stocks_with_returns, 
                key=lambda x: x["price_performance"]["change_20d"],
                                  reverse=True)[:20]

            # RSI 기준 적정 매수 구간 (30-70)
            good_rsi_stocks = [s for s in all_stocks 
                if s.get("technical_indicators", {}).get("momentum", {}).get("rsi")
                             and 30 <= s["technical_indicators"]["momentum"]["rsi"] <= 70]

            return {
                "top_20_returns": top_20_returns[:5],  # 상위 5개만
                "good_rsi_stocks": good_rsi_stocks[:10]  # 상위 10개만
            }

        except Exception as e:
            logger.error(f"상위 성과 종목 식별 실패: {e}")
            return {"top_20_returns": [], "good_rsi_stocks": []}

    def _analyze_technical_patterns(self, all_stocks: List[Dict]) -> Dict[str, any]:
        """기술적 패턴 분석"""
        try:
            patterns = {
                "bullish_signals": 0,
                "bearish_signals": 0,
                "neutral_signals": 0,
                "strong_momentum": [],
                "oversold_opportunities": [],
                "breakout_candidates": []
            }

            for stock in all_stocks:
                tech = stock.get("technical_indicators", {})
                perf = stock.get("price_performance", {})

                rsi = tech.get("momentum", {}).get("rsi")
                macd = tech.get("trend", {}).get("macd")
                change_20d = perf.get("change_20d")

                # 강한 모멘텀 (RSI > 60, 20일 수익률 > 10%)
                if rsi and rsi > 60 and change_20d and change_20d > 10:
                    patterns["strong_momentum"].append(stock["symbol"])
                    patterns["bullish_signals"] += 1

                # 과매도 기회 (RSI < 35, 하지만 기본적으로 건전)
                elif rsi and rsi < 35 and change_20d and change_20d > -15:
                    patterns["oversold_opportunities"].append(stock["symbol"])

                # 중립
                else:
                    patterns["neutral_signals"] += 1

            return patterns

        except Exception as e:
            logger.error(f"기술적 패턴 분석 실패: {e}")
            return {"error": str(e)}

    def _analyze_by_sectors(self, all_stocks: List[Dict]) -> Dict[str, any]:
        """섹터별 간단 분석"""
        # 간단한 섹터 분류 (심볼 기반)
        tech_keywords = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META"]
        finance_keywords = ["JPM", "BAC", "WFC", "GS", "MS"]

        sectors = {"Technology": 0, "Finance": 0, "Others": 0}

        for stock in all_stocks:
            symbol = stock.get("symbol", "")
            if any(keyword in symbol for keyword in tech_keywords):
                sectors["Technology"] += 1
            elif any(keyword in symbol for keyword in finance_keywords):
                sectors["Finance"] += 1
            else:
                sectors["Others"] += 1

        return sectors

    def _generate_quality_summary(self, market_data: Dict[str, List[StockData]]) -> Dict[str, any]:
        """데이터 품질 요약"""
        total_stocks = sum(len(stocks) for stocks in market_data.values())
        quality_scores = []
        
        for stocks in market_data.values():
            for stock in stocks:
                if stock.data_quality.overall_quality > 0:
                    quality_scores.append(stock.data_quality.overall_quality)
        
        return {
            "total_stocks": total_stocks,
            "avg_quality_score": np.mean(quality_scores) if quality_scores else 0,
            "high_quality_count": len([s for s in quality_scores if s >= 80]),
            "medium_quality_count": len([s for s in quality_scores if 60 <= s < 80]),
            "low_quality_count": len([s for s in quality_scores if s < 60]),
        }
    
    def _calculate_avg_quality(self, all_stocks: List[Dict]) -> float:
        """평균 품질 점수 계산"""
        if not all_stocks:
            return 0.0

        total_quality = sum(stock.get('data_quality_score', 0) for stock in all_stocks)
        return total_quality / len(all_stocks)

    # GUI 인터페이스를 위한 개별 시장 데이터 수집 메서드들
    async def collect_kospi_data(self) -> List[StockData]:
        """코스피200 데이터 수집"""
        try:
            logger.info("🇰🇷 코스피200 데이터 수집 시작")
            kospi_collector = KospiCollector()
            stocks = await kospi_collector.collect_all_data()
            logger.info(f"✅ 코스피200 데이터 수집 완료: {len(stocks)}개 종목")
            return stocks
        except Exception as e:
            logger.error(f"❌ 코스피200 데이터 수집 실패: {e}")
            return []

    async def collect_nasdaq_data(self) -> List[StockData]:
        """나스닥100 데이터 수집"""
        try:
            logger.info("🇺🇸 나스닥100 데이터 수집 시작")
            nasdaq_collector = NasdaqCollector()
            stocks = await nasdaq_collector.collect_all_data()
            logger.info(f"✅ 나스닥100 데이터 수집 완료: {len(stocks)}개 종목")
            return stocks
        except Exception as e:
            logger.error(f"❌ 나스닥100 데이터 수집 실패: {e}")
            return []

    async def collect_sp500_data(self) -> List[StockData]:
        """S&P500 데이터 수집"""
        try:
            logger.info("🇺🇸 S&P500 데이터 수집 시작")
            sp500_collector = SP500Collector()
            stocks = await sp500_collector.collect_all_data()
            logger.info(f"✅ S&P500 데이터 수집 완료: {len(stocks)}개 종목")
            return stocks
        except Exception as e:
            logger.error(f"❌ S&P500 데이터 수집 실패: {e}")
            return []