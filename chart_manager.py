"""
📊 차트 분석 및 이미지 생성 클래스 (Gemini API 호환)
- 캔들스틱 차트 생성 (mplfinance)
- 기술적 지표 계산 (이동평균선, 일목균형표, RSI, MACD)
- 차트 패턴 감지 (헤드앤숄더, 삼각형, 쌍바닥/쌍천정)
- 지지/저항선 자동 계산
- base64 인코딩으로 Gemini API 호환
"""
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import base64
import io
import logging
from typing import Dict, List, Tuple, Optional, Any
import requests
from dataclasses import dataclass
import warnings
import os # 파일 저장을 위해 추가
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = ['Malgun Gothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

logger = logging.getLogger(__name__)

@dataclass
class TechnicalIndicators:
    """기술적 지표 데이터 클래스"""
    sma_5: np.ndarray = None      # 5일 이동평균
    sma_20: np.ndarray = None     # 20일 이동평균
    sma_60: np.ndarray = None     # 60일 이동평균
    ema_12: np.ndarray = None     # 12일 지수이동평균
    ema_26: np.ndarray = None     # 26일 지수이동평균
    
    # 일목균형표
    tenkan_sen: np.ndarray = None  # 전환선
    kijun_sen: np.ndarray = None   # 기준선
    senkou_span_a: np.ndarray = None  # 선행스팬A
    senkou_span_b: np.ndarray = None  # 선행스팬B
    chikou_span: np.ndarray = None    # 후행스팬
    
    # 모멘텀 지표
    rsi: np.ndarray = None        # RSI
    macd: np.ndarray = None       # MACD
    macd_signal: np.ndarray = None # MACD 시그널
    macd_histogram: np.ndarray = None # MACD 히스토그램
    
    # 볼린저 밴드
    bb_upper: np.ndarray = None   # 상단밴드
    bb_middle: np.ndarray = None  # 중간밴드
    bb_lower: np.ndarray = None   # 하단밴드

@dataclass
class ChartPattern:
    """차트 패턴 데이터 클래스"""
    pattern_type: str             # 패턴 유형
    confidence: float             # 신뢰도 (0-1)
    start_idx: int               # 시작 인덱스
    end_idx: int                 # 종료 인덱스
    target_price: float = None   # 목표가
    stop_loss: float = None      # 손절가
    description: str = ""        # 패턴 설명

@dataclass
class SupportResistance:
    """지지/저항선 데이터 클래스"""
    support_levels: List[float]   # 지지선 레벨들
    resistance_levels: List[float] # 저항선 레벨들
    current_trend: str           # 현재 추세 (상승/하락/횡보)
    strength_scores: Dict[float, float] # 각 레벨의 강도

class ChartManager:
    """📊 차트 분석 및 이미지 생성 클래스"""
    
    def __init__(self, kis_api_key: str = None, kis_secret: str = None, trader_instance=None):
        """ChartManager 초기화"""
        self.kis_api_key = kis_api_key
        self.kis_secret = kis_secret
        self.trader = trader_instance # CoreTrader 인스턴스
        
        # 한국투자증권 API 설정
        self.base_url = "https://openapi.koreainvestment.com:9443"
        self.session = requests.Session()
        
        # 차트 저장 경로
        self.chart_dir = "charts"
        os.makedirs(self.chart_dir, exist_ok=True)
        
        # 차트 스타일 설정
        self.chart_style = {
            'figsize': (12, 8),
            'volume': True,
            'mav': (5, 20, 60),  # 이동평균선
            'style': 'charles',   # 차트 스타일
            'marketcolors': mpf.make_marketcolors(
                up='red', down='blue',  # 한국식 색상
                edge='inherit',
                wick={'up': 'red', 'down': 'blue'},
                volume='in'
            )
        }
        
        logger.info("📊 ChartManager 초기화 완료")
    
    def get_stock_data(self, stock_code: str, period: str = '1D') -> pd.DataFrame:
        """주식 데이터 수집"""
        try:
            # 기간 설정
            period_map = {
                '1D': 1,    # 1일
                '1W': 7,    # 1주
                '1M': 30,   # 1개월
                '3M': 90,   # 3개월
                '6M': 180,  # 6개월
                '1Y': 365   # 1년
            }
            
            days = period_map.get(period, 30)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days + 50)  # 지표 계산을 위한 추가 데이터
            
            # yfinance를 사용한 데이터 수집 (한국 주식)
            ticker = f"{stock_code}.KS"
            if stock_code.startswith('A'):  # 코스닥
                ticker = f"{stock_code[1:]}.KQ"
            elif len(stock_code) == 6:  # 표준 6자리 코드
                ticker = f"{stock_code}.KS"
            
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)
            
            if df.empty:
                # 대체 방법: 샘플 데이터 생성
                df = self._generate_sample_data(stock_code, days)
            
            # 컬럼명 표준화
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            df = df.dropna()
            
            # 최근 기간만 선택
            df = df.tail(days)
            
            logger.info(f"📊 {stock_code} 데이터 수집 완료: {len(df)}일")
            return df
            
        except Exception as e:
            logger.warning(f"⚠️ 실제 데이터 수집 실패, 샘플 데이터 생성: {e}")
            return self._generate_sample_data(stock_code, days)
    
    def _generate_sample_data(self, stock_code: str, days: int) -> pd.DataFrame:
        """샘플 주식 데이터 생성"""
        np.random.seed(42)  # 재현 가능한 랜덤
        
        # 기본 가격 설정 (종목별)
        base_prices = {
            '005930': 70000,   # 삼성전자
            '000660': 120000,  # SK하이닉스
            '035420': 180000,  # NAVER
            '051910': 400000,  # LG화학
        }
        
        base_price = base_prices.get(stock_code, 50000)
        
        # 가격 데이터 생성 (현실적인 패턴)
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        
        # 트렌드와 변동성 적용
        trend = np.random.choice([-0.001, 0, 0.001], size=days, p=[0.3, 0.4, 0.3])
        volatility = 0.02  # 2% 일일 변동성
        
        returns = np.random.normal(trend, volatility, days)
        prices = [base_price]
        
        for r in returns[1:]:
            new_price = prices[-1] * (1 + r)
            prices.append(max(new_price, prices[-1] * 0.9))  # 최대 10% 하락 제한
        
        # OHLC 데이터 생성
        data = []
        for i, price in enumerate(prices):
            high = price * (1 + abs(np.random.normal(0, 0.01)))
            low = price * (1 - abs(np.random.normal(0, 0.01)))
            open_price = prices[i-1] if i > 0 else price
            close_price = price
            volume = np.random.randint(100000, 1000000)
            
            data.append([open_price, high, low, close_price, volume])
        
        df = pd.DataFrame(data, columns=['Open', 'High', 'Low', 'Close', 'Volume'])
        df.index = dates
        
        return df
    
    # === 기술적 지표 계산 (Numpy 구현) ===
    
    def _calculate_sma(self, data: np.ndarray, period: int) -> np.ndarray:
        """단순이동평균 계산"""
        sma = np.full_like(data, np.nan)
        for i in range(period-1, len(data)):
            sma[i] = np.mean(data[i-period+1:i+1])
        return sma
    
    def _calculate_ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """지수이동평균 계산"""
        ema = np.full_like(data, np.nan)
        alpha = 2.0 / (period + 1)
        
        # 첫 번째 값은 SMA로 시작
        ema[period-1] = np.mean(data[:period])
        
        # EMA 계산
        for i in range(period, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        
        return ema
    
    def _calculate_rsi(self, close: np.ndarray, period: int = 14) -> np.ndarray:
        """RSI 계산"""
        rsi = np.full_like(close, np.nan)
        
        # 가격 변화 계산
        delta = np.diff(close, prepend=close[0])
        
        # 상승/하락 분리
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)
        
        # 평균 상승/하락 계산
        avg_gains = np.full_like(close, np.nan)
        avg_losses = np.full_like(close, np.nan)
        
        # 초기 평균 계산
        if len(gains) >= period:
            avg_gains[period] = np.mean(gains[1:period+1])
            avg_losses[period] = np.mean(losses[1:period+1])
            
            # RSI 계산
            for i in range(period+1, len(close)):
                avg_gains[i] = (avg_gains[i-1] * (period-1) + gains[i]) / period
                avg_losses[i] = (avg_losses[i-1] * (period-1) + losses[i]) / period
                
                if avg_losses[i] != 0:
                    rs = avg_gains[i] / avg_losses[i]
                    rsi[i] = 100 - (100 / (1 + rs))
                else:
                    rsi[i] = 100
        
        return rsi
    
    def _calculate_macd(self, close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        """MACD 계산"""
        ema_fast = self._calculate_ema(close, fast)
        ema_slow = self._calculate_ema(close, slow)
        
        macd = ema_fast - ema_slow
        macd_signal = self._calculate_ema(macd[~np.isnan(macd)], signal)
        
        # 신호선 길이 맞추기
        signal_full = np.full_like(macd, np.nan)
        valid_idx = ~np.isnan(macd)
        signal_full[valid_idx] = np.pad(macd_signal, (np.sum(valid_idx) - len(macd_signal), 0), 
                                       mode='constant', constant_values=np.nan)[:np.sum(valid_idx)]
        
        macd_histogram = macd - signal_full
        
        return macd, signal_full, macd_histogram
    
    def _calculate_bollinger_bands(self, close: np.ndarray, period: int = 20, std_dev: float = 2) -> tuple:
        """볼린저 밴드 계산"""
        middle = self._calculate_sma(close, period)
        
        # 표준편차 계산
        std = np.full_like(close, np.nan)
        for i in range(period-1, len(close)):
            std[i] = np.std(close[i-period+1:i+1])
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return upper, middle, lower

    def calculate_technical_indicators(self, stock_code: str, period: str = '1M') -> TechnicalIndicators:
        """📈 기술적 지표 계산"""
        try:
            # 주식 데이터 수집
            df = self.get_stock_data(stock_code, period)
            
            if len(df) < 60:
                logger.warning(f"⚠️ 데이터 부족: {len(df)}일 (최소 60일 필요)")
                return TechnicalIndicators()
            
            high = df['High'].values
            low = df['Low'].values
            close = df['Close'].values
            volume = df['Volume'].values
            
            indicators = TechnicalIndicators()
            
            # 이동평균선
            indicators.sma_5 = self._calculate_sma(close, 5)
            indicators.sma_20 = self._calculate_sma(close, 20)
            indicators.sma_60 = self._calculate_sma(close, 60)
            
            # 지수이동평균선
            indicators.ema_12 = self._calculate_ema(close, 12)
            indicators.ema_26 = self._calculate_ema(close, 26)
            
            # 일목균형표
            indicators.tenkan_sen = self._calculate_tenkan_sen(high, low)
            indicators.kijun_sen = self._calculate_kijun_sen(high, low)
            indicators.senkou_span_a = self._calculate_senkou_span_a(indicators.tenkan_sen, indicators.kijun_sen)
            indicators.senkou_span_b = self._calculate_senkou_span_b(high, low)
            indicators.chikou_span = np.roll(close, -26)  # 26일 후행
            
            # RSI
            indicators.rsi = self._calculate_rsi(close, 14)
            
            # MACD
            indicators.macd, indicators.macd_signal, indicators.macd_histogram = self._calculate_macd(close)
            
            # 볼린저 밴드
            indicators.bb_upper, indicators.bb_middle, indicators.bb_lower = self._calculate_bollinger_bands(close)
            
            logger.info(f"📈 {stock_code} 기술적 지표 계산 완료")
            return indicators
            
        except Exception as e:
            logger.error(f"❌ 기술적 지표 계산 실패: {e}")
            return TechnicalIndicators()
    
    def _calculate_tenkan_sen(self, high: np.ndarray, low: np.ndarray, period: int = 9) -> np.ndarray:
        """전환선 계산 (9일)"""
        tenkan = np.full_like(high, np.nan)
        for i in range(period-1, len(high)):
            period_high = np.max(high[i-period+1:i+1])
            period_low = np.min(low[i-period+1:i+1])
            tenkan[i] = (period_high + period_low) / 2
        return tenkan
    
    def _calculate_kijun_sen(self, high: np.ndarray, low: np.ndarray, period: int = 26) -> np.ndarray:
        """기준선 계산 (26일)"""
        kijun = np.full_like(high, np.nan)
        for i in range(period-1, len(high)):
            period_high = np.max(high[i-period+1:i+1])
            period_low = np.min(low[i-period+1:i+1])
            kijun[i] = (period_high + period_low) / 2
        return kijun
    
    def _calculate_senkou_span_a(self, tenkan: np.ndarray, kijun: np.ndarray) -> np.ndarray:
        """선행스팬A 계산"""
        span_a = (tenkan + kijun) / 2
        return np.roll(span_a, 26)  # 26일 선행
    
    def _calculate_senkou_span_b(self, high: np.ndarray, low: np.ndarray, period: int = 52) -> np.ndarray:
        """선행스팬B 계산 (52일)"""
        span_b = np.full_like(high, np.nan)
        for i in range(period-1, len(high)):
            period_high = np.max(high[i-period+1:i+1])
            period_low = np.min(low[i-period+1:i+1])
            span_b[i] = (period_high + period_low) / 2
        return np.roll(span_b, 26)  # 26일 선행
    
    def detect_patterns(self, stock_code: str, period: str = '3M') -> List[ChartPattern]:
        """🔍 차트 패턴 감지"""
        try:
            df = self.get_stock_data(stock_code, period)
            patterns = []
            
            if len(df) < 30:
                return patterns
            
            close = df['Close'].values
            high = df['High'].values
            low = df['Low'].values
            
            # 1. 헤드앤숄더 패턴 감지
            head_shoulder = self._detect_head_and_shoulders(high, low, close)
            if head_shoulder:
                patterns.append(head_shoulder)
            
            # 2. 쌍바닥/쌍천정 패턴 감지
            double_patterns = self._detect_double_patterns(high, low, close)
            patterns.extend(double_patterns)
            
            # 3. 삼각형 패턴 감지
            triangle = self._detect_triangle_pattern(high, low)
            if triangle:
                patterns.append(triangle)
            
            # 4. 상승/하락 웨지 패턴
            wedge = self._detect_wedge_pattern(high, low, close)
            if wedge:
                patterns.append(wedge)
            
            # 5. 플래그/페넌트 패턴
            flag = self._detect_flag_pattern(close)
            if flag:
                patterns.append(flag)
            
            logger.info(f"🔍 {stock_code} 패턴 감지 완료: {len(patterns)}개")
            return patterns
            
        except Exception as e:
            logger.error(f"❌ 패턴 감지 실패: {e}")
            return []
    
    def _detect_head_and_shoulders(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Optional[ChartPattern]:
        """헤드앤숄더 패턴 감지"""
        if len(high) < 20:
            return None
        
        # 간단한 헤드앤숄더 감지 로직
        peaks = []
        for i in range(2, len(high)-2):
            if high[i] > high[i-1] and high[i] > high[i+1] and high[i] > high[i-2] and high[i] > high[i+2]:
                peaks.append((i, high[i]))
        
        if len(peaks) >= 3:
            # 마지막 3개 피크 확인
            recent_peaks = peaks[-3:]
            left_shoulder, head, right_shoulder = recent_peaks
            
            # 헤드앤숄더 조건 확인
            if (head[1] > left_shoulder[1] and head[1] > right_shoulder[1] and
                abs(left_shoulder[1] - right_shoulder[1]) / left_shoulder[1] < 0.05):  # 5% 오차 허용
                
                return ChartPattern(
                    pattern_type="헤드앤숄더",
                    confidence=0.7,
                    start_idx=left_shoulder[0],
                    end_idx=right_shoulder[0],
                    target_price=close[-1] * 0.9,  # 10% 하락 목표
                    description="헤드앤숄더 패턴 감지: 하락 신호"
                )
        
        return None
    
    def _detect_double_patterns(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> List[ChartPattern]:
        """쌍바닥/쌍천정 패턴 감지"""
        patterns = []
        
        # 쌍천정 감지
        peaks = []
        for i in range(1, len(high)-1):
            if high[i] > high[i-1] and high[i] > high[i+1]:
                peaks.append((i, high[i]))
        
        for i in range(len(peaks)-1):
            peak1, peak2 = peaks[i], peaks[i+1]
            if (abs(peak1[1] - peak2[1]) / peak1[1] < 0.03 and  # 3% 오차 허용
                peak2[0] - peak1[0] >= 10):  # 최소 10일 간격
                
                patterns.append(ChartPattern(
                    pattern_type="쌍천정",
                    confidence=0.6,
                    start_idx=peak1[0],
                    end_idx=peak2[0],
                    target_price=close[-1] * 0.92,
                    description="쌍천정 패턴: 하락 신호"
                ))
                break
        
        # 쌍바닥 감지
        troughs = []
        for i in range(1, len(low)-1):
            if low[i] < low[i-1] and low[i] < low[i+1]:
                troughs.append((i, low[i]))
        
        for i in range(len(troughs)-1):
            trough1, trough2 = troughs[i], troughs[i+1]
            if (abs(trough1[1] - trough2[1]) / trough1[1] < 0.03 and
                trough2[0] - trough1[0] >= 10):
                
                patterns.append(ChartPattern(
                    pattern_type="쌍바닥",
                    confidence=0.6,
                    start_idx=trough1[0],
                    end_idx=trough2[0],
                    target_price=close[-1] * 1.08,
                    description="쌍바닥 패턴: 상승 신호"
                ))
                break
        
        return patterns
    
    def _detect_triangle_pattern(self, high: np.ndarray, low: np.ndarray) -> Optional[ChartPattern]:
        """삼각형 패턴 감지"""
        if len(high) < 20:
            return None
        
        # 최근 20일 데이터로 삼각형 패턴 확인
        recent_high = high[-20:]
        recent_low = low[-20:]
        
        # 고점 연결선의 기울기
        high_slope = np.polyfit(range(len(recent_high)), recent_high, 1)[0]
        # 저점 연결선의 기울기
        low_slope = np.polyfit(range(len(recent_low)), recent_low, 1)[0]
        
        # 수렴 삼각형 (고점은 하락, 저점은 상승)
        if high_slope < -0.1 and low_slope > 0.1:
            return ChartPattern(
                pattern_type="수렴삼각형",
                confidence=0.5,
                start_idx=len(high)-20,
                end_idx=len(high)-1,
                description="수렴삼각형 패턴: 돌파 대기"
            )
        
        return None
    
    def _detect_wedge_pattern(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Optional[ChartPattern]:
        """웨지 패턴 감지"""
        if len(high) < 15:
            return None
        
        # 최근 15일 추세
        recent_trend = np.polyfit(range(15), close[-15:], 1)[0]
        high_trend = np.polyfit(range(15), high[-15:], 1)[0]
        low_trend = np.polyfit(range(15), low[-15:], 1)[0]
        
        # 상승 웨지 (상승 추세에서 고점과 저점이 모두 상승하지만 폭이 좁아짐)
        if (recent_trend > 0 and high_trend > 0 and low_trend > 0 and 
            high_trend < low_trend * 2):  # 수렴 조건
            
            return ChartPattern(
                pattern_type="상승웨지",
                confidence=0.4,
                start_idx=len(high)-15,
                end_idx=len(high)-1,
                target_price=close[-1] * 0.95,
                description="상승웨지 패턴: 조정 가능성"
            )
        
        return None
    
    def _detect_flag_pattern(self, close: np.ndarray) -> Optional[ChartPattern]:
        """플래그 패턴 감지"""
        if len(close) < 10:
            return None
        
        # 최근 10일간 횡보 여부 확인
        recent_prices = close[-10:]
        volatility = np.std(recent_prices) / np.mean(recent_prices)
        
        # 변동성이 낮으면 플래그 패턴으로 간주
        if volatility < 0.02:  # 2% 미만 변동성
            return ChartPattern(
                pattern_type="플래그",
                confidence=0.3,
                start_idx=len(close)-10,
                end_idx=len(close)-1,
                description="플래그 패턴: 횡보 후 돌파 대기"
            )
        
        return None
    
    def get_support_resistance(self, stock_code: str, period: str = '3M') -> SupportResistance:
        """📊 지지/저항선 계산"""
        try:
            df = self.get_stock_data(stock_code, period)
            
            if len(df) < 20:
                return SupportResistance([], [], "횡보", {})
            
            high = df['High'].values
            low = df['Low'].values
            close = df['Close'].values
            
            # 지지선 찾기 (저점들)
            support_levels = self._find_support_levels(low, close)
            
            # 저항선 찾기 (고점들)
            resistance_levels = self._find_resistance_levels(high, close)
            
            # 현재 추세 판단
            current_trend = self._determine_trend(close)
            
            # 각 레벨의 강도 계산
            strength_scores = self._calculate_level_strength(
                support_levels + resistance_levels, high, low, close
            )
            
            logger.info(f"📊 {stock_code} 지지/저항선 계산 완료")
            return SupportResistance(
                support_levels=support_levels,
                resistance_levels=resistance_levels,
                current_trend=current_trend,
                strength_scores=strength_scores
            )
            
        except Exception as e:
            logger.error(f"❌ 지지/저항선 계산 실패: {e}")
            return SupportResistance([], [], "횡보", {})
    
    def _find_support_levels(self, low: np.ndarray, close: np.ndarray) -> List[float]:
        """지지선 레벨 찾기"""
        support_levels = []
        current_price = close[-1]
        
        # 최근 저점들 찾기
        for i in range(2, len(low)-2):
            if (low[i] < low[i-1] and low[i] < low[i+1] and 
                low[i] < low[i-2] and low[i] < low[i+2]):
                
                # 현재가 아래에 있는 지지선만 선택
                if low[i] < current_price * 0.98:  # 2% 아래
                    support_levels.append(low[i])
        
        # 중복 제거 및 정렬
        support_levels = sorted(list(set([round(level, -1) for level in support_levels])))
        
        # 상위 3개만 선택
        return support_levels[-3:] if len(support_levels) > 3 else support_levels
    
    def _find_resistance_levels(self, high: np.ndarray, close: np.ndarray) -> List[float]:
        """저항선 레벨 찾기"""
        resistance_levels = []
        current_price = close[-1]
        
        # 최근 고점들 찾기
        for i in range(2, len(high)-2):
            if (high[i] > high[i-1] and high[i] > high[i+1] and 
                high[i] > high[i-2] and high[i] > high[i+2]):
                
                # 현재가 위에 있는 저항선만 선택
                if high[i] > current_price * 1.02:  # 2% 위
                    resistance_levels.append(high[i])
        
        # 중복 제거 및 정렬
        resistance_levels = sorted(list(set([round(level, -1) for level in resistance_levels])))
        
        # 상위 3개만 선택
        return resistance_levels[:3] if len(resistance_levels) > 3 else resistance_levels
    
    def _determine_trend(self, close: np.ndarray) -> str:
        """현재 추세 판단"""
        if len(close) < 20:
            return "횡보"
        
        # 최근 20일 추세선
        trend_slope = np.polyfit(range(20), close[-20:], 1)[0]
        
        # 추세 강도 계산
        price_change = (close[-1] - close[-20]) / close[-20]
        
        if trend_slope > 0 and price_change > 0.05:  # 5% 이상 상승
            return "상승"
        elif trend_slope < 0 and price_change < -0.05:  # 5% 이상 하락
            return "하락"
        else:
            return "횡보"
    
    def _calculate_level_strength(self, levels: List[float], high: np.ndarray, 
                                 low: np.ndarray, close: np.ndarray) -> Dict[float, float]:
        """지지/저항선 강도 계산"""
        strength_scores = {}
        
        for level in levels:
            touches = 0
            
            # 해당 레벨 근처에서의 터치 횟수 계산
            for i in range(len(close)):
                price_range = abs(high[i] - low[i])
                tolerance = price_range * 0.5  # 범위의 50%를 허용 오차로
                
                if abs(high[i] - level) <= tolerance or abs(low[i] - level) <= tolerance:
                    touches += 1
            
            # 터치 횟수에 따른 강도 계산 (0-1 스케일)
            strength_scores[level] = min(touches / 5.0, 1.0)  # 최대 5회 터치를 1.0으로
        
        return strength_scores
    
    def generate_chart_image(self, stock_code: str, period: str = '1D') -> str:
        """📊 차트 이미지 생성 및 base64 인코딩"""
        try:
            # 주식 데이터 및 지표 수집
            df = self.get_stock_data(stock_code, period)
            indicators = self.calculate_technical_indicators(stock_code, period)
            patterns = self.detect_patterns(stock_code, period)
            support_resistance = self.get_support_resistance(stock_code, period)
            
            if len(df) < 5:
                raise ValueError("데이터 부족")
            
            # 차트 생성을 위한 추가 설정
            additional_plots = []
            
            # 이동평균선 추가
            if indicators.sma_5 is not None:
                additional_plots.append(
                    mpf.make_addplot(indicators.sma_5, color='orange', width=1, label='SMA5')
                )
            if indicators.sma_20 is not None:
                additional_plots.append(
                    mpf.make_addplot(indicators.sma_20, color='blue', width=1, label='SMA20')
                )
            if indicators.sma_60 is not None:
                additional_plots.append(
                    mpf.make_addplot(indicators.sma_60, color='purple', width=1, label='SMA60')
                )
            
            # 일목균형표 구름대 (선행스팬A, B)
            if indicators.senkou_span_a is not None and indicators.senkou_span_b is not None:
                # 구름대 영역 표시를 위한 fill_between 효과
                cloud_data = pd.DataFrame({
                    'span_a': indicators.senkou_span_a,
                    'span_b': indicators.senkou_span_b
                }, index=df.index)
                
                additional_plots.append(
                    mpf.make_addplot(indicators.senkou_span_a, color='green', 
                                   width=0.5, alpha=0.3, label='선행스팬A')
                )
                additional_plots.append(
                    mpf.make_addplot(indicators.senkou_span_b, color='red', 
                                   width=0.5, alpha=0.3, label='선행스팬B')
                )
            
            # 볼린저 밴드
            if indicators.bb_upper is not None:
                additional_plots.append(
                    mpf.make_addplot(indicators.bb_upper, color='gray', 
                                   width=0.5, alpha=0.7, label='볼린저 상단')
                )
                additional_plots.append(
                    mpf.make_addplot(indicators.bb_lower, color='gray', 
                                   width=0.5, alpha=0.7, label='볼린저 하단')
                )
            
            # 지지/저항선 추가
            current_price = df['Close'].iloc[-1]
            for level in support_resistance.support_levels:
                if abs(level - current_price) / current_price < 0.2:  # 현재가 20% 범위 내
                    line_data = [level] * len(df)
                    additional_plots.append(
                        mpf.make_addplot(line_data, color='green', 
                                       width=1, linestyle='--', alpha=0.7)
                    )
            
            for level in support_resistance.resistance_levels:
                if abs(level - current_price) / current_price < 0.2:
                    line_data = [level] * len(df)
                    additional_plots.append(
                        mpf.make_addplot(line_data, color='red', 
                                       width=1, linestyle='--', alpha=0.7)
                    )
            
            # 차트 스타일 설정
            mc = mpf.make_marketcolors(
                up='red', down='blue',
                edge='inherit',
                wick={'up': 'red', 'down': 'blue'},
                volume='in'
            )
            
            s = mpf.make_mpf_style(
                marketcolors=mc,
                gridstyle='-',
                y_on_right=True
            )
            
            # 메모리 내 이미지 생성
            buf = io.BytesIO()
            
            # 차트 생성
            mpf.plot(
                df,
                type='candle',
                style=s,
                volume=True,
                addplot=additional_plots if additional_plots else None,
                figsize=(12, 8),
                title=f'{stock_code} 주가 차트 ({period})',
                ylabel='가격 (원)',
                ylabel_lower='거래량',
                savefig=dict(fname=buf, format='png', dpi=100, bbox_inches='tight'),
                returnfig=False
            )
            
            # base64 인코딩
            buf.seek(0)
            image_base64 = base64.b64encode(buf.read()).decode('utf-8')
            buf.close()
            
            logger.info(f"📊 {stock_code} 차트 이미지 생성 완료")
            return image_base64
            
        except Exception as e:
            logger.error(f"❌ 차트 이미지 생성 실패: {e}")
            # 오류 시 간단한 차트 생성
            return self._generate_simple_chart(stock_code, period)
    
    def _generate_simple_chart(self, stock_code: str, period: str) -> str:
        """간단한 차트 생성 (오류 시 백업)"""
        try:
            df = self.get_stock_data(stock_code, period)
            
            plt.figure(figsize=(10, 6))
            plt.plot(df.index, df['Close'], linewidth=1.5, color='blue')
            plt.title(f'{stock_code} 주가 차트 ({period})', fontsize=14)
            plt.xlabel('날짜')
            plt.ylabel('주가 (원)')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # base64 인코딩
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            image_base64 = base64.b64encode(buf.read()).decode('utf-8')
            buf.close()
            plt.close()
            
            return image_base64
            
        except Exception as e:
            logger.error(f"❌ 간단한 차트 생성도 실패: {e}")
            return ""
    
    def get_chart_analysis_summary(self, stock_code: str, period: str = '1M') -> Dict[str, Any]:
        """📊 차트 분석 종합 요약"""
        try:
            # 모든 분석 실행
            indicators = self.calculate_technical_indicators(stock_code, period)
            patterns = self.detect_patterns(stock_code, period)
            support_resistance = self.get_support_resistance(stock_code, period)
            df = self.get_stock_data(stock_code, period)
            
            if len(df) == 0:
                return {}
            
            current_price = df['Close'].iloc[-1]
            
            # 기술적 분석 신호 종합
            signals = []
            
            # 이동평균선 신호
            if indicators.sma_5 is not None and indicators.sma_20 is not None:
                sma5_current = indicators.sma_5[-1]
                sma20_current = indicators.sma_20[-1]
                
                if not np.isnan(sma5_current) and not np.isnan(sma20_current):
                    if sma5_current > sma20_current:
                        signals.append("단기 상승 신호 (SMA5 > SMA20)")
                    else:
                        signals.append("단기 하락 신호 (SMA5 < SMA20)")
            
            # RSI 신호
            if indicators.rsi is not None:
                rsi_current = indicators.rsi[-1]
                if not np.isnan(rsi_current):
                    if rsi_current > 70:
                        signals.append("과매수 구간 (RSI > 70)")
                    elif rsi_current < 30:
                        signals.append("과매도 구간 (RSI < 30)")
            
            # 패턴 신호
            for pattern in patterns:
                signals.append(f"{pattern.pattern_type} 패턴 감지 (신뢰도: {pattern.confidence:.1%})")
            
            # 종합 분석 결과
            summary = {
                'stock_code': stock_code,
                'current_price': current_price,
                'period': period,
                'analysis_time': datetime.now().isoformat(),
                'trend': support_resistance.current_trend,
                'support_levels': support_resistance.support_levels,
                'resistance_levels': support_resistance.resistance_levels,
                'detected_patterns': [p.pattern_type for p in patterns],
                'technical_signals': signals,
                'data_points': len(df)
            }
            
            # RSI, MACD 현재 값 추가
            if indicators.rsi is not None and not np.isnan(indicators.rsi[-1]):
                summary['rsi'] = round(indicators.rsi[-1], 2)
            
            if indicators.macd is not None and not np.isnan(indicators.macd[-1]):
                summary['macd'] = round(indicators.macd[-1], 2)
            
            logger.info(f"📊 {stock_code} 차트 분석 요약 완료")
            return summary
            
        except Exception as e:
            logger.error(f"❌ 차트 분석 요약 실패: {e}")
            return {}
    
    def cleanup(self):
        """리소스 정리"""
        try:
            # matplotlib 리소스 정리
            plt.close('all')
            logger.info("🧹 ChartManager 리소스 정리 완료")
        except Exception as e:
            logger.error(f"❌ 리소스 정리 실패: {e}")

    # ===================================================================
    # chart_generator.py에서 가져온 차트 생성 기능들
    # ===================================================================
    
    def create_comprehensive_chart(self, symbol: str, period_days: int = 30, save_path: str = None) -> str:
        """🔥 종합 주식 차트 생성 (캔들스틱 + 거래량 + 지표)"""
        try:
            logger.info(f"📊 {symbol} 종합 차트 생성 시작 (기간: {period_days}일)")
            
            # 1. 데이터 수집 (기존 get_stock_data 활용)
            chart_data = self.get_stock_data(symbol, period=f'{int(period_days/30)}M' if period_days >= 30 else f'{int(period_days/7)}W')

            if chart_data.empty:
                logger.error(f"❌ {symbol} 데이터 수집 실패")
                return None
            
            # 2. 기술적 지표 계산 (기존 메서드 재활용)
            indicators = self.calculate_technical_indicators(df=chart_data)
            chart_data['sma_5'] = indicators.sma_5
            chart_data['sma_20'] = indicators.sma_20
            chart_data['sma_60'] = indicators.sma_60
            chart_data['rsi'] = indicators.rsi
            
            # 3. 차트 그리기
            fig = plt.figure(figsize=(16, 12))
            
            # 레이아웃: 주가차트(70%) + 거래량(30%)
            gs = fig.add_gridspec(3, 1, height_ratios=[2.5, 1, 0.5], hspace=0.1)
            
            # 메인 차트 (캔들스틱 + 이동평균)
            ax1 = fig.add_subplot(gs[0])
            self._draw_candlestick_chart(ax1, chart_data, symbol)
            
            # 거래량 차트
            ax2 = fig.add_subplot(gs[1], sharex=ax1)
            self._draw_volume_chart(ax2, chart_data)
            
            # 기술적 지표 (RSI)
            ax3 = fig.add_subplot(gs[2], sharex=ax1)
            self._draw_rsi_chart(ax3, chart_data)
            
            # 차트 스타일링
            self._style_chart(fig, ax1, ax2, ax3, symbol)
            
            # 차트 저장
            if not save_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(self.chart_dir, f"{symbol}_chart_{timestamp}.png")
            
            plt.savefig(save_path, dpi=150, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close()
            
            logger.info(f"✅ 차트 저장 완료: {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"❌ 차트 생성 실패: {e}")
            import traceback
            traceback.print_exc()
            return None

    def create_market_heatmap(self, symbols: List[str], save_path: str = None) -> str:
        """🌡️ 시장 히트맵 생성 (여러 종목 상승률 비교)"""
        try:
            logger.info(f"🌡️ 시장 히트맵 생성: {len(symbols)}개 종목")
            
            market_data = []
            for symbol in symbols:
                try:
                    if self.trader:
                        # CoreTrader를 통해 현재가 정보 가져오기
                        price_info = self.trader.get_current_price(symbol)
                        if price_info:
                            market_data.append({
                                'symbol': symbol,
                                'name': price_info.get('name', symbol),
                                'change_pct': float(price_info.get('prdy_ctrt', 0.0)) # 전일 대비 등락률
                            })
                except Exception as e:
                    logger.warning(f"⚠️ {symbol} 히트맵 데이터 수집 실패: {e}")
                    continue
            
            if len(market_data) < 1:
                logger.warning("히트맵 생성을 위한 데이터가 부족합니다.")
                return None
            
            df = pd.DataFrame(market_data)
            df = df.sort_values('change_pct', ascending=False)
            
            fig, ax = plt.subplots(figsize=(12, max(8, len(df) * 0.5)))
            colors = ['#27AE60' if x >= 0 else '#E74C3C' for x in df['change_pct']]
            bars = ax.barh(df['name'], df['change_pct'], color=colors, alpha=0.8)
            
            ax.set_xlabel('등락률 (%)', fontsize=12)
            ax.set_title('📊 실시간 시장 등락률 히트맵', fontsize=16, fontweight='bold', pad=20)
            ax.grid(True, axis='x', linestyle='--', alpha=0.5)
            ax.axvline(x=0, color='black', linewidth=1.2)

            for bar in bars:
                width = bar.get_width()
                label_x_pos = width + 0.1 if width >= 0 else width - 0.1
                ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', 
                        va='center', ha='left' if width >= 0 else 'right')

            plt.tight_layout()
            
            if not save_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(self.chart_dir, f"market_heatmap_{timestamp}.png")
            
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✅ 히트맵 저장: {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"❌ 히트맵 생성 실패: {e}")
            return None

    def _draw_candlestick_chart(self, ax, df: pd.DataFrame, symbol: str):
        """캔들스틱 및 이동평균선 그리기"""
        for i, row in df.iterrows():
            color = '#27AE60' if row['Close'] >= row['Open'] else '#E74C3C'
            ax.add_patch(Rectangle((mdates.date2num(i), row['Open']), 0.8, row['Close']-row['Open'], 
                                   facecolor=color, edgecolor=color, zorder=3))
            ax.plot([mdates.date2num(i)+0.4, mdates.date2num(i)+0.4], [row['Low'], row['High']], 
                    color=color, zorder=2)

        # 이동평균선
        ax.plot(df.index, df['sma_5'], label='5일선', color='#FFA500', linestyle='--', linewidth=1.5)
        ax.plot(df.index, df['sma_20'], label='20일선', color='#2E86AB', linewidth=1.5)
        ax.plot(df.index, df['sma_60'], label='60일선', color='#8E44AD', linewidth=1.5)
        
        ax.legend()
        ax.set_ylabel('주가 (원)')
        ax.set_title(f'{symbol} 종합 차트', fontsize=16, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.5)

    def _draw_volume_chart(self, ax, df: pd.DataFrame):
        """거래량 차트 그리기"""
        colors = ['#27AE60' if c >= o else '#E74C3C' for o, c in zip(df['Open'], df['Close'])]
        ax.bar(df.index, df['Volume'], color=colors, alpha=0.7, width=0.8)
        ax.set_ylabel('거래량')
        ax.grid(True, linestyle='--', alpha=0.5)

    def _draw_rsi_chart(self, ax, df: pd.DataFrame):
        """RSI 차트 그리기"""
        ax.plot(df.index, df['rsi'], label='RSI', color='#C0392B')
        ax.axhline(70, color='red', linestyle=':', linewidth=1, label='과매수(70)')
        ax.axhline(30, color='blue', linestyle=':', linewidth=1, label='과매도(30)')
        ax.set_ylim(0, 100)
        ax.set_ylabel('RSI')
        ax.legend(fontsize='small')
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # 날짜 포맷 설정
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

    def _style_chart(self, fig, ax1, ax2, ax3, symbol: str):
        """차트 전반 스타일링"""
        fig.suptitle(f'종합 기술적 분석: {symbol}', fontsize=20, fontweight='bold', y=0.98)
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.set_p(ax2.get_xticklabels(), visible=False)
        
        # 워터마크
        fig.text(0.5, 0.5, 'AI Trader Analysis', fontsize=40, color='gray', 
                 ha='center', va='center', alpha=0.1, rotation=30) 