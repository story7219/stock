"""
기술적 분석 모듈
기술적 지표 계산 및 차트 분석
"""

import asyncio
import logging
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

from .data_collector import StockData

logger = logging.getLogger(__name__)

@dataclass
class TechnicalSignal:
    """기술적 분석 신호"""
    indicator: str
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    strength: float   # 0-100 신호 강도
    description: str
    timestamp: datetime

@dataclass
class ChartPattern:
    """차트 패턴 정보"""
    pattern_name: str
    confidence: float  # 0-100 신뢰도
    target_price: Optional[float]
    stop_loss: Optional[float]
    description: str
    detected_at: datetime

class TechnicalIndicators:
    """기술적 지표 계산기"""
    
    @staticmethod
    def calculate_advanced_rsi(prices: pd.Series, period: int = 14, 
                             smoothing_period: int = 3) -> pd.Series:
        """고급 RSI 계산 (스무딩 적용)"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # 스무딩 적용
            smoothed_rsi = rsi.rolling(window=smoothing_period).mean()
            return smoothed_rsi
        except Exception as e:
            logger.error(f"고급 RSI 계산 실패: {e}")
            return pd.Series([np.nan] * len(prices))
    
    @staticmethod
    def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                           k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """스토캐스틱 오실레이터 계산"""
        try:
            lowest_low = low.rolling(window=k_period).min()
            highest_high = high.rolling(window=k_period).max()
            
            k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
            d_percent = k_percent.rolling(window=d_period).mean()
            
            return k_percent, d_percent
        except Exception as e:
            logger.error(f"스토캐스틱 계산 실패: {e}")
            return pd.Series([np.nan] * len(close)), pd.Series([np.nan] * len(close))
    
    @staticmethod
    def calculate_williams_r(high: pd.Series, low: pd.Series, close: pd.Series,
                           period: int = 14) -> pd.Series:
        """윌리엄스 %R 계산"""
        try:
            highest_high = high.rolling(window=period).max()
            lowest_low = low.rolling(window=period).min()
            
            williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
            return williams_r
        except Exception as e:
            logger.error(f"윌리엄스 %R 계산 실패: {e}")
            return pd.Series([np.nan] * len(close))
    
    @staticmethod
    def calculate_cci(high: pd.Series, low: pd.Series, close: pd.Series,
                     period: int = 20) -> pd.Series:
        """상품 채널 지수(CCI) 계산"""
        try:
            typical_price = (high + low + close) / 3
            sma = typical_price.rolling(window=period).mean()
            mean_deviation = typical_price.rolling(window=period).apply(
                lambda x: np.mean(np.abs(x - x.mean()))
            )
            
            cci = (typical_price - sma) / (0.015 * mean_deviation)
            return cci
        except Exception as e:
            logger.error(f"CCI 계산 실패: {e}")
            return pd.Series([np.nan] * len(close))
    
    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series,
                     period: int = 14) -> pd.Series:
        """평균 실제 범위(ATR) 계산"""
        try:
            high_low = high - low
            high_close = np.abs(high - close.shift())
            low_close = np.abs(low - close.shift())
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=period).mean()
            
            return atr
        except Exception as e:
            logger.error(f"ATR 계산 실패: {e}")
            return pd.Series([np.nan] * len(close))
    
    @staticmethod
    def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series,
                     period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """평균 방향성 지수(ADX) 계산"""
        try:
            # True Range 계산
            high_low = high - low
            high_close = np.abs(high - close.shift())
            low_close = np.abs(low - close.shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            
            # Directional Movement 계산
            plus_dm = high.diff()
            minus_dm = low.diff()
            
            plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
            minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
            minus_dm = minus_dm.abs()
            
            # Smoothed values
            atr = true_range.rolling(window=period).mean()
            plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
            
            # ADX 계산
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(window=period).mean()
            
            return adx, plus_di, minus_di
        except Exception as e:
            logger.error(f"ADX 계산 실패: {e}")
            return (pd.Series([np.nan] * len(close)), 
                   pd.Series([np.nan] * len(close)), 
                   pd.Series([np.nan] * len(close)))
    
    @staticmethod
    def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """거래량 균형 지표(OBV) 계산"""
        try:
            obv = pd.Series(index=close.index, dtype=float)
            obv.iloc[0] = volume.iloc[0]
            
            for i in range(1, len(close)):
                if close.iloc[i] > close.iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
                elif close.iloc[i] < close.iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
                else:
                    obv.iloc[i] = obv.iloc[i-1]
            
            return obv
        except Exception as e:
            logger.error(f"OBV 계산 실패: {e}")
            return pd.Series([np.nan] * len(close))
    
    @staticmethod
    def calculate_vwap(high: pd.Series, low: pd.Series, close: pd.Series,
                      volume: pd.Series) -> pd.Series:
        """거래량 가중 평균 가격(VWAP) 계산"""
        try:
            typical_price = (high + low + close) / 3
            vwap = (typical_price * volume).cumsum() / volume.cumsum()
            return vwap
        except Exception as e:
            logger.error(f"VWAP 계산 실패: {e}")
            return pd.Series([np.nan] * len(close))

class ChartAnalyzer:
    """차트 패턴 분석기"""
    
    def __init__(self):
        self.technical_indicators = TechnicalIndicators()
        
    def analyze_stock_technical(self, stock_data: List[StockData]) -> Dict[str, Any]:
        """종목의 기술적 분석 수행"""
        if not stock_data:
            return {}
        
        # 데이터 준비
        df = self._prepare_dataframe(stock_data)
        
        # 기술적 지표 계산
        technical_indicators = self._calculate_all_indicators(df)
        
        # 신호 생성
        signals = self._generate_signals(df, technical_indicators)
        
        # 차트 패턴 감지
        patterns = self._detect_chart_patterns(df)
        
        # 종합 분석
        summary = self._create_analysis_summary(technical_indicators, signals, patterns)
        
        return {
            'symbol': stock_data[0].symbol,
            'analysis_timestamp': datetime.now(),
            'technical_indicators': technical_indicators,
            'signals': signals,
            'chart_patterns': patterns,
            'summary': summary
        }
    
    def _prepare_dataframe(self, stock_data: List[StockData]) -> pd.DataFrame:
        """StockData를 DataFrame으로 변환"""
        data = []
        for stock in stock_data:
            data.append({
                'timestamp': stock.timestamp,
                'open': stock.price,  # 실제로는 시가 데이터가 필요
                'high': stock.price * 1.02,  # 임시 데이터
                'low': stock.price * 0.98,   # 임시 데이터
                'close': stock.price,
                'volume': stock.volume
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        return df
    
    def _calculate_all_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """모든 기술적 지표 계산"""
        indicators = {}
        
        try:
            # 기본 지표
            indicators['sma_20'] = df['close'].rolling(window=20).mean()
            indicators['sma_50'] = df['close'].rolling(window=50).mean()
            indicators['ema_12'] = df['close'].ewm(span=12).mean()
            indicators['ema_26'] = df['close'].ewm(span=26).mean()
            
            # MACD
            indicators['macd'] = indicators['ema_12'] - indicators['ema_26']
            indicators['macd_signal'] = indicators['macd'].ewm(span=9).mean()
            indicators['macd_histogram'] = indicators['macd'] - indicators['macd_signal']
            
            # 볼린저 밴드
            sma_20 = indicators['sma_20']
            std_20 = df['close'].rolling(window=20).std()
            indicators['bb_upper'] = sma_20 + (std_20 * 2)
            indicators['bb_lower'] = sma_20 - (std_20 * 2)
            indicators['bb_width'] = indicators['bb_upper'] - indicators['bb_lower']
            
            # 고급 지표
            indicators['rsi'] = self.technical_indicators.calculate_advanced_rsi(df['close'])
            indicators['stoch_k'], indicators['stoch_d'] = self.technical_indicators.calculate_stochastic(
                df['high'], df['low'], df['close']
            )
            indicators['williams_r'] = self.technical_indicators.calculate_williams_r(
                df['high'], df['low'], df['close']
            )
            indicators['cci'] = self.technical_indicators.calculate_cci(
                df['high'], df['low'], df['close']
            )
            indicators['atr'] = self.technical_indicators.calculate_atr(
                df['high'], df['low'], df['close']
            )
            indicators['adx'], indicators['plus_di'], indicators['minus_di'] = self.technical_indicators.calculate_adx(
                df['high'], df['low'], df['close']
            )
            indicators['obv'] = self.technical_indicators.calculate_obv(df['close'], df['volume'])
            indicators['vwap'] = self.technical_indicators.calculate_vwap(
                df['high'], df['low'], df['close'], df['volume']
            )
            
            logger.info("기술적 지표 계산 완료")
            
        except Exception as e:
            logger.error(f"기술적 지표 계산 실패: {e}")
        
        return indicators
    
    def _generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any]) -> List[TechnicalSignal]:
        """기술적 신호 생성"""
        signals = []
        current_price = df['close'].iloc[-1]
        
        try:
            # RSI 신호
            current_rsi = indicators['rsi'].iloc[-1] if not pd.isna(indicators['rsi'].iloc[-1]) else 50
            if current_rsi < 30:
                signals.append(TechnicalSignal(
                    indicator='RSI',
                    signal_type='BUY',
                    strength=min(100, (30 - current_rsi) * 3),
                    description=f'RSI {current_rsi:.1f}로 과매도 구간',
                    timestamp=datetime.now()
                ))
            elif current_rsi > 70:
                signals.append(TechnicalSignal(
                    indicator='RSI',
                    signal_type='SELL',
                    strength=min(100, (current_rsi - 70) * 3),
                    description=f'RSI {current_rsi:.1f}로 과매수 구간',
                    timestamp=datetime.now()
                ))
            
            # MACD 신호
            current_macd = indicators['macd'].iloc[-1]
            current_signal = indicators['macd_signal'].iloc[-1]
            prev_macd = indicators['macd'].iloc[-2]
            prev_signal = indicators['macd_signal'].iloc[-2]
            
            if (prev_macd <= prev_signal and current_macd > current_signal):
                signals.append(TechnicalSignal(
                    indicator='MACD',
                    signal_type='BUY',
                    strength=75,
                    description='MACD 골든크로스 발생',
                    timestamp=datetime.now()
                ))
            elif (prev_macd >= prev_signal and current_macd < current_signal):
                signals.append(TechnicalSignal(
                    indicator='MACD',
                    signal_type='SELL',
                    strength=75,
                    description='MACD 데드크로스 발생',
                    timestamp=datetime.now()
                ))
            
            # 볼린저 밴드 신호
            bb_upper = indicators['bb_upper'].iloc[-1]
            bb_lower = indicators['bb_lower'].iloc[-1]
            
            if current_price <= bb_lower:
                signals.append(TechnicalSignal(
                    indicator='Bollinger Bands',
                    signal_type='BUY',
                    strength=70,
                    description='볼린저 밴드 하단 터치',
                    timestamp=datetime.now()
                ))
            elif current_price >= bb_upper:
                signals.append(TechnicalSignal(
                    indicator='Bollinger Bands',
                    signal_type='SELL',
                    strength=70,
                    description='볼린저 밴드 상단 터치',
                    timestamp=datetime.now()
                ))
            
            # 스토캐스틱 신호
            current_stoch_k = indicators['stoch_k'].iloc[-1]
            current_stoch_d = indicators['stoch_d'].iloc[-1]
            
            if current_stoch_k < 20 and current_stoch_d < 20:
                signals.append(TechnicalSignal(
                    indicator='Stochastic',
                    signal_type='BUY',
                    strength=60,
                    description='스토캐스틱 과매도 구간',
                    timestamp=datetime.now()
                ))
            elif current_stoch_k > 80 and current_stoch_d > 80:
                signals.append(TechnicalSignal(
                    indicator='Stochastic',
                    signal_type='SELL',
                    strength=60,
                    description='스토캐스틱 과매수 구간',
                    timestamp=datetime.now()
                ))
            
        except Exception as e:
            logger.error(f"신호 생성 실패: {e}")
        
        return signals
    
    def _detect_chart_patterns(self, df: pd.DataFrame) -> List[ChartPattern]:
        """차트 패턴 감지"""
        patterns = []
        
        try:
            # 간단한 패턴 감지 로직
            prices = df['close'].values
            
            # 상승 삼각형 패턴 감지
            if len(prices) >= 20:
                recent_highs = df['high'].tail(10).values
                recent_lows = df['low'].tail(10).values
                
                # 저점 상승 추세
                if self._is_ascending_trend(recent_lows[-5:]):
                    patterns.append(ChartPattern(
                        pattern_name='상승 삼각형',
                        confidence=70,
                        target_price=prices[-1] * 1.1,
                        stop_loss=prices[-1] * 0.95,
                        description='저점이 상승하는 삼각형 패턴',
                        detected_at=datetime.now()
                    ))
                
                # 하락 삼각형 패턴
                if self._is_descending_trend(recent_highs[-5:]):
                    patterns.append(ChartPattern(
                        pattern_name='하락 삼각형',
                        confidence=70,
                        target_price=prices[-1] * 0.9,
                        stop_loss=prices[-1] * 1.05,
                        description='고점이 하락하는 삼각형 패턴',
                        detected_at=datetime.now()
                    ))
            
            # 이중바닥 패턴 감지
            if self._detect_double_bottom(df):
                patterns.append(ChartPattern(
                    pattern_name='이중바닥',
                    confidence=80,
                    target_price=prices[-1] * 1.15,
                    stop_loss=prices[-1] * 0.92,
                    description='강력한 상승 반전 신호',
                    detected_at=datetime.now()
                ))
            
            # 이중천정 패턴 감지
            if self._detect_double_top(df):
                patterns.append(ChartPattern(
                    pattern_name='이중천정',
                    confidence=80,
                    target_price=prices[-1] * 0.85,
                    stop_loss=prices[-1] * 1.08,
                    description='강력한 하락 반전 신호',
                    detected_at=datetime.now()
                ))
            
        except Exception as e:
            logger.error(f"차트 패턴 감지 실패: {e}")
        
        return patterns
    
    def _is_ascending_trend(self, values: np.ndarray) -> bool:
        """상승 추세 확인"""
        if len(values) < 3:
            return False
        
        # 선형 회귀로 기울기 확인
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        return slope > 0
    
    def _is_descending_trend(self, values: np.ndarray) -> bool:
        """하락 추세 확인"""
        if len(values) < 3:
            return False
        
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        return slope < 0
    
    def _detect_double_bottom(self, df: pd.DataFrame) -> bool:
        """이중바닥 패턴 감지"""
        if len(df) < 30:
            return False
        
        try:
            lows = df['low'].tail(30).values
            
            # 최근 30일 중 최저점 2개 찾기
            min_indices = []
            for i in range(5, len(lows) - 5):
                if (lows[i] < lows[i-5:i].min() and 
                    lows[i] < lows[i+1:i+6].min()):
                    min_indices.append(i)
            
            if len(min_indices) >= 2:
                # 두 저점이 비슷한 수준인지 확인
                last_two_mins = [lows[i] for i in min_indices[-2:]]
                if abs(last_two_mins[0] - last_two_mins[1]) / last_two_mins[0] < 0.03:  # 3% 이내
                    return True
            
        except Exception as e:
            logger.error(f"이중바닥 패턴 감지 실패: {e}")
        
        return False
    
    def _detect_double_top(self, df: pd.DataFrame) -> bool:
        """이중천정 패턴 감지"""
        if len(df) < 30:
            return False
        
        try:
            highs = df['high'].tail(30).values
            
            # 최근 30일 중 최고점 2개 찾기
            max_indices = []
            for i in range(5, len(highs) - 5):
                if (highs[i] > highs[i-5:i].max() and 
                    highs[i] > highs[i+1:i+6].max()):
                    max_indices.append(i)
            
            if len(max_indices) >= 2:
                # 두 고점이 비슷한 수준인지 확인
                last_two_maxs = [highs[i] for i in max_indices[-2:]]
                if abs(last_two_maxs[0] - last_two_maxs[1]) / last_two_maxs[0] < 0.03:  # 3% 이내
                    return True
            
        except Exception as e:
            logger.error(f"이중천정 패턴 감지 실패: {e}")
        
        return False
    
    def _create_analysis_summary(self, indicators: Dict[str, Any], 
                               signals: List[TechnicalSignal], 
                               patterns: List[ChartPattern]) -> Dict[str, Any]:
        """기술적 분석 요약 생성"""
        summary = {
            'overall_trend': 'NEUTRAL',
            'strength': 50,
            'buy_signals': 0,
            'sell_signals': 0,
            'hold_signals': 0,
            'key_levels': {},
            'recommendations': []
        }
        
        try:
            # 신호 집계
            for signal in signals:
                if signal.signal_type == 'BUY':
                    summary['buy_signals'] += 1
                elif signal.signal_type == 'SELL':
                    summary['sell_signals'] += 1
                else:
                    summary['hold_signals'] += 1
            
            # 전체 추세 판단
            if summary['buy_signals'] > summary['sell_signals']:
                summary['overall_trend'] = 'BULLISH'
                summary['strength'] = min(100, 50 + (summary['buy_signals'] - summary['sell_signals']) * 15)
            elif summary['sell_signals'] > summary['buy_signals']:
                summary['overall_trend'] = 'BEARISH'
                summary['strength'] = min(100, 50 + (summary['sell_signals'] - summary['buy_signals']) * 15)
            
            # 주요 레벨 설정
            if 'bb_upper' in indicators and 'bb_lower' in indicators:
                summary['key_levels']['resistance'] = indicators['bb_upper'].iloc[-1]
                summary['key_levels']['support'] = indicators['bb_lower'].iloc[-1]
            
            if 'vwap' in indicators:
                summary['key_levels']['vwap'] = indicators['vwap'].iloc[-1]
            
            # 추천사항 생성
            if summary['overall_trend'] == 'BULLISH':
                summary['recommendations'].append('기술적 지표가 상승세를 시사합니다.')
            elif summary['overall_trend'] == 'BEARISH':
                summary['recommendations'].append('기술적 지표가 하락세를 시사합니다.')
            else:
                summary['recommendations'].append('현재 기술적 지표는 중립적입니다.')
            
            # 패턴 기반 추천
            for pattern in patterns:
                if pattern.confidence > 70:
                    summary['recommendations'].append(f'{pattern.pattern_name} 패턴이 감지되었습니다.')
            
        except Exception as e:
            logger.error(f"분석 요약 생성 실패: {e}")
        
        return summary
    
    def create_technical_chart(self, 
                             stock_data: List[StockData],
                             analysis_result: Dict[str, Any]) -> plt.Figure:
        """기술적 분석 차트 생성"""
        try:
            if not stock_data:
                logger.warning("차트 생성을 위한 주식 데이터가 없습니다")
                return plt.figure()
            
            # 기본 차트 생성 (matplotlib 사용)
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # 주가 차트
            dates = [i for i in range(len(stock_data))]
            prices = [stock.price for stock in stock_data]
            
            ax1.plot(dates, prices, label='주가', linewidth=2)
            ax1.set_title('주가 차트')
            ax1.set_ylabel('가격')
            ax1.legend()
            ax1.grid(True)
            
            # 거래량 차트
            volumes = [stock.volume for stock in stock_data]
            ax2.bar(dates, volumes, alpha=0.7, label='거래량')
            ax2.set_title('거래량')
            ax2.set_xlabel('시간')
            ax2.set_ylabel('거래량')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            logger.error(f"기술적 분석 차트 생성 실패: {e}")
            return plt.figure()

class TechnicalAnalyzer:
    """기술적 분석 통합 클래스"""
    
    def __init__(self):
        self.chart_analyzer = ChartAnalyzer()
        self.logger = logging.getLogger(__name__)
    
    async def analyze_all_markets(self, market_data: Dict[str, List[StockData]]) -> Dict[str, Any]:
        """모든 시장 데이터에 대한 기술적 분석 수행"""
        results = {}
        
        try:
            for market_name, stocks in market_data.items():
                self.logger.info(f"🔍 {market_name} 기술적 분석 시작")
                market_results = {}
                
                for stock in stocks[:10]:  # 상위 10개 종목만 분석
                    try:
                        # 개별 종목 기술적 분석
                        analysis = self.chart_analyzer.analyze_stock_technical([stock])
                        market_results[stock.symbol] = analysis
                        
                    except Exception as e:
                        self.logger.error(f"{stock.symbol} 기술적 분석 실패: {e}")
                        continue
                
                results[market_name] = market_results
                self.logger.info(f"✅ {market_name} 기술적 분석 완료: {len(market_results)}개 종목")
            
        except Exception as e:
            self.logger.error(f"기술적 분석 실패: {e}")
            raise
        
        return results
    
    def create_market_chart(self, market_data: Dict[str, List[StockData]]) -> plt.Figure:
        """시장별 종합 차트 생성"""
        try:
            fig, axes = plt.subplots(len(market_data), 1, figsize=(12, 4 * len(market_data)))
            
            if len(market_data) == 1:
                axes = [axes]
            
            for i, (market_name, stocks) in enumerate(market_data.items()):
                ax = axes[i]
                
                # 시장 대표 종목들의 평균 가격 변화
                if stocks:
                    avg_prices = []
                    for j in range(min(30, len(stocks))):  # 최대 30개 데이터 포인트
                        if j < len(stocks):
                            avg_prices.append(stocks[j].price)
                    
                    ax.plot(range(len(avg_prices)), avg_prices, label=market_name, linewidth=2)
                    ax.set_title(f'{market_name} 대표 종목 가격 추이')
                    ax.set_ylabel('평균 가격')
                    ax.legend()
                    ax.grid(True)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.logger.error(f"시장 차트 생성 실패: {e}")
            return plt.figure() 