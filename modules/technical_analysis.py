#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📈 차트 전문가 기술적 분석 엔진
세계 최고 차트 분석가들의 기법을 구현
Gemini AI 최적화 고품질 기술적 지표 시스템
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import talib
from investment_strategies import StockData

logger = logging.getLogger(__name__)

@dataclass
class TechnicalSignal:
    """기술적 분석 신호"""
    indicator_name: str
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    strength: float  # 0-100
    description: str
    confidence: float

@dataclass
class TechnicalAnalysisResult:
    """기술적 분석 종합 결과"""
    symbol: str
    signals: List[TechnicalSignal]
    overall_score: float
    recommendation: str
    support_levels: List[float]
    resistance_levels: List[float]
    trend_direction: str
    volatility_score: float

class TechnicalIndicatorCalculator:
    """🔧 기술적 지표 계산기"""
    
    @staticmethod
    def calculate_rsi(prices: np.array, period: int = 14) -> np.array:
        """RSI 계산"""
        try:
            return talib.RSI(prices, timeperiod=period)
        except:
            # Fallback 구현
            delta = np.diff(prices)
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)
            
            avg_gain = np.convolve(gain, np.ones(period)/period, mode='valid')
            avg_loss = np.convolve(loss, np.ones(period)/period, mode='valid')
            
            rs = avg_gain / (avg_loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            
            return np.concatenate([np.full(period, 50), rsi])
    
    @staticmethod
    def calculate_macd(prices: np.array, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.array, np.array, np.array]:
        """MACD 계산"""
        try:
            macd, signal_line, histogram = talib.MACD(prices, fastperiod=fast, slowperiod=slow, signalperiod=signal)
            return macd, signal_line, histogram
        except:
            # Fallback 구현
            ema_fast = pd.Series(prices).ewm(span=fast).mean().values
            ema_slow = pd.Series(prices).ewm(span=slow).mean().values
            macd = ema_fast - ema_slow
            signal_line = pd.Series(macd).ewm(span=signal).mean().values
            histogram = macd - signal_line
            return macd, signal_line, histogram
    
    @staticmethod
    def calculate_bollinger_bands(prices: np.array, period: int = 20, std_dev: float = 2) -> Tuple[np.array, np.array, np.array]:
        """볼린저 밴드 계산"""
        try:
            upper, middle, lower = talib.BBANDS(prices, timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev)
            return upper, middle, lower
        except:
            # Fallback 구현
            sma = pd.Series(prices).rolling(period).mean().values
            std = pd.Series(prices).rolling(period).std().values
            upper = sma + (std * std_dev)
            lower = sma - (std * std_dev)
            return upper, sma, lower
    
    @staticmethod
    def calculate_stochastic(high: np.array, low: np.array, close: np.array, k_period: int = 14, d_period: int = 3) -> Tuple[np.array, np.array]:
        """스토캐스틱 계산"""
        try:
            slowk, slowd = talib.STOCH(high, low, close, fastk_period=k_period, slowk_period=d_period, slowd_period=d_period)
            return slowk, slowd
        except:
            # Fallback 구현
            lowest_low = pd.Series(low).rolling(k_period).min().values
            highest_high = pd.Series(high).rolling(k_period).max().values
            k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low + 1e-10))
            d_percent = pd.Series(k_percent).rolling(d_period).mean().values
            return k_percent, d_percent

class ElliottWaveAnalyzer:
    """🌊 엘리엇 파동 분석가 - 랄프 엘리엇"""
    
    def __init__(self):
        self.name = "Elliott Wave Theory"
        self.description = "5파동 상승, 3파동 조정 패턴 분석"
    
    def analyze_wave_pattern(self, prices: np.array) -> TechnicalSignal:
        """파동 패턴 분석"""
        try:
            # 가격 변화 패턴 분석
            price_changes = np.diff(prices)
            
            # 5파동 상승 패턴 감지
            rising_waves = self._count_rising_waves(price_changes[-50:])
            falling_waves = self._count_falling_waves(price_changes[-50:])
            
            if rising_waves >= 3:
                signal_type = "BUY"
                strength = min(70 + rising_waves * 5, 95)
                description = f"상승 {rising_waves}파동 확인, 추가 상승 예상"
            elif falling_waves >= 2:
                signal_type = "SELL" 
                strength = min(60 + falling_waves * 10, 90)
                description = f"하락 {falling_waves}파동 확인, 조정 진행 중"
            else:
                signal_type = "HOLD"
                strength = 40
                description = "명확한 파동 패턴 미확인"
            
            return TechnicalSignal(
                indicator_name="Elliott Wave",
                signal_type=signal_type,
                strength=strength,
                description=description,
                confidence=strength / 100
            )
        except Exception as e:
            logger.warning(f"엘리엇 파동 분석 실패: {e}")
            return TechnicalSignal("Elliott Wave", "HOLD", 30, "분석 불가", 0.3)
    
    def _count_rising_waves(self, changes: np.array) -> int:
        """상승 파동 카운트"""
        waves = 0
        in_rising = False
        
        for change in changes:
            if change > 0 and not in_rising:
                waves += 1
                in_rising = True
            elif change < 0:
                in_rising = False
        
        return waves
    
    def _count_falling_waves(self, changes: np.array) -> int:
        """하락 파동 카운트"""
        waves = 0
        in_falling = False
        
        for change in changes:
            if change < 0 and not in_falling:
                waves += 1
                in_falling = True
            elif change > 0:
                in_falling = False
        
        return waves

class CandlestickPatternAnalyzer:
    """🕯️ 캔들스틱 패턴 분석가 - 스티브 니슨"""
    
    def __init__(self):
        self.name = "Candlestick Patterns"
        self.description = "일본 캔들차트 패턴 분석"
    
    def analyze_patterns(self, open_prices: np.array, high: np.array, low: np.array, close: np.array) -> List[TechnicalSignal]:
        """캔들스틱 패턴 분석"""
        signals = []
        
        try:
            # 망치형 패턴
            hammer_signal = self._detect_hammer(open_prices[-3:], high[-3:], low[-3:], close[-3:])
            if hammer_signal:
                signals.append(hammer_signal)
            
            # 도지 패턴
            doji_signal = self._detect_doji(open_prices[-3:], high[-3:], low[-3:], close[-3:])
            if doji_signal:
                signals.append(doji_signal)
            
            # 삼천아래삼법 패턴
            engulfing_signal = self._detect_engulfing(open_prices[-5:], high[-5:], low[-5:], close[-5:])
            if engulfing_signal:
                signals.append(engulfing_signal)
            
        except Exception as e:
            logger.warning(f"캔들스틱 패턴 분석 실패: {e}")
        
        return signals
    
    def _detect_hammer(self, open_p: np.array, high: np.array, low: np.array, close: np.array) -> Optional[TechnicalSignal]:
        """망치형 패턴 감지"""
        if len(close) < 2:
            return None
        
        last_close = close[-1]
        last_open = open_p[-1]
        last_high = high[-1]
        last_low = low[-1]
        
        body_size = abs(last_close - last_open)
        lower_shadow = min(last_close, last_open) - last_low
        upper_shadow = last_high - max(last_close, last_open)
        
        # 망치형 조건: 아래 그림자가 몸통의 2배 이상, 위 그림자는 작음
        if lower_shadow >= body_size * 2 and upper_shadow <= body_size * 0.3:
            return TechnicalSignal(
                indicator_name="Hammer Pattern",
                signal_type="BUY",
                strength=75,
                description="망치형 반전 패턴 확인",
                confidence=0.75
            )
        return None
    
    def _detect_doji(self, open_p: np.array, high: np.array, low: np.array, close: np.array) -> Optional[TechnicalSignal]:
        """도지 패턴 감지"""
        if len(close) < 1:
            return None
        
        last_close = close[-1]
        last_open = open_p[-1]
        last_high = high[-1]
        last_low = low[-1]
        
        body_size = abs(last_close - last_open)
        total_range = last_high - last_low
        
        # 도지 조건: 몸통이 전체 범위의 5% 이하
        if body_size <= total_range * 0.05:
            return TechnicalSignal(
                indicator_name="Doji Pattern",
                signal_type="HOLD",
                strength=60,
                description="도지 패턴 - 방향성 불분명",
                confidence=0.6
            )
        return None
    
    def _detect_engulfing(self, open_p: np.array, high: np.array, low: np.array, close: np.array) -> Optional[TechnicalSignal]:
        """포용형 패턴 감지"""
        if len(close) < 2:
            return None
        
        prev_open, prev_close = open_p[-2], close[-2]
        curr_open, curr_close = open_p[-1], close[-1]
        
        # 강세 포용형
        if (prev_close < prev_open and  # 전일 음봉
            curr_close > curr_open and  # 당일 양봉
            curr_open < prev_close and  # 당일 시가가 전일 종가보다 낮음
            curr_close > prev_open):    # 당일 종가가 전일 시가보다 높음
            
            return TechnicalSignal(
                indicator_name="Bullish Engulfing",
                signal_type="BUY",
                strength=80,
                description="강세 포용형 - 상승 반전 신호",
                confidence=0.8
            )
        
        # 약세 포용형
        elif (prev_close > prev_open and  # 전일 양봉
              curr_close < curr_open and  # 당일 음봉
              curr_open > prev_close and  # 당일 시가가 전일 종가보다 높음
              curr_close < prev_open):    # 당일 종가가 전일 시가보다 낮음
            
            return TechnicalSignal(
                indicator_name="Bearish Engulfing",
                signal_type="SELL",
                strength=80,
                description="약세 포용형 - 하락 반전 신호",
                confidence=0.8
            )
        
        return None

class MovingAverageAnalyzer:
    """📊 이동평균 분석가 - 그랜빌의 법칙"""
    
    def __init__(self):
        self.name = "Moving Average Analysis"
        self.description = "이동평균선 배열과 그랜빌의 8법칙"
    
    def analyze_ma_signals(self, prices: np.array) -> List[TechnicalSignal]:
        """이동평균 신호 분석"""
        signals = []
        
        try:
            # 각종 이동평균 계산
            ma5 = pd.Series(prices).rolling(5).mean().values
            ma20 = pd.Series(prices).rolling(20).mean().values
            ma60 = pd.Series(prices).rolling(60).mean().values
            ma120 = pd.Series(prices).rolling(120).mean().values
            
            # 정배열/역배열 확인
            arrangement_signal = self._check_ma_arrangement(ma5, ma20, ma60, ma120, prices)
            if arrangement_signal:
                signals.append(arrangement_signal)
            
            # 골든크로스/데드크로스
            cross_signal = self._detect_ma_cross(ma20, ma60)
            if cross_signal:
                signals.append(cross_signal)
            
            # 그랜빌의 법칙
            granville_signal = self._apply_granville_rules(prices, ma20)
            if granville_signal:
                signals.append(granville_signal)
            
        except Exception as e:
            logger.warning(f"이동평균 분석 실패: {e}")
        
        return signals
    
    def _check_ma_arrangement(self, ma5: np.array, ma20: np.array, ma60: np.array, ma120: np.array, prices: np.array) -> Optional[TechnicalSignal]:
        """이동평균 배열 확인"""
        if len(ma120) < 5:
            return None
        
        current_price = prices[-1]
        current_ma5 = ma5[-1] if not np.isnan(ma5[-1]) else current_price
        current_ma20 = ma20[-1] if not np.isnan(ma20[-1]) else current_price  
        current_ma60 = ma60[-1] if not np.isnan(ma60[-1]) else current_price
        current_ma120 = ma120[-1] if not np.isnan(ma120[-1]) else current_price
        
        # 정배열 확인 (가격 > 5일선 > 20일선 > 60일선 > 120일선)
        if (current_price > current_ma5 > current_ma20 > current_ma60 > current_ma120):
            return TechnicalSignal(
                indicator_name="MA Arrangement",
                signal_type="BUY",
                strength=85,
                description="이동평균 정배열 - 강한 상승 추세",
                confidence=0.85
            )
        
        # 역배열 확인
        elif (current_price < current_ma5 < current_ma20 < current_ma60 < current_ma120):
            return TechnicalSignal(
                indicator_name="MA Arrangement", 
                signal_type="SELL",
                strength=85,
                description="이동평균 역배열 - 강한 하락 추세",
                confidence=0.85
            )
        
        return None
    
    def _detect_ma_cross(self, ma_short: np.array, ma_long: np.array) -> Optional[TechnicalSignal]:
        """이동평균 교차 감지"""
        if len(ma_short) < 2 or len(ma_long) < 2:
            return None
        
        # 최근 2일 데이터로 교차 확인
        short_prev, short_curr = ma_short[-2], ma_short[-1]
        long_prev, long_curr = ma_long[-2], ma_long[-1]
        
        # 골든크로스 (단기선이 장기선을 상향 돌파)
        if short_prev <= long_prev and short_curr > long_curr:
            return TechnicalSignal(
                indicator_name="Golden Cross",
                signal_type="BUY", 
                strength=75,
                description="골든크로스 발생 - 상승 전환",
                confidence=0.75
            )
        
        # 데드크로스 (단기선이 장기선을 하향 돌파)
        elif short_prev >= long_prev and short_curr < long_curr:
            return TechnicalSignal(
                indicator_name="Dead Cross",
                signal_type="SELL",
                strength=75, 
                description="데드크로스 발생 - 하락 전환",
                confidence=0.75
            )
        
        return None
    
    def _apply_granville_rules(self, prices: np.array, ma: np.array) -> Optional[TechnicalSignal]:
        """그랜빌의 8법칙 적용"""
        if len(prices) < 5 or len(ma) < 5:
            return None
        
        current_price = prices[-1]
        current_ma = ma[-1]
        
        # 법칙 1,2: 이동평균선 위에서 매수
        if current_price > current_ma * 1.02:  # 2% 이상 상회
            return TechnicalSignal(
                indicator_name="Granville Rule 1-2",
                signal_type="BUY",
                strength=70,
                description="그랜빌 매수 신호 - 이평선 위 강세",
                confidence=0.7
            )
        
        # 법칙 5,6: 이동평균선 아래에서 매도
        elif current_price < current_ma * 0.98:  # 2% 이상 하회
            return TechnicalSignal(
                indicator_name="Granville Rule 5-6",
                signal_type="SELL",
                strength=70,
                description="그랜빌 매도 신호 - 이평선 아래 약세",
                confidence=0.7
            )
        
        return None

class TechnicalAnalyzer:
    """📈 통합 기술적 분석 엔진"""
    
    def __init__(self):
        self.elliott_analyzer = ElliottWaveAnalyzer()
        self.candlestick_analyzer = CandlestickPatternAnalyzer()
        self.ma_analyzer = MovingAverageAnalyzer()
        self.indicator_calc = TechnicalIndicatorCalculator()
        
        logger.info("📈 기술적 분석 엔진 초기화 완료")
    
    def analyze_stock(self, stock: StockData, price_history: Dict[str, np.array]) -> TechnicalAnalysisResult:
        """단일 종목 기술적 분석"""
        try:
            all_signals = []
            
            # 가격 데이터 추출
            close_prices = price_history.get('close', np.array([stock.current_price] * 30))
            open_prices = price_history.get('open', close_prices)
            high_prices = price_history.get('high', close_prices) 
            low_prices = price_history.get('low', close_prices)
            
            # 1. 엘리엇 파동 분석
            elliott_signal = self.elliott_analyzer.analyze_wave_pattern(close_prices)
            all_signals.append(elliott_signal)
            
            # 2. 캔들스틱 패턴 분석
            candlestick_signals = self.candlestick_analyzer.analyze_patterns(
                open_prices, high_prices, low_prices, close_prices
            )
            all_signals.extend(candlestick_signals)
            
            # 3. 이동평균 분석
            ma_signals = self.ma_analyzer.analyze_ma_signals(close_prices)
            all_signals.extend(ma_signals)
            
            # 4. 기본 지표들
            rsi_signal = self._analyze_rsi(close_prices)
            if rsi_signal:
                all_signals.append(rsi_signal)
            
            macd_signal = self._analyze_macd(close_prices)
            if macd_signal:
                all_signals.append(macd_signal)
            
            # 5. 지지/저항 레벨 계산
            support_levels, resistance_levels = self._calculate_support_resistance(close_prices)
            
            # 6. 종합 점수 계산
            overall_score = self._calculate_overall_score(all_signals)
            
            # 7. 최종 추천
            recommendation = self._get_recommendation(overall_score)
            
            # 8. 트렌드 방향
            trend_direction = self._determine_trend(close_prices)
            
            # 9. 변동성 점수
            volatility_score = self._calculate_volatility(close_prices)
            
            return TechnicalAnalysisResult(
                symbol=stock.symbol,
                signals=all_signals,
                overall_score=overall_score,
                recommendation=recommendation,
                support_levels=support_levels,
                resistance_levels=resistance_levels,
                trend_direction=trend_direction,
                volatility_score=volatility_score
            )
            
        except Exception as e:
            logger.error(f"{stock.symbol} 기술적 분석 실패: {e}")
            return TechnicalAnalysisResult(
                symbol=stock.symbol,
                signals=[],
                overall_score=50,
                recommendation="HOLD",
                support_levels=[],
                resistance_levels=[],
                trend_direction="SIDEWAYS",
                volatility_score=50
            )
    
    def _analyze_rsi(self, prices: np.array) -> Optional[TechnicalSignal]:
        """RSI 분석"""
        try:
            rsi_values = self.indicator_calc.calculate_rsi(prices)
            current_rsi = rsi_values[-1]
            
            if current_rsi < 30:
                return TechnicalSignal(
                    indicator_name="RSI",
                    signal_type="BUY",
                    strength=80,
                    description=f"RSI 과매도 구간 ({current_rsi:.1f})",
                    confidence=0.8
                )
            elif current_rsi > 70:
                return TechnicalSignal(
                    indicator_name="RSI", 
                    signal_type="SELL",
                    strength=80,
                    description=f"RSI 과매수 구간 ({current_rsi:.1f})",
                    confidence=0.8
                )
            
            return None
        except:
            return None
    
    def _analyze_macd(self, prices: np.array) -> Optional[TechnicalSignal]:
        """MACD 분석"""
        try:
            macd, signal_line, histogram = self.indicator_calc.calculate_macd(prices)
            
            # 시그널 교차 확인
            if len(histogram) >= 2:
                if histogram[-2] < 0 and histogram[-1] > 0:
                    return TechnicalSignal(
                        indicator_name="MACD",
                        signal_type="BUY",
                        strength=75,
                        description="MACD 골든크로스",
                        confidence=0.75
                    )
                elif histogram[-2] > 0 and histogram[-1] < 0:
                    return TechnicalSignal(
                        indicator_name="MACD",
                        signal_type="SELL", 
                        strength=75,
                        description="MACD 데드크로스",
                        confidence=0.75
                    )
            
            return None
        except:
            return None
    
    def _calculate_support_resistance(self, prices: np.array) -> Tuple[List[float], List[float]]:
        """지지/저항선 계산"""
        try:
            # 최근 데이터로 피벗 포인트 계산
            recent_prices = prices[-20:] if len(prices) >= 20 else prices
            
            support_levels = []
            resistance_levels = []
            
            # 간단한 피벗 포인트 방식
            for i in range(2, len(recent_prices) - 2):
                # 지지선 (저점)
                if (recent_prices[i] < recent_prices[i-1] and 
                    recent_prices[i] < recent_prices[i-2] and
                    recent_prices[i] < recent_prices[i+1] and 
                    recent_prices[i] < recent_prices[i+2]):
                    support_levels.append(float(recent_prices[i]))
                
                # 저항선 (고점) 
                if (recent_prices[i] > recent_prices[i-1] and
                    recent_prices[i] > recent_prices[i-2] and
                    recent_prices[i] > recent_prices[i+1] and
                    recent_prices[i] > recent_prices[i+2]):
                    resistance_levels.append(float(recent_prices[i]))
            
            return support_levels[-3:], resistance_levels[-3:]  # 최근 3개씩만
        except:
            return [], []
    
    def _calculate_overall_score(self, signals: List[TechnicalSignal]) -> float:
        """종합 점수 계산"""
        if not signals:
            return 50.0
        
        buy_signals = [s for s in signals if s.signal_type == "BUY"]
        sell_signals = [s for s in signals if s.signal_type == "SELL"]
        
        buy_score = sum(s.strength * s.confidence for s in buy_signals)
        sell_score = sum(s.strength * s.confidence for s in sell_signals)
        
        total_weight = sum(s.confidence for s in signals)
        
        if total_weight == 0:
            return 50.0
        
        # 매수/매도 신호 균형 계산
        net_score = (buy_score - sell_score) / total_weight
        final_score = 50 + (net_score * 0.5)  # -100~100을 0~100으로 정규화
        
        return max(0, min(100, final_score))
    
    def _get_recommendation(self, score: float) -> str:
        """점수 기반 추천"""
        if score >= 70:
            return "STRONG_BUY"
        elif score >= 60:
            return "BUY"
        elif score >= 40:
            return "HOLD"
        elif score >= 30:
            return "SELL"
        else:
            return "STRONG_SELL"
    
    def _determine_trend(self, prices: np.array) -> str:
        """트렌드 방향 결정"""
        if len(prices) < 10:
            return "SIDEWAYS"
        
        recent_prices = prices[-10:]
        first_half = np.mean(recent_prices[:5])
        second_half = np.mean(recent_prices[5:])
        
        change_pct = (second_half - first_half) / first_half
        
        if change_pct > 0.02:
            return "UPTREND"
        elif change_pct < -0.02:
            return "DOWNTREND"
        else:
            return "SIDEWAYS"
    
    def _calculate_volatility(self, prices: np.array) -> float:
        """변동성 점수 계산 (0-100)"""
        if len(prices) < 2:
            return 50.0
        
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns) * np.sqrt(252)  # 연환산
        
        # 0-100 스케일로 정규화 (일반적으로 0.3 이상이면 높은 변동성)
        return min(100, volatility * 300)

if __name__ == "__main__":
    print("📈 차트 전문가 기술적 분석 엔진 v1.0")
    print("=" * 50)
    
    # 테스트 데이터
    test_prices = np.random.normal(100, 5, 100)
    test_prices = np.cumsum(np.random.normal(0, 0.01, 100)) + 100
    
    analyzer = TechnicalAnalyzer()
    
    # 테스트 주식 데이터
    test_stock = StockData(
        symbol="TEST",
        name="테스트 주식",
        current_price=test_prices[-1],
        rsi=50.0
    )
    
    # 가격 히스토리
    price_history = {
        'close': test_prices,
        'open': test_prices * 0.995,
        'high': test_prices * 1.01,
        'low': test_prices * 0.99
    }
    
    result = analyzer.analyze_stock(test_stock, price_history)
    
    print(f"\n📊 분석 결과:")
    print(f"  • 종목: {result.symbol}")
    print(f"  • 종합 점수: {result.overall_score:.1f}")
    print(f"  • 추천: {result.recommendation}")
    print(f"  • 트렌드: {result.trend_direction}")
    print(f"  • 신호 개수: {len(result.signals)}개")
    
    print("\n✅ 기술적 분석 엔진 테스트 완료!") 