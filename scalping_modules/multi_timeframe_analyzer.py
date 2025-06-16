"""
⏰ 멀티타임프레임 분석기
- 다양한 시간대별 추세 분석
- 타임프레임 간 신호 일치성 검증
- 최적 진입 타이밍 포착
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

class TimeFrame(Enum):
    """시간 프레임 정의"""
    MINUTE_1 = "1m"
    MINUTE_5 = "5m" 
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"

class TrendDirection(Enum):
    """추세 방향"""
    BULLISH = "상승"
    BEARISH = "하락"
    SIDEWAYS = "횡보"

@dataclass
class TimeFrameData:
    """타임프레임별 분석 데이터"""
    timeframe: TimeFrame
    trend_direction: TrendDirection
    trend_strength: float  # 0-100
    momentum_score: float  # -100 to +100
    support_resistance: Dict[str, float]
    signal_confidence: float  # 0-100
    last_updated: datetime

@dataclass
class MultiTimeFrameSignal:
    """멀티타임프레임 종합 신호"""
    symbol: str
    overall_trend: TrendDirection
    signal_strength: float  # 0-100
    timeframe_consensus: float  # 0-100 (일치성)
    recommended_action: str  # BUY, SELL, HOLD, WAIT
    entry_confidence: float  # 0-100
    timeframe_data: Dict[TimeFrame, TimeFrameData]
    timestamp: datetime

class MultiTimeframeAnalyzer:
    """멀티타임프레임 기술적 분석기"""
    
    def __init__(self):
        """멀티타임프레임 분석기 초기화"""
        # 분석할 타임프레임들 (스캘핑 중심)
        self.timeframes = [
            TimeFrame.MINUTE_1,
            TimeFrame.MINUTE_5,
            TimeFrame.MINUTE_15
        ]
        
        # 타임프레임별 가중치 (짧은 타임프레임 우선)
        self.timeframe_weights = {
            TimeFrame.MINUTE_1: 0.5,   # 단기 신호 중요
            TimeFrame.MINUTE_5: 0.3,   # 중기 확인
            TimeFrame.MINUTE_15: 0.2   # 장기 방향
        }
        
        # 이동평균 기간
        self.ma_periods = {
            'short': 5,
            'medium': 20,
            'long': 50
        }
        
        logger.info("⏰ 멀티타임프레임 분석기 초기화 완료")
    
    def analyze_timeframe(self, 
                         timeframe: TimeFrame,
                         prices: List[float],
                         volumes: List[int],
                         high_prices: List[float] = None,
                         low_prices: List[float] = None) -> Optional[TimeFrameData]:
        """
        단일 타임프레임 분석
        
        Args:
            timeframe: 분석할 타임프레임
            prices: 종가 데이터
            volumes: 거래량 데이터
            high_prices: 고가 데이터 (선택)
            low_prices: 저가 데이터 (선택)
            
        Returns:
            타임프레임 분석 결과
        """
        if len(prices) < self.ma_periods['long']:
            logger.warning(f"⚠️ {timeframe.value} 데이터 부족: {len(prices)}개")
            return None
        
        try:
            # 추세 방향 분석
            trend_direction = self._analyze_trend_direction(prices)
            
            # 추세 강도 계산
            trend_strength = self._calculate_trend_strength(prices, volumes)
            
            # 모멘텀 스코어 계산
            momentum_score = self._calculate_momentum_score(prices)
            
            # 지지/저항 계산
            support_resistance = self._calculate_support_resistance(
                prices, high_prices, low_prices
            )
            
            # 신호 신뢰도 계산
            signal_confidence = self._calculate_signal_confidence(
                trend_strength, momentum_score, len(prices)
            )
            
            return TimeFrameData(
                timeframe=timeframe,
                trend_direction=trend_direction,
                trend_strength=trend_strength,
                momentum_score=momentum_score,
                support_resistance=support_resistance,
                signal_confidence=signal_confidence,
                last_updated=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"❌ {timeframe.value} 분석 실패: {e}")
            return None
    
    def _analyze_trend_direction(self, prices: List[float]) -> TrendDirection:
        """추세 방향 분석 (이동평균 기반)"""
        try:
            if len(prices) < self.ma_periods['long']:
                return TrendDirection.SIDEWAYS
            
            # 이동평균 계산
            ma_short = sum(prices[-self.ma_periods['short']:]) / self.ma_periods['short']
            ma_medium = sum(prices[-self.ma_periods['medium']:]) / self.ma_periods['medium']
            ma_long = sum(prices[-self.ma_periods['long']:]) / self.ma_periods['long']
            
            current_price = prices[-1]
            
            # 추세 판단
            if (current_price > ma_short > ma_medium > ma_long):
                return TrendDirection.BULLISH
            elif (current_price < ma_short < ma_medium < ma_long):
                return TrendDirection.BEARISH
            else:
                return TrendDirection.SIDEWAYS
                
        except Exception as e:
            logger.error(f"❌ 추세 방향 분석 실패: {e}")
            return TrendDirection.SIDEWAYS
    
    def _calculate_trend_strength(self, prices: List[float], volumes: List[int]) -> float:
        """추세 강도 계산 (0-100)"""
        try:
            if len(prices) < 20:
                return 50.0
            
            # 가격 변화율
            price_change = (prices[-1] - prices[-20]) / prices[-20] * 100
            
            # 거래량 변화율
            recent_volume = sum(volumes[-5:]) / 5 if len(volumes) >= 5 else 0
            past_volume = sum(volumes[-20:-5]) / 15 if len(volumes) >= 20 else 1
            volume_ratio = recent_volume / past_volume if past_volume > 0 else 1
            
            # 변동성 계산 (표준편차)
            recent_prices = prices[-20:]
            avg_price = sum(recent_prices) / len(recent_prices)
            variance = sum((p - avg_price) ** 2 for p in recent_prices) / len(recent_prices)
            volatility = (variance ** 0.5) / avg_price * 100
            
            # 추세 강도 계산
            strength = (
                abs(price_change) * 30 +  # 가격 변화의 영향 (30%)
                min(volume_ratio * 20, 40) +  # 거래량 증가의 영향 (최대 40%)
                min(volatility * 10, 30)  # 변동성의 영향 (최대 30%)
            )
            
            return min(100, max(0, strength))
            
        except Exception as e:
            logger.error(f"❌ 추세 강도 계산 실패: {e}")
            return 50.0
    
    def _calculate_momentum_score(self, prices: List[float]) -> float:
        """모멘텀 스코어 계산 (-100 to +100)"""
        try:
            if len(prices) < 10:
                return 0.0
            
            # RSI 계산
            rsi = self._calculate_rsi(prices, period=14)
            
            # 가격 모멘텀 (단기/장기 비교)
            short_avg = sum(prices[-5:]) / 5
            long_avg = sum(prices[-20:]) / 20 if len(prices) >= 20 else sum(prices) / len(prices)
            price_momentum = (short_avg - long_avg) / long_avg * 100
            
            # 모멘텀 스코어 계산
            momentum = (
                (rsi - 50) * 1.0 +  # RSI 기반 모멘텀
                price_momentum * 2.0  # 가격 기반 모멘텀
            )
            
            return max(-100, min(100, momentum))
            
        except Exception as e:
            logger.error(f"❌ 모멘텀 스코어 계산 실패: {e}")
            return 0.0
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """RSI (Relative Strength Index) 계산"""
        try:
            if len(prices) < period + 1:
                return 50.0
            
            gains = []
            losses = []
            
            for i in range(1, len(prices)):
                change = prices[i] - prices[i-1]
                if change > 0:
                    gains.append(change)
                    losses.append(0)
                else:
                    gains.append(0)
                    losses.append(abs(change))
            
            if len(gains) < period:
                return 50.0
            
            avg_gain = sum(gains[-period:]) / period
            avg_loss = sum(losses[-period:]) / period
            
            if avg_loss == 0:
                return 100.0
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except Exception as e:
            logger.error(f"❌ RSI 계산 실패: {e}")
            return 50.0
    
    def _calculate_support_resistance(self, 
                                    prices: List[float],
                                    high_prices: List[float] = None,
                                    low_prices: List[float] = None) -> Dict[str, float]:
        """지지/저항선 계산"""
        try:
            if len(prices) < 20:
                current_price = prices[-1] if prices else 0
                return {
                    'support_1': current_price * 0.99,
                    'support_2': current_price * 0.98,
                    'resistance_1': current_price * 1.01,
                    'resistance_2': current_price * 1.02
                }
            
            # 최근 20개 데이터에서 지지/저항 찾기
            recent_prices = prices[-20:]
            recent_highs = high_prices[-20:] if high_prices and len(high_prices) >= 20 else recent_prices
            recent_lows = low_prices[-20:] if low_prices and len(low_prices) >= 20 else recent_prices
            
            # 지지선 계산 (최근 저점들의 평균)
            sorted_lows = sorted(recent_lows)
            support_1 = sum(sorted_lows[:5]) / 5  # 하위 5개 평균
            support_2 = sum(sorted_lows[:3]) / 3  # 하위 3개 평균
            
            # 저항선 계산 (최근 고점들의 평균)
            sorted_highs = sorted(recent_highs, reverse=True)
            resistance_1 = sum(sorted_highs[:5]) / 5  # 상위 5개 평균
            resistance_2 = sum(sorted_highs[:3]) / 3  # 상위 3개 평균
            
            return {
                'support_1': support_1,
                'support_2': support_2,
                'resistance_1': resistance_1,
                'resistance_2': resistance_2
            }
            
        except Exception as e:
            logger.error(f"❌ 지지/저항선 계산 실패: {e}")
            current_price = prices[-1] if prices else 0
            return {
                'support_1': current_price * 0.99,
                'support_2': current_price * 0.98,
                'resistance_1': current_price * 1.01,
                'resistance_2': current_price * 1.02
            }
    
    def _calculate_signal_confidence(self, 
                                   trend_strength: float,
                                   momentum_score: float,
                                   data_points: int) -> float:
        """신호 신뢰도 계산 (0-100)"""
        try:
            # 데이터 충분성 점수
            data_score = min(100, (data_points / 50) * 100)
            
            # 추세 명확성 점수
            trend_score = trend_strength
            
            # 모멘텀 일관성 점수
            momentum_consistency = 100 - abs(momentum_score)  # 극단적 모멘텀은 불안정
            
            # 종합 신뢰도
            confidence = (
                data_score * 0.2 +
                trend_score * 0.4 +
                momentum_consistency * 0.4
            )
            
            return min(100, max(0, confidence))
            
        except Exception as e:
            logger.error(f"❌ 신호 신뢰도 계산 실패: {e}")
            return 50.0
    
    def analyze_multiple_timeframes(self, 
                                  symbol: str,
                                  timeframe_data: Dict[TimeFrame, Dict[str, List]]) -> Optional[MultiTimeFrameSignal]:
        """
        멀티타임프레임 종합 분석
        
        Args:
            symbol: 종목 코드
            timeframe_data: {timeframe: {'prices': [...], 'volumes': [...], ...}}
            
        Returns:
            멀티타임프레임 종합 신호
        """
        try:
            analyzed_timeframes = {}
            
            # 각 타임프레임별 분석
            for timeframe in self.timeframes:
                if timeframe not in timeframe_data:
                    logger.warning(f"⚠️ {symbol} {timeframe.value} 데이터 없음")
                    continue
                
                data = timeframe_data[timeframe]
                prices = data.get('prices', [])
                volumes = data.get('volumes', [])
                highs = data.get('highs', None)
                lows = data.get('lows', None)
                
                timeframe_analysis = self.analyze_timeframe(
                    timeframe, prices, volumes, highs, lows
                )
                
                if timeframe_analysis:
                    analyzed_timeframes[timeframe] = timeframe_analysis
            
            if not analyzed_timeframes:
                logger.warning(f"⚠️ {symbol} 분석 가능한 타임프레임 없음")
                return None
            
            # 종합 신호 계산
            overall_signal = self._calculate_overall_signal(symbol, analyzed_timeframes)
            
            return overall_signal
            
        except Exception as e:
            logger.error(f"❌ {symbol} 멀티타임프레임 분석 실패: {e}")
            return None
    
    def _calculate_overall_signal(self, 
                                symbol: str,
                                timeframe_analyses: Dict[TimeFrame, TimeFrameData]) -> MultiTimeFrameSignal:
        """종합 신호 계산"""
        try:
            # 가중 평균 계산을 위한 변수들
            weighted_momentum = 0.0
            weighted_strength = 0.0
            total_weight = 0.0
            
            # 추세 방향별 투표
            trend_votes = {
                TrendDirection.BULLISH: 0.0,
                TrendDirection.BEARISH: 0.0,
                TrendDirection.SIDEWAYS: 0.0
            }
            
            # 신뢰도 수집
            confidences = []
            
            for timeframe, analysis in timeframe_analyses.items():
                weight = self.timeframe_weights.get(timeframe, 0.1)
                
                # 가중 평균 계산
                weighted_momentum += analysis.momentum_score * weight
                weighted_strength += analysis.trend_strength * weight
                total_weight += weight
                
                # 추세 투표
                trend_votes[analysis.trend_direction] += weight
                
                # 신뢰도 수집
                confidences.append(analysis.signal_confidence)
            
            # 정규화
            if total_weight > 0:
                weighted_momentum /= total_weight
                weighted_strength /= total_weight
            
            # 전체 추세 결정
            overall_trend = max(trend_votes, key=trend_votes.get)
            
            # 타임프레임 간 일치성 계산
            max_vote = max(trend_votes.values())
            consensus = (max_vote / sum(trend_votes.values())) * 100 if sum(trend_votes.values()) > 0 else 0
            
            # 신호 강도 계산
            signal_strength = (weighted_strength + abs(weighted_momentum)) / 2
            
            # 진입 신뢰도 계산
            avg_confidence = sum(confidences) / len(confidences) if confidences else 50
            entry_confidence = (avg_confidence + consensus + signal_strength) / 3
            
            # 권장 액션 결정
            recommended_action = self._determine_action(
                overall_trend, signal_strength, consensus, entry_confidence
            )
            
            return MultiTimeFrameSignal(
                symbol=symbol,
                overall_trend=overall_trend,
                signal_strength=signal_strength,
                timeframe_consensus=consensus,
                recommended_action=recommended_action,
                entry_confidence=entry_confidence,
                timeframe_data=timeframe_analyses,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"❌ {symbol} 종합 신호 계산 실패: {e}")
            # 기본 신호 반환
            return MultiTimeFrameSignal(
                symbol=symbol,
                overall_trend=TrendDirection.SIDEWAYS,
                signal_strength=50.0,
                timeframe_consensus=50.0,
                recommended_action="WAIT",
                entry_confidence=50.0,
                timeframe_data=timeframe_analyses,
                timestamp=datetime.now()
            )
    
    def _determine_action(self, 
                         trend: TrendDirection,
                         strength: float,
                         consensus: float,
                         confidence: float) -> str:
        """권장 액션 결정"""
        try:
            # 최소 기준값들
            MIN_STRENGTH = 60.0
            MIN_CONSENSUS = 70.0
            MIN_CONFIDENCE = 65.0
            
            # 강한 신호 조건
            strong_signal = (
                strength >= MIN_STRENGTH and
                consensus >= MIN_CONSENSUS and
                confidence >= MIN_CONFIDENCE
            )
            
            if not strong_signal:
                return "WAIT"
            
            # 추세에 따른 액션
            if trend == TrendDirection.BULLISH:
                return "BUY"
            elif trend == TrendDirection.BEARISH:
                return "SELL"
            else:
                return "HOLD"
                
        except Exception as e:
            logger.error(f"❌ 액션 결정 실패: {e}")
            return "WAIT"
    
    def get_best_entry_timeframe(self, signal: MultiTimeFrameSignal) -> Optional[TimeFrame]:
        """최적 진입 타임프레임 선택"""
        try:
            if not signal.timeframe_data:
                return None
            
            # 신뢰도와 추세 강도가 가장 높은 타임프레임 선택
            best_timeframe = None
            best_score = 0.0
            
            for timeframe, data in signal.timeframe_data.items():
                # 스코어 = 신뢰도 * 0.6 + 추세강도 * 0.4
                score = data.signal_confidence * 0.6 + data.trend_strength * 0.4
                
                if score > best_score:
                    best_score = score
                    best_timeframe = timeframe
            
            return best_timeframe
            
        except Exception as e:
            logger.error(f"❌ 최적 진입 타임프레임 선택 실패: {e}")
            return None
    
    def get_trading_levels(self, signal: MultiTimeFrameSignal) -> Dict[str, float]:
        """멀티타임프레임 기반 매매 수준 계산"""
        try:
            if not signal.timeframe_data:
                return {}
            
            # 모든 타임프레임의 지지/저항을 종합
            all_supports = []
            all_resistances = []
            
            for timeframe, data in signal.timeframe_data.items():
                weight = self.timeframe_weights.get(timeframe, 0.1)
                sr = data.support_resistance
                
                # 가중치 적용
                all_supports.extend([sr['support_1']] * int(weight * 10))
                all_supports.extend([sr['support_2']] * int(weight * 5))
                all_resistances.extend([sr['resistance_1']] * int(weight * 10))
                all_resistances.extend([sr['resistance_2']] * int(weight * 5))
            
            if not all_supports or not all_resistances:
                return {}
            
            # 통계적 수준 계산
            supports_sorted = sorted(all_supports)
            resistances_sorted = sorted(all_resistances, reverse=True)
            
            return {
                'strong_support': supports_sorted[len(supports_sorted)//4],  # 하위 25%
                'support': sum(supports_sorted[:len(supports_sorted)//2]) / (len(supports_sorted)//2),
                'resistance': sum(resistances_sorted[:len(resistances_sorted)//2]) / (len(resistances_sorted)//2),
                'strong_resistance': resistances_sorted[len(resistances_sorted)//4]  # 상위 25%
            }
            
        except Exception as e:
            logger.error(f"❌ 매매 수준 계산 실패: {e}")
            return {} 