"""
⏰ 멀티타임프레임 분석기
- 다양한 시간대별 추세 분석
- 타임프레임 간 신호 일치성 검증
- 최적 진입 타이밍 포착
- v1.1.0 (2024-07-26): 리팩토링 및 구조 개선
"""
import logging
from typing import Dict, List, Optional, NamedTuple, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from statistics import mean, stdev

logger = logging.getLogger(__name__)

# --- 열거형 및 데이터 클래스 정의 ---

class TimeFrame(Enum):
    """분석할 시간 프레임 정의"""
    MINUTE_1 = "1m"
    MINUTE_5 = "5m" 
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"

class Trend(Enum):
    """추세 방향"""
    BULLISH = "상승"
    BEARISH = "하락"
    SIDEWAYS = "횡보"

@dataclass
class TimeFrameAnalysis:
    """단일 타임프레임 분석 결과를 담는 데이터 클래스"""
    timeframe: TimeFrame
    trend: Trend
    trend_strength: float  # 0-100
    momentum_score: float  # -100 to +100
    confidence: float  # 0-100
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class MultiTimeFrameSignal:
    """멀티타임프레임 종합 분석 결과를 담는 데이터 클래스"""
    symbol: str
    overall_trend: Trend
    trend_consensus: float  # 0-100 (추세 일치도)
    action_confidence: float # 0-100 (매매 행위 신뢰도)
    recommended_action: str  # BUY, SELL, HOLD
    details: Dict[TimeFrame, TimeFrameAnalysis] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

# --- 메인 분석기 클래스 ---

class MultiTimeframeAnalyzer:
    """
    여러 시간대의 가격 데이터를 종합적으로 분석하여,
    추세의 방향성과 일치성을 평가하고 최종 매매 결정을 내립니다.
    """
    # 스캘핑에 사용할 타임프레임과 가중치 설정
    TIMEFRAMES_FOR_ANALYSIS = [TimeFrame.MINUTE_1, TimeFrame.MINUTE_5, TimeFrame.MINUTE_15]
    TIMEFRAME_WEIGHTS = {TimeFrame.MINUTE_1: 0.5, TimeFrame.MINUTE_5: 0.3, TimeFrame.MINUTE_15: 0.2}
    
    # 분석에 사용할 이동평균 기간 설정
    MA_PERIODS = {'short': 5, 'medium': 20, 'long': 50}
    
    def __init__(self):
        logger.info(f"⏰ 멀티타임프레임 분석기 초기화 (대상: {[tf.value for tf in self.TIMEFRAMES_FOR_ANALYSIS]})")
    
    # --- Public API ---

    def analyze_single_timeframe(self, timeframe: TimeFrame, 
                                 prices: List[float]) -> Optional[TimeFrameAnalysis]:
        """
        단일 타임프레임의 추세와 모멘텀을 분석합니다.
        
        Args:
            timeframe: 분석할 시간 프레임 (e.g., TimeFrame.MINUTE_5)
            prices: 해당 시간 프레임의 종가 리스트
            
        Returns:
            분석된 TimeFrameAnalysis 객체 또는 데이터 부족 시 None
        """
        if len(prices) < self.MA_PERIODS['long']:
            logger.debug(f"{timeframe.value}: 분석을 위한 데이터 부족")
            return None
        
        try:
            trend = self._determine_trend(prices)
            trend_strength = self._calculate_trend_strength(prices)
            momentum = self._calculate_momentum(prices)
            confidence = (trend_strength * 0.7) + ((momentum + 100) / 2 * 0.3) # 추세 강도 70%, 모멘텀 30%
            
            return TimeFrameAnalysis(
                timeframe=timeframe,
                trend=trend,
                trend_strength=round(trend_strength, 1),
                momentum_score=round(momentum, 1),
                confidence=round(confidence, 1)
            )
        except Exception as e:
            logger.error(f"❌ {timeframe.value} 분석 실패: {e}", exc_info=True)
            return None
    
    def create_multi_timeframe_signal(self, symbol: str, 
                                      analyses: Dict[TimeFrame, TimeFrameAnalysis]) -> MultiTimeFrameSignal:
        """
        여러 타임프레임의 분석 결과를 종합하여 최종 신호를 생성합니다.
        
        Args:
            symbol: 종목 코드
            analyses: 각 타임프레임별 분석 결과 딕셔너리
            
        Returns:
            종합적인 MultiTimeFrameSignal 객체
        """
        # 가중치를 적용하여 전체 추세와 신뢰도 계산
        weighted_trend_score = 0
        total_weight = 0
        for tf, data in analyses.items():
            weight = self.TIMEFRAME_WEIGHTS.get(tf, 0)
            if weight > 0:
                trend_value = 1 if data.trend == Trend.BULLISH else -1 if data.trend == Trend.BEARISH else 0
                weighted_trend_score += trend_value * data.confidence * weight
                total_weight += weight
                
        overall_confidence = abs(weighted_trend_score)
        overall_trend = self._score_to_trend(weighted_trend_score)
            
        # 타임프레임 간 추세 일치도 계산
        trend_consensus = self._calculate_trend_consensus(analyses)
        
        # 최종 매매 결정
        action = self._determine_final_action(overall_trend, overall_confidence, trend_consensus)
            
            return MultiTimeFrameSignal(
                symbol=symbol,
                overall_trend=overall_trend,
            trend_consensus=round(trend_consensus, 1),
            action_confidence=round(overall_confidence, 1),
            recommended_action=action,
            details=analyses
        )

    # --- Private Calculation Methods ---

    def _determine_trend(self, prices: List[float]) -> Trend:
        """이동평균선의 배열을 기반으로 현재 추세를 판단합니다."""
        mas = {p_name: mean(prices[-p_len:]) for p_name, p_len in self.MA_PERIODS.items()}
        
        is_bullish = mas['short'] > mas['medium'] > mas['long']
        is_bearish = mas['short'] < mas['medium'] < mas['long']
        
        if is_bullish: return Trend.BULLISH
        if is_bearish: return Trend.BEARISH
        return Trend.SIDEWAYS

    def _calculate_trend_strength(self, prices: List[float]) -> float:
        """추세의 강도를 ADX(Average Directional Index)와 유사한 방식으로 계산합니다."""
        n = self.MA_PERIODS['medium'] # 중기 추세 강도
        if len(prices) < n: return 0.0

        # 가격 변화율의 절대값을 통해 추세의 강도 측정
        abs_price_changes = [abs(prices[i] - prices[i-1]) for i in range(1, len(prices))]
        avg_change = mean(abs_price_changes[-n:])
        avg_price = mean(prices[-n:])
        
        # 변동성 대비 추세 강도 (0~100 정규화)
        strength = math.tanh(avg_change / (avg_price * 0.01 + 1e-6)) * 100
        return strength
        
    def _calculate_momentum(self, prices: List[float]) -> float:
        """RSI(Relative Strength Index)를 기반으로 모멘텀을 계산합니다."""
        period = 14
        if len(prices) < period + 1: return 0.0 # 중립

        gains = []
        losses = []
        for i in range(len(prices) - period, len(prices)):
            change = prices[i] - prices[i-1]
            gains.append(change if change > 0 else 0)
            losses.append(abs(change) if change < 0 else 0)
        
        avg_gain = mean(gains)
        avg_loss = mean(losses)

        if avg_loss == 0: return 100.0 # 과매수
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # RSI를 -100 ~ 100 범위의 모멘텀 점수로 변환
        return (rsi - 50) * 2

    def _calculate_trend_consensus(self, analyses: Dict[TimeFrame, TimeFrameAnalysis]) -> float:
        """여러 타임프레임 간 추세 방향이 얼마나 일치하는지 계산합니다."""
        trends = [d.trend for d in analyses.values()]
        if not trends: return 0.0
        
        bullish_count = trends.count(Trend.BULLISH)
        bearish_count = trends.count(Trend.BEARISH)
        
        # 가장 우세한 추세의 비율을 일치도 점수로 변환
        max_consensus = max(bullish_count, bearish_count) / len(trends)
        return max_consensus * 100

    def _score_to_trend(self, score: float) -> Trend:
        """가중치가 적용된 종합 점수를 최종 추세로 변환합니다."""
        if score > 20: return Trend.BULLISH  # 신뢰도 20 이상
        if score < -20: return Trend.BEARISH # 신뢰도 -20 이하
        return Trend.SIDEWAYS

    def _determine_final_action(self, trend: Trend, confidence: float, consensus: float) -> str:
        """종합된 분석 결과를 바탕으로 최종 매매 행위를 결정합니다."""
        if trend == Trend.BULLISH and confidence > 50 and consensus > 60:
            return "BUY" # 강한 상승 추세 및 높은 일치도
        if trend == Trend.BEARISH and confidence > 50 and consensus > 60:
            return "SELL" # 강한 하락 추세 및 높은 일치도
        
        return "HOLD" # 조건 미충족 시 관망 