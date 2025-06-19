"""
📊 ATR (Average True Range) 분석기
- 변동성 기반 매매 신호 생성
- 스캘핑에 적합한 종목 선별
- 진입/청산 타이밍 최적화
- v1.1.0 (2024-07-26): 리팩토링 및 구조 개선
"""

import logging
import math
from typing import List, Dict, Optional, NamedTuple
from dataclasses import dataclass, field
from datetime import datetime
from statistics import mean

logger = logging.getLogger(__name__)

# --- 데이터 클래스 정의 ---

@dataclass
class ATRData:
    """ATR 분석 결과를 담는 데이터 클래스"""
    symbol: str
    atr_value: float
    atr_percentage: float
    volatility_level: str  # LOW, MEDIUM, HIGH, EXTREME
    scalping_suitability: float  # 0-100 점수
    timestamp: datetime = field(default_factory=datetime.now)

class TradingSignalLevels(NamedTuple):
    """ATR 기반 매매 신호 레벨을 담는 튜플"""
    resistance_2: float
    resistance_1: float
    pivot: float
    support_1: float
    support_2: float
    position_size_multiplier: float

# --- 메인 분석기 클래스 ---

class ATRAnalyzer:
    """
    ATR(Average True Range)을 기반으로 자산의 변동성을 분석하고,
    스캘핑 적합도를 평가하여 매매 신호 생성을 돕습니다.
    """
    
    def __init__(self, 
                 optimal_atr_min: float = 0.5,
                 optimal_atr_max: float = 3.0,
                 period: int = 14):
        """
        ATR 분석기 초기화
        
        Args:
            optimal_atr_min: 스캘핑에 최적인 ATR 최소값 (%)
            optimal_atr_max: 스캘핑에 최적인 ATR 최대값 (%)
            period: ATR 계산 기간
        """
        if not 0 < optimal_atr_min < optimal_atr_max:
            raise ValueError("ATR 최적 범위 설정이 유효하지 않습니다.")
        self.optimal_atr_min = optimal_atr_min
        self.optimal_atr_max = optimal_atr_max
        self.period = period
        
        logger.info(f"📊 ATR 분석기 초기화: 최적 범위 {optimal_atr_min}%-{optimal_atr_max}%, 기간 {period}일")

    # --- Public API ---
    
    def analyze_volatility(self, symbol: str, 
                           high_prices: List[float],
                           low_prices: List[float], 
                           close_prices: List[float]) -> Optional[ATRData]:
        """
        주어진 가격 데이터로 변동성 분석을 수행합니다.
        
        Args:
            symbol: 종목 코드
            high_prices: 고가 리스트
            low_prices: 저가 리스트
            close_prices: 종가 리스트
            
        Returns:
            분석된 ATR 데이터 객체 또는 데이터 부족 시 None
        """
        if len(close_prices) <= self.period:
            logger.debug(f"⚠️ {symbol}: ATR 계산을 위한 데이터 부족 ({len(close_prices)}/{self.period + 1})")
            return None
        
        try:
            atr_value = self.calculate_atr(high_prices, low_prices, close_prices)
            if atr_value is None or atr_value == 0:
                return None
            
            current_price = close_prices[-1]
            atr_percentage = (atr_value / current_price) * 100
            
            volatility_level = self._classify_volatility(atr_percentage)
            scalping_score = self._calculate_scalping_suitability(atr_percentage)
            
            return ATRData(
                symbol=symbol,
                atr_value=round(atr_value, 4),
                atr_percentage=round(atr_percentage, 2),
                volatility_level=volatility_level,
                scalping_suitability=round(scalping_score, 1)
            )
            
        except Exception as e:
            logger.error(f"❌ {symbol} ATR 분석 중 오류 발생: {e}", exc_info=True)
            return None

    def calculate_atr(self, high_prices: List[float], 
                      low_prices: List[float], 
                      close_prices: List[float]) -> Optional[float]:
        """
        ATR(Average True Range) 값을 계산합니다.
        
        Returns:
            계산된 ATR 값 또는 실패 시 None
        """
        try:
            true_ranges = self._calculate_true_ranges(high_prices, low_prices, close_prices)
            if not true_ranges:
                return None
            
            # EMA(지수 이동 평균) 방식의 ATR이 더 일반적이나, 여기서는 SMA(단순 이동 평균) 사용
            # 참고: 첫 ATR은 SMA, 이후는 EMA로 계산하는 방식도 널리 쓰임
            atr = mean(true_ranges[-self.period:])
            return atr
                
        except Exception as e:
            logger.error(f"❌ ATR 값 계산 실패: {e}", exc_info=True)
            return None

    def get_trading_signal_levels(self, atr_data: ATRData, current_price: float) -> TradingSignalLevels:
        """
        ATR 분석 데이터를 기반으로 지지/저항 레벨을 생성합니다.
        
        Args:
            atr_data: `analyze_volatility`에서 얻은 ATR 데이터
            current_price: 현재가
            
        Returns:
            지지/저항 레벨이 담긴 `TradingSignalLevels` 객체
        """
        atr = atr_data.atr_value
        multiplier = self._get_position_size_multiplier(atr_data.volatility_level)
        
        return TradingSignalLevels(
            resistance_2=current_price + (atr * 1.5),
            resistance_1=current_price + (atr * 0.75),
            pivot=current_price,
            support_1=current_price - (atr * 0.75),
            support_2=current_price - (atr * 1.5),
            position_size_multiplier=multiplier
        )

    # --- Private Helper Methods ---

    @staticmethod
    def _calculate_true_ranges(high: List[float], low: List[float], close: List[float]) -> List[float]:
        """True Range 값들의 리스트를 계산합니다."""
        if len(close) < 2:
            return []
            
        true_ranges = []
        for i in range(1, len(close)):
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i-1])
            tr3 = abs(low[i] - close[i-1])
            true_ranges.append(max(tr1, tr2, tr3))
        return true_ranges

    def _classify_volatility(self, atr_percentage: float) -> str:
        """ATR 퍼센티지를 기반으로 변동성 수준을 분류합니다."""
        if atr_percentage < self.optimal_atr_min: return "LOW"
        if atr_percentage <= self.optimal_atr_max: return "MEDIUM"
        if atr_percentage < self.optimal_atr_max * 2: return "HIGH"
        return "EXTREME"
    
    def _calculate_scalping_suitability(self, atr_percentage: float) -> float:
        """스캘핑 적합성 점수를 0-100 사이의 값으로 계산합니다."""
        min_opt, max_opt = self.optimal_atr_min, self.optimal_atr_max
        
        if atr_percentage < min_opt:
            return self._score_below_optimal(atr_percentage)
        elif atr_percentage <= max_opt:
            return self._score_in_optimal_range(atr_percentage)
        else:
            return self._score_above_optimal(atr_percentage)

    def _score_in_optimal_range(self, atr_percentage: float) -> float:
        """최적 범위 내에서의 점수를 계산합니다."""
        min_opt, max_opt = self.optimal_atr_min, self.optimal_atr_max
        center = (min_opt + max_opt) / 2
        max_dist = (max_opt - min_opt) / 2
        
        # 중심에서 멀어질수록 점수 감소
        normalized_dist = abs(atr_percentage - center) / max_dist
        score = 100 * (1 - normalized_dist * 0.2) # 최적 범위 내에서는 최소 80점 보장
        return max(80, score)

    def _score_below_optimal(self, atr_percentage: float) -> float:
        """최적 범위보다 낮을 때의 점수를 계산합니다. (변동성 부족)"""
        # 0에 가까울수록 점수가 낮아짐
        score = 80 * (atr_percentage / self.optimal_atr_min) ** 0.5 # 제곱근을 취해 완만하게 감소
        return max(10, score)

    def _score_above_optimal(self, atr_percentage: float) -> float:
        """최적 범위보다 높을 때의 점수를 계산합니다. (과도한 변동성)"""
        # 최적 범위를 초과하는 비율이 클수록 점수가 급격히 감소
        excess_ratio = (atr_percentage - self.optimal_atr_max) / self.optimal_atr_max
        penalty = excess_ratio * 120 # 패널티 강화
        score = 80 - penalty
        return max(0, score)

    @staticmethod
    def _get_position_size_multiplier(volatility_level: str) -> float:
        """변동성 수준에 따른 포지션 크기 배율을 반환합니다."""
        multipliers = {
            "LOW": 1.2,
            "MEDIUM": 1.0,
            "HIGH": 0.8,
            "EXTREME": 0.5,
        }
        return multipliers.get(volatility_level, 1.0)
    
    # --- Deprecated / Helper for other modules ---

    def calculate_quick_atr(self, prices: List[float], symbol: str = "Unknown") -> Dict[str, float]:
        """
        [다른 모듈과의 호환성을 위한 헬퍼 함수]
        간단한 종가 리스트만으로 ATR 분석을 모의 수행합니다.
        정확한 high/low 데이터가 없어 추정치이므로, 테스트 용도로만 사용해야 합니다.
        """
        if len(prices) < self.period:
            return {'atr_percentage': 0, 'scalping_suitability': 0}
        
        # 가상의 high/low 데이터 생성 (단순 추정)
        high_prices = [p * 1.005 for p in prices]
        low_prices = [p * 0.995 for p in prices]
        
        analysis = self.analyze_volatility(symbol, high_prices, low_prices, prices)
        
        if analysis:
            return {
                'atr_percentage': analysis.atr_percentage,
                'scalping_suitability': analysis.scalping_suitability
            }
        return {'atr_percentage': 0, 'scalping_suitability': 0}

    def analyze_multiple_symbols(self, symbols_data: Dict[str, Dict[str, List[float]]]) -> List[ATRData]:
        """
        여러 종목의 ATR 분석
        
        Args:
            symbols_data: {symbol: {'high': [...], 'low': [...], 'close': [...]}}
            
        Returns:
            ATR 분석 결과 리스트 (스캘핑 적합성 순으로 정렬)
        """
        results = []
        
        for symbol, data in symbols_data.items():
            try:
                high_prices = data.get('high', [])
                low_prices = data.get('low', [])
                close_prices = data.get('close', [])
                
                atr_result = self.analyze_volatility(symbol, high_prices, low_prices, close_prices)
                if atr_result:
                    results.append(atr_result)
                    
            except Exception as e:
                logger.warning(f"⚠️ {symbol} ATR 분석 건너뜀: {e}")
                continue
        
        # 스캘핑 적합성 점수로 정렬
        results.sort(key=lambda x: x.scalping_suitability, reverse=True)
        
        logger.info(f"📊 ATR 분석 완료: {len(results)}개 종목")
        return results
    
    def get_optimal_symbols(self, atr_results: List[ATRData], 
                           max_count: int = 10) -> List[ATRData]:
        """
        스캘핑에 최적인 종목 선별
        
        Args:
            atr_results: ATR 분석 결과 리스트
            max_count: 최대 선별 종목 수
            
        Returns:
            최적 종목 리스트
        """
        # 최소 점수 기준 (60점 이상)
        MIN_SCORE = 60.0
        
        optimal_symbols = [
            result for result in atr_results 
            if result.scalping_suitability >= MIN_SCORE
        ]
        
        # 상위 종목만 선택
        optimal_symbols = optimal_symbols[:max_count]
        
        logger.info(f"🎯 스캘핑 최적 종목: {len(optimal_symbols)}개 선별")
        for result in optimal_symbols:
            logger.info(f"  - {result.symbol}: {result.scalping_suitability:.1f}점 "
                       f"(ATR: {result.atr_percentage:.2f}%, {result.volatility_level})")
        
        return optimal_symbols 