"""
⚡ 모멘텀 스코어링 시스템
- 실시간 모멘텀 강도 측정
- 단기 매매 신호 생성
- 가격/거래량 기반 복합 분석
- v1.1.0 (2024-07-26): 리팩토링 및 구조 개선
"""

import logging
import math
from typing import List, Dict, Optional, NamedTuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
from statistics import mean, stdev

logger = logging.getLogger(__name__)

# --- 데이터 클래스 정의 ---

@dataclass
class MomentumData:
    """모멘텀 분석 결과를 담는 데이터 클래스"""
    symbol: str
    price_momentum: float
    volume_momentum: float
    combined_score: float
    strength: str  # WEAK, MODERATE, STRONG, EXTREME
    direction: str  # BULLISH, BEARISH, NEUTRAL
    acceleration: float  # 모멘텀 가속도
    timestamp: datetime = field(default_factory=datetime.now)

class MomentumSignal(NamedTuple):
    """모멘텀 기반 매매 신호를 담는 튜플"""
    action: str  # 'BUY', 'SELL', 'HOLD'
    strength_score: float # 0-100, 신호 강도
    message: str

@dataclass
class SymbolDataBuffer:
    """종목별 시계열 데이터 및 모멘텀 기록을 관리하는 버퍼"""
    prices: deque
    volumes: deque
    momentum_history: deque

# --- 메인 분석기 클래스 ---

class MomentumScorer:
    """
    가격 및 거래량 데이터를 기반으로 자산의 모멘텀을 정량화하고,
    이를 바탕으로 스캘핑 전략에 활용 가능한 분석 데이터를 생성합니다.
    """
    PRICE_MOMENTUM_WEIGHTS = {'short': 0.5, 'medium': 0.3, 'long': 0.2}
    VOLUME_MOMENTUM_WEIGHTS = {'base': 0.7, 'trend': 0.3}
    COMBINED_SCORE_WEIGHTS = {'price': 0.65, 'volume': 0.35}

    def __init__(self, short_period: int = 5, medium_period: int = 20, long_period: int = 50):
        self.periods = {'short': short_period, 'medium': medium_period, 'long': long_period}
        
        # 종목별 데이터 버퍼 관리
        self.buffers: Dict[str, SymbolDataBuffer] = {}
        
        logger.info(f"⚡ 모멘텀 스코어러 초기화 (기간: {short_period}/{medium_period}/{long_period})")

    # --- Public API ---

    def calculate_batch_momentum(self, symbol: str, prices: List[float], 
                                 volumes: List[int]) -> Optional[MomentumData]:
        """일괄(batch) 데이터를 사용하여 모멘텀을 한 번에 계산합니다."""
        if len(prices) < self.periods['long']:
            logger.debug(f"{symbol}: 배치 모멘텀 계산을 위한 데이터 부족")
            return None
        
        try:
            price_mom = self._calculate_price_momentum(prices)
            volume_mom = self._calculate_volume_momentum(volumes)
            
            # 시너지 보너스 적용
            synergy_bonus = self._get_synergy_bonus(price_mom, volume_mom)
            
            # 종합 점수 계산
            combined_score = (price_mom * self.COMBINED_SCORE_WEIGHTS['price'] +
                            volume_mom * self.COMBINED_SCORE_WEIGHTS['volume'] +
                            synergy_bonus)
            
            # 결과 정규화 및 생성
            return self._build_momentum_data(symbol, price_mom, volume_mom, combined_score)

        except Exception as e:
            logger.error(f"❌ {symbol} 배치 모멘텀 계산 실패: {e}", exc_info=True)
            return None

    def get_trading_signal(self, momentum_data: MomentumData) -> MomentumSignal:
        """모멘텀 데이터를 해석하여 구체적인 매매 신호를 생성합니다."""
        score = momentum_data.combined_score
        strength_map = {'WEAK': 20, 'MODERATE': 50, 'STRONG': 80, 'EXTREME': 95}
        strength_score = strength_map.get(momentum_data.strength, 0)
        
        if momentum_data.direction == 'BULLISH' and score > 20:
            action = 'BUY'
            message = f"강세 모멘텀({score:.1f}) 포착. {momentum_data.strength} 강도."
        elif momentum_data.direction == 'BEARISH' and score < -20:
            action = 'SELL'
            message = f"약세 모멘텀({score:.1f}) 포착. {momentum_data.strength} 강도."
        else:
            action = 'HOLD'
            message = f"중립 상태({score:.1f}). 뚜렷한 모멘텀 부재."
        
        # 가속도 정보 추가
        if abs(momentum_data.acceleration) > 0.1:
            accel_text = "가속" if momentum_data.acceleration > 0 else "감속"
            message += f" (모멘텀 {accel_text} 중)"

        return MomentumSignal(action=action, strength_score=strength_score, message=message)

    # --- Private Calculation Methods ---

    def _calculate_price_momentum(self, prices: List[float]) -> float:
        """여러 기간의 가격 변화율을 가중 평균하여 가격 모멘텀을 계산합니다."""
        total_momentum = 0
        for period_name, weight in self.PRICE_MOMENTUM_WEIGHTS.items():
            period_len = self.periods[period_name]
            if len(prices) >= period_len:
                change_pct = (prices[-1] - prices[-period_len]) / prices[-period_len] * 100
                total_momentum += change_pct * weight
        
        # 표준편차를 이용한 변동성 정규화 (tanh 함수로 -100 ~ 100 범위 매핑)
        price_stdev = stdev(prices[-self.periods['medium']:]) if len(prices) > 1 else 1
        normalized_momentum = math.tanh(total_momentum / (price_stdev * 2 + 1e-6)) * 100
        return normalized_momentum

    def _calculate_volume_momentum(self, volumes: List[int]) -> float:
        """거래량의 상대적 크기와 최근 추세를 종합하여 거래량 모멘텀을 계산합니다."""
        short_p, medium_p = self.periods['short'], self.periods['medium']
        if len(volumes) < medium_p:
            return 0.0

        # 최근 거래량과 과거 거래량의 비율 계산
        recent_avg_vol = mean(volumes[-short_p:])
        past_avg_vol = mean(volumes[-medium_p:-short_p])
        
        if past_avg_vol == 0: return 100.0 # 과거 거래량 0이면 급등으로 간주
        
        volume_ratio = (recent_avg_vol - past_avg_vol) / past_avg_vol * 100
        
        # 거래량 추세 분석 (최근 단기 데이터)
        trend_slope = self._calculate_linear_regression_slope(volumes[-short_p:])
        # 평균 거래량으로 정규화하여 추세 강도 계산
        normalized_trend = (trend_slope / (mean(volumes[-short_p:]) + 1e-6)) * 100
        
        # 기본 점수와 추세 점수 가중합
        base_score = math.tanh(volume_ratio / 50) * 100  # 50% 변화 시 약 76점
        trend_score = math.tanh(normalized_trend / 20) * 100 # 20% 기울기 시 약 76점

        return base_score * self.VOLUME_MOMENTUM_WEIGHTS['base'] + \
               trend_score * self.VOLUME_MOMENTUM_WEIGHTS['trend']

    def _get_synergy_bonus(self, price_mom: float, vol_mom: float) -> float:
        """가격과 거래량 모멘텀이 같은 방향일 때 시너지 보너스를 계산합니다."""
        if price_mom * vol_mom > 0:  # 같은 부호 (방향 일치)
            # 두 모멘텀의 기하 평균을 보너스 점수로 활용
            synergy = math.sqrt(abs(price_mom * vol_mom)) * math.copysign(1, price_mom)
            return synergy * 0.2 # 보너스는 전체 점수에 20% 정도의 영향력
        return 0.0

    def _build_momentum_data(self, symbol: str, price_mom: float, vol_mom: float, 
                              combined_score: float) -> MomentumData:
        """계산된 값들을 바탕으로 최종 MomentumData 객체를 생성합니다."""
        # 점수 정규화 (-100 ~ 100)
        final_score = math.tanh(combined_score / 75) * 100

        # 가속도 계산 (실시간 분석 시에만 의미 있음)
        # 배치 분석에서는 히스토리가 없으므로 0으로 처리
        acceleration = 0.0
        
        return MomentumData(
            symbol=symbol,
            price_momentum=round(price_mom, 2),
            volume_momentum=round(vol_mom, 2),
            combined_score=round(final_score, 2),
            strength=self._classify_strength(final_score),
            direction=self._classify_direction(final_score),
            acceleration=acceleration
        )

    # --- Private Helper Methods ---

    @staticmethod
    def _calculate_linear_regression_slope(data: List[int]) -> float:
        """선형 회귀를 사용하여 데이터의 추세(기울기)를 계산합니다."""
        n = len(data)
        if n < 2: return 0.0
        
        x_sum = sum(range(n))
        y_sum = sum(data)
        xy_sum = sum(x * y for x, y in enumerate(data))
        x_sq_sum = sum(x**2 for x in range(n))
        
        numerator = n * xy_sum - x_sum * y_sum
        denominator = n * x_sq_sum - x_sum**2
        
        return numerator / denominator if denominator != 0 else 0.0

    @staticmethod
    def _classify_strength(score: float) -> str:
        """종합 점수를 기반으로 모멘텀 강도를 분류합니다."""
        abs_score = abs(score)
        if abs_score < 20: return "WEAK"
        if abs_score < 50: return "MODERATE"
        if abs_score < 80: return "STRONG"
        return "EXTREME"

    @staticmethod
    def _classify_direction(score: float) -> str:
        """종합 점수를 기반으로 모멘텀 방향을 분류합니다."""
        if score > 10: return "BULLISH"
        if score < -10: return "BEARISH"
        return "NEUTRAL"
    
    def add_data_point(self, symbol: str, price: float, volume: int) -> None:
        """새로운 데이터 포인트 추가"""
        try:
            # 버퍼 초기화 (필요시)
            if symbol not in self.buffers:
                self.buffers[symbol] = SymbolDataBuffer(deque(maxlen=self.periods['long'] * 2), deque(maxlen=self.periods['long'] * 2), deque(maxlen=20))
            
            # 데이터 추가
            self.buffers[symbol].prices.append(price)
            self.buffers[symbol].volumes.append(volume)
            
        except Exception as e:
            logger.error(f"❌ {symbol} 데이터 포인트 추가 실패: {e}")
    
    def calculate_momentum_score(self, symbol: str) -> Optional[MomentumData]:
        """
        종목의 모멘텀 스코어 계산
        
        Args:
            symbol: 종목 코드
            
        Returns:
            모멘텀 분석 결과
        """
        if symbol not in self.buffers:
            logger.warning(f"⚠️ {symbol} 데이터 없음")
            return None
        
        prices = list(self.buffers[symbol].prices)
        volumes = list(self.buffers[symbol].volumes)
        
        if len(prices) < self.periods['short']:
            logger.warning(f"⚠️ {symbol} 데이터 부족: {len(prices)}개")
            return None
        
        try:
            # 가격 모멘텀 계산
            price_momentum = self._calculate_price_momentum(prices)
            
            # 거래량 모멘텀 계산
            volume_momentum = self._calculate_volume_momentum(volumes)
            
            # 복합 모멘텀 스코어
            combined_score = self._calculate_combined_score(price_momentum, volume_momentum)
            
            # 모멘텀 강도 분류
            momentum_strength = self._classify_momentum_strength(abs(combined_score))
            
            # 모멘텀 방향 분류
            momentum_direction = self._classify_momentum_direction(combined_score)
            
            # 모멘텀 가속도 계산
            acceleration = self._calculate_momentum_acceleration(symbol, combined_score)
            
            momentum_data = MomentumData(
                symbol=symbol,
                price_momentum=price_momentum,
                volume_momentum=volume_momentum,
                combined_score=combined_score,
                strength=momentum_strength,
                direction=momentum_direction,
                acceleration=acceleration,
                timestamp=datetime.now()
            )
            
            # 히스토리에 추가 (가속도 계산용)
            self.buffers[symbol].momentum_history.append(combined_score)
            
            return momentum_data
            
        except Exception as e:
            logger.error(f"❌ {symbol} 모멘텀 스코어 계산 실패: {e}")
            return None
    
    def _calculate_combined_score(self, price_momentum: float, volume_momentum: float) -> float:
        """가격과 거래량 모멘텀을 결합한 종합 스코어 계산"""
        try:
            # 가격 모멘텀에 더 높은 가중치 (70%)
            # 거래량 모멘텀은 확인 지표 역할 (30%)
            combined = price_momentum * 0.7 + volume_momentum * 0.3
            
            # 가격과 거래량 모멘텀이 같은 방향일 때 보너스
            if (price_momentum > 0 and volume_momentum > 0) or \
               (price_momentum < 0 and volume_momentum < 0):
                # 같은 방향일 때 10% 보너스
                bonus = abs(combined) * 0.1
                combined = combined + (bonus if combined > 0 else -bonus)
            
            return max(-100, min(100, combined))
            
        except Exception as e:
            logger.error(f"❌ 복합 스코어 계산 실패: {e}")
            return 0.0
    
    def _classify_momentum_strength(self, abs_score: float) -> str:
        """모멘텀 강도 분류"""
        if abs_score < 20:
            return "WEAK"
        elif abs_score < 40:
            return "MODERATE"
        elif abs_score < 70:
            return "STRONG"
        else:
            return "EXTREME"
    
    def _classify_momentum_direction(self, score: float) -> str:
        """모멘텀 방향 분류"""
        if score > 10:
            return "BULLISH"
        elif score < -10:
            return "BEARISH"
        else:
            return "NEUTRAL"
    
    def _calculate_momentum_acceleration(self, symbol: str, current_score: float) -> float:
        """
        모멘텀 가속도 계산
        
        모멘텀의 변화율을 측정하여 추세의 가속/감속 판단
        """
        try:
            if symbol not in self.buffers:
                return 0.0
            
            history = list(self.buffers[symbol].momentum_history)
            if len(history) < 5:
                return 0.0
            
            # 최근 5개 모멘텀 값의 변화율 계산
            recent_momentum = history[-5:]
            
            # 선형 회귀를 통한 가속도 계산
            x_values = list(range(len(recent_momentum)))
            
            n = len(recent_momentum)
            sum_x = sum(x_values)
            sum_y = sum(recent_momentum)
            sum_xy = sum(x * y for x, y in zip(x_values, recent_momentum))
            sum_x2 = sum(x * x for x in x_values)
            
            # 기울기 (가속도)
            if n * sum_x2 - sum_x * sum_x != 0:
                acceleration = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
                return max(-50, min(50, acceleration))
            
            return 0.0
            
        except Exception as e:
            logger.error(f"❌ 모멘텀 가속도 계산 실패: {e}")
            return 0.0
    
    def analyze_multiple_symbols(self, symbols_data: Dict[str, Dict[str, List]]) -> List[MomentumData]:
        """
        여러 종목의 모멘텀 분석
        
        Args:
            symbols_data: {symbol: {'prices': [...], 'volumes': [...]}}
            
        Returns:
            모멘텀 분석 결과 리스트 (점수 순 정렬)
        """
        results = []
        
        for symbol, data in symbols_data.items():
            try:
                prices = data.get('prices', [])
                volumes = data.get('volumes', [])
                
                momentum_result = self.calculate_batch_momentum(symbol, prices, volumes)
                if momentum_result:
                    results.append(momentum_result)
                    
            except Exception as e:
                logger.warning(f"⚠️ {symbol} 모멘텀 분석 건너뜀: {e}")
                continue
        
        # 모멘텀 스코어 절댓값으로 정렬 (강한 모멘텀 우선)
        results.sort(key=lambda x: abs(x.combined_score), reverse=True)
        
        logger.info(f"⚡ 모멘텀 분석 완료: {len(results)}개 종목")
        return results
    
    def get_top_momentum_symbols(self, 
                                momentum_results: List[MomentumData],
                                direction: str = 'BULLISH',
                                min_strength: str = 'MODERATE',
                                max_count: int = 10) -> List[MomentumData]:
        """
        조건에 맞는 상위 모멘텀 종목 선별
        
        Args:
            momentum_results: 모멘텀 분석 결과 리스트
            direction: 원하는 방향 ('BULLISH', 'BEARISH', 'ANY')
            min_strength: 최소 강도 ('WEAK', 'MODERATE', 'STRONG', 'EXTREME')
            max_count: 최대 선별 종목 수
            
        Returns:
            조건에 맞는 종목 리스트
        """
        try:
            # 강도 순서 정의
            strength_order = ['WEAK', 'MODERATE', 'STRONG', 'EXTREME']
            min_strength_idx = strength_order.index(min_strength)
            
            # 필터링
            filtered_results = []
            for result in momentum_results:
                # 방향 필터
                if direction != 'ANY' and result.direction != direction:
                    continue
                
                # 강도 필터
                result_strength_idx = strength_order.index(result.strength)
                if result_strength_idx < min_strength_idx:
                    continue
                
                filtered_results.append(result)
            
            # 상위 종목만 선택
            top_symbols = filtered_results[:max_count]
            
            logger.info(f"🎯 상위 모멘텀 종목: {len(top_symbols)}개 선별 "
                       f"(방향: {direction}, 최소강도: {min_strength})")
            
            for result in top_symbols:
                logger.info(f"  - {result.symbol}: {result.combined_score:.1f}점 "
                           f"({result.direction}, {result.strength})")
            
            return top_symbols
            
        except Exception as e:
            logger.error(f"❌ 상위 모멘텀 종목 선별 실패: {e}")
            return []
    
    def clear_symbol_data(self, symbol: str) -> None:
        """특정 종목의 데이터 버퍼 정리"""
        try:
            if symbol in self.buffers:
                del self.buffers[symbol]
                
            logger.debug(f"🗑️ {symbol} 데이터 버퍼 정리 완료")
            
        except Exception as e:
            logger.error(f"❌ {symbol} 데이터 정리 실패: {e}")
    
    def get_buffer_status(self) -> Dict[str, int]:
        """버퍼 상태 정보 반환"""
        return {
            'symbols_count': len(self.buffers),
            'total_price_points': sum(len(buf.prices) for buf in self.buffers.values()),
            'total_volume_points': sum(len(buf.volumes) for buf in self.buffers.values()),
            'total_momentum_history': sum(len(buf.momentum_history) for buf in self.buffers.values())
        } 