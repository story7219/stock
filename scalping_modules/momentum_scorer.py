"""
⚡ 모멘텀 스코어링 시스템
- 실시간 모멘텀 강도 측정
- 단기 매매 신호 생성
- 가격/거래량 기반 복합 분석
"""

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from collections import deque
import statistics

logger = logging.getLogger(__name__)

@dataclass
class MomentumData:
    """모멘텀 분석 데이터 클래스"""
    symbol: str
    price_momentum: float  # -100 to +100
    volume_momentum: float  # -100 to +100
    combined_score: float  # -100 to +100
    momentum_strength: str  # WEAK, MODERATE, STRONG, EXTREME
    momentum_direction: str  # BULLISH, BEARISH, NEUTRAL
    acceleration: float  # 모멘텀 가속도
    timestamp: datetime

class MomentumScorer:
    """고급 모멘텀 분석 및 스코어링 시스템"""
    
    def __init__(self, 
                 short_period: int = 5,
                 medium_period: int = 20,
                 long_period: int = 50):
        """
        모멘텀 스코어러 초기화
        
        Args:
            short_period: 단기 모멘텀 계산 기간
            medium_period: 중기 모멘텀 계산 기간  
            long_period: 장기 모멘텀 계산 기간
        """
        self.short_period = short_period
        self.medium_period = medium_period
        self.long_period = long_period
        
        # 실시간 데이터 버퍼
        self.price_buffers: Dict[str, deque] = {}
        self.volume_buffers: Dict[str, deque] = {}
        
        # 모멘텀 히스토리 (가속도 계산용)
        self.momentum_history: Dict[str, deque] = {}
        
        logger.info(f"⚡ 모멘텀 스코어러 초기화: {short_period}/{medium_period}/{long_period}")
    
    def add_data_point(self, symbol: str, price: float, volume: int) -> None:
        """새로운 데이터 포인트 추가"""
        try:
            # 버퍼 초기화 (필요시)
            if symbol not in self.price_buffers:
                self.price_buffers[symbol] = deque(maxlen=self.long_period * 2)
                self.volume_buffers[symbol] = deque(maxlen=self.long_period * 2)
                self.momentum_history[symbol] = deque(maxlen=20)  # 최근 20개 모멘텀 기록
            
            # 데이터 추가
            self.price_buffers[symbol].append(price)
            self.volume_buffers[symbol].append(volume)
            
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
        if symbol not in self.price_buffers:
            logger.warning(f"⚠️ {symbol} 데이터 없음")
            return None
        
        prices = list(self.price_buffers[symbol])
        volumes = list(self.volume_buffers[symbol])
        
        if len(prices) < self.short_period:
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
                momentum_strength=momentum_strength,
                momentum_direction=momentum_direction,
                acceleration=acceleration,
                timestamp=datetime.now()
            )
            
            # 히스토리에 추가 (가속도 계산용)
            self.momentum_history[symbol].append(combined_score)
            
            return momentum_data
            
        except Exception as e:
            logger.error(f"❌ {symbol} 모멘텀 스코어 계산 실패: {e}")
            return None
    
    def _calculate_price_momentum(self, prices: List[float]) -> float:
        """
        가격 기반 모멘텀 계산 (-100 to +100)
        
        여러 기간의 가격 변화율을 가중 평균하여 계산
        """
        try:
            if len(prices) < self.short_period:
                return 0.0
            
            momentum_scores = []
            
            # 단기 모멘텀 (가중치 높음)
            if len(prices) >= self.short_period:
                short_change = (prices[-1] - prices[-self.short_period]) / prices[-self.short_period] * 100
                momentum_scores.append(short_change * 0.5)  # 50% 가중치
            
            # 중기 모멘텀
            if len(prices) >= self.medium_period:
                medium_change = (prices[-1] - prices[-self.medium_period]) / prices[-self.medium_period] * 100
                momentum_scores.append(medium_change * 0.3)  # 30% 가중치
            
            # 장기 모멘텀
            if len(prices) >= self.long_period:
                long_change = (prices[-1] - prices[-self.long_period]) / prices[-self.long_period] * 100
                momentum_scores.append(long_change * 0.2)  # 20% 가중치
            
            # 가중 평균 계산
            total_momentum = sum(momentum_scores)
            
            # -100 ~ +100 범위로 정규화
            return max(-100, min(100, total_momentum))
            
        except Exception as e:
            logger.error(f"❌ 가격 모멘텀 계산 실패: {e}")
            return 0.0
    
    def _calculate_volume_momentum(self, volumes: List[int]) -> float:
        """
        거래량 기반 모멘텀 계산 (-100 to +100)
        
        거래량 증가/감소 패턴으로 모멘텀 방향성 판단
        """
        try:
            if len(volumes) < self.short_period * 2:
                return 0.0
            
            # 최근 거래량과 과거 거래량 비교
            recent_volume = sum(volumes[-self.short_period:]) / self.short_period
            
            # 비교 기간 설정
            if len(volumes) >= self.medium_period:
                past_volume = sum(volumes[-self.medium_period:-self.short_period]) / (self.medium_period - self.short_period)
            else:
                available_past = len(volumes) - self.short_period
                if available_past <= 0:
                    return 0.0
                past_volume = sum(volumes[:-self.short_period]) / available_past
            
            if past_volume == 0:
                return 0.0
            
            # 거래량 변화율 계산
            volume_ratio = recent_volume / past_volume
            
            # 거래량 모멘텀 스코어 계산
            if volume_ratio > 1:
                # 거래량 증가 → 긍정적 모멘텀
                volume_momentum = min(50, (volume_ratio - 1) * 100)
            else:
                # 거래량 감소 → 부정적 모멘텀
                volume_momentum = max(-50, (volume_ratio - 1) * 100)
            
            # 거래량 트렌드 추가 고려
            if len(volumes) >= self.short_period:
                volume_trend = self._calculate_volume_trend(volumes[-self.short_period:])
                volume_momentum += volume_trend * 0.5
            
            return max(-100, min(100, volume_momentum))
            
        except Exception as e:
            logger.error(f"❌ 거래량 모멘텀 계산 실패: {e}")
            return 0.0
    
    def _calculate_volume_trend(self, recent_volumes: List[int]) -> float:
        """최근 거래량의 트렌드 계산 (-50 to +50)"""
        try:
            if len(recent_volumes) < 3:
                return 0.0
            
            # 선형 회귀를 통한 트렌드 계산
            x_values = list(range(len(recent_volumes)))
            y_values = recent_volumes
            
            n = len(recent_volumes)
            sum_x = sum(x_values)
            sum_y = sum(y_values)
            sum_xy = sum(x * y for x, y in zip(x_values, y_values))
            sum_x2 = sum(x * x for x in x_values)
            
            # 기울기 계산
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            
            # 평균 거래량 대비 기울기 정규화
            avg_volume = sum_y / n
            if avg_volume > 0:
                normalized_slope = (slope / avg_volume) * 100
                return max(-50, min(50, normalized_slope))
            
            return 0.0
            
        except Exception as e:
            logger.error(f"❌ 거래량 트렌드 계산 실패: {e}")
            return 0.0
    
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
            if symbol not in self.momentum_history:
                return 0.0
            
            history = list(self.momentum_history[symbol])
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
    
    def calculate_batch_momentum(self, 
                                symbol: str,
                                prices: List[float], 
                                volumes: List[int]) -> Optional[MomentumData]:
        """
        배치 방식으로 모멘텀 계산 (실시간 버퍼 없이)
        
        Args:
            symbol: 종목 코드
            prices: 가격 데이터 리스트
            volumes: 거래량 데이터 리스트
            
        Returns:
            모멘텀 분석 결과
        """
        if len(prices) < self.short_period or len(volumes) < self.short_period:
            logger.warning(f"⚠️ {symbol} 배치 분석용 데이터 부족")
            return None
        
        try:
            # 가격 모멘텀 계산
            price_momentum = self._calculate_price_momentum(prices)
            
            # 거래량 모멘텀 계산  
            volume_momentum = self._calculate_volume_momentum(volumes)
            
            # 복합 스코어
            combined_score = self._calculate_combined_score(price_momentum, volume_momentum)
            
            # 분류
            momentum_strength = self._classify_momentum_strength(abs(combined_score))
            momentum_direction = self._classify_momentum_direction(combined_score)
            
            # 배치 모드에서는 가속도 계산 안함
            acceleration = 0.0
            
            return MomentumData(
                symbol=symbol,
                price_momentum=price_momentum,
                volume_momentum=volume_momentum,
                combined_score=combined_score,
                momentum_strength=momentum_strength,
                momentum_direction=momentum_direction,
                acceleration=acceleration,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"❌ {symbol} 배치 모멘텀 계산 실패: {e}")
            return None
    
    def get_trading_signals(self, momentum_data: MomentumData) -> Dict[str, any]:
        """
        모멘텀 기반 매매 신호 생성
        
        Args:
            momentum_data: 모멘텀 분석 결과
            
        Returns:
            매매 신호 정보
        """
        try:
            signals = {
                'symbol': momentum_data.symbol,
                'timestamp': momentum_data.timestamp.isoformat(),
                'momentum_score': momentum_data.combined_score,
                'signal_strength': momentum_data.momentum_strength,
                'direction': momentum_data.momentum_direction,
                'acceleration': momentum_data.acceleration
            }
            
            # 매매 신호 생성
            score = momentum_data.combined_score
            strength = momentum_data.momentum_strength
            acceleration = momentum_data.acceleration
            
            # 강한 상승 모멘텀
            if score > 30 and strength in ['STRONG', 'EXTREME']:
                signals['action'] = 'STRONG_BUY'
                signals['confidence'] = min(95, 70 + abs(score) * 0.3)
                
            # 중간 상승 모멘텀
            elif score > 15 and strength in ['MODERATE', 'STRONG']:
                signals['action'] = 'BUY'
                signals['confidence'] = min(85, 60 + abs(score) * 0.4)
                
            # 강한 하락 모멘텀
            elif score < -30 and strength in ['STRONG', 'EXTREME']:
                signals['action'] = 'STRONG_SELL'
                signals['confidence'] = min(95, 70 + abs(score) * 0.3)
                
            # 중간 하락 모멘텀
            elif score < -15 and strength in ['MODERATE', 'STRONG']:
                signals['action'] = 'SELL'
                signals['confidence'] = min(85, 60 + abs(score) * 0.4)
                
            # 약한 모멘텀
            else:
                signals['action'] = 'HOLD'
                signals['confidence'] = 50
            
            # 가속도 고려 (추가 보정)
            if acceleration > 5 and score > 0:
                signals['acceleration_bonus'] = '상승 가속'
            elif acceleration < -5 and score < 0:
                signals['acceleration_bonus'] = '하락 가속'
            elif acceleration > 5 and score < 0:
                signals['acceleration_warning'] = '하락 둔화'
            elif acceleration < -5 and score > 0:
                signals['acceleration_warning'] = '상승 둔화'
            
            return signals
            
        except Exception as e:
            logger.error(f"❌ 매매 신호 생성 실패: {e}")
            return {'action': 'HOLD', 'confidence': 50}
    
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
                if direction != 'ANY' and result.momentum_direction != direction:
                    continue
                
                # 강도 필터
                result_strength_idx = strength_order.index(result.momentum_strength)
                if result_strength_idx < min_strength_idx:
                    continue
                
                filtered_results.append(result)
            
            # 상위 종목만 선택
            top_symbols = filtered_results[:max_count]
            
            logger.info(f"🎯 상위 모멘텀 종목: {len(top_symbols)}개 선별 "
                       f"(방향: {direction}, 최소강도: {min_strength})")
            
            for result in top_symbols:
                logger.info(f"  - {result.symbol}: {result.combined_score:.1f}점 "
                           f"({result.momentum_direction}, {result.momentum_strength})")
            
            return top_symbols
            
        except Exception as e:
            logger.error(f"❌ 상위 모멘텀 종목 선별 실패: {e}")
            return []
    
    def clear_symbol_data(self, symbol: str) -> None:
        """특정 종목의 데이터 버퍼 정리"""
        try:
            if symbol in self.price_buffers:
                del self.price_buffers[symbol]
            if symbol in self.volume_buffers:
                del self.volume_buffers[symbol]
            if symbol in self.momentum_history:
                del self.momentum_history[symbol]
                
            logger.debug(f"🗑️ {symbol} 데이터 버퍼 정리 완료")
            
        except Exception as e:
            logger.error(f"❌ {symbol} 데이터 정리 실패: {e}")
    
    def get_buffer_status(self) -> Dict[str, int]:
        """버퍼 상태 정보 반환"""
        return {
            'symbols_count': len(self.price_buffers),
            'total_price_points': sum(len(buf) for buf in self.price_buffers.values()),
            'total_volume_points': sum(len(buf) for buf in self.volume_buffers.values()),
            'total_momentum_history': sum(len(buf) for buf in self.momentum_history.values())
        } 