"""
📊 ATR (Average True Range) 분석기
- 변동성 기반 매매 신호 생성
- 스캘핑에 적합한 종목 선별
- 진입/청산 타이밍 최적화
"""

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class ATRData:
    """ATR 분석 데이터 클래스"""
    symbol: str
    atr_value: float
    atr_percentage: float
    volatility_level: str  # LOW, MEDIUM, HIGH, EXTREME
    scalping_suitability: float  # 0-100 점수
    timestamp: datetime

class ATRAnalyzer:
    """ATR 기반 변동성 분석기"""
    
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
        self.optimal_atr_min = optimal_atr_min
        self.optimal_atr_max = optimal_atr_max
        self.period = period
        
        logger.info(f"📊 ATR 분석기 초기화: 최적 범위 {optimal_atr_min}%-{optimal_atr_max}%")
    
    def calculate_atr(self, high_prices: List[float], 
                     low_prices: List[float], 
                     close_prices: List[float]) -> float:
        """
        ATR 계산
        
        Args:
            high_prices: 고가 리스트
            low_prices: 저가 리스트  
            close_prices: 종가 리스트
            
        Returns:
            ATR 값
        """
        if len(high_prices) < self.period + 1:
            logger.warning("⚠️ ATR 계산을 위한 데이터 부족")
            return 0.0
        
        try:
            true_ranges = []
            
            for i in range(1, len(close_prices)):
                # True Range 계산
                tr1 = high_prices[i] - low_prices[i]  # 당일 고가 - 저가
                tr2 = abs(high_prices[i] - close_prices[i-1])  # 당일 고가 - 전일 종가
                tr3 = abs(low_prices[i] - close_prices[i-1])   # 당일 저가 - 전일 종가
                
                true_range = max(tr1, tr2, tr3)
                true_ranges.append(true_range)
            
            # ATR = True Range의 이동평균
            if len(true_ranges) >= self.period:
                atr = sum(true_ranges[-self.period:]) / self.period
                return atr
            else:
                return sum(true_ranges) / len(true_ranges)
                
        except Exception as e:
            logger.error(f"❌ ATR 계산 실패: {e}")
            return 0.0
    
    def analyze_volatility(self, symbol: str, 
                          high_prices: List[float],
                          low_prices: List[float], 
                          close_prices: List[float]) -> Optional[ATRData]:
        """
        변동성 분석 수행
        
        Args:
            symbol: 종목 코드
            high_prices: 고가 데이터
            low_prices: 저가 데이터
            close_prices: 종가 데이터
            
        Returns:
            ATR 분석 결과
        """
        if not close_prices:
            logger.warning(f"⚠️ {symbol} 가격 데이터 없음")
            return None
        
        try:
            # ATR 계산
            atr_value = self.calculate_atr(high_prices, low_prices, close_prices)
            if atr_value == 0:
                return None
            
            # ATR 퍼센티지 계산 (현재가 대비)
            current_price = close_prices[-1]
            atr_percentage = (atr_value / current_price) * 100
            
            # 변동성 수준 분류
            volatility_level = self._classify_volatility(atr_percentage)
            
            # 스캘핑 적합성 점수 계산
            scalping_score = self._calculate_scalping_suitability(atr_percentage)
            
            return ATRData(
                symbol=symbol,
                atr_value=atr_value,
                atr_percentage=atr_percentage,
                volatility_level=volatility_level,
                scalping_suitability=scalping_score,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"❌ {symbol} ATR 분석 실패: {e}")
            return None
    
    def _classify_volatility(self, atr_percentage: float) -> str:
        """변동성 수준 분류"""
        if atr_percentage < 1.0:
            return "LOW"
        elif atr_percentage < 2.0:
            return "MEDIUM"
        elif atr_percentage < 4.0:
            return "HIGH"
        else:
            return "EXTREME"
    
    def _calculate_scalping_suitability(self, atr_percentage: float) -> float:
        """
        스캘핑 적합성 점수 계산 (0-100)
        
        Args:
            atr_percentage: ATR 퍼센티지
            
        Returns:
            적합성 점수 (높을수록 스캘핑에 적합)
        """
        try:
            # 최적 범위 내에 있는 경우 높은 점수
            if self.optimal_atr_min <= atr_percentage <= self.optimal_atr_max:
                # 최적 범위 중앙값에 가까울수록 높은 점수
                center = (self.optimal_atr_min + self.optimal_atr_max) / 2
                distance_from_center = abs(atr_percentage - center)
                max_distance = (self.optimal_atr_max - self.optimal_atr_min) / 2
                
                normalized_distance = distance_from_center / max_distance
                score = 100 * (1 - normalized_distance)
                return max(80, score)  # 최적 범위 내는 최소 80점
            
            # 최적 범위를 벗어난 경우
            elif atr_percentage < self.optimal_atr_min:
                # 너무 낮은 변동성 - 거래 기회 부족
                ratio = atr_percentage / self.optimal_atr_min
                return max(20, 80 * ratio)
            
            else:  # atr_percentage > self.optimal_atr_max
                # 너무 높은 변동성 - 위험 증가
                excess_ratio = (atr_percentage - self.optimal_atr_max) / self.optimal_atr_max
                penalty = min(60, excess_ratio * 100)  # 최대 60점 감점
                return max(10, 80 - penalty)
                
        except Exception as e:
            logger.error(f"❌ 스캘핑 적합성 점수 계산 실패: {e}")
            return 0.0
    
    def get_trading_signals(self, atr_data: ATRData, 
                           current_price: float) -> Dict[str, float]:
        """
        ATR 기반 매매 신호 생성
        
        Args:
            atr_data: ATR 분석 데이터
            current_price: 현재가
            
        Returns:
            매매 신호 정보 (진입가, 손절가, 목표가 등)
        """
        try:
            atr_value = atr_data.atr_value
            
            # ATR 기반 수준 계산
            resistance_1 = current_price + (atr_value * 0.5)
            resistance_2 = current_price + atr_value
            support_1 = current_price - (atr_value * 0.5)
            support_2 = current_price - atr_value
            
            # 스캘핑 전략별 신호
            signals = {
                'current_price': current_price,
                'atr_value': atr_value,
                'resistance_1': resistance_1,  # 1차 저항
                'resistance_2': resistance_2,  # 2차 저항
                'support_1': support_1,        # 1차 지지
                'support_2': support_2,        # 2차 지지
                
                # 매수 신호
                'buy_entry': support_1,        # 매수 진입가
                'buy_stop_loss': support_2,    # 매수 손절가
                'buy_take_profit': resistance_1, # 매수 목표가
                
                # 매도 신호  
                'sell_entry': resistance_1,     # 매도 진입가
                'sell_stop_loss': resistance_2, # 매도 손절가
                'sell_take_profit': support_1,  # 매도 목표가
                
                # 리스크 관리
                'position_size_multiplier': self._get_position_size_multiplier(atr_data.volatility_level)
            }
            
            return signals
            
        except Exception as e:
            logger.error(f"❌ ATR 매매 신호 생성 실패: {e}")
            return {}
    
    def _get_position_size_multiplier(self, volatility_level: str) -> float:
        """변동성 수준에 따른 포지션 크기 조절"""
        multipliers = {
            'LOW': 1.2,      # 낮은 변동성 → 포지션 크기 증가
            'MEDIUM': 1.0,   # 보통 변동성 → 기본 포지션
            'HIGH': 0.8,     # 높은 변동성 → 포지션 크기 감소
            'EXTREME': 0.5   # 극한 변동성 → 포지션 크기 대폭 감소
        }
        return multipliers.get(volatility_level, 1.0)
    
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