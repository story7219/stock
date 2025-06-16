"""
기술적 분석 도구
시장 상황 분석, 매수 신호 감지 등
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import logging

class TechnicalAnalyzer:
    """기술적 분석 도구"""
    
    def __init__(self):
        self.name = "기술적 분석기"
    
    def analyze_market_situation(self, price_data: Dict) -> str:
        """시장 상황 분석하여 최적 전략 결정"""
        try:
            current_price = price_data.get('current_price', 0)
            recent_high = price_data.get('recent_high', 0)
            recent_low = price_data.get('recent_low', 0)
            volume_ratio = price_data.get('volume_ratio', 1.0)
            price_history = price_data.get('price_history', [])
            
            if not all([current_price, recent_high, recent_low]) or len(price_history) < 20:
                return "NEUTRAL"
            
            # 현재가 위치 분석
            price_position = (current_price - recent_low) / (recent_high - recent_low) if recent_high != recent_low else 0.5
            
            # 추세 강도 분석
            ma5 = np.mean(price_history[-5:])
            ma20 = np.mean(price_history[-20:])
            trend_strength = (ma5 - ma20) / ma20 if ma20 > 0 else 0
            
            # 상황별 우선 전략 결정
            if abs(trend_strength) < 0.02:  # 횡보 구간
                if price_position < 0.4:  # 저점 근처
                    return "TREND_CHANGE_PRIORITY"  # 추세전환 우선 대기
                else:
                    return "PULLBACK_PRIORITY"  # 눌림목 우선
            elif trend_strength > 0.05:  # 강한 상승 추세
                if volume_ratio > 2.0:  # 거래량 급증
                    return "BREAKOUT_PRIORITY"  # 돌파 우선
                else:
                    return "PULLBACK_PRIORITY"  # 눌림목 우선
            else:  # 약한 추세 또는 불확실
                return "TREND_CHANGE_PRIORITY"  # 추세전환 우선
                
        except Exception as e:
            logging.error(f"❌ 시장 상황 분석 오류: {e}")
            return "NEUTRAL"
    
    def detect_trend_change(self, price_history: List[float], volume_history: List[float]) -> Tuple[bool, str]:
        """추세전환 신호 감지"""
        try:
            if len(price_history) < 20 or len(volume_history) < 20:
                return False, ""
            
            # 이동평균 교차 확인
            ma5 = np.mean(price_history[-5:])
            ma20 = np.mean(price_history[-20:])
            prev_ma5 = np.mean(price_history[-6:-1])
            prev_ma20 = np.mean(price_history[-21:-1])
            
            # 골든크로스
            if ma5 > ma20 and prev_ma5 <= prev_ma20:
                return True, "골든크로스"
            
            # 거래량 급증 + 모멘텀
            avg_volume = np.mean(volume_history[-20:])
            current_volume = volume_history[-1]
            volume_spike = current_volume > avg_volume * 1.5
            
            momentum = (price_history[-1] - price_history[-5]) / price_history[-5] * 100
            
            if volume_spike and momentum > 2:
                return True, "거래량급증+모멘텀"
            
            return False, ""
            
        except Exception as e:
            logging.error(f"❌ 추세전환 감지 오류: {e}")
            return False, ""
    
    def detect_pullback_opportunity(self, current_price: float, recent_high: float, recent_low: float) -> Tuple[bool, str]:
        """눌림목 매수 기회 감지"""
        try:
            if not all([current_price, recent_high, recent_low]):
                return False, ""
            
            price_range = recent_high - recent_low
            if price_range <= 0:
                return False, ""
            
            # 피보나치 되돌림 레벨
            fib_levels = {
                0.236: "23.6%",
                0.382: "38.2%", 
                0.618: "61.8%"
            }
            
            for ratio, level_name in fib_levels.items():
                fib_level = recent_high - (price_range * ratio)
                
                # 현재가가 피보나치 레벨 근처인지 확인 (±2%)
                if abs(current_price - fib_level) / fib_level <= 0.02:
                    return True, level_name
            
            return False, ""
            
        except Exception as e:
            logging.error(f"❌ 눌림목 기회 감지 오류: {e}")
            return False, ""
    
    def detect_breakout_opportunity(self, current_price: float, recent_high: float, volume_ratio: float) -> bool:
        """돌파 매수 기회 감지"""
        try:
            if not all([current_price, recent_high]):
                return False
            
            # 전고점 돌파 확인 (1% 이상)
            breakout_threshold = recent_high * 1.01
            
            # 거래량 조건 (1.5배 이상)
            return current_price >= breakout_threshold and volume_ratio >= 1.5
            
        except Exception as e:
            logging.error(f"❌ 돌파 기회 감지 오류: {e}")
            return False
    
    def calculate_support_resistance(self, price_history: List[float], window: int = 20) -> Dict[str, float]:
        """지지/저항선 계산"""
        try:
            if len(price_history) < window:
                return {"support": 0, "resistance": 0}
            
            recent_prices = price_history[-window:]
            
            # 단순히 최고/최저가로 계산 (실제로는 더 복잡한 알고리즘 사용)
            support = min(recent_prices)
            resistance = max(recent_prices)
            
            return {"support": support, "resistance": resistance}
            
        except Exception as e:
            logging.error(f"❌ 지지/저항선 계산 오류: {e}")
            return {"support": 0, "resistance": 0}
    
    def calculate_rsi(self, price_history: List[float], period: int = 14) -> float:
        """RSI 계산"""
        try:
            if len(price_history) < period + 1:
                return 50.0  # 중립값
            
            deltas = np.diff(price_history)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gain = np.mean(gains[-period:])
            avg_loss = np.mean(losses[-period:])
            
            if avg_loss == 0:
                return 100.0
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except Exception as e:
            logging.error(f"❌ RSI 계산 오류: {e}")
            return 50.0
    
    def calculate_bollinger_bands(self, price_history: List[float], period: int = 20, std_dev: float = 2.0) -> Dict[str, float]:
        """볼린저 밴드 계산"""
        try:
            if len(price_history) < period:
                return {"upper": 0, "middle": 0, "lower": 0}
            
            recent_prices = price_history[-period:]
            middle = np.mean(recent_prices)
            std = np.std(recent_prices)
            
            upper = middle + (std * std_dev)
            lower = middle - (std * std_dev)
            
            return {"upper": upper, "middle": middle, "lower": lower}
            
        except Exception as e:
            logging.error(f"❌ 볼린저 밴드 계산 오류: {e}")
            return {"upper": 0, "middle": 0, "lower": 0} 