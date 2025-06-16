"""
피보나치 분할매수 전략
추세전환, 눌림목, 돌파 3가지 전략을 우선순위에 따라 실행
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np
import logging

from .base_strategy import BaseStrategy, StrategySignal

@dataclass
class FibonacciConfig:
    """피보나치 전략 설정"""
    enabled: bool = True
    
    # 매수 전략 우선순위 (낮은 숫자가 높은 우선순위)
    strategy_priority: Dict[str, int] = field(default_factory=lambda: {
        'TREND_CHANGE': 1,  # 최우선: 추세전환 매수
        'PULLBACK': 2,      # 2순위: 눌림목 매수  
        'BREAKOUT': 3       # 3순위: 전고점 돌파 매수
    })
    
    # 피보나치 비율 및 배수
    pullback_ratios: List[float] = field(default_factory=lambda: [0.236, 0.382, 0.618])
    breakout_multipliers: List[float] = field(default_factory=lambda: [1, 2, 3])
    trend_change_signals: List[str] = field(default_factory=lambda: ['MA_CROSS', 'VOLUME_SPIKE', 'MOMENTUM'])
    
    # 피보나치 수열 기반 매수 수량 (1, 1, 2, 3, 5, 8...)
    fibonacci_sequence: List[int] = field(default_factory=lambda: [1, 1, 2, 3, 5, 8, 13])
    
    # 각 전략별 현재 단계
    pullback_stage: Dict[str, int] = field(default_factory=dict)
    breakout_stage: Dict[str, int] = field(default_factory=dict)
    trend_change_stage: Dict[str, int] = field(default_factory=dict)

class FibonacciStrategyManager(BaseStrategy):
    """피보나치 분할매수 전략 관리자"""
    
    def __init__(self, config: FibonacciConfig = None):
        super().__init__("피보나치 분할매수")
        self.config = config or FibonacciConfig()
    
    async def analyze(self, stock_code: str, market_data: Dict) -> Optional[StrategySignal]:
        """피보나치 전략 분석"""
        if not self.config.enabled:
            return None
        
        # 모든 매수 신호 분석
        available_strategies = []
        
        # 1. 추세전환 매수 분석
        trend_signal = await self._analyze_trend_change(stock_code, market_data)
        if trend_signal:
            available_strategies.append(trend_signal)
        
        # 2. 눌림목 매수 분석
        pullback_signal = await self._analyze_pullback(stock_code, market_data)
        if pullback_signal:
            available_strategies.append(pullback_signal)
        
        # 3. 돌파 매수 분석
        breakout_signal = await self._analyze_breakout(stock_code, market_data)
        if breakout_signal:
            available_strategies.append(breakout_signal)
        
        if not available_strategies:
            return None
        
        # 시장 상황에 따른 최적 전략 선택
        market_situation = market_data.get('market_situation', 'NEUTRAL')
        optimal_signal = self._select_optimal_strategy(available_strategies, market_situation)
        
        return optimal_signal
    
    async def _analyze_trend_change(self, stock_code: str, market_data: Dict) -> Optional[StrategySignal]:
        """추세전환 매수 분석"""
        try:
            price_history = market_data.get('price_history', [])
            volume_history = market_data.get('volume_history', [])
            
            if len(price_history) < 20 or len(volume_history) < 20:
                return None
            
            # 이동평균 교차 확인
            ma5 = np.mean(price_history[-5:])
            ma20 = np.mean(price_history[-20:])
            prev_ma5 = np.mean(price_history[-6:-1])
            prev_ma20 = np.mean(price_history[-21:-1])
            
            # 골든크로스 확인
            is_golden_cross = (ma5 > ma20) and (prev_ma5 <= prev_ma20)
            
            # 거래량 급증 확인
            avg_volume = np.mean(volume_history[-20:])
            current_volume = volume_history[-1]
            volume_spike = current_volume > avg_volume * 1.5
            
            # 모멘텀 확인
            momentum = (price_history[-1] - price_history[-5]) / price_history[-5] * 100
            
            if is_golden_cross or (volume_spike and momentum > 2):
                stage = self.config.trend_change_stage.get(stock_code, 0)
                quantity = self._get_fibonacci_quantity(stage)
                
                signal_type = "골든크로스" if is_golden_cross else "거래량급증+모멘텀"
                
                return StrategySignal(
                    action="BUY",
                    confidence=0.9,
                    reason=f"추세전환 신호: {signal_type}",
                    priority=self.config.strategy_priority['TREND_CHANGE'],
                    quantity=quantity,
                    metadata={
                        "strategy_type": "TREND_CHANGE",
                        "signal_type": signal_type,
                        "stage": stage,
                        "momentum": momentum
                    }
                )
            
            return None
        except Exception as e:
            logging.error(f"❌ 추세전환 분석 오류 ({stock_code}): {e}")
            return None
    
    async def _analyze_pullback(self, stock_code: str, market_data: Dict) -> Optional[StrategySignal]:
        """눌림목 매수 분석"""
        try:
            current_price = market_data.get('current_price', 0)
            recent_high = market_data.get('recent_high', 0)
            recent_low = market_data.get('recent_low', 0)
            
            if not all([current_price, recent_high, recent_low]):
                return None
            
            # 피보나치 되돌림 레벨 계산
            price_range = recent_high - recent_low
            
            for ratio in self.config.pullback_ratios:
                fib_level = recent_high - (price_range * ratio)
                
                # 현재가가 피보나치 레벨 근처인지 확인 (±2%)
                if abs(current_price - fib_level) / fib_level <= 0.02:
                    stage = self.config.pullback_stage.get(stock_code, 0)
                    quantity = self._get_fibonacci_quantity(stage)
                    
                    return StrategySignal(
                        action="BUY",
                        confidence=0.8,
                        reason=f"피보나치 {ratio} 레벨 눌림목",
                        priority=self.config.strategy_priority['PULLBACK'],
                        quantity=quantity,
                        metadata={
                            "strategy_type": "PULLBACK",
                            "fib_ratio": ratio,
                            "fib_level": fib_level,
                            "stage": stage
                        }
                    )
            
            return None
        except Exception as e:
            logging.error(f"❌ 눌림목 분석 오류 ({stock_code}): {e}")
            return None
    
    async def _analyze_breakout(self, stock_code: str, market_data: Dict) -> Optional[StrategySignal]:
        """돌파 매수 분석"""
        try:
            current_price = market_data.get('current_price', 0)
            recent_high = market_data.get('recent_high', 0)
            volume_ratio = market_data.get('volume_ratio', 1.0)
            
            if not all([current_price, recent_high]):
                return None
            
            # 전고점 돌파 확인 (1% 이상)
            breakout_threshold = recent_high * 1.01
            
            if current_price >= breakout_threshold and volume_ratio >= 1.5:
                stage = self.config.breakout_stage.get(stock_code, 0)
                quantity = self._get_fibonacci_quantity(stage)
                
                return StrategySignal(
                    action="BUY",
                    confidence=0.7,
                    reason=f"전고점 {recent_high:,}원 돌파 (거래량 {volume_ratio:.1f}배)",
                    priority=self.config.strategy_priority['BREAKOUT'],
                    quantity=quantity,
                    metadata={
                        "strategy_type": "BREAKOUT",
                        "breakout_price": recent_high,
                        "volume_ratio": volume_ratio,
                        "stage": stage
                    }
                )
            
            return None
        except Exception as e:
            logging.error(f"❌ 돌파 분석 오류 ({stock_code}): {e}")
            return None
    
    def _select_optimal_strategy(self, available_strategies: List[StrategySignal], market_situation: str) -> StrategySignal:
        """시장 상황에 따른 최적 전략 선택"""
        if len(available_strategies) == 1:
            return available_strategies[0]
        
        # 시장 상황별 우선순위 조정
        situation_weights = {
            "TREND_CHANGE_PRIORITY": {'TREND_CHANGE': 0.5, 'PULLBACK': 0.3, 'BREAKOUT': 0.2},
            "PULLBACK_PRIORITY": {'PULLBACK': 0.5, 'TREND_CHANGE': 0.3, 'BREAKOUT': 0.2},
            "BREAKOUT_PRIORITY": {'BREAKOUT': 0.5, 'PULLBACK': 0.3, 'TREND_CHANGE': 0.2}
        }
        
        weights = situation_weights.get(market_situation, {
            'TREND_CHANGE': 0.4, 'PULLBACK': 0.35, 'BREAKOUT': 0.25
        })
        
        # 각 전략의 점수 계산
        for signal in available_strategies:
            strategy_type = signal.metadata.get('strategy_type', '')
            
            # 점수 계산 (낮을수록 좋음)
            priority_score = signal.priority
            confidence_score = (1 - signal.confidence) * 5
            situation_score = (1 - weights.get(strategy_type, 0.1)) * 3
            
            signal.metadata['total_score'] = priority_score + confidence_score + situation_score
        
        # 가장 낮은 점수(최적) 전략 선택
        optimal_signal = min(available_strategies, key=lambda x: x.metadata.get('total_score', 999))
        
        logging.info(f"🎯 전략 선택 결과:")
        for signal in available_strategies:
            status = "✅ 선택됨" if signal == optimal_signal else "⏸️ 대기"
            strategy_type = signal.metadata.get('strategy_type', 'UNKNOWN')
            score = signal.metadata.get('total_score', 0)
            logging.info(f"   {strategy_type}: 점수 {score:.2f} {status}")
        
        return optimal_signal
    
    def _get_fibonacci_quantity(self, stage: int) -> int:
        """피보나치 수열 기반 매수 수량 계산"""
        if stage < len(self.config.fibonacci_sequence):
            return self.config.fibonacci_sequence[stage]
        else:
            # 수열을 넘어서면 마지막 값 사용
            return self.config.fibonacci_sequence[-1]
    
    def update_stage(self, stock_code: str, strategy_type: str):
        """전략별 단계 업데이트"""
        if strategy_type == 'TREND_CHANGE':
            self.config.trend_change_stage[stock_code] = self.config.trend_change_stage.get(stock_code, 0) + 1
        elif strategy_type == 'PULLBACK':
            self.config.pullback_stage[stock_code] = self.config.pullback_stage.get(stock_code, 0) + 1
        elif strategy_type == 'BREAKOUT':
            self.config.breakout_stage[stock_code] = self.config.breakout_stage.get(stock_code, 0) + 1
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """전략 정보 반환"""
        return {
            "name": self.name,
            "enabled": self.config.enabled,
            "strategy_priority": self.config.strategy_priority,
            "fibonacci_sequence": self.config.fibonacci_sequence,
            "active_stages": {
                "trend_change": len(self.config.trend_change_stage),
                "pullback": len(self.config.pullback_stage),
                "breakout": len(self.config.breakout_stage)
            }
        } 