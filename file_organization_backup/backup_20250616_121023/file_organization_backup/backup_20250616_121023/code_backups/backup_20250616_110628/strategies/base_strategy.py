"""
기본 전략 인터페이스 및 공통 기능
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

@dataclass
class StrategySignal:
    """전략 신호 데이터 클래스"""
    action: str  # BUY, SELL, HOLD
    confidence: float  # 0.0 ~ 1.0
    reason: str
    priority: int  # 1(최고) ~ 10(최저)
    quantity: int = 1
    target_price: Optional[int] = None
    stop_loss: Optional[int] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class BaseStrategy(ABC):
    """모든 전략의 기본 클래스"""
    
    def __init__(self, name: str):
        self.name = name
        self.enabled = True
        self.last_signal_time = None
        self.performance_metrics = {
            'total_signals': 0,
            'successful_signals': 0,
            'win_rate': 0.0
        }
    
    @abstractmethod
    async def analyze(self, stock_code: str, market_data: Dict) -> Optional[StrategySignal]:
        """전략 분석 실행"""
        pass
    
    @abstractmethod
    def get_strategy_info(self) -> Dict[str, Any]:
        """전략 정보 반환"""
        pass
    
    def update_performance(self, signal: StrategySignal, success: bool):
        """성과 업데이트"""
        self.performance_metrics['total_signals'] += 1
        if success:
            self.performance_metrics['successful_signals'] += 1
        
        self.performance_metrics['win_rate'] = (
            self.performance_metrics['successful_signals'] / 
            self.performance_metrics['total_signals']
        )
    
    def is_signal_valid(self, signal: StrategySignal) -> bool:
        """신호 유효성 검증"""
        if not signal:
            return False
        
        if signal.action not in ['BUY', 'SELL', 'HOLD']:
            return False
        
        if not (0.0 <= signal.confidence <= 1.0):
            return False
        
        return True
    
    def log_signal(self, signal: StrategySignal, stock_code: str):
        """신호 로깅"""
        logging.info(
            f"🎯 [{self.name}] {stock_code}: {signal.action} "
            f"(신뢰도: {signal.confidence:.2f}, 우선순위: {signal.priority}) - {signal.reason}"
        ) 