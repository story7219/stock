"""
📈 피보나치 분할매수 전략
"""

import logging
from typing import List, Optional

class FibonacciStrategy:
    """피보나치 분할매수 전략"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.fibonacci_sequence = [1, 1, 2, 3, 5, 8, 13, 21]
        self.is_initialized = False
    
    async def initialize(self):
        """전략 초기화"""
        if self.is_initialized:
            return
            
        self.logger.info("📈 피보나치 전략 초기화")
        self.is_initialized = True
    
    def get_next_quantity(self, current_total: int) -> int:
        """다음 매수 수량 계산"""
        try:
            # 현재까지 매수한 총 수량을 기준으로 다음 피보나치 수량 결정
            cumulative = 0
            for i, fib_num in enumerate(self.fibonacci_sequence):
                cumulative += fib_num
                if cumulative > current_total:
                    next_quantity = fib_num
                    self.logger.info(f"📊 다음 피보나치 수량: {next_quantity}주")
                    return next_quantity
            
            # 시퀀스 끝에 도달한 경우
            self.logger.warning("⚠️ 피보나치 시퀀스 완료")
            return 0
            
        except Exception as e:
            self.logger.error(f"❌ 피보나치 수량 계산 오류: {e}")
            return 1  # 기본값 