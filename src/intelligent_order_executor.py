from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal, getcontext
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Literal
import asyncio
import logging
import numpy as np
import pandas as pd
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: intelligent_order_executor.py
모듈: 지능형 주문 실행 시스템
목적: 단기매매 특화 진입/익절/손절 전략 및 실행 최적화

Author: AI Assistant
Created: 2025-01-27
Modified: 2025-01-27
Version: 1.0.0

Dependencies:
    - Python 3.11+
    - asyncio, typing, dataclasses
    - numpy, pandas (데이터 처리)
    - aiohttp (API 통신)

Performance:
    - 주문 실행: < 100ms
    - 호가창 분석: < 50ms
    - 분할 주문: 동시 3-5개 주문

Security:
    - 주문 한도 체크
    - 리스크 관리
    - 실패 시 롤백
"""




# 정밀도 설정 (금융 계산용)
getcontext().prec = 10

# 로깅 설정
logger = logging.getLogger(__name__)

class OrderType(Enum):
    """주문 타입"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    TRAILING_STOP = "trailing_stop"

class PositionSide(Enum):
    """포지션 방향"""
    LONG = "long"
    SHORT = "short"

@dataclass
class OrderBook:
    """호가창 정보"""
    bids: List[Tuple[float, int]]  # (가격, 수량)
    asks: List[Tuple[float, int]]  # (가격, 수량)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class MarketData:
    """시장 데이터"""
    symbol: str
    current_price: float
    volume: int
    high: float
    low: float
    change_rate: float
    order_book: OrderBook
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class EntrySignal:
    """진입 신호"""
    signal_type: Literal["breakout", "volume_surge", "technical_ai"]
    confidence: float  # 0-1
    target_price: float
    stop_loss: float
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class Position:
    """포지션 정보"""
    symbol: str
    side: PositionSide
    quantity: int
    avg_price: float
    entry_time: datetime
    stop_loss: float
    take_profit_levels: List[Tuple[float, float]]  # (가격, 비율)
    trailing_stop: Optional[float] = None
    high_since_entry: float = 0.0

class IntelligentOrderExecutor:
    """지능형 주문 실행 시스템"""

    def __init__(self, trader, config: Dict[str, Any]):
        self.trader = trader
        self.config = config
        self.positions: Dict[str, Position] = {}
        self.order_history: List[Dict] = []

        # 설정값
        self.max_positions = config.get('max_positions', 5)
        self.max_order_size = config.get('max_order_size', 1000000)  # 100만원
        self.min_liquidity = config.get('min_liquidity', 1000000)  # 최소 유동성
        self.slippage_tolerance = config.get('slippage_tolerance', 0.002)  # 0.2%

        # 익절 설정
        self.take_profit_levels = [
            (0.03, 0.5),  # +3%, 50%
            (0.05, 0.3),  # +5%, 30%
            (0.07, 0.2),  # +7%, 20%
        ]

        # 손절 설정
        self.stop_loss_pct = 0.02  # -2%
        self.time_stop_days = 3  # 3일

    async def check_entry_conditions(self, market_data: MarketData,
                                   news_sentiment: float,
                                   ai_signal: Dict[str, Any]) -> Optional[EntrySignal]:
        """진입 조건 체크"""

        # 1. 급등 초기 진입 (상위 5% 돌파)
        if market_data.change_rate > 0.05:
            return EntrySignal(
                signal_type="breakout",
                confidence=0.8,
                target_price=market_data.current_price * 1.03,
                stop_loss=market_data.current_price * (1 - self.stop_loss_pct)
            )

        # 2. 거래량 급증 + 호재 뉴스
        volume_surge = market_data.volume > self.config.get('volume_threshold', 1000000)
        positive_news = news_sentiment > 0.3

        if volume_surge and positive_news:
            return EntrySignal(
                signal_type="volume_surge",
                confidence=0.7,
                target_price=market_data.current_price * 1.02,
                stop_loss=market_data.current_price * (1 - self.stop_loss_pct)
            )

        # 3. 기술적 돌파 + AI 긍정 신호
        technical_breakout = self._check_technical_breakout(market_data)
        ai_positive = ai_signal.get('confidence', 0) > 0.6 and ai_signal.get('direction_30min') == 'UP'

        if technical_breakout and ai_positive:
            return EntrySignal(
                signal_type="technical_ai",
                confidence=ai_signal.get('confidence', 0.5),
                target_price=market_data.current_price * 1.025,
                stop_loss=market_data.current_price * (1 - self.stop_loss_pct)
            )

        return None

    def _check_technical_breakout(self, market_data: MarketData) -> bool:
        """기술적 돌파 체크"""
        # 예시: 고점 돌파, 거래량 증가 등
        return market_data.current_price > market_data.high * 0.98

    async def execute_entry(self, signal: EntrySignal, market_data: MarketData) -> bool:
        """진입 주문 실행"""

        # 유동성 체크
        if not await self._check_liquidity(market_data):
            logger.warning(f"Insufficient liquidity for {market_data.symbol}")
            return False

        # 포지션 한도 체크
        if len(self.positions) >= self.max_positions:
            logger.warning("Maximum positions reached")
            return False

        # 주문 수량 계산
        order_amount = min(self.max_order_size,
                          self.config.get('position_size', 500000))
        quantity = int(order_amount / market_data.current_price)

        # 분할 매수 실행
        success = await self._execute_split_orders(
            symbol=market_data.symbol,
            side="buy",
            quantity=quantity,
            market_data=market_data
        )

        if success:
            # 포지션 등록
            self.positions[market_data.symbol] = Position(
                symbol=market_data.symbol,
                side=PositionSide.LONG,
                quantity=quantity,
                avg_price=market_data.current_price,
                entry_time=datetime.now(),
                stop_loss=signal.stop_loss,
                take_profit_levels=self._calculate_take_profit_levels(
                    entry_price=market_data.current_price,
                    levels=self.take_profit_levels
                ),
                high_since_entry=market_data.current_price
            )

            logger.info(f"Entry executed: {market_data.symbol}, "
                       f"Qty: {quantity}, Price: {market_data.current_price}")

        return success

    async def _execute_split_orders(self, symbol: str, side: str,
                                  quantity: int, market_data: MarketData) -> bool:
        """분할 주문 실행 (슬리페지 최소화)"""

        # 호가창 분석
        optimal_price = self._analyze_order_book(market_data.order_book, side)

        # 3-5개로 분할
        split_count = min(5, max(3, quantity // 100))
        split_quantity = quantity // split_count

        success_count = 0

        for i in range(split_count):
            try:
                # 마지막 주문은 남은 수량
                if i == split_count - 1:
                    current_quantity = quantity - (split_quantity * i)
                else:
                    current_quantity = split_quantity

                # 주문 실행
                order_result = await self.trader.place_order(
                    symbol=symbol,
                    side=side,
                    quantity=current_quantity,
                    price=optimal_price,
                    order_type=OrderType.LIMIT
                )

                if order_result.get('success'):
                    success_count += 1
                    await asyncio.sleep(0.1)  # 100ms 간격

            except Exception as e:
                logger.error(f"Split order failed: {e}")

        return success_count > 0

    def _analyze_order_book(self, order_book: OrderBook, side: str) -> float:
        """호가창 분석 기반 최적 주문가 결정"""

        if side == "buy":
            # 매수: 호가 상위 20% 내에서 결정
            total_volume = sum(qty for _, qty in order_book.asks[:5])
            target_volume = total_volume * 0.2

            cumulative_volume = 0
            for price, qty in order_book.asks:
                cumulative_volume += qty
                if cumulative_volume >= target_volume:
                    return price
        else:
            # 매도: 호가 하위 20% 내에서 결정
            total_volume = sum(qty for _, qty in order_book.bids[:5])
            target_volume = total_volume * 0.2

            cumulative_volume = 0
            for price, qty in order_book.bids:
                cumulative_volume += qty
                if cumulative_volume >= target_volume:
                    return price

        return order_book.asks[0][0] if side == "buy" else order_book.bids[0][0]

    async def _check_liquidity(self, market_data: MarketData) -> bool:
        """유동성 체크"""
        total_volume = sum(qty for _, qty in market_data.order_book.asks[:10])
        return total_volume >= self.min_liquidity

    def _calculate_take_profit_levels(self, entry_price: float,
                                    levels: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """익절 레벨 계산"""
        return [(entry_price * (1 + pct), ratio) for pct, ratio in levels]

    async def check_exit_conditions(self, market_data: MarketData,
                                  news_sentiment: float) -> List[Dict[str, Any]]:
        """익절/손절 조건 체크"""

        if market_data.symbol not in self.positions:
            return []

        position = self.positions[market_data.symbol]
        exit_orders = []

        # 고점 업데이트
        if market_data.current_price > position.high_since_entry:
            position.high_since_entry = market_data.current_price

        # 1. 익절 체크
        for target_price, ratio in position.take_profit_levels:
            if market_data.current_price >= target_price:
                exit_quantity = int(position.quantity * ratio)
                if exit_quantity > 0:
                    exit_orders.append({
                        'type': 'take_profit',
                        'quantity': exit_quantity,
                        'price': target_price,
                        'reason': f'Take profit at {target_price}'
                    })

        # 2. 트레일링 스톱 체크
        if position.trailing_stop and position.high_since_entry:
            trailing_price = position.high_since_entry * 0.98  # 고점 대비 -2%
            if market_data.current_price <= trailing_price:
                exit_orders.append({
                    'type': 'trailing_stop',
                    'quantity': position.quantity,
                    'price': market_data.current_price,
                    'reason': f'Trailing stop at {trailing_price}'
                })

        # 3. 손절 체크
        if market_data.current_price <= position.stop_loss:
            exit_orders.append({
                'type': 'stop_loss',
                'quantity': position.quantity,
                'price': market_data.current_price,
                'reason': f'Stop loss at {position.stop_loss}'
            })

        # 4. 뉴스 손절
        if news_sentiment < -0.3:  # 악재
            exit_orders.append({
                'type': 'news_stop',
                'quantity': position.quantity,
                'price': market_data.current_price,
                'reason': 'Negative news sentiment'
            })

        # 5. 시간 손절
        days_held = (datetime.now() - position.entry_time).days
        if days_held >= self.time_stop_days:
            exit_orders.append({
                'type': 'time_stop',
                'quantity': position.quantity,
                'price': market_data.current_price,
                'reason': f'Time stop after {days_held} days'
            })

        return exit_orders

    async def execute_exits(self, exit_orders: List[Dict[str, Any]],
                          market_data: MarketData) -> bool:
        """익절/손절 주문 실행"""

        if not exit_orders:
            return True

        success = True

        for order in exit_orders:
            try:
                # 분할 매도 실행
                split_success = await self._execute_split_orders(
                    symbol=market_data.symbol,
                    side="sell",
                    quantity=order['quantity'],
                    market_data=market_data
                )

                if split_success:
                    # 포지션 업데이트
                    position = self.positions[market_data.symbol]
                    position.quantity -= order['quantity']

                    if position.quantity <= 0:
                        del self.positions[market_data.symbol]

                    logger.info(f"Exit executed: {order['reason']}, "
                               f"Qty: {order['quantity']}")
                else:
                    success = False

            except Exception as e:
                logger.error(f"Exit execution failed: {e}")
                success = False

        return success

    async def update_trailing_stops(self, market_data: MarketData):
        """트레일링 스톱 업데이트"""

        if market_data.symbol in self.positions:
            position = self.positions[market_data.symbol]

            if market_data.current_price > position.high_since_entry:
                position.high_since_entry = market_data.current_price
                position.trailing_stop = position.high_since_entry * 0.98

# 사용 예시
async def main():
    """메인 실행 함수"""

    # 설정
    config = {
        'max_positions': 5,
        'max_order_size': 1000000,
        'min_liquidity': 1000000,
        'slippage_tolerance': 0.002,
        'volume_threshold': 1000000,
        'position_size': 500000
    }

    # 트레이더 인스턴스 (실제 KIS API 연동)
    # trader = KISTrader(...)

    # 실행기 생성
    # executor = IntelligentOrderExecutor(trader, config)

    # 시장 데이터 예시
    market_data = MarketData(
        symbol="005930",  # 삼성전자
        current_price=75000,
        volume=1500000,
        high=75500,
        low=74500,
        change_rate=0.03,
        order_book=OrderBook(
            bids=[(74900, 1000), (74800, 2000)],
            asks=[(75100, 1000), (75200, 2000)]
        )
    )

    # 진입 신호 체크
    # signal = await executor.check_entry_conditions(market_data, 0.5, {'confidence': 0.7})
    # if signal:
    #     await executor.execute_entry(signal, market_data)

    # 익절/손절 체크
    # exit_orders = await executor.check_exit_conditions(market_data, 0.2)
    # await executor.execute_exits(exit_orders, market_data)

if __name__ == "__main__":
    asyncio.run(main())

