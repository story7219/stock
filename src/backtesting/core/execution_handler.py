#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: src/backtesting/core/execution_handler.py
"""
from __future__ import annotations
from typing import Any
import Dict, List, Deque
from ..utils.helpers import apply_tick_size
import calculate_commission
from ..events import OrderEvent
import FillEvent, SignalEvent
from .data_handler import DataHandler # 데이터 핸들러 참조 추가

class ExecutionHandler:
    def __init__(self, config: Dict[str, Any], data_handler: DataHandler):
        self.config = config
        self.data_handler = data_handler # 데이터 핸들러 인스턴스 저장
        self.trades: List[Dict[str, Any]] = []
        self.orders: List[Dict[str, Any]] = []

    def reset(self):
        self.trades.clear()
        self.orders.clear()

    def on_signal_event(self, event: SignalEvent, event_queue: Deque):
        """
        Handles a SignalEvent and creates an OrderEvent.
        """
        order = OrderEvent(
            symbol=event.symbol,
            datetime=event.datetime,
            quantity=event.quantity,
            side=event.side,
            order_type='MARKET'
        )
        event_queue.append(order)

    def on_order_event(self, event: OrderEvent, event_queue: Deque):
        """
        Handles an OrderEvent and simulates execution based on market volume,
        creating a FillEvent for partially or fully filled orders.
        """
        # Get current market data for the order's timestamp
        market_data = self.data_handler.get_latest_data(event.symbol, event.datetime)
        if market_data is None:
            return

        available_volume = market_data.get('volume', 0)
        market_price = market_data.get('close', 0) # Use close price as reference

        max_fill_ratio = self.config.get("max_fill_ratio", 0.25)
        fillable_quantity = available_volume * max_fill_ratio

        filled_quantity = min(event.quantity, fillable_quantity)

        if filled_quantity <= 0:
            return

        adjusted_price = apply_tick_size(market_price, self.config.get("tick_sizes", {}))
        slippage_pct = self._calculate_slippage(event, filled_quantity, available_volume)

        if event.side == 'buy':
            fill_price = adjusted_price * (1 + slippage_pct)
        else: # sell
            fill_price = adjusted_price * (1 - slippage_pct)

        commission = calculate_commission(
            fill_price, filled_quantity, self.config, event.side, event.datetime
        )

        fill_event = FillEvent(
            symbol=event.symbol,
            datetime=event.datetime,
            quantity=filled_quantity,
            side=event.side,
            fill_price=fill_price,
            commission=commission,
            slippage=slippage_pct
        )
        event_queue.append(fill_event)

        self.trades.append({
            "symbol": event.symbol, "side": event.side, "qty": filled_quantity,
            "price": fill_price, "fee": commission, "datetime": event.datetime,
        })


    def get_trades(self) -> List[Dict[str, Any]]:
        return self.trades

    def _calculate_slippage(self, order: OrderEvent, filled_quantity: float, available_volume: float) -> float:
        base_slippage = self.config.get("base_slippage", 0.0005)
        volume_impact = (filled_quantity / (available_volume + 1e-6)) * 0.1
        return base_slippage + volume_impact
