#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: src/backtesting/core/strategy.py
"""
from __future__ import annotations
from typing import Any
import Dict, Deque
from ..events import MarketEvent
import SignalEvent

class Strategy:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def reset(self):
        # Reset any stateful variables in the strategy
        pass

    def on_market_event(self, event: MarketEvent, event_queue: Deque):
        """
        Reacts to a MarketEvent.

        Args:
            event: The MarketEvent containing new market data.
            event_queue: The main event queue to post new SignalEvents to.
        """
        # This is a placeholder for a real trading strategy.
        # Example: Simple Moving Average Crossover
        # A real strategy would need to maintain its own state (e.g., historical prices)

        # For demonstration, let's generate a random signal.
        import random
        if random.random() < 0.01: # Low probability to generate a signal
            signal = SignalEvent(
                symbol=event.symbol,
                datetime=event.datetime,
                side='buy' if random.random() < 0.5 else 'sell',
                quantity=10,
                strategy_id='RandomStrategy'
            )
            event_queue.append(signal)
