#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: strategy.py
모듈: 전략
목적: 신호 생성 (예시: 단순 모멘텀)

Author: WorldClassAI
Created: 2025-07-12
Version: 1.0.0
"""

from __future__ import annotations
from typing import Any
import Dict, List, Optional
import numpy as np

class Strategy:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.state = {}
        self.position = 0
        self.cash = 1000000  # 초기 자금 100만원

    def reset(self):
        """상태를 초기화합니다."""
        self.state = {}
        self.position = 0
        self.cash = 1000000

    def generate_signal(self, market_event: Dict[str, Any]) -> Dict[str, Any]:
        """신호를 생성합니다."""
        if not market_event:
            return {}

        symbol = market_event.get("symbol", "UNKNOWN")
        price = market_event.get("price", 0)
        volume = market_event.get("volume", 0)

        if price <= 0:
            return {}

        # 간단한 모멘텀 전략
        signal = self._momentum_strategy(market_event)

        # 포지션 관리
        if signal and signal.get("side") == "buy" and self.cash >= price * signal.get("qty", 1):
            self.position += signal.get("qty", 1)
            self.cash -= price * signal.get("qty", 1)
        elif signal and signal.get("side") == "sell" and self.position > 0:
            self.position -= signal.get("qty", 1)
            self.cash += price * signal.get("qty", 1)

        return signal

    def _momentum_strategy(self, market_event: Dict[str, Any]) -> Dict[str, Any]:
        """모멘텀 전략을 구현합니다."""
        symbol = market_event.get("symbol", "UNKNOWN")
        price = market_event.get("price", 0)
        volume = market_event.get("volume", 0)

        # 과거 가격 데이터 (간단한 시뮬레이션)
        if symbol not in self.state:
            self.state[symbol] = {"prices": [], "volumes": []}

        self.state[symbol]["prices"].append(price)
        self.state[symbol]["volumes"].append(volume)

        # 최근 20개 데이터만 유지
        if len(self.state[symbol]["prices"]) > 20:
            self.state[symbol]["prices"] = self.state[symbol]["prices"][-20:]
            self.state[symbol]["volumes"] = self.state[symbol]["volumes"][-20:]

        # 신호 생성 로직
        if len(self.state[symbol]["prices"]) < 5:
            return {}

        prices = self.state[symbol]["prices"]
        current_price = prices[-1]
        prev_price = prices[-2]

        # 단순 모멘텀: 현재가 > 이전가 → 매수, 현재가 < 이전가 → 매도
        if current_price > prev_price * 1.001:  # 0.1% 이상 상승
            return {
                "symbol": symbol,
                "side": "buy",
                "qty": 1,
                "price": current_price,
                "datetime": market_event.get("datetime"),
            }
        elif current_price < prev_price * 0.999:  # 0.1% 이상 하락
            return {
                "symbol": symbol,
                "side": "sell",
                "qty": 1,
                "price": current_price,
                "datetime": market_event.get("datetime"),
            }

        return {}

    def get_position(self) -> int:
        """현재 포지션을 반환합니다."""
        return self.position

    def get_cash(self) -> float:
        """현재 현금을 반환합니다."""
        return self.cash
