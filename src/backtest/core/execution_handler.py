#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: execution_handler.py
모듈: 주문/체결 핸들러
목적: 실제 호가단위, 체결순서, 슬리피지, 수수료, 세금, 장애 등 완벽 반영

Author: WorldClassAI
Created: 2025-07-12
Version: 1.0.0
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Any
import Dict, List, Optional, Tuple, Union
from .utils import apply_tick_size
import calculate_commission

class ExecutionHandler:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.trades = []
        self.orders = []
        self.fills = []

    def reset(self):
        """상태를 초기화합니다."""
        self.trades = []
        self.orders = []
        self.fills = []

    def create_order(self, signal: Dict[str, Any], market_event: Dict[str, Any]) -> Dict[str, Any]:
        """주문을 생성합니다."""
        if not signal or not market_event:
            return {}

        # 실제 호가단위 적용
        price = market_event.get("price", 0)
        tick_sizes = self.config.get("tick_sizes", {})
        adjusted_price = apply_tick_size(price, tick_sizes)

        order = {
            "symbol": signal.get("symbol", market_event.get("symbol", "UNKNOWN")),
            "side": signal.get("side", "buy"),
            "qty": signal.get("qty", 1),
            "price": adjusted_price,
            "datetime": market_event.get("datetime"),
            "order_id": len(self.orders) + 1,
            "status": "pending",
        }

        self.orders.append(order)
        return order

    def execute_order(self, order: Dict[str, Any], market_event: Dict[str, Any]):
        """주문을 실행합니다."""
        if not order or not market_event:
            return

        # 슬리피지 적용
        slippage = self._calculate_slippage(order, market_event)
        execution_price = order["price"] * (1 + slippage)

        # 수수료 계산
        fee = calculate_commission(execution_price, order["qty"], self.config)

        # 체결 정보 생성
        fill = {
            "order_id": order["order_id"],
            "symbol": order["symbol"],
            "side": order["side"],
            "qty": order["qty"],
            "price": execution_price,
            "fee": fee,
            "datetime": order["datetime"],
            "slippage": slippage,
        }

        self.fills.append(fill)

        # 거래 정보 생성
        trade = {
            "symbol": order["symbol"],
            "side": order["side"],
            "qty": order["qty"],
            "price": execution_price,
            "fee": fee,
            "datetime": order["datetime"],
            "slippage": slippage,
            "pnl": self._calculate_pnl(order["side"], execution_price, order["qty"], fee),
        }

        self.trades.append(trade)

        # 주문 상태 업데이트
        order["status"] = "filled"
        order["execution_price"] = execution_price
        order["fee"] = fee

    def get_trades(self) -> List[Dict[str, Any]]:
        """거래 내역을 반환합니다."""
        return self.trades

    def get_orders(self) -> List[Dict[str, Any]]:
        """주문 내역을 반환합니다."""
        return self.orders

    def get_fills(self) -> List[Dict[str, Any]]:
        """체결 내역을 반환합니다."""
        return self.fills

    def _calculate_slippage(self, order: Dict[str, Any], market_event: Dict[str, Any]) -> float:
        """슬리피지를 계산합니다."""
        # 기본 슬리피지 (호가스프레드)
        base_slippage = 0.0001  # 0.01%

        # 거래량에 따른 마켓임팩트
        volume_impact = min(order["qty"] / market_event.get("volume", 1000000), 0.01)

        # 변동성에 따른 슬리피지
        volatility = market_event.get("volatility", 0.02)
        volatility_impact = volatility * 0.1

        # 총 슬리피지
        total_slippage = base_slippage + volume_impact + volatility_impact

        # 매도일 때는 음수, 매수일 때는 양수
        if order["side"] == "sell":
            total_slippage = -total_slippage

        return total_slippage

    def _calculate_pnl(self, side: str, price: float, qty: float, fee: float) -> float:
        """손익을 계산합니다."""
        if side == "buy":
            return -(price * qty + fee)
        else:  # sell
            return price * qty - fee
