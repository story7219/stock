#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: src/backtesting/events.py
설명: 백테스팅 시스템에서 사용할 이벤트 클래스 정의
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Event:
    """
    모든 이벤트의 기본 클래스
    """
    pass

@dataclass
class MarketEvent(Event):
    """
    새로운 시장 데이터가 도착했을 때 발생하는 이벤트.
    DataHandler에 의해 생성되어 Strategy로 전달됨.
    """
    type: str = 'MARKET'
    symbol: str = ''
    datetime: datetime = None
    data: dict = None  # OHLCV, volume, volatility 등

@dataclass
class SignalEvent(Event):
    """
    Strategy 객체가 매매 신호를 생성했을 때 발생하는 이벤트.
    BacktestEngine에 의해 ExecutionHandler로 전달됨.
    """
    type: str = 'SIGNAL'
    symbol: str = ''
    datetime: datetime = None
    side: str = ''  # 'buy', 'sell'
    quantity: float = 0.0
    strategy_id: str = ''

@dataclass
class OrderEvent(Event):
    """
    SignalEvent를 받아 실제 주문을 생성했을 때 발생하는 이벤트.
    ExecutionHandler에 의해 생성되며, 실제 체결을 시뮬레이션하는 데 사용됨.
    """
    type: str = 'ORDER'
    symbol: str = ''
    datetime: datetime = None
    quantity: float = 0.0
    side: str = ''  # 'buy', 'sell'
    order_type: str = 'MARKET'  # 'MARKET', 'LIMIT'
    limit_price: float = 0.0

@dataclass
class FillEvent(Event):
    """
    주문이 성공적으로 체결되었을 때 발생하는 이벤트.
    ExecutionHandler에 의해 생성되며, 포트폴리오 관리에 사용됨.
    """
    type: str = 'FILL'
    symbol: str = ''
    datetime: datetime = None
    quantity: float = 0.0
    side: str = ''  # 'buy', 'sell'
    fill_price: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
