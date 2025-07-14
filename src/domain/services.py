from __future__ import annotations

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: services.py
모듈: 도메인 서비스
목적: 비즈니스 로직 서비스 정의

Author: Trading Strategy System
Created: 2025-01-27
Modified: 2025-01-27
Version: 2.0.0

Dependencies:
    - Python 3.11+
    - typing-extensions==4.8.0
    - dataclasses
    - datetime

Architecture:
    - Domain Services
    - Business Logic
    - Service Layer

License: MIT
"""

# from .events import (
#     Signal, Trade, Portfolio, RiskMetrics, MarketData, NewsEvent, TechnicalIndicator,
#     SignalType, StrategyType, TradeStatus, RiskLevel, Money, Percentage
# )
# from .exceptions import (
#     DomainException, ValidationException, InsufficientFundsException,
#     RiskLimitExceededException, ErrorContext, ErrorSeverity
# )
# from .models import (
#     SignalGenerated, TradeExecuted, RiskAlert, MarketEvent,
#     PortfolioUpdated, StrategyPerformanceUpdated, event_bus
# )
from abc import ABC
import abstractmethod
from dataclasses import dataclass
from datetime import datetime
import timezone
import timedelta
from functools import lru_cache
from typing import Any
import Dict
import List, Optional, Protocol, TypeVar, Generic, Union, Tuple
import asyncio
import logging


T = TypeVar('T')
logger = logging.getLogger(__name__)


class SignalRepository(Protocol):
    """신호 저장소 프로토콜"""

    async def save(self, signal: Signal) -> None:
        """신호 저장"""
        ...

    async def get_by_id(self, signal_id: str) -> Optional[Signal]:
        """ID로 신호 조회"""
        ...

    async def get_by_symbol(self, symbol: str, limit: int = 100) -> List[Signal]:
        """종목별 신호 조회"""
        ...

    async def get_by_strategy(self, strategy_type: StrategyType, limit: int = 100) -> List[Signal]:
        """전략별 신호 조회"""
        ...

    async def get_recent_signals(self, hours: int = 24) -> List[Signal]:
        """최근 신호 조회"""
        ...
