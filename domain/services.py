from .events import (
from .exceptions import (
from .models import (
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from functools import lru_cache
from typing import Any, Dict, List, Optional, Protocol, TypeVar, Generic, Union, Tuple
import asyncio
import logging
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: services.py
모듈: 도메인 서비스 정의
목적: 핵심 비즈니스 로직을 담당하는 서비스 클래스들

Author: Trading Strategy System
Created: 2025-01-27
Modified: 2025-01-27
Version: 2.0.0

Dependencies:
    - Python 3.11+
    - pydantic==2.5.0
    - typing-extensions==4.8.0
    - asyncio
    - abc

Architecture:
    - Domain Services
    - Business Logic
    - Strategy Pattern
    - Dependency Injection

License: MIT
"""



    Signal, Trade, Portfolio, RiskMetrics, MarketData, NewsEvent, TechnicalIndicator,
    SignalType, StrategyType, TradeStatus, RiskLevel, Money, Percentage
)
    DomainException, ValidationException, InsufficientFundsException,
    RiskLimitExceededException, ErrorContext, ErrorSeverity
)
    SignalGenerated, TradeExecuted, RiskAlert, MarketEvent,
    PortfolioUpdated, StrategyPerformanceUpdated, event_bus
)

# 타입 변수
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


# ... (나머지 코드)
