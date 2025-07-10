from .exceptions import ErrorContext
from .models import Signal, Trade, Portfolio, RiskMetrics, MarketData, NewsEvent
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Protocol, TypeVar, Generic, Union
from uuid import uuid4
import asyncio
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: events.py
모듈: 도메인 이벤트 정의
목적: 이벤트 기반 아키텍처를 위한 이벤트 및 핸들러 정의

Author: Trading Strategy System
Created: 2025-01-27
Modified: 2025-01-27
Version: 2.0.0

Dependencies:
    - Python 3.11+
    - typing-extensions==4.8.0
    - dataclasses
    - datetime
    - abc
    - asyncio

Architecture:
    - Event-Driven Architecture
    - Domain Events
    - Event Handlers
    - Event Bus

License: MIT
"""





class EventType(Enum):
    """이벤트 타입 열거형"""
    SIGNAL_GENERATED = auto()
    TRADE_EXECUTED = auto()
    RISK_ALERT = auto()
    MARKET_EVENT = auto()
    PORTFOLIO_UPDATED = auto()
    STRATEGY_PERFORMANCE_UPDATED = auto()
    SYSTEM_HEALTH_CHECK = auto()
    CONFIGURATION_CHANGED = auto()


class EventPriority(Enum):
    """이벤트 우선순위 열거형"""
    LOW = auto()
    NORMAL = auto()
    HIGH = auto()
    CRITICAL = auto()


@dataclass(kw_only=True)
class DomainEvent(ABC):
    event_type: EventType
    event_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    priority: EventPriority = field(default=EventPriority.NORMAL)
    source: str = field(default="system")
    correlation_id: Optional[str] = field(default=None)
    user_id: Optional[str] = field(default=None)
    session_id: Optional[str] = field(default=None)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.correlation_id:
            self.correlation_id = self.event_id

    @abstractmethod
    def get_summary(self) -> str:
        pass

    def to_dict(self) -> Dict[str, Any]:
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.name,
            'timestamp': self.timestamp.isoformat(),
            'priority': self.priority.name,
            'source': self.source,
            'correlation_id': self.correlation_id,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'metadata': self.metadata,
            'summary': self.get_summary()
        }


# ... (나머지 코드)
