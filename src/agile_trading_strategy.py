from __future__ import annotations
from aiohttp import ClientSession
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from .kis_config import KISConfig
from .kis_trader import KISTrader
from .push_notifications import PushNotificationService
from typing import Dict, List, Optional, Any, Tuple
import asyncio
import json
import logging
import numpy as np
import pandas as pd
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: agile_trading_strategy.py
모듈: 소액 투자 민첩성 전략
목적: 소액의 장점 극대화 (민첩성, 즉시 진입, 빠른 청산, 소형주 집중)

Author: AI Assistant
Created: 2025-01-27
Modified: 2025-01-27
Version: 1.0.0

Dependencies:
    - Python 3.11+
    - asyncio, aiohttp, pandas, numpy
    - kis_config, kis_trader
"""





logger = logging.getLogger(__name__)


class EntrySpeed(Enum):
    """진입 속도"""
    INSTANT = "instant"  # 30초 내
    FAST = "fast"  # 1분 내
    NORMAL = "normal"  # 3분 내


class ExitSpeed(Enum):
    """청산 속도"""
    EMERGENCY = "emergency"  # 즉시
    FAST = "fast"  # 30초 내
    NORMAL = "normal"  # 1분 내


@dataclass
class AgileSignal:
    """민첩성 신호"""
    symbol: str
    name: str
    signal_type: str  # 'buy', 'sell', 'hold'
    confidence: float
    entry_speed: EntrySpeed
    exit_speed: ExitSpeed
    reason: str
    target_price: float
    stop_loss: float
    expected_hold_time: timedelta
    market_cap: float  # 시가총액
    volume_surge: float  # 거래량 급증률
    news_impact: float  # 뉴스 영향도
    theme_strength: float  # 테마 강도
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SmallCapOpportunity:
    """소형주 기회"""
    symbol: str
    name: str
    market_cap: float
    institutional_ratio: float  # 기관 보유 비율
    retail_ratio: float  # 개인 보유 비율
    volume_surge: float
    price_momentum: float
    news_count: int
    theme_keywords: List[str]
    opportunity_score: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ThemeOpportunity:
    """테마 기회"""
    theme_name: str
    keywords: List[str]
    related_stocks: List[str]
    momentum_score: float
    news_count: int
    volume_increase: float
    price_increase: float
    early_stage: bool  # 초기 단계 여부
    opportunity_score: float
    timestamp: datetime = field(default_factory=datetime.now)


class AgileTradingStrategy:
    """소액 투자 민첩성 전략"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.kis_config = KISConfig()
        self.kis_trader = KISTrader(self.kis_config)
        self.max_position_size = config.get('max_position_size', 1000000)
        self.min_market_cap = config.get('min_market_cap', 100000000000)
        self.max_market_cap = config.get('max_market_cap', 5000000000000)
        self.volume_surge_threshold = config.get('volume_surge_threshold', 2.0)
        self.news_impact_threshold = config.get('news_impact_threshold', 0.7)
        self.theme_strength_threshold = config.get('theme_strength_threshold', 0.6)
        self.instant_entry_timeout = config.get('instant_entry_timeout', 30)
        self.fast_exit_timeout = config.get('fast_exit_timeout', 30)
        self.max_hold_time = config.get('max_hold_time', 7)
        self.push_service = PushNotificationService(config.get('push_config', {}))
        self.opportunities = {}
        self.themes = {}
        self.active_signals = []
        self.session = None

    # ... (rest of the code)
