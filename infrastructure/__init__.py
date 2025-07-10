from .di import DependencyContainer
from __future__ import annotations
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: __init__.py
모듈: 인프라스트럭처 계층 초기화
목적: 외부 의존성 및 기술적 구현

Author: Trading Strategy System
Created: 2025-01-27
Modified: 2025-01-27
Version: 2.0.0

Dependencies:
    - Python 3.11+
    - pydantic==2.5.0
    - typing-extensions==4.8.0

Architecture:
    - Infrastructure Layer
    - Dependency Injection
    - Repository Pattern
    - Adapter Pattern

License: MIT
"""


# 의존성 주입

# 저장소 구현들
# from .repositories import (
#     SignalRepositoryImpl,
#     TradeRepositoryImpl,
#     PortfolioRepositoryImpl,
# )

# 외부 서비스 어댑터들
# from .adapters import (
#     KISAPIClient,
#     DARTAPIClient,
#     NewsAPIClient
# )

# 데이터베이스 관련
# from .database import DatabaseManager

# 캐싱
# from .cache import CacheManager

# 메시징
# from .messaging import MessageBus

__all__ = [
    'DependencyContainer',
    'SignalRepositoryImpl',
    'TradeRepositoryImpl',
    'PortfolioRepositoryImpl',
    'MarketDataRepositoryImpl',
    'KISAPIClient',
    'DARTAPIClient',
    'NewsAPIClient',
    'DatabaseManager',
    'CacheManager',
    'MessageBus'
]

