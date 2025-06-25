#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 최적화된 투자 분석 시스템 - 핵심 모듈
====================================
"""

from .optimized_core import (
    get_core,
    initialize_core,
    OptimizedCore,
    SystemConfig,
    PerformanceMonitor,
    MultiLevelCache,
    OptimizedConnectionPool,
    AsyncTaskManager,
    MemoryOptimizer
)

from .base_interfaces import (
    StockData,
    MarketData,
    AnalysisResult,
    TechnicalIndicators,
    InvestmentStrategy
)

from .config import Config as CoreConfig
from .cache_manager import CacheManager
from .connection_pool import ConnectionPool
from .memory_optimizer import MemoryOptimizer as MemOpt
from .async_executor import AsyncExecutor

__all__ = [
    # 핵심 시스템
    'get_core',
    'initialize_core',
    'OptimizedCore',
    'SystemConfig',
    
    # 성능 최적화
    'PerformanceMonitor',
    'MultiLevelCache',
    'OptimizedConnectionPool',
    'AsyncTaskManager',
    'MemoryOptimizer',
    
    # 인터페이스
    'StockData',
    'MarketData',
    'AnalysisResult',
    'TechnicalIndicators',
    'InvestmentStrategy',
    
    # 개별 컴포넌트
    'CoreConfig',
    'CacheManager',
    'ConnectionPool',
    'MemOpt',
    'AsyncExecutor'
]

# 버전 정보
__version__ = "3.0.0" 