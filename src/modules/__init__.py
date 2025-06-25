#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📊 최적화된 투자 분석 시스템 - 모듈
================================
"""

# 핵심 데이터 처리
from .unified_data_processor import (
    get_processor,
    UnifiedDataProcessor,
    MarketDataManager,
    AsyncDataCollector,
    TechnicalAnalyzer,
    GeminiAIAnalyzer
)

# 투자 전략
from .optimized_investment_strategies import (
    get_strategy_engine,
    OptimizedStrategyEngine,
    StrategyFactory
)

# 기본 인터페이스에서 InvestmentStrategy import
from ..core.base_interfaces import InvestmentStrategy

# 알림 시스템
from .notification_system import NotificationSystem

# 리포트 생성
from .report_generator import ReportGenerator

# 백테스팅
from .backtesting_engine import BacktestingEngine

# 포트폴리오 관리
from .portfolio_manager import PortfolioManager

# 성능 최적화
from .performance_optimizer import PerformanceOptimizer

# Gemini 분석
from .gemini_analyzer import GeminiAnalyzer

# 뉴스 분석
from .news_collector import NewsCollector
from .news_analyzer import NewsAnalyzer

# 기술적 분석
from .technical_analysis import TechnicalAnalyzer

# 파생상품 모니터링
from .derivatives_monitor import (
    DerivativeData,
    MarketSignal, 
    DerivativesMonitor
)

# 한국투자증권 API
from .kis_derivatives_api import (
    KISDerivativeData,
    KISDerivativesAPI,
    get_kis_derivatives_api
)

__all__ = [
    # 데이터 처리
    'get_processor',
    'UnifiedDataProcessor',
    'MarketDataManager',
    'AsyncDataCollector',
    'TechnicalAnalyzer',
    'GeminiAIAnalyzer',
    
    # 투자 전략
    'get_strategy_engine',
    'OptimizedStrategyEngine',
    'StrategyFactory',
    
    # 시스템 모듈
    'NotificationSystem',
    'ReportGenerator',
    'BacktestingEngine',
    'PortfolioManager',
    'PerformanceOptimizer',
    'GeminiAnalyzer',
    'NewsCollector',
    'NewsAnalyzer',
    'TechnicalAnalyzer',
    
    # 파생상품 모니터링
    'DerivativeData',
    'MarketSignal',
    'DerivativesMonitor',
    
    # 한국투자증권 API
    'KISDerivativeData',
    'KISDerivativesAPI',
    'get_kis_derivatives_api'
]

# 버전 정보
__version__ = "3.0.0"