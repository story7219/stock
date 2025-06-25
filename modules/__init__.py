#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📦 분석 모듈 패키지
투자 분석에 필요한 모든 핵심 모듈들을 포함합니다.
"""

__version__ = "1.1.0"
__author__ = "Investment Analysis System"

# 주요 모듈들 임포트
try:
    from .ai_analyzer import GeminiAnalyzer
    from .data_collector import DataCollector
    from .investment_strategies import InvestmentStrategy
    from .news_analyzer import NewsAnalyzer
    from .technical_analysis import TechnicalAnalyzer
    from .news_collector import NewsCollector
    from .world_chart_experts import WorldChartExperts
    
    __all__ = [
        'GeminiAnalyzer',
        'DataCollector', 
        'InvestmentStrategy',
        'NewsAnalyzer',
        'TechnicalAnalyzer',
        'NewsCollector',
        'WorldChartExperts'
    ]
    
except ImportError as e:
    print(f"⚠️ 일부 모듈 임포트 실패: {e}")
    __all__ = []
