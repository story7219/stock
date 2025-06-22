"""
Ultra HTS v5.0 - 코스피200·나스닥100·S&P500 전체 종목 분석 시스템
Gemini AI 기반 Top5 종목 자동 선정 시스템

This package provides:
- 코스피200·나스닥100·S&P500 전체 종목 데이터 자동 수집
- 투자 대가 전략별 필터링 및 스코어링 (워런 버핏, 피터 린치, 벤저민 그레이엄)
- Gemini AI의 고급 추론 및 Top5 종목 자동 선정
- 기술적 분석 기반 종목 선정 (재무정보 제외)
- 자동화된 테스트 및 리포트 생성
"""

__version__ = "5.0.0"
__author__ = "Ultra HTS Team"
__description__ = "Gemini AI 기반 코스피200·나스닥100·S&P500 Top5 종목 자동 선정 시스템"

from .data_collector import (
    KospiCollector,
    NasdaqCollector,
    SP500Collector,
    DataCollector
)

from .strategies import (
    WarrenBuffettStrategy,
    PeterLynchStrategy,
    BenjaminGrahamStrategy,
    StrategyManager
)

from .gemini_analyzer import (
    GeminiAnalyzer,
    Top5Selector
)

from .technical_analyzer import (
    TechnicalIndicators,
    ChartAnalyzer
)

from .report_generator import (
    ReportGenerator,
    ResultExporter
)

__all__ = [
    # Data Collection
    'KospiCollector',
    'NasdaqCollector',
    'SP500Collector',
    'DataCollector',
    
    # Investment Strategies
    'WarrenBuffettStrategy',
    'PeterLynchStrategy',
    'BenjaminGrahamStrategy',
    'StrategyManager',
    
    # AI Analysis
    'GeminiAnalyzer',
    'Top5Selector',
    
    # Technical Analysis
    'TechnicalIndicators',
    'ChartAnalyzer',
    
    # Report Generation
    'ReportGenerator',
    'ResultExporter'
] 