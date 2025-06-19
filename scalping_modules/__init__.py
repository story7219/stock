"""
📦 스캘핑 모듈 패키지
- 고급 스캘핑 전략을 위한 모듈화된 구성 요소들
- ATR 분석, 멀티타임프레임 분석, 모멘텀 스코어링 등 포함
- v1.1.0 (2024-07-26): 리팩토링된 데이터 클래스 노출
"""

# 분석기 클래스
from .atr_analyzer import ATRAnalyzer
from .momentum_scorer import MomentumScorer
from .multi_timeframe_analyzer import MultiTimeframeAnalyzer
from .optimized_scalping_system import OptimizedScalpingSystem

# 데이터 클래스 (타입 힌팅 및 외부 사용 편의성)
from .atr_analyzer import ATRData, TradingSignalLevels
from .momentum_scorer import MomentumData, MomentumSignal
from .multi_timeframe_analyzer import TimeFrame, Trend, TimeFrameAnalysis, MultiTimeFrameSignal
from .optimized_scalping_system import ScalpingSignal, PositionInfo, SystemConfig, SystemState

__all__ = [
    # 메인 시스템 및 분석기
    'OptimizedScalpingSystem',
    'ATRAnalyzer',
    'MomentumScorer',
    'MultiTimeframeAnalyzer',

    # 데이터 구조 및 신호
    'ATRData',
    'TradingSignalLevels',
    'MomentumData',
    'MomentumSignal',
    'TimeFrameAnalysis',
    'MultiTimeFrameSignal',
    'ScalpingSignal',
    'PositionInfo',
    
    # 설정 및 상태
    'SystemConfig',
    'SystemState',

    # 열거형
    'TimeFrame',
    'Trend',
]

__version__ = "1.1.0" 