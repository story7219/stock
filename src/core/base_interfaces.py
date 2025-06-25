#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📋 기본 인터페이스 및 공통 데이터 클래스 정의 (Base Interfaces)
===========================================================

투자 분석 시스템의 핵심 인터페이스와 데이터 구조를 정의합니다.
모든 모듈에서 공통으로 사용하는 표준 인터페이스와 데이터 클래스를 제공합니다.

주요 구성 요소:
1. 열거형 (Enums):
   - MarketType: 지원하는 시장 유형 (코스피200, 나스닥100, S&P500)
   - StrategyType: 투자 전략 유형 (워런 버핏, 벤저민 그레이엄 등)
   - RiskLevel: 위험도 수준 (낮음, 보통, 높음)
   - InvestmentPeriod: 투자 기간 (단기, 중기, 장기)

2. 데이터 클래스 (Data Classes):
   - StockData: 주식 기본 정보 및 가격 데이터
   - TechnicalIndicators: 기술적 지표 (RSI, MACD, 볼린저밴드 등)
   - TechnicalSignals: 기술적 신호 (매수/매도/중립)
   - TechnicalAnalysisResult: 기술적 분석 종합 결과
   - StrategyScore: 투자 전략별 점수 및 근거
   - InvestmentRecommendation: 최종 투자 추천 결과
   - MarketData: 시장 데이터 표준 클래스

3. 인터페이스 (Interfaces):
   - IDataCollector: 데이터 수집기 인터페이스
   - ITechnicalAnalyzer: 기술적 분석기 인터페이스
   - IInvestmentStrategy: 투자 전략 인터페이스
   - IAIAnalyzer: AI 분석기 인터페이스
   - IReportGenerator: 리포트 생성기 인터페이스

4. 예외 클래스 (Exceptions):
   - AnalysisError: 분석 관련 오류
   - DataCollectionError: 데이터 수집 오류
   - StrategyError: 전략 실행 오류
   - AIAnalysisError: AI 분석 오류

5. 유틸리티 함수:
   - validate_stock_data: 주식 데이터 유효성 검증
   - calculate_expected_return: 기대 수익률 계산
   - determine_position_size: 포지션 크기 결정

이 모듈은 시스템 전체의 표준을 정의하며, 모든 다른 모듈들이 
이 인터페이스를 구현하거나 데이터 클래스를 사용합니다.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import pandas as pd
from enum import Enum


class MarketType(Enum):
    """시장 유형"""
    KOSPI200 = "KOSPI200"
    NASDAQ100 = "NASDAQ100"
    SP500 = "SP500"


class StrategyType(Enum):
    """투자 전략 유형"""
    WARREN_BUFFETT = "워런_버핏"
    BENJAMIN_GRAHAM = "벤저민_그레이엄"
    PETER_LYNCH = "피터_린치"
    GEORGE_SOROS = "조지_소로스"
    JAMES_SIMONS = "제임스_사이먼스"
    RAY_DALIO = "레이_달리오"
    JOEL_GREENBLATT = "조엘_그린블랫"
    WILLIAM_ONEIL = "윌리엄_오닐"
    JESSE_LIVERMORE = "제시_리버모어"
    PAUL_TUDOR_JONES = "폴_튜더_존스"
    RICHARD_DENNIS = "리처드_데니스"
    ED_SEYKOTA = "에드_세이코타"
    LARRY_WILLIAMS = "래리_윌리엄스"
    MARTIN_SCHWARTZ = "마틴_슈바르츠"
    STANLEY_DRUCKENMILLER = "스탠리_드러켄밀러"
    JOHN_HENRY = "존_헨리"
    BRUCE_KOVNER = "브루스_코브너"


class RiskLevel(Enum):
    """위험도 수준"""
    LOW = "낮음"
    MEDIUM = "보통" 
    HIGH = "높음"


class InvestmentPeriod(Enum):
    """투자 기간"""
    SHORT = "단기"  # 단기 (1-3개월)
    MEDIUM = "중기"  # 중기 (3-12개월)
    LONG = "장기"  # 장기 (1년 이상)


@dataclass
class StockData:
    """주식 데이터 표준 클래스"""
    symbol: str
    name: str = ""
    market: Optional[MarketType] = None
    current_price: float = 0.0
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    volume: Optional[int] = None
    change_percent: Optional[float] = None
    historical_data: Optional[pd.DataFrame] = None
    info: Dict[str, Any] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """초기화 후 검증"""
        if not self.symbol:
            raise ValueError("심볼이 비어있을 수 없습니다")
        if self.current_price < 0:
            raise ValueError("현재가는 음수일 수 없습니다")


@dataclass
class MarketData:
    """시장 데이터 표준 클래스"""
    market: MarketType
    stocks: List[StockData] = field(default_factory=list)
    market_index: Optional[float] = None
    market_change: Optional[float] = None
    total_volume: Optional[int] = None
    active_stocks: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def get_top_stocks(self, n: int = 10) -> List[StockData]:
        """상위 N개 종목 반환"""
        return sorted(self.stocks, key=lambda x: x.market_cap or 0, reverse=True)[:n]
    
    def get_stocks_by_performance(self, ascending: bool = False) -> List[StockData]:
        """성과별 정렬된 종목 반환"""
        return sorted(self.stocks, key=lambda x: x.change_percent or 0, reverse=not ascending)


@dataclass
class TechnicalIndicators:
    """기술적 지표 데이터"""
    rsi: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    bb_upper: Optional[float] = None
    bb_middle: Optional[float] = None
    bb_lower: Optional[float] = None
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    ema_20: Optional[float] = None
    volume_sma: Optional[float] = None
    adx: Optional[float] = None
    stoch_k: Optional[float] = None
    stoch_d: Optional[float] = None
    obv: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class TechnicalSignals:
    """기술적 신호"""
    rsi_signal: str = "중립"
    macd_signal: str = "중립"
    bb_signal: str = "중립"
    ma_trend: str = "중립"
    volume_signal: str = "중립"
    overall_trend: str = "중립"
    
    def to_dict(self) -> Dict[str, str]:
        """딕셔너리 변환"""
        return self.__dict__.copy()


@dataclass
class TechnicalAnalysisResult:
    """기술적 분석 결과"""
    symbol: str
    indicators: TechnicalIndicators
    signals: TechnicalSignals
    confidence: float = 0.0
    summary: str = ""
    analysis_date: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            'symbol': self.symbol,
            'indicators': self.indicators.to_dict(),
            'signals': self.signals.to_dict(),
            'confidence': self.confidence,
            'summary': self.summary,
            'analysis_date': self.analysis_date.isoformat()
        }


@dataclass
class StrategyScore:
    """전략 점수 결과"""
    symbol: str
    strategy_name: str
    score: float
    confidence: float
    reasoning: List[str] = field(default_factory=list)
    key_factors: Dict[str, Any] = field(default_factory=dict)
    analysis_date: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """점수 범위 검증"""
        self.score = max(0.0, min(100.0, self.score))
        self.confidence = max(0.0, min(1.0, self.confidence))


@dataclass
class InvestmentRecommendation:
    """투자 추천 결과"""
    symbol: str
    action: str  # 매수, 매도, 보유
    confidence: float  # 0.0 ~ 1.0
    investment_period: InvestmentPeriod
    target_price: Optional[float] = None
    current_price: Optional[float] = None
    expected_return: Optional[float] = None
    risk_level: Optional[str] = None  # 낮음, 보통, 높음
    reasoning: str = ""
    ai_confidence: float = 0.0
    strategy_scores: List[float] = field(default_factory=list)
    technical_signals: Dict[str, Any] = field(default_factory=dict)
    
    # 투자 정보
    position_size_percent: float = 0.0
    
    # 분석 결과
    recommendation_reason: str = ""
    key_indicators: Dict[str, Any] = field(default_factory=dict)
    technical_analysis: Optional[TechnicalAnalysisResult] = None
    
    # 메타 정보
    analysis_date: datetime = field(default_factory=datetime.now)
    confidence_level: str = "보통"
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            'symbol': self.symbol,
            'action': self.action,
            'confidence': self.confidence,
            'target_price': self.target_price,
            'current_price': self.current_price,
            'expected_return': self.expected_return,
            'risk_level': self.risk_level,
            'reasoning': self.reasoning,
            'ai_confidence': self.ai_confidence,
            'strategy_scores': self.strategy_scores,
            'technical_signals': self.technical_signals,
            'investment_period': self.investment_period.value,
            'position_size_percent': self.position_size_percent,
            'recommendation_reason': self.recommendation_reason,
            'key_indicators': self.key_indicators,
            'analysis_date': self.analysis_date.isoformat(),
            'confidence_level': self.confidence_level
        }


@dataclass
class AnalysisResult:
    """분석 결과 표준 클래스"""
    symbol: str
    market: MarketType
    analysis_type: str
    score: float
    confidence: float
    recommendations: List[InvestmentRecommendation] = field(default_factory=list)
    technical_analysis: Optional[TechnicalAnalysisResult] = None
    strategy_scores: List[StrategyScore] = field(default_factory=list)
    summary: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    
    def get_best_recommendation(self) -> Optional[InvestmentRecommendation]:
        """최고 추천 반환"""
        if not self.recommendations:
            return None
        return max(self.recommendations, key=lambda x: x.confidence)
    
    def get_average_strategy_score(self) -> float:
        """평균 전략 점수 반환"""
        if not self.strategy_scores:
            return 0.0
        return sum(score.score for score in self.strategy_scores) / len(self.strategy_scores)


@dataclass
class InvestmentStrategy:
    """투자 전략 표준 클래스"""
    name: str
    strategy_type: StrategyType
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    enabled: bool = True
    risk_tolerance: RiskLevel = RiskLevel.MEDIUM
    
    def __post_init__(self):
        """가중치 검증"""
        self.weight = max(0.0, min(1.0, self.weight))
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            'name': self.name,
            'strategy_type': self.strategy_type.value,
            'description': self.description,
            'parameters': self.parameters,
            'weight': self.weight,
            'enabled': self.enabled,
            'risk_tolerance': self.risk_tolerance.value
        }


# 인터페이스 정의
class IDataCollector(ABC):
    """데이터 수집기 인터페이스"""
    
    @abstractmethod
    async def collect_market_data(self, market: MarketType) -> List[StockData]:
        """시장 데이터 수집"""
        pass
    
    @abstractmethod
    async def get_stock_data(self, symbol: str) -> Optional[StockData]:
        """개별 주식 데이터 수집"""
        pass
    
    @abstractmethod
    def get_market_symbols(self, market: MarketType) -> List[str]:
        """시장별 종목 심볼 리스트 반환"""
        pass


class ITechnicalAnalyzer(ABC):
    """기술적 분석기 인터페이스"""
    
    @abstractmethod
    def analyze(self, stock_data: StockData) -> TechnicalAnalysisResult:
        """기술적 분석 수행"""
        pass
    
    @abstractmethod
    def calculate_indicators(self, price_data: pd.DataFrame) -> TechnicalIndicators:
        """기술적 지표 계산"""
        pass
    
    @abstractmethod
    def generate_signals(self, indicators: TechnicalIndicators, 
                        current_price: float) -> TechnicalSignals:
        """기술적 신호 생성"""
        pass


class IInvestmentStrategy(ABC):
    """투자 전략 인터페이스"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """전략명"""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """전략 설명"""
        pass
    
    @abstractmethod
    def analyze(self, stock_data: StockData, 
                technical_result: TechnicalAnalysisResult) -> StrategyScore:
        """전략 분석 수행"""
        pass
    
    @abstractmethod
    def get_strategy_type(self) -> StrategyType:
        """전략 타입 반환"""
        pass


class IAIAnalyzer(ABC):
    """AI 분석기 인터페이스"""
    
    @abstractmethod
    async def analyze_recommendations(self, 
                                    stocks: List[StockData],
                                    strategy_scores: List[StrategyScore],
                                    technical_results: List[TechnicalAnalysisResult]) -> List[InvestmentRecommendation]:
        """AI 기반 종합 분석 및 추천"""
        pass
    
    @abstractmethod
    async def generate_market_insight(self, 
                                    market: MarketType,
                                    recommendations: List[InvestmentRecommendation]) -> Dict[str, Any]:
        """시장 인사이트 생성"""
        pass


class IReportGenerator(ABC):
    """리포트 생성기 인터페이스"""
    
    @abstractmethod
    async def generate_analysis_report(self, 
                                     recommendations: List[InvestmentRecommendation],
                                     market_insight: Dict[str, Any]) -> str:
        """분석 리포트 생성"""
        pass
    
    @abstractmethod
    async def save_report(self, report_content: str, 
                         format_type: str = "html") -> str:
        """리포트 저장"""
        pass


# 예외 클래스
class AnalysisError(Exception):
    """분석 관련 예외"""
    pass


class DataCollectionError(Exception):
    """데이터 수집 관련 예외"""
    pass


class StrategyError(Exception):
    """전략 관련 예외"""
    pass


class AIAnalysisError(Exception):
    """AI 분석 관련 예외"""
    pass


# 유틸리티 함수
def validate_stock_data(stock_data: StockData) -> bool:
    """주식 데이터 유효성 검증"""
    try:
        if not stock_data.symbol:
            return False
        if stock_data.current_price < 0:
            return False
        if stock_data.historical_data is not None and stock_data.historical_data.empty:
            return False
        return True
    except Exception:
        return False


def calculate_expected_return(current_price: float, target_price: float) -> float:
    """기대 수익률 계산"""
    if current_price <= 0:
        return 0.0
    return ((target_price - current_price) / current_price) * 100


def determine_position_size(risk_level: RiskLevel, confidence: float) -> float:
    """포지션 크기 결정"""
    base_sizes = {
        RiskLevel.LOW: 5.0,    # 낮음
        RiskLevel.MEDIUM: 10.0, # 보통
        RiskLevel.HIGH: 15.0    # 높음
    }
    
    base_size = base_sizes.get(risk_level, 5.0)
    confidence_multiplier = min(max(confidence, 0.5), 1.0)
    
    return base_size * confidence_multiplier 