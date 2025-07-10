from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, getcontext
from enum import Enum, auto
from pydantic import Field, validator, model_validator
from typing import Any, Dict, List, Optional, Union, Literal, Final
from typing_extensions import Annotated
import pydantic
import uuid
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: models.py
모듈: 도메인 모델 정의
목적: 핵심 비즈니스 엔티티와 값 객체 정의

Author: Trading Strategy System
Created: 2025-01-27
Modified: 2025-01-27
Version: 2.0.0

Dependencies:
    - Python 3.11+
    - pydantic==2.5.0
    - typing-extensions==4.8.0
    - uuid
    - datetime
    - decimal

Architecture:
    - Domain-Driven Design (DDD)
    - Value Objects
    - Entities
    - Aggregates

License: MIT
"""




# 정밀도 설정 (금융 계산용)
getcontext().prec = 10

# 상수 정의
MAX_SIGNAL_STRENGTH: Final = 100.0
MIN_SIGNAL_STRENGTH: Final = 0.0
MAX_POSITION_SIZE: Final = 1.0
MIN_POSITION_SIZE: Final = 0.0
MAX_RISK_SCORE: Final = 100.0
MIN_RISK_SCORE: Final = 0.0


class SignalType(Enum):
    """신호 타입 열거형"""
    BUY = auto()
    SELL = auto()
    HOLD = auto()
    STRONG_BUY = auto()
    STRONG_SELL = auto()


class StrategyType(Enum):
    """전략 타입 열거형"""
    NEWS_MOMENTUM = auto()
    TECHNICAL_PATTERN = auto()
    THEME_ROTATION = auto()
    RISK_MANAGEMENT = auto()
    SENTIMENT_PSYCHOLOGY = auto()
    AGILE_SMALL_CAP = auto()
    HYBRID_AI = auto()


class TradeStatus(Enum):
    """거래 상태 열거형"""
    PENDING = auto()
    EXECUTED = auto()
    CANCELLED = auto()
    FAILED = auto()
    PARTIALLY_FILLED = auto()


class RiskLevel(Enum):
    """리스크 레벨 열거형"""
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()


class MarketSession(Enum):
    """시장 세션 열거형"""
    PRE_MARKET = auto()
    REGULAR = auto()
    AFTER_HOURS = auto()
    CLOSED = auto()


@dataclass(frozen=True)
class Money:
    """금액 값 객체 (불변)"""
    amount: Decimal
    currency: str = "KRW"

    def __post_init__(self) -> None:
        if not isinstance(self.amount, Decimal):
            object.__setattr__(self, 'amount', Decimal(str(self.amount)))

        if self.amount < 0:
            raise ValueError("금액은 0 이상이어야 합니다")

    def __add__(self, other: Money) -> Money:
        if self.currency != other.currency:
            raise ValueError("다른 통화는 더할 수 없습니다")
        return Money(self.amount + other.amount, self.currency)

    def __sub__(self, other: Money) -> Money:
        if self.currency != other.currency:
            raise ValueError("다른 통화는 뺄 수 없습니다")
        return Money(self.amount - other.amount, self.currency)

    def __mul__(self, multiplier: Union[int, float, Decimal]) -> Money:
        return Money(self.amount * Decimal(str(multiplier)), self.currency)

    def __truediv__(self, divisor: Union[int, float, Decimal]) -> Money:
        if divisor == 0:
            raise ValueError("0으로 나눌 수 없습니다")
        return Money(self.amount / Decimal(str(divisor)), self.currency)

    def __str__(self) -> str:
        return f"{self.amount:,.2f} {self.currency}"


@dataclass(frozen=True)
class Percentage:
    """퍼센트 값 객체 (불변)"""
    value: Decimal

    def __post_init__(self) -> None:
        if not isinstance(self.value, Decimal):
            object.__setattr__(self, 'value', Decimal(str(self.value)))

        if not (Decimal('0') <= self.value <= Decimal('100')):
            raise ValueError("퍼센트는 0-100 범위여야 합니다")

    def as_decimal(self) -> Decimal:
        """소수점으로 변환 (예: 5% -> 0.05)"""
        return self.value / Decimal('100')

    def __str__(self) -> str:
        return f"{self.value:.2f}%"


class Signal(pydantic.BaseModel):
    """거래 신호 엔티티"""

    # 기본 식별자
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="신호 고유 ID")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="신호 생성 시간")

    # 신호 정보
    symbol: str = Field(..., min_length=1, max_length=20, description="종목 코드")
    signal_type: SignalType = Field(..., description="신호 타입")
    strength: float = Field(..., ge=MIN_SIGNAL_STRENGTH, le=MAX_SIGNAL_STRENGTH, description="신호 강도 (0-100)")

    # 전략 정보
    strategy_type: StrategyType = Field(..., description="생성 전략")
    confidence: float = Field(..., ge=0.0, le=1.0, description="신뢰도 (0-1)")

    # 가격 정보
    current_price: Optional[float] = Field(None, ge=0, description="현재 가격")
    target_price: Optional[float] = Field(None, ge=0, description="목표 가격")
    stop_loss: Optional[float] = Field(None, ge=0, description="손절가")
    take_profit: Optional[float] = Field(None, ge=0, description="익절가")

    # 메타데이터
    metadata: Dict[str, Any] = Field(default_factory=dict, description="추가 메타데이터")
    tags: List[str] = Field(default_factory=list, description="태그 목록")

    class Config:
        validate_assignment = True
        str_strip_whitespace = True
        use_enum_values = False

    @validator('symbol')
    def validate_symbol(cls, v: str) -> str:
        """종목 코드 검증"""
        if not v.isalnum():
            raise ValueError("종목 코드는 영숫자만 허용됩니다")
        return v.upper()

    @model_validator(mode='after')
    def validate_price_logic(self) -> 'Signal':
        """가격 로직 검증"""
        if self.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
            if self.target_price and self.current_price and self.target_price <= self.current_price:
                raise ValueError("매수 신호의 목표가격은 현재가격보다 높아야 합니다")
        elif self.signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
            if self.target_price and self.current_price and self.target_price >= self.current_price:
                raise ValueError("매도 신호의 목표가격은 현재가격보다 낮아야 합니다")

        return self

    def is_valid(self) -> bool:
        """신호 유효성 검사"""
        return (
            self.strength >= MIN_SIGNAL_STRENGTH and
            self.strength <= MAX_SIGNAL_STRENGTH and
            self.confidence >= 0.0 and
            self.confidence <= 1.0 and
            len(self.symbol) > 0
        )

    def get_priority_score(self) -> float:
        """우선순위 점수 계산"""
        return self.strength * self.confidence


class Trade(pydantic.BaseModel):
    """거래 엔티티"""

    # 기본 식별자
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="거래 고유 ID")
    signal_id: str = Field(..., description="연관된 신호 ID")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="거래 시간")

    # 거래 정보
    symbol: str = Field(..., min_length=1, max_length=20, description="종목 코드")
    side: Literal["BUY", "SELL"] = Field(..., description="거래 방향")
    quantity: int = Field(..., gt=0, description="거래 수량")
    price: float = Field(..., gt=0, description="거래 가격")

    # 상태 정보
    status: TradeStatus = Field(default=TradeStatus.PENDING, description="거래 상태")
    executed_at: Optional[datetime] = Field(None, description="실행 시간")
    executed_price: Optional[float] = Field(None, gt=0, description="실행 가격")
    executed_quantity: Optional[int] = Field(None, ge=0, description="실행 수량")

    # 메타데이터
    commission: float = Field(default=0.0, ge=0, description="수수료")
    slippage: float = Field(default=0.0, ge=0, description="슬리피지")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="추가 메타데이터")

    class Config:
        validate_assignment = True
        str_strip_whitespace = True
        use_enum_values = False

    @property
    def total_amount(self) -> float:
        """총 거래 금액"""
        return self.quantity * self.price

    @property
    def executed_amount(self) -> float:
        """실행된 거래 금액"""
        if self.executed_price and self.executed_quantity:
            return self.executed_quantity * self.executed_price
        return 0.0

    @property
    def is_completed(self) -> bool:
        """거래 완료 여부"""
        return self.status in [TradeStatus.EXECUTED, TradeStatus.PARTIALLY_FILLED]

    @property
    def is_failed(self) -> bool:
        """거래 실패 여부"""
        return self.status in [TradeStatus.FAILED, TradeStatus.CANCELLED]


class Strategy(pydantic.BaseModel):
    """전략 엔티티"""

    # 기본 식별자
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="전략 고유 ID")
    name: str = Field(..., min_length=1, max_length=100, description="전략 이름")
    strategy_type: StrategyType = Field(..., description="전략 타입")

    # 설정 정보
    weight: float = Field(..., ge=0.0, le=1.0, description="전략 가중치")
    enabled: bool = Field(default=True, description="활성화 여부")

    # 성능 메트릭
    total_signals: int = Field(default=0, ge=0, description="총 신호 수")
    successful_signals: int = Field(default=0, ge=0, description="성공 신호 수")
    win_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="승률")

    # 메타데이터
    description: Optional[str] = Field(None, max_length=500, description="전략 설명")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="전략 파라미터")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="생성 시간")
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="수정 시간")

    class Config:
        validate_assignment = True
        str_strip_whitespace = True
        use_enum_values = False

    @property
    def success_rate(self) -> float:
        """성공률 계산"""
        if self.total_signals == 0:
            return 0.0
        return self.successful_signals / self.total_signals

    def update_performance(self, signal_success: bool) -> None:
        """성능 업데이트"""
        self.total_signals += 1
        if signal_success:
            self.successful_signals += 1
        self.win_rate = self.success_rate
        self.updated_at = datetime.now(timezone.utc)


class Portfolio(pydantic.BaseModel):
    """포트폴리오 엔티티"""

    # 기본 식별자
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="포트폴리오 고유 ID")
    name: str = Field(..., min_length=1, max_length=100, description="포트폴리오 이름")

    # 자산 정보
    initial_capital: float = Field(..., gt=0, description="초기 자본")
    current_capital: float = Field(..., ge=0, description="현재 자본")
    cash: float = Field(..., ge=0, description="현금")

    # 포지션 정보
    positions: Dict[str, int] = Field(default_factory=dict, description="보유 종목별 수량")
    position_values: Dict[str, float] = Field(default_factory=dict, description="보유 종목별 평가금액")

    # 성과 메트릭
    total_return: float = Field(default=0.0, description="총 수익률")
    daily_return: float = Field(default=0.0, description="일간 수익률")
    max_drawdown: float = Field(default=0.0, description="최대 낙폭")

    # 메타데이터
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="생성 시간")
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="수정 시간")

    class Config:
        validate_assignment = True
        str_strip_whitespace = True

    @property
    def total_value(self) -> float:
        """총 포트폴리오 가치"""
        return self.cash + sum(self.position_values.values())

    @property
    def total_return_pct(self) -> float:
        """총 수익률 (퍼센트)"""
        if self.initial_capital == 0:
            return 0.0
        return ((self.total_value - self.initial_capital) / self.initial_capital) * 100

    def update_position(self, symbol: str, quantity: int, price: float) -> None:
        """포지션 업데이트"""
        current_quantity = self.positions.get(symbol, 0)
        new_quantity = current_quantity + quantity

        if new_quantity == 0:
            self.positions.pop(symbol, None)
            self.position_values.pop(symbol, None)
        else:
            self.positions[symbol] = new_quantity
            self.position_values[symbol] = new_quantity * price

        self.updated_at = datetime.now(timezone.utc)


class RiskMetrics(pydantic.BaseModel):
    """리스크 메트릭 값 객체"""

    # 변동성 메트릭
    volatility: float = Field(..., ge=0, description="변동성")
    beta: float = Field(default=1.0, description="베타")
    sharpe_ratio: float = Field(default=0.0, description="샤프 비율")

    # 위험 메트릭
    var_95: float = Field(..., ge=0, description="VaR (95%)")
    cvar_95: float = Field(..., ge=0, description="CVaR (95%)")
    max_drawdown: float = Field(..., ge=0, le=1, description="최대 낙폭")

    # 포지션 메트릭
    concentration_risk: float = Field(..., ge=0, le=1, description="집중도 위험")
    sector_exposure: Dict[str, float] = Field(default_factory=dict, description="섹터별 노출도")

    # 계산 시간
    calculated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="계산 시간")

    class Config:
        validate_assignment = True

    @property
    def risk_score(self) -> float:
        """종합 리스크 점수 (0-100)"""
        # 가중 평균으로 리스크 점수 계산
        weights = {
            'volatility': 0.2,
            'var_95': 0.25,
            'max_drawdown': 0.25,
            'concentration_risk': 0.3
        }

        score = (
            min(self.volatility * 100, 100) * weights['volatility'] +
            min(self.var_95 * 100, 100) * weights['var_95'] +
            min(self.max_drawdown * 100, 100) * weights['max_drawdown'] +
            min(self.concentration_risk * 100, 100) * weights['concentration_risk']
        )

        return min(score, 100.0)


class MarketData(pydantic.BaseModel):
    """시장 데이터 값 객체"""

    # 기본 정보
    symbol: str = Field(..., min_length=1, max_length=20, description="종목 코드")
    timestamp: datetime = Field(..., description="데이터 시간")

    # 가격 데이터
    open: float = Field(..., gt=0, description="시가")
    high: float = Field(..., gt=0, description="고가")
    low: float = Field(..., gt=0, description="저가")
    close: float = Field(..., gt=0, description="종가")
    volume: int = Field(..., ge=0, description="거래량")

    # 추가 데이터
    vwap: Optional[float] = Field(None, gt=0, description="거래량가중평균가")
    market_cap: Optional[float] = Field(None, gt=0, description="시가총액")

    class Config:
        validate_assignment = True
        str_strip_whitespace = True

    @validator('high')
    def validate_high(cls, v: float, values: Dict[str, Any]) -> float:
        """고가 검증"""
        if 'low' in values and v < values['low']:
            raise ValueError("고가는 저가보다 높아야 합니다")
        return v

    @property
    def price_change(self) -> float:
        """가격 변화"""
        return self.close - self.open

    @property
    def price_change_pct(self) -> float:
        """가격 변화율 (%)"""
        if self.open == 0:
            return 0.0
        return (self.price_change / self.open) * 100

    @property
    def is_green(self) -> bool:
        """상승 여부"""
        return self.close > self.open

    @property
    def is_red(self) -> bool:
        """하락 여부"""
        return self.close < self.open


class NewsEvent(pydantic.BaseModel):
    """뉴스 이벤트 값 객체"""

    # 기본 정보
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="뉴스 고유 ID")
    title: str = Field(..., min_length=1, max_length=500, description="뉴스 제목")
    content: str = Field(..., min_length=1, description="뉴스 내용")
    source: str = Field(..., min_length=1, max_length=100, description="뉴스 출처")

    # 시간 정보
    published_at: datetime = Field(..., description="발행 시간")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="수집 시간")

    # 감정 분석
    sentiment_score: float = Field(..., ge=-1.0, le=1.0, description="감정 점수 (-1: 부정, 1: 긍정)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="감정 분석 신뢰도")

    # 관련 종목
    related_symbols: List[str] = Field(default_factory=list, description="관련 종목 코드")
    impact_score: float = Field(default=0.0, ge=0.0, le=10.0, description="시장 영향도 점수")

    # 메타데이터
    url: Optional[str] = Field(None, description="뉴스 URL")
    tags: List[str] = Field(default_factory=list, description="태그 목록")

    class Config:
        validate_assignment = True
        str_strip_whitespace = True

    @property
    def is_positive(self) -> bool:
        """긍정적 뉴스 여부"""
        return self.sentiment_score > 0.1

    @property
    def is_negative(self) -> bool:
        """부정적 뉴스 여부"""
        return self.sentiment_score < -0.1

    @property
    def is_neutral(self) -> bool:
        """중립적 뉴스 여부"""
        return -0.1 <= self.sentiment_score <= 0.1


class TechnicalIndicator(pydantic.BaseModel):
    """기술적 지표 값 객체"""

    # 기본 정보
    name: str = Field(..., min_length=1, max_length=50, description="지표 이름")
    symbol: str = Field(..., min_length=1, max_length=20, description="종목 코드")
    timestamp: datetime = Field(..., description="계산 시간")

    # 지표 값
    value: float = Field(..., description="지표 값")
    signal: Literal["BUY", "SELL", "NEUTRAL"] = Field(..., description="신호")
    strength: float = Field(..., ge=0.0, le=1.0, description="신호 강도")

    # 메타데이터
    parameters: Dict[str, Any] = Field(default_factory=dict, description="계산 파라미터")

    class Config:
        validate_assignment = True
        str_strip_whitespace = True
        use_enum_values = False

