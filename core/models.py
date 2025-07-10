#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: models.py
모듈: 공통 데이터 모델
목적: 시스템 전반에서 사용하는 데이터 구조 정의

Author: Trading Strategy System
Created: 2025-01-27
Modified: 2025-01-27
Version: 1.0.0

Dependencies:
    - Python 3.11+
    - pydantic==2.5.0
    - datetime
    - typing

Performance:
    - 모델 검증: < 1ms
    - 메모리사용량: 최적화

Security:
    - 입력 검증
    - 타입 안전성

License: MIT
"""

from __future__ import annotations

import enum
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Union

import pydantic
from pydantic import Field, validator


class TradeType(enum.Enum):
    """거래 타입"""
    BUY = "buy"
    SELL = "sell"


class OrderType(enum.Enum):
    """주문 타입"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class StrategyType(enum.Enum):
    """전략 타입"""
    NEWS_MOMENTUM = "news_momentum"
    TECHNICAL_PATTERN = "technical_pattern"
    THEME_ROTATION = "theme_rotation"
    RISK_MANAGEMENT = "risk_management"
    SHORT_TERM_OPTIMIZED = "short_term_optimized"
    COMBINED = "combined"


class SentimentType(enum.Enum):
    """감성 분석 타입"""
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class NewsCategory(enum.Enum):
    """뉴스 카테고리"""
    FINANCIAL = "financial"
    ECONOMIC = "economic"
    POLITICAL = "political"
    TECHNOLOGICAL = "technological"
    SOCIAL = "social"
    OTHER = "other"


class BaseModel(pydantic.BaseModel):
    """기본 모델 클래스"""

    class Config:
        validate_assignment = True
        str_strip_whitespace = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            Decimal: lambda v: float(v)
        }


class Stock(BaseModel):
    """주식 정보 모델"""

    code: str = Field(..., description="종목 코드")
    name: str = Field(..., description="종목명")
    market: str = Field(..., description="시장 구분 (KOSPI/KOSDAQ)")
    sector: Optional[str] = Field(None, description="섹터")
    market_cap: Optional[float] = Field(None, description="시가총액")
    price: Optional[float] = Field(None, description="현재가")
    volume: Optional[int] = Field(None, description="거래량")
    change_pct: Optional[float] = Field(None, description="등락률")

    @validator('code')
    def validate_code(cls, v: str) -> str:
        """종목 코드 검증"""
        if not v.isalnum():
            raise ValueError("종목 코드는 영숫자만 가능합니다")
        return v.upper()


class News(BaseModel):
    """뉴스 정보 모델"""

    id: str = Field(..., description="뉴스 ID")
    title: str = Field(..., description="뉴스 제목")
    content: str = Field(..., description="뉴스 내용")
    url: str = Field(..., description="뉴스 URL")
    source: str = Field(..., description="뉴스 출처")
    published_at: datetime = Field(..., description="발행 시간")
    category: NewsCategory = Field(..., description="뉴스 카테고리")
    sentiment: SentimentType = Field(..., description="감성 분석 결과")
    sentiment_score: float = Field(..., ge=-1.0, le=1.0, description="감성 점수")
    related_stocks: List[str] = Field(default_factory=list, description="관련 종목 코드")
    importance_score: float = Field(..., ge=0.0, le=1.0, description="중요도 점수")

    @validator('sentiment_score')
    def validate_sentiment_score(cls, v: float) -> float:
        """감성 점수 검증"""
        if not -1.0 <= v <= 1.0:
            raise ValueError("감성 점수는 -1.0에서 1.0 사이여야 합니다")
        return v


class Disclosure(BaseModel):
    """공시 정보 모델"""

    id: str = Field(..., description="공시 ID")
    corp_code: str = Field(..., description="기업 코드")
    corp_name: str = Field(..., description="기업명")
    stock_code: str = Field(..., description="종목 코드")
    title: str = Field(..., description="공시 제목")
    content: str = Field(..., description="공시 내용")
    disclosure_date: datetime = Field(..., description="공시 날짜")
    disclosure_type: str = Field(..., description="공시 유형")


class TechnicalIndicator(BaseModel):
    """기술적 지표 모델"""

    stock_code: str = Field(..., description="종목 코드")
    date: datetime = Field(..., description="날짜")

    # 가격 정보
    open_price: float = Field(..., description="시가")
    high_price: float = Field(..., description="고가")
    low_price: float = Field(..., description="저가")
    close_price: float = Field(..., description="종가")
    volume: int = Field(..., description="거래량")

    # 기술적 지표
    ma_5: Optional[float] = Field(None, description="5일 이동평균")
    ma_20: Optional[float] = Field(None, description="20일 이동평균")
    ma_60: Optional[float] = Field(None, description="60일 이동평균")
    rsi: Optional[float] = Field(None, description="RSI")
    macd: Optional[float] = Field(None, description="MACD")
    macd_signal: Optional[float] = Field(None, description="MACD 시그널")
    bollinger_upper: Optional[float] = Field(None, description="볼린저 상단")
    bollinger_lower: Optional[float] = Field(None, description="볼린저 하단")

    # 패턴 정보
    patterns: List[str] = Field(default_factory=list, description="차트 패턴")
    volume_ratio: Optional[float] = Field(None, description="거래량 비율")

    @validator('volume_ratio')
    def validate_volume_ratio(cls, v: Optional[float]) -> Optional[float]:
        """거래량 비율 검증"""
        if v is not None and v < 0:
            raise ValueError("거래량 비율은 0 이상이어야 합니다")
        return v


class Theme(BaseModel):
    """테마 정보 모델"""

    id: str = Field(..., description="테마 ID")
    name: str = Field(..., description="테마명")
    description: str = Field(..., description="테마 설명")
    category: str = Field(..., description="테마 카테고리")
    related_stocks: List[str] = Field(default_factory=list, description="관련 종목 코드")
    momentum_score: float = Field(..., ge=0.0, le=1.0, description="모멘텀 점수")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Signal(BaseModel):
    """매매 신호 모델"""

    id: str = Field(..., description="신호 ID")
    stock_code: str = Field(..., description="종목 코드")
    strategy_type: StrategyType = Field(..., description="전략 타입")
    signal_type: TradeType = Field(..., description="신호 타입")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="신뢰도 점수")
    target_price: Optional[float] = Field(None, description="목표가")
    stop_loss: Optional[float] = Field(None, description="손절가")
    take_profit: Optional[float] = Field(None, description="익절가")
    reasoning: str = Field(..., description="신호 근거")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @validator('confidence_score')
    def validate_confidence_score(cls, v: float) -> float:
        """신뢰도 점수 검증"""
        if not 0.0 <= v <= 1.0:
            raise ValueError("신뢰도 점수는 0.0에서 1.0 사이여야 합니다")
        return v


class Trade(BaseModel):
    """거래 정보 모델"""

    id: str = Field(..., description="거래 ID")
    stock_code: str = Field(..., description="종목 코드")
    trade_type: TradeType = Field(..., description="거래 타입")
    order_type: OrderType = Field(..., description="주문 타입")
    quantity: int = Field(..., gt=0, description="수량")
    price: float = Field(..., gt=0, description="가격")
    amount: float = Field(..., gt=0, description="거래 금액")
    commission: float = Field(default=0.0, ge=0, description="수수료")
    tax: float = Field(default=0.0, ge=0, description="세금")
    net_amount: float = Field(..., description="순 거래 금액")
    trade_date: datetime = Field(..., description="거래 날짜")
    signal_id: Optional[str] = Field(None, description="연관 신호 ID")
    strategy_type: StrategyType = Field(..., description="전략 타입")

    @validator('net_amount')
    def validate_net_amount(cls, v: float, values: Dict[str, Any]) -> float:
        """순 거래 금액 검증"""
        amount = values.get('amount', 0)
        commission = values.get('commission', 0)
        tax = values.get('tax', 0)
        trade_type = values.get('trade_type')

        # 매수 거래: 음수 (현금이 빠져나감)
        if trade_type == TradeType.BUY:
            expected_net = -(amount + commission + tax)
        # 매도 거래: 양수 (현금이 들어옴)
        else:
            expected_net = amount - commission - tax

        if abs(v - expected_net) > 0.01:  # 부동소수점 오차 허용
            raise ValueError(f"순 거래 금액이 일치하지 않습니다: {v} != {expected_net}")

        return v


class Portfolio(BaseModel):
    """포트폴리오 모델"""

    id: str = Field(..., description="포트폴리오 ID")
    name: str = Field(..., description="포트폴리오명")
    initial_capital: float = Field(..., gt=0, description="초기 자본금")
    current_capital: float = Field(..., ge=0, description="현재 자본금")
    total_return: float = Field(..., description="총 수익률")
    daily_return: float = Field(..., description="일간 수익률")
    max_drawdown: float = Field(..., description="최대 낙폭")
    sharpe_ratio: float = Field(..., description="샤프 비율")
    win_rate: float = Field(..., ge=0.0, le=1.0, description="승률")
    total_trades: int = Field(..., ge=0, description="총 거래 횟수")
    positions: List[Dict[str, Any]] = Field(default_factory=list, description="보유 포지션")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class BacktestResult(BaseModel):
    """백테스트 결과 모델"""

    strategy_name: str = Field(..., description="전략명")
    start_date: datetime = Field(..., description="시작 날짜")
    end_date: datetime = Field(..., description="종료 날짜")
    initial_capital: float = Field(..., gt=0, description="초기 자본금")
    final_capital: float = Field(..., ge=0, description="최종 자본금")
    total_return: float = Field(..., description="총 수익률")
    annual_return: float = Field(..., description="연간 수익률")
    max_drawdown: float = Field(..., description="최대 낙폭")
    sharpe_ratio: float = Field(..., description="샤프 비율")
    win_rate: float = Field(..., ge=0.0, le=1.0, description="승률")
    total_trades: int = Field(..., ge=0, description="총 거래 횟수")
    avg_trade_return: float = Field(..., description="평균 거래 수익률")
    profit_factor: float = Field(..., description="수익 팩터")
    max_consecutive_losses: int = Field(..., ge=0, description="최대 연속 손실")
    trades: List[Trade] = Field(default_factory=list, description="거래 내역")
    equity_curve: List[Dict[str, Any]] = Field(default_factory=list, description="자본 곡선")

    @validator('win_rate')
    def validate_win_rate(cls, v: float) -> float:
        """승률 검증"""
        if not 0.0 <= v <= 1.0:
            raise ValueError("승률은 0.0에서 1.0 사이여야 합니다")
        return v


class StrategyConfig(BaseModel):
    """전략 설정 모델"""

    strategy_type: StrategyType = Field(..., description="전략 타입")
    enabled: bool = Field(default=True, description="활성화 여부")
    weight: float = Field(..., ge=0.0, le=1.0, description="가중치")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="전략 파라미터")

    @validator('weight')
    def validate_weight(cls, v: float) -> float:
        """가중치 검증"""
        if not 0.0 <= v <= 1.0:
            raise ValueError("가중치는 0.0에서 1.0 사이여야 합니다")
        return v

