#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: short_term_optimized.py
모듈: 단기매매 최적화 전략 엔진
목적: 2-3회/일, 소형주/중형주, 테마주, 1-7일 보유 특화 전략

Author: Trading Strategy System
Created: 2025-01-27
Modified: 2025-01-27
Version: 1.0.0

Dependencies:
    - Python 3.11+
    - pandas==2.1.4
    - numpy==1.24.0
    - scikit-learn==1.3.2

Performance:
    - 신호 생성: < 2초 (100종목 기준)
    - 메모리사용량: < 100MB

Security:
    - 데이터 검증
    - 에러 처리
    - 로깅

License: MIT
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from core.config import config
from core.logger import get_logger, log_function_call
from core.models import Signal, StrategyType, TradeType, News, Theme

# Optional imports with graceful degradation
try:
    from .news_momentum import NewsMomentumStrategy
    from .risk import RiskManagementStrategy
    from .technical import TechnicalPatternStrategy
    from .theme_rotation import ThemeRotationStrategy
    STRATEGY_IMPORTS_AVAILABLE = True
except ImportError:
    STRATEGY_IMPORTS_AVAILABLE = False
    print("⚠️ 전략 모듈들을 사용할 수 없습니다.")

logger = get_logger(__name__)


class ShortTermOptimizedStrategy:
    """단기매매 최적화 전략 (2-3회/일, 소형주/중형주, 테마주, 1-7일 보유)"""

    def __init__(self):
        """초기화"""
        self.strategy_type = StrategyType.SHORT_TERM_OPTIMIZED
        self.name = "Short-Term Optimized Strategy"
        
        # 전략 컴포넌트 초기화
        if STRATEGY_IMPORTS_AVAILABLE:
            self.news_strategy = NewsMomentumStrategy()
            self.technical_strategy = TechnicalPatternStrategy()
            self.theme_strategy = ThemeRotationStrategy()
            self.risk_strategy = RiskManagementStrategy()
        else:
            self.news_strategy = None
            self.technical_strategy = None
            self.theme_strategy = None
            self.risk_strategy = None
        
        # 전략 파라미터
        self.max_daily_trades = 3
        self.min_holding_days = 1
        self.max_holding_days = 7
        self.min_market_cap = 100_000_000_000  # 1000억
        self.max_market_cap = 5_000_000_000_000  # 50조
        self.min_confidence_score = 0.6
        self.min_volume_ratio = 2.0
        self.min_price_change = 0.03
        self.theme_correlation_threshold = 0.7
        self.theme_min_stocks = 3
        self.theme_momentum_threshold = 0.5
        self.base_position_size = 0.05
        self.max_position_size = 0.15
        self.portfolio_diversification = 0.3
        
        # ML 모델
        self.signal_quality_model = None
        self.feature_scaler = StandardScaler()
        
        # 상태 관리
        self.daily_trades = 0
        self.daily_trades_reset_date = datetime.now(timezone.utc).date()
        self.daily_signals = []
        self.performance_metrics = {
            'total_signals': 0,
            'successful_signals': 0,
            'avg_holding_days': 0,
            'avg_return': 0.0,
            'win_rate': 0.0
        }

    @log_function_call
    async def generate_signals(self, news_list: List[News], themes: List[Theme], stock_data: Dict[str, Any], market_data: Optional[Dict[str, Any]] = None, target_stocks: Optional[List[str]] = None) -> List[Signal]:
        """단기매매 최적화 신호 생성"""
        try:
            self._reset_daily_trades_if_needed()
            
            if self.daily_trades >= self.max_daily_trades:
                logger.warning("일일 거래 한도 도달", extra={'daily_trades': self.daily_trades})
                return []
            
            # 각 전략별 신호 생성
            news_signals = await self._generate_news_signals(news_list, stock_data, target_stocks)
            technical_signals = await self._generate_technical_signals(stock_data, target_stocks)
            theme_signals = await self._generate_theme_signals(themes, stock_data, target_stocks)
            
            # 소형/중형주 필터링
            filtered_stocks = await self._filter_small_mid_caps(list(stock_data.keys()), market_data)
            
            # 신호 결합
            combined_signals = self._combine_signals(news_signals, technical_signals, theme_signals, filtered_stocks)
            
            # 테마 패턴 감지
            theme_enhanced_signals = await self._detect_theme_patterns(combined_signals, themes, stock_data)
            
            # 신호 품질 평가
            quality_scored_signals = self._evaluate_signal_quality(theme_enhanced_signals, stock_data, market_data)
            
            # 최종 필터 적용
            final_signals = self._apply_final_filters(quality_scored_signals)
            
            # 리스크 관리 적용
            if self.risk_strategy:
                risk_adjusted_signals = self.risk_strategy.apply_risk_management(final_signals, self._get_portfolio_state(), market_data)
            else:
                risk_adjusted_signals = final_signals
            
            # 거래 한도 적용
            limited_signals = self._apply_trading_limits(risk_adjusted_signals)
            
            # 상태 업데이트
            self.daily_trades += len(limited_signals)
            self.daily_signals.extend(limited_signals)
            
            logger.info("단기매매 최적화 신호 생성 완료", extra={
                'generated_signals': len(limited_signals),
                'daily_trades': self.daily_trades
            })
            
            return limited_signals
            
        except Exception as e:
            logger.error(f"신호 생성 실패: {e}", exc_info=True)
            return []

    def _reset_daily_trades_if_needed(self) -> None:
        """일일 거래 수 초기화"""
        current_date = datetime.now(timezone.utc).date()
        if current_date != self.daily_trades_reset_date:
            self.daily_trades = 0
            self.daily_trades_reset_date = current_date
            logger.info("일일 거래 수 초기화", extra={'date': current_date})

    async def _generate_news_signals(self, news_list: List[News], stock_data: Dict[str, Any], target_stocks: Optional[List[str]] = None) -> List[Signal]:
        """뉴스 기반 신호 생성"""
        if not self.news_strategy:
            return []
        
        try:
            return await self.news_strategy.generate_signals(news_list, stock_data, target_stocks)
        except Exception as e:
            logger.error(f"뉴스 신호 생성 실패: {e}")
            return []

    async def _generate_technical_signals(self, stock_data: Dict[str, Any], target_stocks: Optional[List[str]] = None) -> List[Signal]:
        """기술적 분석 신호 생성"""
        if not self.technical_strategy:
            return []
        
        try:
            return await self.technical_strategy.generate_signals(stock_data, target_stocks)
        except Exception as e:
            logger.error(f"기술적 신호 생성 실패: {e}")
            return []

    async def _generate_theme_signals(self, themes: List[Theme], stock_data: Dict[str, Any], target_stocks: Optional[List[str]] = None) -> List[Signal]:
        """테마 기반 신호 생성"""
        if not self.theme_strategy:
            return []
        
        try:
            return await self.theme_strategy.generate_signals(themes, stock_data, target_stocks)
        except Exception as e:
            logger.error(f"테마 신호 생성 실패: {e}")
            return []

    async def _filter_small_mid_caps(self, stock_codes: List[str], market_data: Optional[Dict[str, Any]] = None) -> List[str]:
        """소형/중형주 필터링"""
        try:
            filtered_stocks = []
            
            for stock_code in stock_codes:
                if market_data and stock_code in market_data:
                    market_cap = market_data[stock_code].get('market_cap', 0)
                    if self.min_market_cap <= market_cap <= self.max_market_cap:
                        filtered_stocks.append(stock_code)
                else:
                    # 시장 데이터가 없으면 기본적으로 포함
                    filtered_stocks.append(stock_code)
            
            logger.debug(f"소형/중형주 필터링 완료: {len(filtered_stocks)}/{len(stock_codes)}")
            return filtered_stocks
            
        except Exception as e:
            logger.error(f"소형/중형주 필터링 실패: {e}")
            return stock_codes

    def _combine_signals(self, news_signals: List[Signal], technical_signals: List[Signal], theme_signals: List[Signal], filtered_stocks: List[str]) -> List[Signal]:
        """신호 결합"""
        try:
            all_signals = news_signals + technical_signals + theme_signals
            
            # 필터링된 종목만 선택
            filtered_signals = [signal for signal in all_signals if signal.stock_code in filtered_stocks]
            
            # 중복 제거 (같은 종목의 가장 높은 신뢰도 신호 선택)
            signal_dict = {}
            for signal in filtered_signals:
                if signal.stock_code not in signal_dict or signal.confidence_score > signal_dict[signal.stock_code].confidence_score:
                    signal_dict[signal.stock_code] = signal
            
            combined_signals = list(signal_dict.values())
            
            logger.debug(f"신호 결합 완료: {len(combined_signals)}개")
            return combined_signals
            
        except Exception as e:
            logger.error(f"신호 결합 실패: {e}")
            return []

    async def _detect_theme_patterns(self, signals: List[Signal], themes: List[Theme], stock_data: Dict[str, Any]) -> List[Signal]:
        """테마 패턴 감지"""
        try:
            enhanced_signals = []
            
            for signal in signals:
                # 관련 테마 찾기
                related_themes = [theme for theme in themes if signal.stock_code in theme.related_stocks]
                
                if related_themes:
                    # 테마 모멘텀 계산
                    theme_momentum = sum(theme.momentum_score for theme in related_themes) / len(related_themes)
                    
                    # 신뢰도 조정
                    enhanced_confidence = signal.confidence_score * (1.0 + theme_momentum * 0.2)
                    enhanced_confidence = min(1.0, enhanced_confidence)
                    
                    # 신호 업데이트
                    enhanced_signal = Signal(
                        id=signal.id,
                        stock_code=signal.stock_code,
                        strategy_type=signal.strategy_type,
                        signal_type=signal.signal_type,
                        confidence_score=enhanced_confidence,
                        target_price=signal.target_price,
                        stop_loss=signal.stop_loss,
                        take_profit=signal.take_profit,
                        reasoning=f"{signal.reasoning} [테마모멘텀: {theme_momentum:.2f}]"
                    )
                    enhanced_signals.append(enhanced_signal)
                else:
                    enhanced_signals.append(signal)
            
            return enhanced_signals
            
        except Exception as e:
            logger.error(f"테마 패턴 감지 실패: {e}")
            return signals

    def _evaluate_signal_quality(self, signals: List[Signal], stock_data: Dict[str, Any], market_data: Optional[Dict[str, Any]] = None) -> List[Signal]:
        """신호 품질 평가"""
        try:
            quality_signals = []
            
            for signal in signals:
                quality_score = self._calculate_quality_score(signal, stock_data, market_data)
                
                if quality_score >= self.min_confidence_score:
                    # 품질 점수로 신뢰도 조정
                    adjusted_confidence = signal.confidence_score * quality_score
                    
                    quality_signal = Signal(
                        id=signal.id,
                        stock_code=signal.stock_code,
                        strategy_type=signal.strategy_type,
                        signal_type=signal.signal_type,
                        confidence_score=adjusted_confidence,
                        target_price=signal.target_price,
                        stop_loss=signal.stop_loss,
                        take_profit=signal.take_profit,
                        reasoning=f"{signal.reasoning} [품질점수: {quality_score:.2f}]"
                    )
                    quality_signals.append(quality_signal)
            
            return quality_signals
            
        except Exception as e:
            logger.error(f"신호 품질 평가 실패: {e}")
            return signals

    def _calculate_quality_score(self, signal: Signal, stock_data: Dict[str, Any], market_data: Optional[Dict[str, Any]] = None) -> float:
        """품질 점수 계산"""
        try:
            score = 1.0
            
            # 거래량 비율 확인
            if signal.stock_code in stock_data:
                volume_ratio = stock_data[signal.stock_code].get('volume_ratio', 1.0)
                if volume_ratio < self.min_volume_ratio:
                    score *= 0.8
            
            # 가격 변동 확인
            if signal.stock_code in stock_data:
                price_change = stock_data[signal.stock_code].get('price_change', 0.0)
                if abs(price_change) < self.min_price_change:
                    score *= 0.9
            
            return score
            
        except Exception as e:
            logger.error(f"품질 점수 계산 실패: {e}")
            return 0.5

    def _apply_final_filters(self, signals: List[Signal]) -> List[Signal]:
        """최종 필터 적용"""
        try:
            # 신뢰도 기준 필터링
            filtered_signals = [signal for signal in signals if signal.confidence_score >= self.min_confidence_score]
            
            # 신뢰도 순으로 정렬
            filtered_signals.sort(key=lambda x: x.confidence_score, reverse=True)
            
            return filtered_signals
            
        except Exception as e:
            logger.error(f"최종 필터 적용 실패: {e}")
            return signals

    def _apply_trading_limits(self, signals: List[Signal]) -> List[Signal]:
        """거래 한도 적용"""
        try:
            remaining_trades = self.max_daily_trades - self.daily_trades
            limited_signals = signals[:remaining_trades]
            
            return limited_signals
            
        except Exception as e:
            logger.error(f"거래 한도 적용 실패: {e}")
            return signals

    def _get_portfolio_state(self) -> Dict[str, Any]:
        """포트폴리오 상태 반환"""
        return {
            'current_capital': 10000000,  # 임시 값
            'positions': [],
            'max_drawdown': 0.0
        }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """성능 메트릭 반환"""
        return self.performance_metrics.copy()
