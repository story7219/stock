#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: risk.py
모듈: 리스크 관리 전략 엔진
목적: 포지션 크기 제한 + 손절/익절 자동화 + 일일 거래 제한

Author: Trading Strategy System
Created: 2025-01-27
Modified: 2025-01-27
Version: 1.0.0

Dependencies:
    - Python 3.11+
    - pandas==2.1.4
    - numpy==1.24.0

Performance:
    - 리스크 계산: < 100ms
    - 메모리사용량: < 10MB

Security:
    - 데이터 검증
    - 에러 처리
    - 로깅

License: MIT
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from core.config import config
from core.logger import get_logger, log_function_call
from core.models import Signal, StrategyType, TradeType

logger = get_logger(__name__)


class RiskManagementStrategy:
    """리스크 관리 전략 (단기매매 특화)"""

    def __init__(self):
        """초기화"""
        self.strategy_type = StrategyType.RISK_MANAGEMENT
        self.weight = config.trading.risk_management_weight if config.trading else 0.1
        self.max_position_size = 0.1
        self.max_daily_trades = 3
        self.stop_loss_pct = 0.05
        self.take_profit_pct = 0.15
        self.max_drawdown_limit = 0.20
        self.max_holding_days = 7
        self.volatility_multiplier = 1.0
        self.correlation_threshold = 0.7
        self.daily_trades = 0
        self.daily_trades_reset_date = datetime.now(timezone.utc).date()

    @log_function_call
    def apply_risk_management(self, signals: List[Signal], portfolio: Dict[str, Any], market_data: Optional[Dict[str, Any]] = None) -> List[Signal]:
        """리스크 관리 적용"""
        logger.info("리스크 관리 적용 시작", extra={
            'original_signals': len(signals),
            'portfolio_value': portfolio.get('current_capital', 0)
        })
        
        self._reset_daily_trades_if_needed()
        
        if self.daily_trades >= self.max_daily_trades:
            logger.warning("일일 거래 한도 도달", extra={'daily_trades': self.daily_trades})
            return []
        
        risk_adjusted_signals = []
        for signal in signals:
            try:
                adjusted_signal = self._adjust_signal_risk(signal, portfolio, market_data)
                if adjusted_signal:
                    risk_adjusted_signals.append(adjusted_signal)
            except Exception as e:
                logger.error(f"신호 리스크 조정 실패: {signal.id}", extra={'error': str(e)})
                continue
        
        final_signals = self._validate_portfolio_risk(risk_adjusted_signals, portfolio)
        self.daily_trades += len(final_signals)
        
        logger.info("리스크 관리 적용 완료", extra={
            'final_signals': len(final_signals),
            'daily_trades': self.daily_trades
        })
        return final_signals

    def _reset_daily_trades_if_needed(self) -> None:
        """일일 거래 수 초기화 (새로운 날짜인 경우)"""
        current_date = datetime.now(timezone.utc).date()
        if current_date != self.daily_trades_reset_date:
            self.daily_trades = 0
            self.daily_trades_reset_date = current_date
            logger.info("일일 거래 수 초기화", extra={'date': current_date})

    def _adjust_signal_risk(self, signal: Signal, portfolio: Dict[str, Any], market_data: Optional[Dict[str, Any]] = None) -> Optional[Signal]:
        """개별 신호 리스크 조정"""
        position_size = self._calculate_position_size(signal, portfolio, market_data)
        if position_size <= 0:
            logger.debug(f"포지션 크기 0: {signal.stock_code}")
            return None
        
        stop_loss, take_profit = self._calculate_risk_levels(signal, market_data)
        adjusted_confidence = self._adjust_confidence_for_risk(signal, portfolio, market_data)
        
        if adjusted_confidence < 0.3:
            logger.debug(f"신뢰도 부족: {signal.stock_code} ({adjusted_confidence:.2f})")
            return None
        
        return Signal(
            id=signal.id,
            stock_code=signal.stock_code,
            strategy_type=signal.strategy_type,
            signal_type=signal.signal_type,
            confidence_score=adjusted_confidence,
            target_price=signal.target_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reasoning=f"{signal.reasoning} [리스크조정: 포지션{position_size:.1%}]"
        )

    def _calculate_position_size(self, signal: Signal, portfolio: Dict[str, Any], market_data: Optional[Dict[str, Any]] = None) -> float:
        """포지션 크기 계산"""
        try:
            current_capital = portfolio.get('current_capital', 0)
            if current_capital <= 0:
                return 0.0
            
            # 기본 포지션 크기
            base_size = self.max_position_size
            
            # 변동성 조정
            if market_data and signal.stock_code in market_data:
                volatility = market_data[signal.stock_code].get('volatility', 1.0)
                base_size *= (1.0 / volatility) * self.volatility_multiplier
            
            # 신뢰도 조정
            confidence_multiplier = signal.confidence_score
            adjusted_size = base_size * confidence_multiplier
            
            # 최대 포지션 크기 제한
            return min(adjusted_size, self.max_position_size)
            
        except Exception as e:
            logger.error(f"포지션 크기 계산 실패: {e}")
            return 0.0

    def _calculate_risk_levels(self, signal: Signal, market_data: Optional[Dict[str, Any]] = None) -> Tuple[float, float]:
        """손절/익절 레벨 계산"""
        try:
            current_price = signal.target_price or 0.0
            if current_price <= 0:
                return 0.0, 0.0
            
            stop_loss = current_price * (1.0 - self.stop_loss_pct)
            take_profit = current_price * (1.0 + self.take_profit_pct)
            
            return stop_loss, take_profit
            
        except Exception as e:
            logger.error(f"리스크 레벨 계산 실패: {e}")
            return 0.0, 0.0

    def _adjust_confidence_for_risk(self, signal: Signal, portfolio: Dict[str, Any], market_data: Optional[Dict[str, Any]] = None) -> float:
        """리스크를 고려한 신뢰도 조정"""
        try:
            base_confidence = signal.confidence_score
            
            # 포트폴리오 분산도 조정
            diversification_penalty = self._calculate_diversification_penalty(signal, portfolio)
            
            # 변동성 조정
            volatility_penalty = self._calculate_volatility_penalty(signal, market_data)
            
            # 최종 신뢰도 계산
            adjusted_confidence = base_confidence * (1.0 - diversification_penalty) * (1.0 - volatility_penalty)
            
            return max(0.0, min(1.0, adjusted_confidence))
            
        except Exception as e:
            logger.error(f"신뢰도 조정 실패: {e}")
            return signal.confidence_score

    def _calculate_diversification_penalty(self, signal: Signal, portfolio: Dict[str, Any]) -> float:
        """분산도 페널티 계산"""
        try:
            positions = portfolio.get('positions', [])
            if not positions:
                return 0.0
            
            # 같은 섹터/테마의 포지션 수 확인
            same_sector_count = sum(1 for pos in positions if pos.get('sector') == signal.stock_code[:2])
            
            # 페널티 계산 (같은 섹터가 많을수록 페널티 증가)
            penalty = min(0.3, same_sector_count * 0.1)
            
            return penalty
            
        except Exception as e:
            logger.error(f"분산도 페널티 계산 실패: {e}")
            return 0.0

    def _calculate_volatility_penalty(self, signal: Signal, market_data: Optional[Dict[str, Any]] = None) -> float:
        """변동성 페널티 계산"""
        try:
            if not market_data or signal.stock_code not in market_data:
                return 0.0
            
            volatility = market_data[signal.stock_code].get('volatility', 1.0)
            
            # 변동성이 높을수록 페널티 증가
            penalty = min(0.2, (volatility - 1.0) * 0.1)
            
            return max(0.0, penalty)
            
        except Exception as e:
            logger.error(f"변동성 페널티 계산 실패: {e}")
            return 0.0

    def _validate_portfolio_risk(self, signals: List[Signal], portfolio: Dict[str, Any]) -> List[Signal]:
        """포트폴리오 리스크 검증"""
        try:
            if not signals:
                return []
            
            # 최대 낙폭 확인
            current_drawdown = portfolio.get('max_drawdown', 0.0)
            if current_drawdown >= self.max_drawdown_limit:
                logger.warning("최대 낙폭 한도 도달", extra={'current_drawdown': current_drawdown})
                return []
            
            # 포지션 수 제한
            current_positions = len(portfolio.get('positions', []))
            max_positions = 10  # 최대 10개 포지션
            
            if current_positions >= max_positions:
                logger.warning("최대 포지션 수 도달", extra={'current_positions': current_positions})
                return []
            
            return signals
            
        except Exception as e:
            logger.error(f"포트폴리오 리스크 검증 실패: {e}")
            return []

    def get_risk_metrics(self) -> Dict[str, Any]:
        """리스크 메트릭 반환"""
        return {
            'daily_trades': self.daily_trades,
            'max_daily_trades': self.max_daily_trades,
            'max_position_size': self.max_position_size,
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct,
            'max_drawdown_limit': self.max_drawdown_limit,
            'max_holding_days': self.max_holding_days
        }
