from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal, getcontext
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Set
import asyncio
import json
import logging
import numpy as np
import pandas as pd
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: real_time_risk_manager.py
모듈: 실시간 리스크 관리 시스템
목적: 단기매매 특성에 맞는 실시간 리스크 모니터링 및 자동 방어

Author: AI Assistant
Created: 2025-01-27
Modified: 2025-01-27
Version: 1.0.0

Dependencies:
    - Python 3.11+
    - asyncio, typing, dataclasses
    - numpy, pandas (데이터 분석)
    - aiohttp (API 통신)

Performance:
    - 리스크 체크: < 50ms
    - 실시간 모니터링: 1초 간격
    - 자동 방어: < 100ms

Security:
    - 다중 리스크 한도 체크
    - 자동 손절 및 포지션 축소
    - 시장 충격 이벤트 대응
"""




# 정밀도 설정 (금융 계산용)
getcontext().prec = 10

# 로깅 설정
logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """리스크 레벨"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class MarketEvent(Enum):
    """시장 이벤트"""
    NORMAL = "normal"
    VOLATILE = "volatile"
    CRASH = "crash"
    RECOVERY = "recovery"

@dataclass
class PositionRisk:
    """포지션별 리스크 정보"""
    symbol: str
    quantity: int
    avg_price: float
    current_price: float
    pnl: float
    pnl_pct: float
    market_value: float
    portfolio_pct: float
    volatility: float
    beta: float
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class PortfolioRisk:
    """포트폴리오 리스크 정보"""
    total_value: float
    total_pnl: float
    total_pnl_pct: float
    daily_pnl: float
    daily_pnl_pct: float
    weekly_pnl: float
    weekly_pnl_pct: float
    monthly_pnl: float
    monthly_pnl_pct: float
    max_drawdown: float
    sharpe_ratio: float
    correlation_matrix: pd.DataFrame
    vix_level: float
    market_regime: MarketEvent
    risk_level: RiskLevel
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class RiskLimits:
    """리스크 한도 설정"""
    daily_loss_limit: float = -0.03  # -3%
    weekly_loss_limit: float = -0.07  # -7%
    monthly_loss_limit: float = -0.15  # -15%
    individual_position_limit: float = 0.10  # 10%
    max_positions: int = 10
    max_correlation: float = 0.7
    min_liquidity: float = 1000000
    vix_threshold: float = 30.0

@dataclass
class PerformanceMetrics:
    """성과 지표"""
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    max_consecutive_losses: int
    strategy_contribution: Dict[str, float]
    market_performance: Dict[str, float]
    improvement_suggestions: List[str]

class RealTimeRiskManager:
    """실시간 리스크 관리 시스템"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.risk_limits = RiskLimits(**config.get('risk_limits', {}))

        # 상태 관리
        self.positions: Dict[str, PositionRisk] = {}
        self.portfolio_risk: Optional[PortfolioRisk] = None
        self.performance_history: List[Dict] = []
        self.trading_allowed: bool = True
        self.defense_mode: bool = False

        # 모니터링 설정
        self.monitoring_interval = config.get('monitoring_interval', 1.0)  # 1초
        self.correlation_window = config.get('correlation_window', 30)  # 30일
        self.volatility_window = config.get('volatility_window', 20)  # 20일

        # 성과 추적
        self.daily_trades: List[Dict] = []
        self.weekly_trades: List[Dict] = []
        self.monthly_trades: List[Dict] = []

    async def start_monitoring(self):
        """실시간 모니터링 시작"""
        logger.info("Starting real-time risk monitoring...")

        while True:
            try:
                await self._monitor_risk()
                await asyncio.sleep(self.monitoring_interval)
            except Exception as e:
                logger.error(f"Risk monitoring error: {e}")
                await asyncio.sleep(5.0)  # 에러 시 5초 대기

    async def _monitor_risk(self):
        """리스크 모니터링"""

        # 1. 포지션별 리스크 업데이트
        await self._update_position_risks()

        # 2. 포트폴리오 리스크 계산
        await self._calculate_portfolio_risk()

        # 3. 리스크 한도 체크
        await self._check_risk_limits()

        # 4. 상관관계 모니터링
        await self._monitor_correlations()

        # 5. 유동성 리스크 체크
        await self._check_liquidity_risk()

        # 6. 시장 충격 이벤트 감지
        await self._detect_market_shocks()

        # 7. 자동 방어 실행
        await self._execute_defense_actions()

    async def _update_position_risks(self):
        """포지션별 리스크 업데이트"""

        for symbol, position in self.positions.items():
            try:
                # 실시간 가격 업데이트 (실제로는 API 호출)
                current_price = await self._get_current_price(symbol)

                # 손익 계산
                pnl = (current_price - position.avg_price) * position.quantity
                pnl_pct = (current_price - position.avg_price) / position.avg_price
                market_value = current_price * position.quantity

                # 포트폴리오 비중 계산
                total_value = sum(p.market_value for p in self.positions.values())
                portfolio_pct = market_value / total_value if total_value > 0 else 0

                # 변동성 계산 (실제로는 히스토리 데이터 필요)
                volatility = await self._calculate_volatility(symbol)

                # 베타 계산 (실제로는 시장 대비 계산)
                beta = await self._calculate_beta(symbol)

                # 포지션 리스크 업데이트
                self.positions[symbol] = PositionRisk(
                    symbol=symbol,
                    quantity=position.quantity,
                    avg_price=position.avg_price,
                    current_price=current_price,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    market_value=market_value,
                    portfolio_pct=portfolio_pct,
                    volatility=volatility,
                    beta=beta
                )

            except Exception as e:
                logger.error(f"Failed to update position risk for {symbol}: {e}")

    async def _calculate_portfolio_risk(self):
        """포트폴리오 리스크 계산"""

        if not self.positions:
            return

        # 기본 지표 계산
        total_value = sum(p.market_value for p in self.positions.values())
        total_pnl = sum(p.pnl for p in self.positions.values())
        total_pnl_pct = total_pnl / total_value if total_value > 0 else 0

        # 기간별 손익 계산
        daily_pnl = await self._calculate_period_pnl('daily')
        weekly_pnl = await self._calculate_period_pnl('weekly')
        monthly_pnl = await self._calculate_period_pnl('monthly')

        daily_pnl_pct = daily_pnl / total_value if total_value > 0 else 0
        weekly_pnl_pct = weekly_pnl / total_value if total_value > 0 else 0
        monthly_pnl_pct = monthly_pnl / total_value if total_value > 0 else 0

        # 최대 낙폭 계산
        max_drawdown = await self._calculate_max_drawdown()

        # 샤프 비율 계산
        sharpe_ratio = await self._calculate_sharpe_ratio()

        # 상관관계 행렬 계산
        correlation_matrix = await self._calculate_correlation_matrix()

        # VIX 레벨 (실제로는 API 호출)
        vix_level = await self._get_vix_level()

        # 시장 상황 판단
        market_regime = await self._determine_market_regime()

        # 리스크 레벨 결정
        risk_level = self._determine_risk_level(
            total_pnl_pct, daily_pnl_pct, weekly_pnl_pct, monthly_pnl_pct, vix_level
        )

        # 포트폴리오 리스크 업데이트
        self.portfolio_risk = PortfolioRisk(
            total_value=total_value,
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            daily_pnl=daily_pnl,
            daily_pnl_pct=daily_pnl_pct,
            weekly_pnl=weekly_pnl,
            weekly_pnl_pct=weekly_pnl_pct,
            monthly_pnl=monthly_pnl,
            monthly_pnl_pct=monthly_pnl_pct,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            correlation_matrix=correlation_matrix,
            vix_level=vix_level,
            market_regime=market_regime,
            risk_level=risk_level
        )

    async def _check_risk_limits(self):
        """리스크 한도 체크"""

        if not self.portfolio_risk:
            return

        violations = []

        # 1. 일일 손실 한도 체크
        if self.portfolio_risk.daily_pnl_pct < self.risk_limits.daily_loss_limit:
            violations.append({
                'type': 'daily_loss_limit',
                'current': self.portfolio_risk.daily_pnl_pct,
                'limit': self.risk_limits.daily_loss_limit,
                'action': 'reduce_positions'
            })

        # 2. 주간 손실 한도 체크
        if self.portfolio_risk.weekly_pnl_pct < self.risk_limits.weekly_loss_limit:
            violations.append({
                'type': 'weekly_loss_limit',
                'current': self.portfolio_risk.weekly_pnl_pct,
                'limit': self.risk_limits.weekly_loss_limit,
                'action': 'stop_trading'
            })

        # 3. 월간 손실 한도 체크
        if self.portfolio_risk.monthly_pnl_pct < self.risk_limits.monthly_loss_limit:
            violations.append({
                'type': 'monthly_loss_limit',
                'current': self.portfolio_risk.monthly_pnl_pct,
                'limit': self.risk_limits.monthly_loss_limit,
                'action': 'emergency_exit'
            })

        # 4. 개별 종목 한도 체크
        for symbol, position in self.positions.items():
            if position.portfolio_pct > self.risk_limits.individual_position_limit:
                violations.append({
                    'type': 'individual_position_limit',
                    'symbol': symbol,
                    'current': position.portfolio_pct,
                    'limit': self.risk_limits.individual_position_limit,
                    'action': 'reduce_position'
                })

        # 5. 포지션 수 한도 체크
        if len(self.positions) > self.risk_limits.max_positions:
            violations.append({
                'type': 'max_positions',
                'current': len(self.positions),
                'limit': self.risk_limits.max_positions,
                'action': 'close_oldest_position'
            })

        # 위반 사항 처리
        if violations:
            await self._handle_risk_violations(violations)

    async def _monitor_correlations(self):
        """상관관계 모니터링"""

        if len(self.positions) < 2:
            return

        # 상관관계 계산
        correlation_matrix = self.portfolio_risk.correlation_matrix

        # 높은 상관관계 체크
        high_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr = correlation_matrix.iloc[i, j]
                if abs(corr) > self.risk_limits.max_correlation:
                    high_correlations.append({
                        'symbol1': correlation_matrix.columns[i],
                        'symbol2': correlation_matrix.columns[j],
                        'correlation': corr
                    })

        # 높은 상관관계 처리
        if high_correlations:
            await self._handle_high_correlations(high_correlations)

    async def _check_liquidity_risk(self):
        """유동성 리스크 체크"""

        for symbol, position in self.positions.items():
            try:
                # 유동성 체크 (실제로는 거래량, 호가창 분석)
                liquidity = await self._check_symbol_liquidity(symbol)

                if liquidity < self.risk_limits.min_liquidity:
                    logger.warning(f"Low liquidity for {symbol}: {liquidity}")
                    await self._handle_liquidity_risk(symbol, liquidity)

            except Exception as e:
                logger.error(f"Liquidity check failed for {symbol}: {e}")

    async def _detect_market_shocks(self):
        """시장 충격 이벤트 감지"""

        # VIX 급등 체크
        if self.portfolio_risk and self.portfolio_risk.vix_level > self.risk_limits.vix_threshold:
            await self._handle_vix_spike()

        # 급락 체크
        if self.portfolio_risk and self.portfolio_risk.daily_pnl_pct < -0.05:  # -5%
            await self._handle_market_crash()

        # 연속 손실 체크
        consecutive_losses = await self._count_consecutive_losses()
        if consecutive_losses > 5:  # 5회 연속 손실
            await self._handle_consecutive_losses(consecutive_losses)

    async def _execute_defense_actions(self):
        """자동 방어 실행"""

        if not self.defense_mode:
            return

        # 1. 전체 포지션 축소
        if self.portfolio_risk and self.portfolio_risk.risk_level == RiskLevel.CRITICAL:
            await self._reduce_all_positions(0.5)  # 50% 축소

        # 2. 거래 중단
        if self.portfolio_risk and self.portfolio_risk.vix_level > 40:
            self.trading_allowed = False
            logger.warning("Trading suspended due to high VIX")

        # 3. 거래량 감소
        if await self._count_consecutive_losses() > 3:
            await self._reduce_trading_volume(0.5)  # 50% 감소

        # 4. 현금 전환
        if self.portfolio_risk and self.portfolio_risk.market_regime == MarketEvent.CRASH:
            await self._convert_to_cash()

    async def _handle_risk_violations(self, violations: List[Dict]):
        """리스크 위반 처리"""

        for violation in violations:
            logger.warning(f"Risk violation: {violation}")

            if violation['action'] == 'reduce_positions':
                await self._reduce_all_positions(0.3)  # 30% 축소
            elif violation['action'] == 'stop_trading':
                self.trading_allowed = False
                logger.warning("Trading stopped due to risk violation")
            elif violation['action'] == 'emergency_exit':
                await self._emergency_exit_all_positions()
            elif violation['action'] == 'reduce_position':
                await self._reduce_position(violation['symbol'], 0.5)
            elif violation['action'] == 'close_oldest_position':
                await self._close_oldest_position()

    async def _handle_high_correlations(self, high_correlations: List[Dict]):
        """높은 상관관계 처리"""

        for corr in high_correlations:
            logger.warning(f"High correlation: {corr['symbol1']} - {corr['symbol2']}: {corr['correlation']:.3f}")

            # 상관관계가 높은 종목 중 하나 축소
            await self._reduce_position(corr['symbol1'], 0.3)

    async def _handle_liquidity_risk(self, symbol: str, liquidity: float):
        """유동성 리스크 처리"""

        logger.warning(f"Liquidity risk for {symbol}: {liquidity}")

        # 유동성이 낮은 종목 축소
        await self._reduce_position(symbol, 0.5)

    async def _handle_vix_spike(self):
        """VIX 급등 처리"""

        logger.warning(f"VIX spike detected: {self.portfolio_risk.vix_level}")

        # 포지션 축소
        await self._reduce_all_positions(0.4)  # 40% 축소

        # 헤지 포지션 추가 (실제로는 선물/옵션)
        await self._add_hedge_positions()

    async def _handle_market_crash(self):
        """시장 급락 처리"""

        logger.warning("Market crash detected")

        # 긴급 현금 전환
        await self._convert_to_cash()

        # 방어 모드 활성화
        self.defense_mode = True

    async def _handle_consecutive_losses(self, count: int):
        """연속 손실 처리"""

        logger.warning(f"Consecutive losses: {count}")

        # 거래량 감소
        reduction_ratio = max(0.1, 1.0 - (count * 0.2))  # 최소 10%
        await self._reduce_trading_volume(reduction_ratio)

    async def _reduce_all_positions(self, ratio: float):
        """전체 포지션 축소"""

        for symbol in list(self.positions.keys()):
            await self._reduce_position(symbol, ratio)

    async def _reduce_position(self, symbol: str, ratio: float):
        """개별 포지션 축소"""

        if symbol not in self.positions:
            return

        position = self.positions[symbol]
        reduce_quantity = int(position.quantity * ratio)

        if reduce_quantity > 0:
            # 실제 주문 실행 (실제로는 trader API 호출)
            logger.info(f"Reducing position {symbol}: {reduce_quantity} shares")

            # 포지션 업데이트
            position.quantity -= reduce_quantity
            if position.quantity <= 0:
                del self.positions[symbol]

    async def _emergency_exit_all_positions(self):
        """긴급 전체 청산"""

        logger.critical("Emergency exit all positions")

        for symbol in list(self.positions.keys()):
            position = self.positions[symbol]

            # 시장가 청산 (실제로는 trader API 호출)
            logger.info(f"Emergency exit {symbol}: {position.quantity} shares")

            del self.positions[symbol]

        self.trading_allowed = False

    async def _convert_to_cash(self):
        """현금 전환"""

        logger.warning("Converting positions to cash")

        # 모든 포지션 청산
        await self._emergency_exit_all_positions()

        # 현금 보유 (실제로는 계좌 잔고 확인)
        logger.info("All positions converted to cash")

    async def _add_hedge_positions(self):
        """헤지 포지션 추가"""

        logger.info("Adding hedge positions")

        # VIX ETF 매수 또는 풋옵션 매수 (실제로는 구현 필요)
        # 예시: VIX ETF 매수
        pass

    async def _reduce_trading_volume(self, ratio: float):
        """거래량 감소"""

        logger.info(f"Reducing trading volume by {ratio:.1%}")

        # 거래량 한도 설정 (실제로는 구현 필요)
        self.config['max_order_size'] *= ratio

    async def _close_oldest_position(self):
        """가장 오래된 포지션 청산"""

        if not self.positions:
            return

        # 가장 오래된 포지션 찾기
        oldest_symbol = min(self.positions.keys(),
                          key=lambda x: self.positions[x].timestamp)

        await self._reduce_position(oldest_symbol, 1.0)  # 100% 청산

    async def _get_current_price(self, symbol: str) -> float:
        """현재가 조회 (실제로는 API 호출)"""
        # 예시: 실제로는 KIS API 호출
        return 75000.0  # 예시 가격

    async def _calculate_volatility(self, symbol: str) -> float:
        """변동성 계산"""
        # 예시: 실제로는 히스토리 데이터 기반 계산
        return 0.02  # 2%

    async def _calculate_beta(self, symbol: str) -> float:
        """베타 계산"""
        # 예시: 실제로는 시장 대비 계산
        return 1.0

    async def _calculate_period_pnl(self, period: str) -> float:
        """기간별 손익 계산"""
        # 예시: 실제로는 거래 히스토리 기반 계산
        return -100000.0  # 예시 손실

    async def _calculate_max_drawdown(self) -> float:
        """최대 낙폭 계산"""
        # 예시: 실제로는 히스토리 기반 계산
        return -0.05  # -5%

    async def _calculate_sharpe_ratio(self) -> float:
        """샤프 비율 계산"""
        # 예시: 실제로는 수익률 기반 계산
        return 1.2

    async def _calculate_correlation_matrix(self) -> pd.DataFrame:
        """상관관계 행렬 계산"""
        # 예시: 실제로는 수익률 기반 계산
        symbols = list(self.positions.keys())
        if len(symbols) < 2:
            return pd.DataFrame()

        # 예시 상관관계 행렬
        data = np.random.rand(len(symbols), len(symbols))
        data = (data + data.T) / 2  # 대칭 행렬
        np.fill_diagonal(data, 1.0)  # 대각선 1

        return pd.DataFrame(data, index=symbols, columns=symbols)

    async def _get_vix_level(self) -> float:
        """VIX 레벨 조회"""
        # 예시: 실제로는 VIX API 호출
        return 25.0  # 예시 VIX

    async def _determine_market_regime(self) -> MarketEvent:
        """시장 상황 판단"""
        # 예시: 실제로는 다양한 지표 기반 판단
        if self.portfolio_risk and self.portfolio_risk.vix_level > 30:
            return MarketEvent.VOLATILE
        elif self.portfolio_risk and self.portfolio_risk.daily_pnl_pct < -0.05:
            return MarketEvent.CRASH
        else:
            return MarketEvent.NORMAL

    def _determine_risk_level(self, total_pnl_pct: float, daily_pnl_pct: float,
                            weekly_pnl_pct: float, monthly_pnl_pct: float,
                            vix_level: float) -> RiskLevel:
        """리스크 레벨 결정"""

        # 위험도 점수 계산
        risk_score = 0

        if total_pnl_pct < -0.10:
            risk_score += 3
        elif total_pnl_pct < -0.05:
            risk_score += 2
        elif total_pnl_pct < 0:
            risk_score += 1

        if daily_pnl_pct < -0.03:
            risk_score += 2

        if weekly_pnl_pct < -0.07:
            risk_score += 2

        if monthly_pnl_pct < -0.15:
            risk_score += 3

        if vix_level > 30:
            risk_score += 2
        elif vix_level > 25:
            risk_score += 1

        # 리스크 레벨 결정
        if risk_score >= 8:
            return RiskLevel.CRITICAL
        elif risk_score >= 5:
            return RiskLevel.HIGH
        elif risk_score >= 3:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    async def _check_symbol_liquidity(self, symbol: str) -> float:
        """종목별 유동성 체크"""
        # 예시: 실제로는 거래량, 호가창 분석
        return 2000000.0  # 예시 유동성

    async def _count_consecutive_losses(self) -> int:
        """연속 손실 횟수 계산"""
        # 예시: 실제로는 거래 히스토리 기반 계산
        return 2  # 예시 연속 손실

    async def get_performance_metrics(self) -> PerformanceMetrics:
        """성과 지표 계산"""

        # 승률 계산
        total_trades = len(self.daily_trades)
        winning_trades = len([t for t in self.daily_trades if t.get('pnl', 0) > 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # 평균 손익 계산
        wins = [t['pnl'] for t in self.daily_trades if t.get('pnl', 0) > 0]
        losses = [t['pnl'] for t in self.daily_trades if t.get('pnl', 0) < 0]

        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0

        # 수익 팩터
        total_wins = sum(wins) if wins else 0
        total_losses = abs(sum(losses)) if losses else 1
        profit_factor = total_wins / total_losses if total_losses > 0 else 0

        # 연속 손실
        max_consecutive_losses = await self._count_consecutive_losses()

        # 전략별 기여도 (예시)
        strategy_contribution = {
            'breakout': 0.4,
            'volume_surge': 0.3,
            'technical_ai': 0.3
        }

        # 시장 상황별 성과 (예시)
        market_performance = {
            'bull_market': 0.15,
            'bear_market': -0.05,
            'sideways': 0.02
        }

        # 개선점 도출
        improvement_suggestions = []

        if win_rate < 0.5:
            improvement_suggestions.append("승률 개선 필요: 진입 조건 강화")

        if profit_factor < 1.5:
            improvement_suggestions.append("손익비 개선 필요: 익절/손절 최적화")

        if max_consecutive_losses > 5:
            improvement_suggestions.append("연속 손실 관리: 리스크 한도 강화")

        return PerformanceMetrics(
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            max_consecutive_losses=max_consecutive_losses,
            strategy_contribution=strategy_contribution,
            market_performance=market_performance,
            improvement_suggestions=improvement_suggestions
        )

# 사용 예시
async def main():
    """메인 실행 함수"""

    # 설정
    config = {
        'risk_limits': {
            'daily_loss_limit': -0.03,
            'weekly_loss_limit': -0.07,
            'monthly_loss_limit': -0.15,
            'individual_position_limit': 0.10,
            'max_positions': 10,
            'max_correlation': 0.7,
            'min_liquidity': 1000000,
            'vix_threshold': 30.0
        },
        'monitoring_interval': 1.0,
        'correlation_window': 30,
        'volatility_window': 20
    }

    # 리스크 매니저 생성
    risk_manager = RealTimeRiskManager(config)

    # 모니터링 시작
    await risk_manager.start_monitoring()

if __name__ == "__main__":
    asyncio.run(main())

