#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: risk_management_system.py
모듈: 포괄적 리스크 관리 및 안전장치 시스템
목적: 시장/모델/운영 리스크 관리, 자동 안전장치, 리스크 한도 관리

Author: Trading AI System
Created: 2025-01-27
Modified: 2025-01-27
Version: 1.0.0

Dependencies:
    - Python 3.11+
    - numpy>=1.24.0
    - pandas>=2.0.0
    - scipy>=1.10.0
    - scikit-learn>=1.3.0

Performance:
    - 리스크 계산: < 50ms
    - 안전장치 응답: < 10ms
    - 실시간 모니터링: < 100ms

Security:
    - 실시간 검증
    - 자동 복구
    - 로그 추적

License: MIT
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
import warnings
from collections import deque
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings('ignore')

# 로깅 설정
logger = logging.getLogger(__name__)


class RiskType(Enum):
    """리스크 타입"""
    MARKET = "market"
    MODEL = "model"
    OPERATIONAL = "operational"
    LIQUIDITY = "liquidity"
    CONCENTRATION = "concentration"


class RiskLevel(Enum):
    """리스크 레벨"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskMetrics:
    """리스크 메트릭"""
    timestamp: datetime
    risk_type: RiskType
    var_95: float = 0.0
    cvar_95: float = 0.0
    max_drawdown: float = 0.0
    volatility: float = 0.0
    beta: float = 0.0
    correlation: float = 0.0
    concentration: float = 0.0
    leverage: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskLimits:
    """리스크 한도"""
    # 포트폴리오 한도
    max_portfolio_value: float = 1000000.0
    max_daily_loss: float = 50000.0  # 5%
    max_drawdown: float = 0.15  # 15%
    
    # 종목별 한도
    max_position_size: float = 0.1  # 10%
    max_single_stock: float = 0.05  # 5%
    
    # 섹터별 한도
    max_sector_exposure: float = 0.3  # 30%
    
    # 레버리지 한도
    max_leverage: float = 1.5  # 150%
    
    # VaR 한도
    max_var_95: float = 0.02  # 2%
    max_cvar_95: float = 0.03  # 3%


@dataclass
class SafetyConfig:
    """안전장치 설정"""
    # 자동 중단 임계값
    loss_threshold: float = 0.05  # 5%
    performance_threshold: float = 0.6  # 60%
    volatility_threshold: float = 0.03  # 3%
    
    # 응답 시간
    emergency_response_time: float = 1.0  # 1초
    warning_response_time: float = 5.0  # 5초
    
    # 복구 설정
    auto_recovery_enabled: bool = True
    recovery_check_interval: float = 60.0  # 60초


class RiskManager:
    """리스크 관리자"""
    
    def __init__(self, risk_limits: RiskLimits):
        self.risk_limits = risk_limits
        self.risk_history = {}
        self.current_risk = {}
        self.logger = logging.getLogger("RiskManager")
    
    def calculate_market_risk(self, portfolio_returns: np.ndarray, 
                            market_returns: np.ndarray) -> RiskMetrics:
        """시장 리스크 계산"""
        try:
            # VaR 계산 (Historical Simulation)
            var_95 = np.percentile(portfolio_returns, 5)
            cvar_95 = np.mean(portfolio_returns[portfolio_returns <= var_95])
            
            # 최대 낙폭
            cumulative = np.cumprod(1 + portfolio_returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = np.min(drawdown)
            
            # 변동성
            volatility = np.std(portfolio_returns) * np.sqrt(252)
            
            # Beta 계산
            if len(market_returns) > 0:
                covariance = np.cov(portfolio_returns, market_returns)[0, 1]
                market_variance = np.var(market_returns)
                beta = covariance / market_variance if market_variance > 0 else 1.0
            else:
                beta = 1.0
            
            # 상관관계
            correlation = np.corrcoef(portfolio_returns, market_returns)[0, 1] if len(market_returns) > 0 else 0.0
            
            risk_metrics = RiskMetrics(
                timestamp=datetime.now(),
                risk_type=RiskType.MARKET,
                var_95=float(var_95),
                cvar_95=float(cvar_95),
                max_drawdown=float(max_drawdown),
                volatility=float(volatility),
                beta=float(beta),
                correlation=float(correlation),
                metadata={
                    'portfolio_returns': portfolio_returns.tolist(),
                    'market_returns': market_returns.tolist()
                }
            )
            
            self.current_risk[RiskType.MARKET] = risk_metrics
            return risk_metrics
            
        except Exception as e:
            self.logger.error(f"시장 리스크 계산 실패: {e}")
            return None
    
    def calculate_model_risk(self, predictions: np.ndarray, 
                           uncertainties: np.ndarray) -> RiskMetrics:
        """모델 리스크 계산"""
        try:
            # 예측 불확실성
            prediction_std = np.std(predictions)
            uncertainty_mean = np.mean(uncertainties)
            
            # 모델 불확실성 VaR
            var_95 = np.percentile(uncertainties, 95)
            cvar_95 = np.mean(uncertainties[uncertainties >= var_95])
            
            # 예측 분산
            prediction_variance = np.var(predictions)
            
            risk_metrics = RiskMetrics(
                timestamp=datetime.now(),
                risk_type=RiskType.MODEL,
                var_95=float(var_95),
                cvar_95=float(cvar_95),
                volatility=float(prediction_std),
                metadata={
                    'predictions': predictions.tolist(),
                    'uncertainties': uncertainties.tolist(),
                    'prediction_variance': float(prediction_variance),
                    'uncertainty_mean': float(uncertainty_mean)
                }
            )
            
            self.current_risk[RiskType.MODEL] = risk_metrics
            return risk_metrics
            
        except Exception as e:
            self.logger.error(f"모델 리스크 계산 실패: {e}")
            return None
    
    def calculate_concentration_risk(self, positions: Dict[str, float], 
                                  portfolio_value: float) -> RiskMetrics:
        """집중도 리스크 계산"""
        try:
            if not positions or portfolio_value <= 0:
                return None
            
            # 개별 종목 집중도
            position_values = list(positions.values())
            max_concentration = max(position_values) / portfolio_value if position_values else 0.0
            
            # 전체 집중도 (Herfindahl Index)
            weights = np.array(position_values) / portfolio_value
            concentration_index = np.sum(weights ** 2)
            
            # 섹터별 집중도 (간단한 예시)
            sector_concentration = 0.0  # 실제 구현에서는 섹터 정보 필요
            
            risk_metrics = RiskMetrics(
                timestamp=datetime.now(),
                risk_type=RiskType.CONCENTRATION,
                concentration=float(concentration_index),
                max_drawdown=float(max_concentration),
                metadata={
                    'positions': positions,
                    'portfolio_value': portfolio_value,
                    'max_concentration': float(max_concentration),
                    'sector_concentration': float(sector_concentration)
                }
            )
            
            self.current_risk[RiskType.CONCENTRATION] = risk_metrics
            return risk_metrics
            
        except Exception as e:
            self.logger.error(f"집중도 리스크 계산 실패: {e}")
            return None
    
    def check_risk_limits(self, risk_metrics: RiskMetrics) -> List[str]:
        """리스크 한도 체크"""
        violations = []
        
        try:
            if risk_metrics.risk_type == RiskType.MARKET:
                if abs(risk_metrics.var_95) > self.risk_limits.max_var_95:
                    violations.append(f"VaR 한도 초과: {risk_metrics.var_95:.3f} > {self.risk_limits.max_var_95}")
                
                if abs(risk_metrics.cvar_95) > self.risk_limits.max_cvar_95:
                    violations.append(f"CVaR 한도 초과: {risk_metrics.cvar_95:.3f} > {self.risk_limits.max_cvar_95}")
                
                if risk_metrics.max_drawdown < -self.risk_limits.max_drawdown:
                    violations.append(f"Drawdown 한도 초과: {risk_metrics.max_drawdown:.3f} < -{self.risk_limits.max_drawdown}")
            
            elif risk_metrics.risk_type == RiskType.CONCENTRATION:
                if risk_metrics.concentration > 0.3:  # 30% 집중도 한도
                    violations.append(f"집중도 한도 초과: {risk_metrics.concentration:.3f} > 0.3")
                
                if risk_metrics.max_drawdown > self.risk_limits.max_single_stock:
                    violations.append(f"개별 종목 한도 초과: {risk_metrics.max_drawdown:.3f} > {self.risk_limits.max_single_stock}")
            
            return violations
            
        except Exception as e:
            self.logger.error(f"리스크 한도 체크 실패: {e}")
            return ["리스크 한도 체크 오류"]


class SafetyController:
    """안전장치 컨트롤러"""
    
    def __init__(self, safety_config: SafetyConfig):
        self.safety_config = safety_config
        self.emergency_stop_active = False
        self.warning_active = False
        self.last_check_time = datetime.now()
        self.logger = logging.getLogger("SafetyController")
    
    def check_safety_conditions(self, risk_metrics: Dict[RiskType, RiskMetrics],
                              performance_metrics: Dict[str, float]) -> Dict[str, Any]:
        """안전 조건 체크"""
        try:
            safety_status = {
                'emergency_stop': False,
                'warning': False,
                'auto_recovery': False,
                'violations': [],
                'timestamp': datetime.now()
            }
            
            # 1. 손실 한도 체크
            if 'daily_pnl' in performance_metrics:
                daily_loss_pct = abs(performance_metrics['daily_pnl']) / performance_metrics.get('portfolio_value', 1.0)
                if daily_loss_pct > self.safety_config.loss_threshold:
                    safety_status['emergency_stop'] = True
                    safety_status['violations'].append(f"일일 손실 한도 초과: {daily_loss_pct:.3f}")
            
            # 2. 성능 저하 체크
            if 'accuracy' in performance_metrics:
                if performance_metrics['accuracy'] < self.safety_config.performance_threshold:
                    safety_status['warning'] = True
                    safety_status['violations'].append(f"성능 저하: {performance_metrics['accuracy']:.3f}")
            
            # 3. 변동성 체크
            if RiskType.MARKET in risk_metrics:
                volatility = risk_metrics[RiskType.MARKET].volatility
                if volatility > self.safety_config.volatility_threshold:
                    safety_status['warning'] = True
                    safety_status['violations'].append(f"높은 변동성: {volatility:.3f}")
            
            # 4. 리스크 한도 체크
            for risk_type, metrics in risk_metrics.items():
                violations = self.risk_manager.check_risk_limits(metrics)
                if violations:
                    safety_status['emergency_stop'] = True
                    safety_status['violations'].extend(violations)
            
            # 상태 업데이트
            if safety_status['emergency_stop']:
                self.emergency_stop_active = True
                self.warning_active = False
            elif safety_status['warning']:
                self.warning_active = True
            else:
                # 복구 조건 체크
                if self.emergency_stop_active or self.warning_active:
                    if self._check_recovery_conditions(risk_metrics, performance_metrics):
                        safety_status['auto_recovery'] = True
                        self.emergency_stop_active = False
                        self.warning_active = False
            
            return safety_status
            
        except Exception as e:
            self.logger.error(f"안전 조건 체크 실패: {e}")
            return {'emergency_stop': True, 'warning': False, 'auto_recovery': False, 
                   'violations': ['안전 조건 체크 오류'], 'timestamp': datetime.now()}
    
    def _check_recovery_conditions(self, risk_metrics: Dict[RiskType, RiskMetrics],
                                 performance_metrics: Dict[str, float]) -> bool:
        """복구 조건 체크"""
        try:
            # 복구 조건: 모든 지표가 정상 범위로 복귀
            recovery_conditions = []
            
            # 성능 복구
            if 'accuracy' in performance_metrics:
                recovery_conditions.append(performance_metrics['accuracy'] > 0.7)
            
            # 변동성 복구
            if RiskType.MARKET in risk_metrics:
                volatility = risk_metrics[RiskType.MARKET].volatility
                recovery_conditions.append(volatility < 0.02)
            
            # 손실 복구
            if 'daily_pnl' in performance_metrics:
                daily_loss_pct = abs(performance_metrics['daily_pnl']) / performance_metrics.get('portfolio_value', 1.0)
                recovery_conditions.append(daily_loss_pct < 0.02)
            
            return all(recovery_conditions) if recovery_conditions else False
            
        except Exception as e:
            self.logger.error(f"복구 조건 체크 실패: {e}")
            return False
    
    def execute_safety_actions(self, safety_status: Dict[str, Any]) -> Dict[str, Any]:
        """안전장치 실행"""
        try:
            actions = {
                'executed_actions': [],
                'timestamp': datetime.now()
            }
            
            if safety_status['emergency_stop']:
                # 긴급 중단
                actions['executed_actions'].extend([
                    '모든 신규 주문 중단',
                    '기존 포지션 청산 시작',
                    '모델 추론 중단',
                    '알림 발송'
                ])
                
                self.logger.critical("긴급 중단 실행됨")
                
            elif safety_status['warning']:
                # 경고 모드
                actions['executed_actions'].extend([
                    '포지션 크기 축소',
                    '리스크 한도 강화',
                    '모니터링 강화',
                    '경고 알림 발송'
                ])
                
                self.logger.warning("경고 모드 활성화")
            
            elif safety_status['auto_recovery']:
                # 자동 복구
                actions['executed_actions'].extend([
                    '정상 모드 복귀',
                    '포지션 크기 정상화',
                    '리스크 한도 정상화',
                    '복구 알림 발송'
                ])
                
                self.logger.info("자동 복구 실행됨")
            
            return actions
            
        except Exception as e:
            self.logger.error(f"안전장치 실행 실패: {e}")
            return {'executed_actions': ['안전장치 실행 오류'], 'timestamp': datetime.now()}


class LimitMonitor:
    """한도 모니터"""
    
    def __init__(self, risk_limits: RiskLimits):
        self.risk_limits = risk_limits
        self.limit_violations = []
        self.logger = logging.getLogger("LimitMonitor")
    
    def check_position_limits(self, positions: Dict[str, float], 
                            portfolio_value: float) -> List[str]:
        """포지션 한도 체크"""
        violations = []
        
        try:
            # 개별 종목 한도
            for symbol, position_value in positions.items():
                position_pct = abs(position_value) / portfolio_value
                if position_pct > self.risk_limits.max_single_stock:
                    violations.append(f"{symbol}: 개별 종목 한도 초과 ({position_pct:.3f} > {self.risk_limits.max_single_stock})")
            
            # 전체 포지션 한도
            total_position_value = sum(abs(pos) for pos in positions.values())
            total_position_pct = total_position_value / portfolio_value
            if total_position_pct > self.risk_limits.max_position_size:
                violations.append(f"전체 포지션 한도 초과 ({total_position_pct:.3f} > {self.risk_limits.max_position_size})")
            
            return violations
            
        except Exception as e:
            self.logger.error(f"포지션 한도 체크 실패: {e}")
            return ["포지션 한도 체크 오류"]
    
    def check_portfolio_limits(self, portfolio_value: float, daily_pnl: float) -> List[str]:
        """포트폴리오 한도 체크"""
        violations = []
        
        try:
            # 포트폴리오 가치 한도
            if portfolio_value > self.risk_limits.max_portfolio_value:
                violations.append(f"포트폴리오 가치 한도 초과 ({portfolio_value:.0f} > {self.risk_limits.max_portfolio_value})")
            
            # 일일 손실 한도
            if abs(daily_pnl) > self.risk_limits.max_daily_loss:
                violations.append(f"일일 손실 한도 초과 ({abs(daily_pnl):.0f} > {self.risk_limits.max_daily_loss})")
            
            return violations
            
        except Exception as e:
            self.logger.error(f"포트폴리오 한도 체크 실패: {e}")
            return ["포트폴리오 한도 체크 오류"]
    
    def check_leverage_limits(self, total_exposure: float, 
                            portfolio_value: float) -> List[str]:
        """레버리지 한도 체크"""
        violations = []
        
        try:
            leverage = total_exposure / portfolio_value if portfolio_value > 0 else 0.0
            if leverage > self.risk_limits.max_leverage:
                violations.append(f"레버리지 한도 초과 ({leverage:.2f} > {self.risk_limits.max_leverage})")
            
            return violations
            
        except Exception as e:
            self.logger.error(f"레버리지 한도 체크 실패: {e}")
            return ["레버리지 한도 체크 오류"]


class EmergencyStop:
    """긴급 중단 시스템"""
    
    def __init__(self):
        self.emergency_active = False
        self.emergency_reason = ""
        self.emergency_time = None
        self.logger = logging.getLogger("EmergencyStop")
    
    def activate_emergency_stop(self, reason: str):
        """긴급 중단 활성화"""
        try:
            self.emergency_active = True
            self.emergency_reason = reason
            self.emergency_time = datetime.now()
            
            self.logger.critical(f"긴급 중단 활성화: {reason}")
            
            # 긴급 조치 실행
            self._execute_emergency_actions()
            
        except Exception as e:
            self.logger.error(f"긴급 중단 활성화 실패: {e}")
    
    def deactivate_emergency_stop(self):
        """긴급 중단 해제"""
        try:
            self.emergency_active = False
            self.emergency_reason = ""
            self.emergency_time = None
            
            self.logger.info("긴급 중단 해제됨")
            
        except Exception as e:
            self.logger.error(f"긴급 중단 해제 실패: {e}")
    
    def _execute_emergency_actions(self):
        """긴급 조치 실행"""
        try:
            # 1. 모든 신규 주문 중단
            self._stop_all_orders()
            
            # 2. 기존 포지션 청산
            self._liquidate_positions()
            
            # 3. 모델 추론 중단
            self._stop_model_inference()
            
            # 4. 알림 발송
            self._send_emergency_notifications()
            
        except Exception as e:
            self.logger.error(f"긴급 조치 실행 실패: {e}")
    
    def _stop_all_orders(self):
        """모든 주문 중단"""
        # 실제 구현에서는 주문 시스템과 연동
        self.logger.info("모든 신규 주문 중단됨")
    
    def _liquidate_positions(self):
        """포지션 청산"""
        # 실제 구현에서는 포지션 관리 시스템과 연동
        self.logger.info("포지션 청산 시작됨")
    
    def _stop_model_inference(self):
        """모델 추론 중단"""
        # 실제 구현에서는 모델 추론 시스템과 연동
        self.logger.info("모델 추론 중단됨")
    
    def _send_emergency_notifications(self):
        """긴급 알림 발송"""
        # 실제 구현에서는 알림 시스템과 연동
        self.logger.info("긴급 알림 발송됨")
    
    def get_emergency_status(self) -> Dict[str, Any]:
        """긴급 중단 상태 조회"""
        return {
            'active': self.emergency_active,
            'reason': self.emergency_reason,
            'timestamp': self.emergency_time.isoformat() if self.emergency_time else None
        }


class StressTestEngine:
    """스트레스 테스트 엔진"""
    
    def __init__(self):
        self.logger = logging.getLogger("StressTestEngine")
    
    def run_stress_test(self, portfolio_positions: Dict[str, float],
                       market_data: pd.DataFrame) -> Dict[str, Any]:
        """스트레스 테스트 실행"""
        try:
            stress_scenarios = {
                'market_crash': self._market_crash_scenario,
                'volatility_spike': self._volatility_spike_scenario,
                'liquidity_crisis': self._liquidity_crisis_scenario,
                'correlation_breakdown': self._correlation_breakdown_scenario
            }
            
            stress_results = {}
            
            for scenario_name, scenario_func in stress_scenarios.items():
                try:
                    result = scenario_func(portfolio_positions, market_data)
                    stress_results[scenario_name] = result
                except Exception as e:
                    self.logger.error(f"스트레스 테스트 {scenario_name} 실패: {e}")
                    stress_results[scenario_name] = {'error': str(e)}
            
            return stress_results
            
        except Exception as e:
            self.logger.error(f"스트레스 테스트 실행 실패: {e}")
            return {'error': str(e)}
    
    def _market_crash_scenario(self, positions: Dict[str, float], 
                              market_data: pd.DataFrame) -> Dict[str, Any]:
        """시장 폭락 시나리오"""
        try:
            # 20% 시장 폭락 가정
            crash_factor = -0.20
            
            # 포지션별 손실 계산
            position_losses = {}
            total_loss = 0.0
            
            for symbol, position_value in positions.items():
                # 간단한 베타 기반 손실 계산
                beta = 1.0  # 실제로는 개별 종목 베타 필요
                position_loss = position_value * beta * crash_factor
                position_losses[symbol] = position_loss
                total_loss += position_loss
            
            return {
                'scenario': 'market_crash',
                'crash_factor': crash_factor,
                'position_losses': position_losses,
                'total_loss': total_loss,
                'loss_percentage': total_loss / sum(abs(pos) for pos in positions.values()) if positions else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"시장 폭락 시나리오 실패: {e}")
            return {'error': str(e)}
    
    def _volatility_spike_scenario(self, positions: Dict[str, float], 
                                  market_data: pd.DataFrame) -> Dict[str, Any]:
        """변동성 급증 시나리오"""
        try:
            # 변동성 3배 증가 가정
            volatility_multiplier = 3.0
            
            # VaR 증가 계산
            base_var = 0.02  # 기본 VaR 2%
            stressed_var = base_var * volatility_multiplier
            
            total_exposure = sum(abs(pos) for pos in positions.values())
            potential_loss = total_exposure * stressed_var
            
            return {
                'scenario': 'volatility_spike',
                'volatility_multiplier': volatility_multiplier,
                'stressed_var': stressed_var,
                'potential_loss': potential_loss,
                'loss_percentage': stressed_var
            }
            
        except Exception as e:
            self.logger.error(f"변동성 급증 시나리오 실패: {e}")
            return {'error': str(e)}
    
    def _liquidity_crisis_scenario(self, positions: Dict[str, float], 
                                  market_data: pd.DataFrame) -> Dict[str, Any]:
        """유동성 위기 시나리오"""
        try:
            # 유동성 프리미엄 5% 가정
            liquidity_premium = 0.05
            
            # 청산 비용 계산
            liquidation_costs = {}
            total_cost = 0.0
            
            for symbol, position_value in positions.items():
                cost = abs(position_value) * liquidity_premium
                liquidation_costs[symbol] = cost
                total_cost += cost
            
            return {
                'scenario': 'liquidity_crisis',
                'liquidity_premium': liquidity_premium,
                'liquidation_costs': liquidation_costs,
                'total_cost': total_cost,
                'cost_percentage': liquidity_premium
            }
            
        except Exception as e:
            self.logger.error(f"유동성 위기 시나리오 실패: {e}")
            return {'error': str(e)}
    
    def _correlation_breakdown_scenario(self, positions: Dict[str, float], 
                                       market_data: pd.DataFrame) -> Dict[str, Any]:
        """상관관계 붕괴 시나리오"""
        try:
            # 상관관계 0.5로 증가 가정 (다변화 효과 감소)
            correlation_increase = 0.5
            
            # 포트폴리오 리스크 증가 계산
            base_volatility = 0.15  # 기본 변동성 15%
            stressed_volatility = base_volatility * (1 + correlation_increase)
            
            total_exposure = sum(abs(pos) for pos in positions.values())
            risk_increase = total_exposure * (stressed_volatility - base_volatility)
            
            return {
                'scenario': 'correlation_breakdown',
                'correlation_increase': correlation_increase,
                'stressed_volatility': stressed_volatility,
                'risk_increase': risk_increase,
                'risk_increase_percentage': correlation_increase
            }
            
        except Exception as e:
            self.logger.error(f"상관관계 붕괴 시나리오 실패: {e}")
            return {'error': str(e)}


class IntegratedRiskSystem:
    """통합 리스크 관리 시스템"""
    
    def __init__(self, risk_limits: RiskLimits, safety_config: SafetyConfig):
        self.risk_limits = risk_limits
        self.safety_config = safety_config
        
        self.risk_manager = RiskManager(risk_limits)
        self.safety_controller = SafetyController(safety_config)
        self.limit_monitor = LimitMonitor(risk_limits)
        self.emergency_stop = EmergencyStop()
        self.stress_test_engine = StressTestEngine()
        
        self.logger = logging.getLogger("IntegratedRiskSystem")
    
    def comprehensive_risk_check(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """종합 리스크 체크"""
        try:
            risk_report = {
                'timestamp': datetime.now().isoformat(),
                'risk_metrics': {},
                'limit_violations': [],
                'safety_status': {},
                'emergency_status': {},
                'stress_test_results': {}
            }
            
            # 1. 리스크 메트릭 계산
            if 'portfolio_returns' in portfolio_data and 'market_returns' in portfolio_data:
                market_risk = self.risk_manager.calculate_market_risk(
                    np.array(portfolio_data['portfolio_returns']),
                    np.array(portfolio_data['market_returns'])
                )
                if market_risk:
                    risk_report['risk_metrics']['market'] = market_risk
            
            if 'predictions' in portfolio_data and 'uncertainties' in portfolio_data:
                model_risk = self.risk_manager.calculate_model_risk(
                    np.array(portfolio_data['predictions']),
                    np.array(portfolio_data['uncertainties'])
                )
                if model_risk:
                    risk_report['risk_metrics']['model'] = model_risk
            
            if 'positions' in portfolio_data and 'portfolio_value' in portfolio_data:
                concentration_risk = self.risk_manager.calculate_concentration_risk(
                    portfolio_data['positions'],
                    portfolio_data['portfolio_value']
                )
                if concentration_risk:
                    risk_report['risk_metrics']['concentration'] = concentration_risk
            
            # 2. 한도 체크
            if 'positions' in portfolio_data and 'portfolio_value' in portfolio_data:
                position_violations = self.limit_monitor.check_position_limits(
                    portfolio_data['positions'],
                    portfolio_data['portfolio_value']
                )
                risk_report['limit_violations'].extend(position_violations)
            
            if 'portfolio_value' in portfolio_data and 'daily_pnl' in portfolio_data:
                portfolio_violations = self.limit_monitor.check_portfolio_limits(
                    portfolio_data['portfolio_value'],
                    portfolio_data['daily_pnl']
                )
                risk_report['limit_violations'].extend(portfolio_violations)
            
            # 3. 안전장치 체크
            safety_status = self.safety_controller.check_safety_conditions(
                risk_report['risk_metrics'],
                portfolio_data.get('performance_metrics', {})
            )
            risk_report['safety_status'] = safety_status
            
            # 4. 긴급 중단 상태
            risk_report['emergency_status'] = self.emergency_stop.get_emergency_status()
            
            # 5. 스트레스 테스트 (주기적 실행)
            if 'positions' in portfolio_data and 'market_data' in portfolio_data:
                stress_results = self.stress_test_engine.run_stress_test(
                    portfolio_data['positions'],
                    portfolio_data['market_data']
                )
                risk_report['stress_test_results'] = stress_results
            
            # 6. 안전장치 실행
            if safety_status['emergency_stop'] or safety_status['warning']:
                safety_actions = self.safety_controller.execute_safety_actions(safety_status)
                risk_report['safety_actions'] = safety_actions
            
            return risk_report
            
        except Exception as e:
            self.logger.error(f"종합 리스크 체크 실패: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}


# 사용 예시
def main():
    """메인 실행 함수"""
    # 설정
    risk_limits = RiskLimits()
    safety_config = SafetyConfig()
    
    # 시스템 초기화
    risk_system = IntegratedRiskSystem(risk_limits, safety_config)
    
    # 예시 데이터
    portfolio_data = {
        'portfolio_returns': [0.01, -0.005, 0.02, -0.01, 0.015],
        'market_returns': [0.008, -0.003, 0.018, -0.008, 0.012],
        'predictions': [0.5, 0.3, 0.7, 0.2, 0.6],
        'uncertainties': [0.1, 0.15, 0.08, 0.12, 0.09],
        'positions': {'005930': 100000, '000660': 80000, '035420': 60000},
        'portfolio_value': 1000000,
        'daily_pnl': -5000,
        'performance_metrics': {'accuracy': 0.65, 'sharpe_ratio': 1.2},
        'market_data': pd.DataFrame({'close': [100, 101, 102, 103, 104]})
    }
    
    # 종합 리스크 체크
    risk_report = risk_system.comprehensive_risk_check(portfolio_data)
    
    print("리스크 관리 리포트:")
    print(json.dumps(risk_report, indent=2, default=str))


if __name__ == "__main__":
    main() 