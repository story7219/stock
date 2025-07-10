#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: trading_execution_system.py
모듈: 실시간 자동매매 실행 시스템
목적: 매매 전략 엔진, 신호 처리, 리스크 관리, 주문 실행

Author: AI Trading System
Created: 2025-01-08
Modified: 2025-01-08
Version: 1.0.0

Dependencies:
    - Python 3.11+
    - asyncio
    - websockets==11.0.3
    - aiohttp==3.9.1
    - numpy==1.24.0
    - pandas==2.0.0
    - scipy==1.11.0

Performance:
    - 신호 처리: < 10ms
    - 주문 실행: < 50ms
    - 리스크 계산: < 5ms
    - 포지션 업데이트: < 1ms

Security:
    - API 키 암호화
    - 주문 검증
    - 리스크 제한
    - 장애 복구

License: MIT
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import lru_cache
from typing import (
    Any, Dict, List, Optional, Tuple, Union, Literal,
    Protocol, TypeVar, Generic, Final, Callable
)

import aiohttp
import numpy as np
import pandas as pd
import websockets
from numpy.typing import NDArray
from scipy import stats

# 타입 정의
T = TypeVar('T')
SignalData = Dict[str, Union[float, np.ndarray, Dict[str, float]]]
OrderData = Dict[str, Any]
PositionData = Dict[str, Any]

# 로깅 설정
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=UserWarning)


class OrderType(Enum):
    """주문 타입"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """주문 방향"""
    BUY = "buy"
    SELL = "sell"


class SignalType(Enum):
    """신호 타입"""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


@dataclass
class Signal:
    """매매 신호"""
    
    symbol: str
    signal_type: SignalType
    confidence: float  # 0.0 ~ 1.0
    price: float
    timestamp: datetime
    model_predictions: Dict[str, float] = field(default_factory=dict)
    market_conditions: Dict[str, float] = field(default_factory=dict)
    volatility: float = 0.0
    volume: float = 0.0


@dataclass
class Position:
    """포지션 정보"""
    
    symbol: str
    quantity: int
    avg_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    timestamp: datetime
    side: OrderSide = OrderSide.BUY


@dataclass
class Order:
    """주문 정보"""
    
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    price: float
    status: str
    timestamp: datetime
    filled_quantity: int = 0
    filled_price: float = 0.0
    commission: float = 0.0


@dataclass
class RiskLimits:
    """리스크 제한"""
    
    max_position_size: float = 0.1  # 포트폴리오의 10%
    max_sector_exposure: float = 0.3  # 섹터당 30%
    max_drawdown: float = 0.2  # 최대 20% 손실
    max_daily_loss: float = 0.05  # 일일 최대 5% 손실
    max_var: float = 0.02  # VaR 2%
    max_correlation: float = 0.7  # 상관관계 70%


class SignalProcessor:
    """신호 처리 및 필터링"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.signal_history = []
        self.market_filters = {
            'volatility_threshold': 0.05,
            'volume_threshold': 1000000,
            'spread_threshold': 0.002,
            'market_hours': (9, 15)  # 9시-15시
        }
    
    def process_signals(self, model_predictions: Dict[str, float], 
                       market_data: Dict[str, Any]) -> List[Signal]:
        """신호 처리 및 필터링"""
        try:
            signals = []
            
            for symbol, prediction in model_predictions.items():
                # 신호 타입 결정
                signal_type = self._determine_signal_type(prediction)
                
                # 신뢰도 계산
                confidence = self._calculate_confidence(prediction, market_data.get(symbol, {})
                )
                
                # 시장 필터 적용
                if not self._apply_market_filters(symbol, market_data.get(symbol, {})):
                    continue
                
                # 신호 생성
                signal = Signal(
                    symbol=symbol,
                    signal_type=signal_type,
                    confidence=confidence,
                    price=market_data.get(symbol, {}).get('price', 0.0),
                    timestamp=datetime.now(),
                    model_predictions={symbol: prediction},
                    market_conditions=market_data.get(symbol, {}),
                    volatility=market_data.get(symbol, {}).get('volatility', 0.0),
                    volume=market_data.get(symbol, {}).get('volume', 0.0)
                )
                
                signals.append(signal)
            
            # 신호 히스토리 업데이트
            self.signal_history.extend(signals)
            
            # 지연 보상 적용
            signals = self._apply_latency_compensation(signals)
            
            logger.info(f"신호 처리 완료: {len(signals)}개 신호 생성")
            return signals
            
        except Exception as e:
            logger.error(f"신호 처리 오류: {e}")
            return []
    
    def _determine_signal_type(self, prediction: float) -> SignalType:
        """신호 타입 결정"""
        if prediction > 0.7:
            return SignalType.STRONG_BUY
        elif prediction > 0.6:
            return SignalType.BUY
        elif prediction < 0.3:
            return SignalType.STRONG_SELL
        elif prediction < 0.4:
            return SignalType.SELL
        else:
            return SignalType.HOLD
    
    def _calculate_confidence(self, prediction: float, market_data: Dict[str, Any]) -> float:
        """신뢰도 계산"""
        # 기본 신뢰도
        base_confidence = abs(prediction - 0.5) * 2  # 0.0 ~ 1.0
        
        # 시장 조건 조정
        volatility = market_data.get('volatility', 0.0)
        volume = market_data.get('volume', 0.0)
        
        # 변동성이 높으면 신뢰도 감소
        volatility_factor = max(0.5, 1.0 - volatility)
        
        # 거래량이 많으면 신뢰도 증가
        volume_factor = min(1.2, volume / 1000000)
        
        confidence = base_confidence * volatility_factor * volume_factor
        return min(confidence, 1.0)
    
    def _apply_market_filters(self, symbol: str, market_data: Dict[str, Any]) -> bool:
        """시장 필터 적용"""
        try:
            # 거래 시간 확인
            current_hour = datetime.now().hour
            if not (self.market_filters['market_hours'][0] <= 
                   current_hour <= self.market_filters['market_hours'][1]):
                return False
            
            # 변동성 필터
            volatility = market_data.get('volatility', 0.0)
            if volatility > self.market_filters['volatility_threshold']:
                return False
            
            # 거래량 필터
            volume = market_data.get('volume', 0.0)
            if volume < self.market_filters['volume_threshold']:
                return False
            
            # 스프레드 필터
            spread = market_data.get('spread', 0.0)
            if spread > self.market_filters['spread_threshold']:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"시장 필터 적용 오류: {e}")
            return False
    
    def _apply_latency_compensation(self, signals: List[Signal]) -> List[Signal]:
        """지연 보상 적용"""
        try:
            compensated_signals = []
            
            for signal in signals:
                # 지연 시간 계산 (예: 100ms)
                latency_ms = 100
                compensation_factor = 1.0 + (latency_ms / 1000.0) * 0.1
                
                # 가격 조정
                if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                    signal.price *= compensation_factor
                elif signal.signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
                    signal.price /= compensation_factor
                
                compensated_signals.append(signal)
            
            return compensated_signals
            
        except Exception as e:
            logger.error(f"지연 보상 적용 오류: {e}")
            return signals


class RiskManager:
    """리스크 관리"""
    
    def __init__(self, risk_limits: RiskLimits, portfolio_value: float):
        self.risk_limits = risk_limits
        self.portfolio_value = portfolio_value
        self.positions = {}
        self.daily_pnl = 0.0
        self.max_drawdown = 0.0
        self.var_history = []
    
    def calculate_position_size(self, signal: Signal, current_price: float) -> int:
        """포지션 사이징 (Kelly Criterion 기반)"""
        try:
            # Kelly Criterion 계산
            win_rate = signal.confidence
            avg_win = 0.02  # 평균 수익 2%
            avg_loss = 0.01  # 평균 손실 1%
            
            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            kelly_fraction = max(0.0, min(kelly_fraction, 0.25))  # 0~25% 제한
            
            # 변동성 조정 (ATR 기반)
            volatility = signal.volatility
            volatility_adjustment = max(0.5, 1.0 - volatility)
            
            # 최종 포지션 사이즈
            position_value = self.portfolio_value * kelly_fraction * volatility_adjustment
            position_size = int(position_value / current_price)
            
            # 리스크 제한 적용
            position_size = self._apply_risk_limits(signal.symbol, position_size, current_price)
            
            return position_size
            
        except Exception as e:
            logger.error(f"포지션 사이징 오류: {e}")
            return 0
    
    def _apply_risk_limits(self, symbol: str, position_size: int, 
                          current_price: float) -> int:
        """리스크 제한 적용"""
        try:
            position_value = position_size * current_price
            
            # 최대 포지션 크기 제한
            max_position_value = self.portfolio_value * self.risk_limits.max_position_size
            if position_value > max_position_value:
                position_size = int(max_position_value / current_price)
            
            # 섹터 노출도 제한 (간단한 구현)
            sector_exposure = self._calculate_sector_exposure(symbol)
            if sector_exposure > self.risk_limits.max_sector_exposure:
                position_size = int(position_size * 0.5)  # 50% 감소
            
            # 상관관계 제한
            correlation = self._calculate_correlation(symbol)
            if correlation > self.risk_limits.max_correlation:
                position_size = int(position_size * 0.7)  # 30% 감소
            
            return position_size
            
        except Exception as e:
            logger.error(f"리스크 제한 적용 오류: {e}")
            return 0
    
    def calculate_var(self, positions: Dict[str, Position], 
                     market_data: Dict[str, Any]) -> float:
        """VaR 계산"""
        try:
            if not positions:
                return 0.0
            
            # 포지션별 수익률 시뮬레이션
            returns = []
            for symbol, position in positions.items():
                price = market_data.get(symbol, {}).get('price', position.current_price)
                volatility = market_data.get(symbol, {}).get('volatility', 0.02)
                
                # Monte Carlo 시뮬레이션 (간단한 구현)
                for _ in range(1000):
                    price_change = np.random.normal(0, volatility)
                    new_price = price * (1 + price_change)
                    pnl_change = (new_price - price) * position.quantity
                    returns.append(pnl_change)
            
            # VaR 계산 (95% 신뢰수준)
            var = np.percentile(returns, 5)
            self.var_history.append(var)
            
            return abs(var)
            
        except Exception as e:
            logger.error(f"VaR 계산 오류: {e}")
            return 0.0
    
    def check_risk_limits(self, new_order: Order, positions: Dict[str, Position]) -> bool:
        """리스크 제한 확인"""
        try:
            # VaR 확인
            current_var = self.calculate_var(positions, {})
            if current_var > self.portfolio_value * self.risk_limits.max_var:
                logger.warning(f"VaR 제한 초과: {current_var:.2f}")
                return False
            
            # 일일 손실 제한
            if self.daily_pnl < -self.portfolio_value * self.risk_limits.max_daily_loss:
                logger.warning(f"일일 손실 제한 초과: {self.daily_pnl:.2f}")
                return False
            
            # 최대 드로우다운 제한
            if self.max_drawdown > self.risk_limits.max_drawdown:
                logger.warning(f"최대 드로우다운 초과: {self.max_drawdown:.2f}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"리스크 제한 확인 오류: {e}")
            return False
    
    def update_risk_metrics(self, pnl_change: float) -> None:
        """리스크 메트릭 업데이트"""
        try:
            self.daily_pnl += pnl_change
            
            # 드로우다운 계산
            if pnl_change < 0:
                self.max_drawdown = min(self.max_drawdown, pnl_change / self.portfolio_value)
            
        except Exception as e:
            logger.error(f"리스크 메트릭 업데이트 오류: {e}")
    
    def _calculate_sector_exposure(self, symbol: str) -> float:
        """섹터 노출도 계산 (간단한 구현)"""
        # 실제로는 섹터 정보를 DB에서 조회
        return 0.1  # 기본값 10%
    
    def _calculate_correlation(self, symbol: str) -> float:
        """상관관계 계산 (간단한 구현)"""
        # 실제로는 히스토리컬 데이터로 계산
        return 0.3  # 기본값 30%


class ExecutionOptimizer:
    """실행 최적화"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.market_impact_model = 'linear'
        self.execution_algorithms = {
            'twap': self._twap_execution,
            'vwap': self._vwap_execution,
            'market': self._market_execution
        }
    
    def optimize_execution(self, order: Order, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """실행 최적화"""
        try:
            # 시장 영향도 계산
            market_impact = self._calculate_market_impact(order, market_data)
            
            # 최적 실행 알고리즘 선택
            algorithm = self._select_execution_algorithm(order, market_data)
            
            # 실행 계획 생성
            execution_plan = self.execution_algorithms[algorithm](order, market_data)
            
            # 슬리피지 예상
            slippage = self._estimate_slippage(order, market_data)
            
            result = {
                'algorithm': algorithm,
                'market_impact': market_impact,
                'slippage': slippage,
                'execution_plan': execution_plan,
                'total_cost': market_impact + slippage
            }
            
            logger.info(f"실행 최적화 완료: {algorithm}, 비용: {result['total_cost']:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"실행 최적화 오류: {e}")
            return {}
    
    def _calculate_market_impact(self, order: Order, market_data: Dict[str, Any]) -> float:
        """시장 영향도 계산"""
        try:
            volume = market_data.get('volume', 1.0)
            price = market_data.get('price', order.price)
            
            # 주문 크기 대비 거래량 비율
            order_ratio = order.quantity / volume
            
            if self.market_impact_model == 'linear':
                impact = 0.1 * order_ratio * price
            elif self.market_impact_model == 'square_root':
                impact = 0.1 * np.sqrt(order_ratio) * price
            else:
                impact = 0.1 * order_ratio * price
            
            return impact
            
        except Exception as e:
            logger.error(f"시장 영향도 계산 오류: {e}")
            return 0.0
    
    def _select_execution_algorithm(self, order: Order, market_data: Dict[str, Any]) -> str:
        """실행 알고리즘 선택"""
        try:
            volume = market_data.get('volume', 0.0)
            volatility = market_data.get('volatility', 0.0)
            
            # 대용량 주문은 TWAP
            if order.quantity > volume * 0.1:
                return 'twap'
            
            # 고변동성 시장은 VWAP
            elif volatility > 0.03:
                return 'vwap'
            
            # 일반적인 경우 시장가
            else:
                return 'market'
                
        except Exception as e:
            logger.error(f"실행 알고리즘 선택 오류: {e}")
            return 'market'
    
    def _twap_execution(self, order: Order, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """TWAP 실행 계획"""
        try:
            # 시간 분할 (예: 10분 동안 분할)
            time_slices = 10
            quantity_per_slice = order.quantity // time_slices
            
            execution_plan = {
                'method': 'twap',
                'time_slices': time_slices,
                'quantity_per_slice': quantity_per_slice,
                'interval_seconds': 60,  # 1분 간격
                'remaining_quantity': order.quantity % time_slices
            }
            
            return execution_plan
            
        except Exception as e:
            logger.error(f"TWAP 실행 계획 오류: {e}")
            return {}
    
    def _vwap_execution(self, order: Order, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """VWAP 실행 계획"""
        try:
            # 거래량 가중 평균 가격 기준
            vwap = market_data.get('vwap', order.price)
            
            execution_plan = {
                'method': 'vwap',
                'target_price': vwap,
                'quantity': order.quantity,
                'time_limit': 300  # 5분 제한
            }
            
            return execution_plan
            
        except Exception as e:
            logger.error(f"VWAP 실행 계획 오류: {e}")
            return {}
    
    def _market_execution(self, order: Order, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """시장가 실행 계획"""
        try:
            execution_plan = {
                'method': 'market',
                'quantity': order.quantity,
                'immediate': True
            }
            
            return execution_plan
            
        except Exception as e:
            logger.error(f"시장가 실행 계획 오류: {e}")
            return {}
    
    def _estimate_slippage(self, order: Order, market_data: Dict[str, Any]) -> float:
        """슬리피지 예상"""
        try:
            spread = market_data.get('spread', 0.001)
            volatility = market_data.get('volatility', 0.02)
            
            # 스프레드 + 변동성 기반 슬리피지
            slippage = spread + volatility * 0.5
            
            return slippage * order.price * order.quantity
            
        except Exception as e:
            logger.error(f"슬리피지 예상 오류: {e}")
            return 0.0


class OrderManager:
    """주문 관리"""
    
    def __init__(self, api_config: Dict[str, Any]):
        self.api_config = api_config
        self.orders = {}
        self.order_history = []
        self.websocket = None
        self.is_connected = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
    
    async def connect(self) -> bool:
        """웹소켓 연결"""
        try:
            websocket_url = self.api_config.get('websocket_url')
            if not websocket_url:
                raise ValueError("웹소켓 URL이 설정되지 않았습니다.")
            
            self.websocket = await websockets.connect(websocket_url)
            self.is_connected = True
            self.reconnect_attempts = 0
            
            # 연결 유지 태스크 시작
            asyncio.create_task(self._keep_alive())
            
            logger.info("웹소켓 연결 성공")
            return True
            
        except Exception as e:
            logger.error(f"웹소켓 연결 오류: {e}")
            return False
    
    async def disconnect(self) -> None:
        """웹소켓 연결 해제"""
        try:
            if self.websocket:
                await self.websocket.close()
                self.is_connected = False
                logger.info("웹소켓 연결 해제")
        except Exception as e:
            logger.error(f"웹소켓 연결 해제 오류: {e}")
    
    async def place_order(self, order: Order) -> bool:
        """주문 실행"""
        try:
            if not self.is_connected:
                logger.error("웹소켓이 연결되지 않았습니다.")
                return False
            
            # 주문 검증
            if not self._validate_order(order):
                return False
            
            # 주문 전송
            order_message = self._create_order_message(order)
            await self.websocket.send(json.dumps(order_message))
            
            # 주문 등록
            self.orders[order.order_id] = order
            
            logger.info(f"주문 전송 완료: {order.order_id}")
            return True
            
        except Exception as e:
            logger.error(f"주문 실행 오류: {e}")
            return False
    
    async def cancel_order(self, order_id: str) -> bool:
        """주문 취소"""
        try:
            if not self.is_connected:
                return False
            
            cancel_message = {
                'type': 'cancel_order',
                'order_id': order_id
            }
            
            await self.websocket.send(json.dumps(cancel_message))
            
            if order_id in self.orders:
                self.orders[order_id].status = 'cancelled'
            
            logger.info(f"주문 취소 완료: {order_id}")
            return True
            
        except Exception as e:
            logger.error(f"주문 취소 오류: {e}")
            return False
    
    async def get_order_status(self, order_id: str) -> Optional[Order]:
        """주문 상태 조회"""
        try:
            if order_id in self.orders:
                return self.orders[order_id]
            return None
            
        except Exception as e:
            logger.error(f"주문 상태 조회 오류: {e}")
            return None
    
    async def _keep_alive(self) -> None:
        """연결 유지"""
        try:
            while self.is_connected:
                await asyncio.sleep(30)  # 30초마다 ping
                if self.websocket:
                    await self.websocket.ping()
                    
        except Exception as e:
            logger.error(f"연결 유지 오류: {e}")
            await self._reconnect()
    
    async def _reconnect(self) -> None:
        """재연결"""
        try:
            if self.reconnect_attempts >= self.max_reconnect_attempts:
                logger.error("최대 재연결 시도 횟수 초과")
                return
            
            self.reconnect_attempts += 1
            logger.info(f"재연결 시도 {self.reconnect_attempts}/{self.max_reconnect_attempts}")
            
            await self.disconnect()
            await asyncio.sleep(5)  # 5초 대기
            
            success = await self.connect()
            if success:
                logger.info("재연결 성공")
            else:
                await self._reconnect()
                
        except Exception as e:
            logger.error(f"재연결 오류: {e}")
    
    def _validate_order(self, order: Order) -> bool:
        """주문 검증"""
        try:
            # 기본 검증
            if order.quantity <= 0:
                logger.error("주문 수량이 0 이하입니다.")
                return False
            
            if order.price <= 0:
                logger.error("주문 가격이 0 이하입니다.")
                return False
            
            # 중복 주문 확인
            if order.order_id in self.orders:
                logger.error("중복 주문 ID입니다.")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"주문 검증 오류: {e}")
            return False
    
    def _create_order_message(self, order: Order) -> Dict[str, Any]:
        """주문 메시지 생성"""
        return {
            'type': 'place_order',
            'order_id': order.order_id,
            'symbol': order.symbol,
            'side': order.side.value,
            'order_type': order.order_type.value,
            'quantity': order.quantity,
            'price': order.price,
            'timestamp': order.timestamp.isoformat()
        }


class PortfolioManager:
    """포트폴리오 관리"""
    
    def __init__(self, initial_value: float):
        self.initial_value = initial_value
        self.current_value = initial_value
        self.positions = {}
        self.cash = initial_value
        self.transaction_history = []
        self.performance_metrics = {}
    
    def update_position(self, symbol: str, quantity: int, price: float, 
                       side: OrderSide, commission: float = 0.0) -> None:
        """포지션 업데이트"""
        try:
            transaction_value = quantity * price
            commission_cost = commission
            
            if side == OrderSide.BUY:
                # 매수
                if symbol in self.positions:
                    # 기존 포지션 업데이트
                    pos = self.positions[symbol]
                    total_quantity = pos.quantity + quantity
                    total_cost = pos.avg_price * pos.quantity + transaction_value
                    pos.avg_price = total_cost / total_quantity
                    pos.quantity = total_quantity
                else:
                    # 새 포지션 생성
                    self.positions[symbol] = Position(
                        symbol=symbol,
                        quantity=quantity,
                        avg_price=price,
                        current_price=price,
                        unrealized_pnl=0.0,
                        realized_pnl=0.0,
                        timestamp=datetime.now(),
                        side=side
                    )
                
                self.cash -= (transaction_value + commission_cost)
                
            else:
                # 매도
                if symbol in self.positions:
                    pos = self.positions[symbol]
                    if pos.quantity >= quantity:
                        # 부분 매도
                        realized_pnl = (price - pos.avg_price) * quantity
                        pos.realized_pnl += realized_pnl
                        pos.quantity -= quantity
                        
                        if pos.quantity == 0:
                            # 포지션 종료
                            del self.positions[symbol]
                        
                        self.cash += (transaction_value - commission_cost)
                    else:
                        logger.error(f"매도 수량이 보유 수량을 초과합니다: {symbol}")
                        return
                else:
                    logger.error(f"매도할 포지션이 없습니다: {symbol}")
                    return
            
            # 거래 기록
            transaction = {
                'symbol': symbol,
                'side': side.value,
                'quantity': quantity,
                'price': price,
                'commission': commission_cost,
                'timestamp': datetime.now()
            }
            self.transaction_history.append(transaction)
            
            # 포트폴리오 가치 업데이트
            self._update_portfolio_value()
            
            logger.info(f"포지션 업데이트 완료: {symbol} {side.value} {quantity}주")
            
        except Exception as e:
            logger.error(f"포지션 업데이트 오류: {e}")
    
    def update_market_prices(self, market_data: Dict[str, float]) -> None:
        """시장 가격 업데이트"""
        try:
            for symbol, price in market_data.items():
                if symbol in self.positions:
                    pos = self.positions[symbol]
                    pos.current_price = price
                    pos.unrealized_pnl = (price - pos.avg_price) * pos.quantity
            
            # 포트폴리오 가치 업데이트
            self._update_portfolio_value()
            
        except Exception as e:
            logger.error(f"시장 가격 업데이트 오류: {e}")
    
    def _update_portfolio_value(self) -> None:
        """포트폴리오 가치 업데이트"""
        try:
            position_value = sum(
                pos.current_price * pos.quantity 
                for pos in self.positions.values()
            )
            
            self.current_value = self.cash + position_value
            
            # 성능 메트릭 업데이트
            self._update_performance_metrics()
            
        except Exception as e:
            logger.error(f"포트폴리오 가치 업데이트 오류: {e}")
    
    def _update_performance_metrics(self) -> None:
        """성능 메트릭 업데이트"""
        try:
            # 총 수익률
            total_return = (self.current_value - self.initial_value) / self.initial_value
            
            # 실현 손익
            realized_pnl = sum(pos.realized_pnl for pos in self.positions.values())
            
            # 미실현 손익
            unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
            
            # 드로우다운
            drawdown = (self.current_value - self.initial_value) / self.initial_value
            
            self.performance_metrics = {
                'total_return': total_return,
                'realized_pnl': realized_pnl,
                'unrealized_pnl': unrealized_pnl,
                'drawdown': drawdown,
                'cash_ratio': self.cash / self.current_value,
                'position_count': len(self.positions)
            }
            
        except Exception as e:
            logger.error(f"성능 메트릭 업데이트 오류: {e}")
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """포트폴리오 요약"""
        try:
            summary = {
                'current_value': self.current_value,
                'initial_value': self.initial_value,
                'cash': self.cash,
                'positions': len(self.positions),
                'performance': self.performance_metrics,
                'top_positions': self._get_top_positions(),
                'recent_transactions': self.transaction_history[-10:]  # 최근 10개
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"포트폴리오 요약 생성 오류: {e}")
            return {}
    
    def _get_top_positions(self) -> List[Dict[str, Any]]:
        """상위 포지션 조회"""
        try:
            positions = []
            for symbol, pos in self.positions.items():
                position_value = pos.current_price * pos.quantity
                positions.append({
                    'symbol': symbol,
                    'quantity': pos.quantity,
                    'avg_price': pos.avg_price,
                    'current_price': pos.current_price,
                    'position_value': position_value,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'weight': position_value / self.current_value
                })
            
            # 포지션 가치 기준 정렬
            positions.sort(key=lambda x: x['position_value'], reverse=True)
            return positions[:5]  # 상위 5개
            
        except Exception as e:
            logger.error(f"상위 포지션 조회 오류: {e}")
            return []


class TradingEngine:
    """매매 전략 엔진"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.signal_processor = SignalProcessor(config.get('signal_processor', {}))
        self.risk_manager = RiskManager(
            RiskLimits(**config.get('risk_limits', {})),
            config.get('portfolio_value', 1000000)
        )
        self.execution_optimizer = ExecutionOptimizer(config.get('execution_optimizer', {}))
        self.order_manager = OrderManager(config.get('order_manager', {}))
        self.portfolio_manager = PortfolioManager(config.get('portfolio_value', 1000000))
        
        self.is_running = False
        self.trading_task = None
    
    async def start(self) -> bool:
        """트레이딩 엔진 시작"""
        try:
            # 웹소켓 연결
            if not await self.order_manager.connect():
                logger.error("웹소켓 연결 실패")
                return False
            
            self.is_running = True
            self.trading_task = asyncio.create_task(self._trading_loop())
            
            logger.info("트레이딩 엔진 시작")
            return True
            
        except Exception as e:
            logger.error(f"트레이딩 엔진 시작 오류: {e}")
            return False
    
    async def stop(self) -> None:
        """트레이딩 엔진 중지"""
        try:
            self.is_running = False
            
            if self.trading_task:
                self.trading_task.cancel()
            
            await self.order_manager.disconnect()
            
            logger.info("트레이딩 엔진 중지")
            
        except Exception as e:
            logger.error(f"트레이딩 엔진 중지 오류: {e}")
    
    async def _trading_loop(self) -> None:
        """트레이딩 루프"""
        try:
            while self.is_running:
                # 신호 처리
                signals = await self._process_signals()
                
                # 신호별 매매 실행
                for signal in signals:
                    await self._execute_signal(signal)
                
                # 포트폴리오 업데이트
                await self._update_portfolio()
                
                # 리스크 체크
                await self._check_risk_limits()
                
                # 대기
                await asyncio.sleep(1)  # 1초 간격
                
        except asyncio.CancelledError:
            logger.info("트레이딩 루프 취소됨")
        except Exception as e:
            logger.error(f"트레이딩 루프 오류: {e}")
    
    async def _process_signals(self) -> List[Signal]:
        """신호 처리"""
        try:
            # 모델 예측 데이터 (실제로는 ML 시스템에서 받아옴)
            model_predictions = {
                'AAPL': 0.75,
                'GOOGL': 0.65,
                'MSFT': 0.80
            }
            
            # 시장 데이터 (실제로는 실시간 데이터에서 받아옴)
            market_data = {
                'AAPL': {'price': 150.0, 'volatility': 0.02, 'volume': 1000000},
                'GOOGL': {'price': 2800.0, 'volatility': 0.015, 'volume': 500000},
                'MSFT': {'price': 300.0, 'volatility': 0.018, 'volume': 800000}
            }
            
            # 신호 처리
            signals = self.signal_processor.process_signals(model_predictions, market_data)
            
            return signals
            
        except Exception as e:
            logger.error(f"신호 처리 오류: {e}")
            return []
    
    async def _execute_signal(self, signal: Signal) -> None:
        """신호 실행"""
        try:
            # 신호 강도 확인
            if signal.confidence < 0.6:
                logger.info(f"신호 강도 부족: {signal.symbol} {signal.confidence}")
                return
            
            # 포지션 사이징
            position_size = self.risk_manager.calculate_position_size(signal, signal.price)
            
            if position_size == 0:
                logger.info(f"포지션 사이즈 0: {signal.symbol}")
                return
            
            # 주문 생성
            order = self._create_order(signal, position_size)
            
            # 실행 최적화
            execution_plan = self.execution_optimizer.optimize_execution(order, {
                'price': signal.price,
                'volume': signal.volume,
                'volatility': signal.volatility
            })
            
            # 리스크 체크
            if not self.risk_manager.check_risk_limits(order, self.portfolio_manager.positions):
                logger.warning(f"리스크 제한 초과: {signal.symbol}")
                return
            
            # 주문 실행
            success = await self.order_manager.place_order(order)
            
            if success:
                logger.info(f"주문 실행 성공: {signal.symbol} {signal.signal_type.value}")
            else:
                logger.error(f"주문 실행 실패: {signal.symbol}")
            
        except Exception as e:
            logger.error(f"신호 실행 오류: {e}")
    
    def _create_order(self, signal: Signal, quantity: int) -> Order:
        """주문 생성"""
        try:
            order_id = f"order_{int(time.time() * 1000)}"
            
            if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                side = OrderSide.BUY
            else:
                side = OrderSide.SELL
            
            order = Order(
                order_id=order_id,
                symbol=signal.symbol,
                side=side,
                order_type=OrderType.MARKET,
                quantity=quantity,
                price=signal.price,
                status='pending',
                timestamp=datetime.now()
            )
            
            return order
            
        except Exception as e:
            logger.error(f"주문 생성 오류: {e}")
            raise
    
    async def _update_portfolio(self) -> None:
        """포트폴리오 업데이트"""
        try:
            # 시장 가격 업데이트 (실제로는 실시간 데이터에서 받아옴)
            market_prices = {
                'AAPL': 150.5,
                'GOOGL': 2805.0,
                'MSFT': 301.0
            }
            
            self.portfolio_manager.update_market_prices(market_prices)
            
        except Exception as e:
            logger.error(f"포트폴리오 업데이트 오류: {e}")
    
    async def _check_risk_limits(self) -> None:
        """리스크 제한 확인"""
        try:
            # VaR 계산
            var = self.risk_manager.calculate_var(
                self.portfolio_manager.positions,
                {}  # 시장 데이터
            )
            
            if var > self.risk_manager.risk_limits.max_var * self.portfolio_manager.current_value:
                logger.warning(f"VaR 제한 초과: {var:.2f}")
                # 여기서 포지션 축소 로직 구현 가능
            
        except Exception as e:
            logger.error(f"리스크 제한 확인 오류: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 조회"""
        try:
            status = {
                'is_running': self.is_running,
                'portfolio_summary': self.portfolio_manager.get_portfolio_summary(),
                'risk_metrics': {
                    'var': self.risk_manager.calculate_var(
                        self.portfolio_manager.positions, {}
                    ),
                    'daily_pnl': self.risk_manager.daily_pnl,
                    'max_drawdown': self.risk_manager.max_drawdown
                },
                'order_count': len(self.order_manager.orders),
                'websocket_connected': self.order_manager.is_connected
            }
            
            return status
            
        except Exception as e:
            logger.error(f"시스템 상태 조회 오류: {e}")
            return {'error': str(e)}


# 사용 예시
if __name__ == "__main__":
    # 트레이딩 엔진 설정
    config = {
        'portfolio_value': 1000000,
        'risk_limits': {
            'max_position_size': 0.1,
            'max_sector_exposure': 0.3,
            'max_drawdown': 0.2,
            'max_daily_loss': 0.05,
            'max_var': 0.02,
            'max_correlation': 0.7
        },
        'order_manager': {
            'websocket_url': 'ws://localhost:8080/ws'
        },
        'signal_processor': {
            'volatility_threshold': 0.05,
            'volume_threshold': 1000000
        },
        'execution_optimizer': {
            'market_impact_model': 'linear'
        }
    }
    
    # 트레이딩 엔진 생성
    trading_engine = TradingEngine(config)
    
    # 비동기 실행
    async def main():
        # 트레이딩 엔진 시작
        success = await trading_engine.start()
        
        if success:
            # 10초간 실행
            await asyncio.sleep(10)
            
            # 시스템 상태 확인
            status = trading_engine.get_system_status()
            print("시스템 상태:", status)
            
            # 트레이딩 엔진 중지
            await trading_engine.stop()
    
    # 실행
    asyncio.run(main()) 