#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📊 시스템 상태 관리 모듈
===================

자동매매 시스템의 상태 정보와 안전장치를 관리합니다.
"""

import logging
from datetime import datetime
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class SystemStatus:
    """📊 시스템 상태 정보"""
    is_running: bool = False
    start_time: Optional[datetime] = None
    total_trades: int = 0
    total_profit_loss: float = 0.0
    daily_trades: int = 0
    daily_profit_loss: float = 0.0
    last_trade_time: Optional[datetime] = None
    error_count: int = 0
    last_error: Optional[str] = None
    emergency_stop: bool = False

class SafetyManager:
    """🛡️ 안전장치 관리 시스템"""
    
    def __init__(self, max_daily_loss: float = -50000, max_daily_trades: int = 50):
        self.max_daily_loss = max_daily_loss  # 일일 최대 손실 (-5만원)
        self.max_daily_trades = max_daily_trades  # 일일 최대 거래수
        self.consecutive_losses = 0
        self.max_consecutive_losses = 5  # 연속 손실 제한
        self.last_reset_date = datetime.now().date()
        
        logger.info(f"🛡️ 안전장치 초기화 - 일일 손실 한도: {max_daily_loss:,}원, 거래 한도: {max_daily_trades}회")
    
    def check_daily_limits(self, status: SystemStatus) -> tuple[bool, str]:
        """📊 일일 한도 확인"""
        current_date = datetime.now().date()
        
        # 날짜가 바뀌면 일일 카운터 리셋
        if current_date != self.last_reset_date:
            status.daily_trades = 0
            status.daily_profit_loss = 0.0
            self.consecutive_losses = 0
            self.last_reset_date = current_date
            logger.info("📅 일일 카운터 리셋 완료")
        
        # 일일 손실 한도 확인
        if status.daily_profit_loss <= self.max_daily_loss:
            return False, f"🚨 일일 손실 한도 초과: {status.daily_profit_loss:,}원 (한도: {self.max_daily_loss:,}원)"
        
        # 일일 거래 한도 확인
        if status.daily_trades >= self.max_daily_trades:
            return False, f"🚨 일일 거래 한도 초과: {status.daily_trades}회 (한도: {self.max_daily_trades}회)"
        
        # 연속 손실 확인
        if self.consecutive_losses >= self.max_consecutive_losses:
            return False, f"🚨 연속 손실 한도 초과: {self.consecutive_losses}회 연속 손실"
        
        return True, "✅ 안전 범위 내"
    
    def record_trade_result(self, profit_loss: float, status: SystemStatus):
        """📈 거래 결과 기록"""
        status.total_trades += 1
        status.daily_trades += 1
        status.total_profit_loss += profit_loss
        status.daily_profit_loss += profit_loss
        status.last_trade_time = datetime.now()
        
        if profit_loss < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        logger.info(f"📊 거래 기록: {profit_loss:+,}원, 일일 누적: {status.daily_profit_loss:+,}원") 