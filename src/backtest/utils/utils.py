#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: utils.py
모듈: 백테스팅 유틸리티
목적: 공통 유틸리티 함수들

Author: WorldClassAI
Created: 2025-07-12
Version: 1.0.0
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import datetime, time
from typing import Dict, Any, List, Optional, Tuple

def apply_tick_size(price: float, tick_sizes: Dict[str, int]) -> float:
    """호가단위를 적용합니다."""
    if price < 1000:
        return round(price / tick_sizes["under_1000"]) * tick_sizes["under_1000"]
    elif price < 5000:
        return round(price / tick_sizes["under_5000"]) * tick_sizes["under_5000"]
    elif price < 10000:
        return round(price / tick_sizes["under_10000"]) * tick_sizes["under_10000"]
    elif price < 50000:
        return round(price / tick_sizes["under_50000"]) * tick_sizes["under_50000"]
    elif price < 100000:
        return round(price / tick_sizes["under_100000"]) * tick_sizes["under_100000"]
    elif price < 500000:
        return round(price / tick_sizes["under_500000"]) * tick_sizes["under_500000"]
    else:
        return round(price / tick_sizes["over_500000"]) * tick_sizes["over_500000"]

def calculate_commission(price: float, qty: float, config: Dict[str, Any]) -> float:
    """수수료를 계산합니다."""
    commission = max(price * qty * config["commission_rate"], config["min_commission"])
    tax = price * qty * config["tax_rate"]
    return commission + tax

def is_trading_time(dt: datetime, trading_hours: Dict[str, str]) -> bool:
    """거래시간인지 확인합니다."""
    current_time = dt.time()

    # 정규장 시간
    morning_start = datetime.strptime(trading_hours["morning_start"], "%H:%M").time()
    morning_end = datetime.strptime(trading_hours["morning_end"], "%H:%M").time()
    afternoon_start = datetime.strptime(trading_hours["afternoon_start"], "%H:%M").time()
    afternoon_end = datetime.strptime(trading_hours["afternoon_end"], "%H:%M").time()

    # 동시호가 시간
    pre_market_start = datetime.strptime(trading_hours["pre_market_start"], "%H:%M").time()
    pre_market_end = datetime.strptime(trading_hours["pre_market_end"], "%H:%M").time()
    post_market_start = datetime.strptime(trading_hours["post_market_start"], "%H:%M").time()
    post_market_end = datetime.strptime(trading_hours["post_market_end"], "%H:%M").time()

    # 정규장 시간 확인
    if (morning_start <= current_time <= morning_end or :
        afternoon_start <= current_time <= afternoon_end):
        return True

    # 동시호가 시간 확인
    if (pre_market_start <= current_time <= pre_market_end or:
        post_market_start <= current_time <= post_market_end):
        return True

    return False

def calculate_drawdown(cumulative_returns: pd.Series) -> Tuple[float, float]:
    """최대 드로우다운을 계산합니다."""
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max.abs()
    max_drawdown = drawdown.min()
    avg_drawdown = drawdown.mean()
    return max_drawdown, avg_drawdown

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """샤프 비율을 계산합니다."""
    excess_returns = returns - risk_free_rate / 252
    if returns.std() == 0:
        return 0
    return excess_returns.mean() / returns.std() * np.sqrt(252)

def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """소르티노 비율을 계산합니다."""
    excess_returns = returns - risk_free_rate / 252
    negative_returns = returns[returns < 0]
    downside_deviation = negative_returns.std() if len(negative_returns) > 0 else 0
    if downside_deviation == 0:
        return 0
    return excess_returns.mean() / downside_deviation * np.sqrt(252)

def calculate_calmar_ratio(total_return: float, max_drawdown: float) -> float:
    """칼마 비율을 계산합니다."""
    if max_drawdown == 0:
        return 0
    return total_return / abs(max_drawdown)

def calculate_omega_ratio(returns: pd.Series, threshold: float = 0) -> float:
    """오메가 비율을 계산합니다."""
    positive_returns = returns[returns > threshold].sum()
    negative_returns_sum = abs(returns[returns < threshold].sum())
    if negative_returns_sum == 0:
        return np.nan
    return positive_returns / negative_returns_sum

def calculate_var_cvar(returns: pd.Series, confidence_level: float = 0.95) -> Tuple[float, float]:
    """VaR와 CVaR를 계산합니다."""
    var = np.percentile(returns, (1 - confidence_level) * 100)
    cvar = returns[returns <= var].mean()
    return var, cvar

def format_percentage(value: float) -> str:
    """백분율로 포맷합니다."""
    return f"{value:.2%}"

def format_currency(value: float) -> str:
    """통화로 포맷합니다."""
    return f"₩{value:,.0f}"

def generate_report_filename(prefix: str = "backtest_report") -> str:
    """보고서 파일명을 생성합니다."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.json"
