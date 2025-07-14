#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: src/backtesting/utils/helpers.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict
import Any, Tuple

def apply_tick_size(price: float, tick_sizes: Dict[str, int]) -> float:
    # (이전과 동일)
    if price < 1000: return round(price / tick_sizes["under_1000"]) * tick_sizes["under_1000"]
    if price < 5000: return round(price / tick_sizes["under_5000"]) * tick_sizes["under_5000"]
    # ...
    return round(price / tick_sizes["over_500000"]) * tick_sizes["over_500000"]

def calculate_commission(price: float, qty: float, config: Dict[str, Any], side: str, dt: datetime) -> float:
    """
    Calculates the total commission and tax for a trade, considering Korean market specifics.
    """
    # 1. Commission
    commission = max(price * qty * config["commission_rate"], config["min_commission"])

    # 2. Taxes (only on 'sell' side)
    tax = 0.0
    if side == 'sell':
        tax_config = config.get("tax_rules", {})

        # Determine which tax rate to use based on the date
        effective_rate = 0.0
        for rule in sorted(tax_config, key=lambda x: x['effective_date'], reverse=True):
            if dt >= datetime.strptime(rule['effective_date'], "%Y-%m-%d"):
                effective_rate = rule['rate']
                break

        tax = price * qty * effective_rate

    return commission + tax

def calculate_drawdown(cumulative_returns: pd.Series) -> Tuple[float, float]:
    if cumulative_returns.empty: return 0.0, 0.0
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max.replace(0, 1)
    return drawdown.min(), drawdown.mean()

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    if returns.std() == 0: return 0.0
    excess_returns = returns - risk_free_rate / 252
    return excess_returns.mean() / returns.std() * np.sqrt(252)

def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    excess_returns = returns - risk_free_rate / 252
    downside_returns = excess_returns[excess_returns < 0]
    if downside_returns.std() == 0: return 0.0
    return excess_returns.mean() / downside_returns.std() * np.sqrt(252)

def calculate_calmar_ratio(total_return: float, max_drawdown: float) -> float:
    return total_return / abs(max_drawdown) if max_drawdown != 0 else 0.0

def calculate_omega_ratio(returns: pd.Series, threshold: float = 0) -> float:
    positive_returns = returns[returns > threshold].sum()
    negative_returns_sum = abs(returns[returns < threshold].sum())
    return positive_returns / negative_returns_sum if negative_returns_sum != 0 else np.inf

def calculate_var_cvar(returns: pd.Series, confidence_level: float = 0.95) -> Tuple[float, float]:
    if returns.empty: return 0.0, 0.0
    var = np.percentile(returns, (1 - confidence_level) * 100)
    cvar = returns[returns <= var].mean()
    return var, cvar

def generate_report_filename(prefix: str = "backtest_report") -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"results/{prefix}_{timestamp}.json"
