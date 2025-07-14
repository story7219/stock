#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: src/backtesting/analysis/performance_analyzer.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Any
import Dict, List
from ..utils.helpers import (
    calculate_sharpe_ratio, calculate_sortino_ratio, calculate_calmar_ratio,
    calculate_omega_ratio, calculate_drawdown, calculate_var_cvar
)

class PerformanceAnalyzer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def analyze(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not trades:
            return self._empty_results()

        df = pd.DataFrame(trades)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime')

        total_pnl = df["pnl"].sum()
        returns = df["pnl"] / (df["price"] * df["qty"]).abs().shift(1).fillna(1)

        max_dd, avg_dd = calculate_drawdown(df["pnl"].cumsum())

        return {
            "total_return": total_pnl / self.config.get("initial_capital", 1_000_000),
            "sharpe_ratio": calculate_sharpe_ratio(returns),
            "sortino_ratio": calculate_sortino_ratio(returns),
            "calmar_ratio": calculate_calmar_ratio(total_pnl, max_dd),
            "omega_ratio": calculate_omega_ratio(returns),
            "max_drawdown": max_dd,
            "avg_drawdown": avg_dd,
            "win_rate": (df["pnl"] > 0).mean(),
            "profit_factor": df[df["pnl"] > 0]["pnl"].sum() / abs(df[df["pnl"] < 0]["pnl"].sum()),
            "volatility": returns.std() * np.sqrt(252),
            "var_95": calculate_var_cvar(returns)[0],
            "cvar_95": calculate_var_cvar(returns)[1],
            "trade_count": len(df),
        }

    def _empty_results(self) -> Dict[str, Any]:
        return {k: 0.0 for k in [
            "total_return", "sharpe_ratio", "sortino_ratio", "calmar_ratio", "omega_ratio",
            "max_drawdown", "avg_drawdown", "win_rate", "profit_factor", "volatility",
            "var_95", "cvar_95", "trade_count"
        ]}
