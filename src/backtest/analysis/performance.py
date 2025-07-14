#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: performance.py
모듈: 성과 분석
목적: 수익률, 리스크, 드로우다운, 승률, 손익비, 회복기간 등 분석

Author: WorldClassAI
Created: 2025-07-12
Version: 1.0.0
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Any
import Dict, List

class PerformanceAnalyzer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def analyze(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not trades:
            return {
                "total_return": 0.0,
                "sharpe": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "calmar_ratio": 0.0,
                "sortino_ratio": 0.0,
                "omega_ratio": 0.0,
                "recovery_time": 0,
                "volatility": 0.0,
                "var_95": 0.0,
                "cvar_95": 0.0,
            }

        df = pd.DataFrame(trades)
        df["pnl"] = np.where(df["side"] == "buy", -df["price"] * df["qty"] - df["fee"], df["price"] * df["qty"] - df["fee"])
        df["cum_pnl"] = df["pnl"].cumsum()

        # 수익률 계산
        total_return = df["cum_pnl"].iloc[-1] if not df.empty else 0
        returns = df["pnl"] / (df["price"] * df["qty"]).abs()
        returns = returns.replace([np.inf, -np.inf], 0)

        # 샤프 비율
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

        # 최대 드로우다운
        cumulative = df["cum_pnl"]
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max.abs()
        max_drawdown = drawdown.min()

        # 승률
        win_rate = (df["pnl"] > 0).mean()

        # 손익비
        profit_factor = df[df["pnl"] > 0]["pnl"].sum() / abs(df[df["pnl"] < 0]["pnl"].sum()) if abs(df[df["pnl"] < 0]["pnl"].sum()) > 0 else np.nan

        # 칼마 비율
        calmar_ratio = total_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # 소르티노 비율
        negative_returns = returns[returns < 0]
        downside_deviation = negative_returns.std() if len(negative_returns) > 0 else 0
        sortino_ratio = returns.mean() / downside_deviation * np.sqrt(252) if downside_deviation > 0 else 0

        # 오메가 비율
        threshold = 0
        positive_returns = returns[returns > threshold].sum()
        negative_returns_sum = abs(returns[returns < threshold].sum())
        omega_ratio = positive_returns / negative_returns_sum if negative_returns_sum > 0 else np.nan

        # 변동성
        volatility = returns.std() * np.sqrt(252)

        # VaR (Value at Risk)
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean()

        # 회복 기간 (간단한 추정)
        recovery_time = len(df[df["cum_pnl"] < 0]) if len(df[df["cum_pnl"] < 0]) > 0 else 0

        return {
            "total_return": total_return,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "calmar_ratio": calmar_ratio,
            "sortino_ratio": sortino_ratio,
            "omega_ratio": omega_ratio,
            "recovery_time": recovery_time,
            "volatility": volatility,
            "var_95": var_95,
            "cvar_95": cvar_95,
        }
