#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: montecarlo.py
모듈: 몬테카를로 시뮬레이션
목적: 1000회 부트스트랩, 신뢰구간, 랜덤 시나리오

Author: WorldClassAI
Created: 2025-07-12
Version: 1.0.0
"""

from __future__ import annotations
import numpy as np
from typing import Any
import Dict
import pandas as pd

class MonteCarloSimulator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def run(self, data_handler, strategy, execution_handler) -> Dict[str, Any]:
        n_simulations = self.config.get("n_simulations", 1000)

        # 원본 데이터 백업
        original_data = data_handler.data.copy()

        results = []

        for i in range(n_simulations):
            # 랜덤 시드 설정
            np.random.seed(i)

            # 데이터 순서 랜덤화 (부트스트랩)
            shuffled_indices = np.random.permutation(len(original_data))
            data_handler.data = original_data.iloc[shuffled_indices].reset_index(drop=True)

            # 백테스트 실행
            data_handler.reset()
            execution_handler.reset()
            strategy.reset()

            while not data_handler.is_end():
                market_event = data_handler.get_next_event()
                signal = strategy.generate_signal(market_event)
                order = execution_handler.create_order(signal, market_event)
                execution_handler.execute_order(order, market_event)

            # 결과 수집
            trades = execution_handler.get_trades()
            if trades:
                df = pd.DataFrame(trades)
                df["pnl"] = np.where(df["side"] == "buy", -df["price"] * df["qty"] - df["fee"], df["price"] * df["qty"] - df["fee"])
                total_return = df["pnl"].sum()
                max_drawdown = self._calculate_max_drawdown(df["pnl"].cumsum())
            else:
                total_return = 0
                max_drawdown = 0

            results.append({
                "simulation": i,
                "total_return": total_return,
                "max_drawdown": max_drawdown,
            })

        # 원본 데이터 복원
        data_handler.data = original_data

        # 통계 분석
        returns = [r["total_return"] for r in results]
        drawdowns = [r["max_drawdown"] for r in results]

        # 신뢰구간 계산
        confidence_level = 0.95
        alpha = 1 - confidence_level

        return_ci_lower = np.percentile(returns, alpha/2 * 100)
        return_ci_upper = np.percentile(returns, (1-alpha/2) * 100)

        drawdown_ci_lower = np.percentile(drawdowns, alpha/2 * 100)
        drawdown_ci_upper = np.percentile(drawdowns, (1-alpha/2) * 100)

        return {
            "total_return": np.mean(returns),
            "max_drawdown": np.mean(drawdowns),
            "return_std": np.std(returns),
            "drawdown_std": np.std(drawdowns),
            "return_ci": (return_ci_lower, return_ci_upper),
            "drawdown_ci": (drawdown_ci_lower, drawdown_ci_upper),
            "positive_probability": np.mean(np.array(returns) > 0),
            "n_simulations": n_simulations,
            "detailed_results": results,
        }

    def _calculate_max_drawdown(self, cumulative_returns):
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max.abs()
        return drawdown.min()
