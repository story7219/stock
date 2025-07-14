#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: src/backtesting/analysis/montecarlo_analyzer.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Any
import Dict, List

class MonteCarloAnalyzer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def run(self, data_handler, strategy, execution_handler) -> Dict[str, Any]:
        n_simulations = self.config.get("n_simulations", 1000)
        original_data = data_handler.data.copy()
        results: List[Dict[str, Any]] = []

        for i in range(n_simulations):
            # Bootstrap resampling of daily returns
            daily_returns = original_data['price'].pct_change().dropna()
            simulated_returns = daily_returns.sample(n=len(original_data), replace=True)

            simulated_prices = (1 + simulated_returns).cumprod() * original_data['price'].iloc[0]

            simulated_data = original_data.copy()
            simulated_data['price'] = simulated_prices
            simulated_data = simulated_data.fillna(method='ffill')

            data_handler.data = simulated_data
            data_handler.reset()
            strategy.reset()
            execution_handler.reset()

            while not data_handler.is_end():
                event = data_handler.get_next_event()
                if not event: continue
                signal = strategy.generate_signal(event)
                order = execution_handler.create_order(signal, event)
                execution_handler.execute_order(order, event)

            pnl = sum(t['pnl'] for t in execution_handler.get_trades())
            results.append({"pnl": pnl})

        data_handler.data = original_data # Restore

        if not results:
            return {"average_pnl": 0, "pnl_std": 0, "positive_prob": 0}

        pnls = [r['pnl'] for r in results]
        return {
            "average_pnl": np.mean(pnls),
            "pnl_std": np.std(pnls),
            "positive_prob": np.mean(np.array(pnls) > 0)
        }
