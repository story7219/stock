#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: src/backtesting/analysis/stress_tester.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Any
import Dict

class StressTester:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def run(self, data_handler, strategy, execution_handler) -> Dict[str, Any]:
        scenarios = self.config.get("stress_scenarios", {})
        results = {}

        for name, params in scenarios.items():
            original_data = data_handler.data.copy()
            stressed_data = self._apply_stress(original_data, params)
            data_handler.data = stressed_data

            # Run backtest on stressed data
            data_handler.reset()
            strategy.reset()
            execution_handler.reset()
            while not data_handler.is_end():
                event = data_handler.get_next_event()
                if not event: continue
                signal = strategy.generate_signal(event)
                order = execution_handler.create_order(signal, event)
                execution_handler.execute_order(order, event)

            trades = execution_handler.get_trades()
            pnl = sum(t['pnl'] for t in trades)
            results[name] = {"total_pnl": pnl, "survived": pnl > -0.5 * self.config.get("initial_capital", 1e6)}

            data_handler.data = original_data # Restore original data

        return {
            "average_pnl": np.mean([r["total_pnl"] for r in results.values()]),
            "survival_rate": np.mean([r["survived"] for r in results.values()]),
            "scenario_results": results
        }

    def _apply_stress(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        stressed = data.copy()
        stressed['price'] *= (1 + params.get('price_shock', 0))
        stressed['volume'] *= params.get('volume_shock', 1)
        return stressed
