#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: src/backtesting/analysis/walkforward_analyzer.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Any
import Dict, List

class WalkForwardAnalyzer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def run(self, data_handler, strategy, execution_handler) -> Dict[str, Any]:
        train_window = self.config.get("train_window", 24) * 30 # 월 -> 일
        test_window = self.config.get("test_window", 3) * 30
        embargo = self.config.get("embargo", 1) * 30

        full_data = data_handler.data
        results: List[Dict[str, Any]] = []

        for i in range(0, len(full_data) - train_window - test_window - embargo, test_window):
            train_data = full_data.iloc[i : i + train_window]
            test_data = full_data.iloc[i + train_window + embargo : i + train_window + embargo + test_window]

            # 1. Train/Optimize strategy on train_data (simplified)
            optimized_params = self._optimize_strategy(train_data)

            # 2. Test strategy on test_data
            data_handler.data = test_data # Temporarily set test data
            data_handler.reset()
            strategy.reset()
            execution_handler.reset()

            while not data_handler.is_end():
                event = data_handler.get_next_event()
                if not event: continue
                signal = strategy.generate_signal(event) # Should use optimized_params
                order = execution_handler.create_order(signal, event)
                execution_handler.execute_order(order, event)

            trades = execution_handler.get_trades()
            pnl = sum(t['pnl'] for t in trades)
            results.append({"pnl": pnl})

        data_handler.data = full_data # Restore full data

        if not results:
            return {"average_pnl": 0, "consistency": 0}

        pnls = [r['pnl'] for r in results]
        return {"average_pnl": np.mean(pnls), "consistency": np.std(pnls)}

    def _optimize_strategy(self, train_data: pd.DataFrame) -> Dict[str, Any]:
        # In a real scenario, you'd optimize strategy parameters here
        return {"ma_period": 10} # Example parameter
