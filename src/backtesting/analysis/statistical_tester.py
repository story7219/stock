#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: src/backtesting/analysis/statistical_tester.py
"""
from __future__ import annotations
import numpy as np
from scipy import stats
from typing import Any
import Dict

class StatisticalTester:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def run(self, base_result, walk_result, mc_result, stress_result) -> Dict[str, Any]:
        returns = [
            res["total_return"] for res in [base_result, walk_result, mc_result, stress_result] if res
        ]
        if not returns:
            return {"p_value": 1.0, "significant": False, "effect_size": 0.0}

        returns_array = np.array(returns)
        t_stat, p_value = stats.ttest_1samp(returns_array, 0)
        effect_size = returns_array.mean() / returns_array.std() if returns_array.std() > 0 else 0

        return {
            "p_value": p_value,
            "significant": p_value < self.config.get("statistical_thresholds", {}).get("significance_level", 0.01),
            "effect_size": effect_size,
        }
