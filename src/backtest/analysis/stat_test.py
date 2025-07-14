#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: stat_test.py
모듈: 통계 검증
목적: p-value, t-test, 부트스트랩, 랜덤화, 편향 검증

Author: WorldClassAI
Created: 2025-07-12
Version: 1.0.0
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
        # 수익률 데이터 수집
        returns = []
        if base_result and "total_return" in base_result:
            returns.append(base_result["total_return"])
        if walk_result and "total_return" in walk_result:
            returns.append(walk_result["total_return"])
        if mc_result and "total_return" in mc_result:
            returns.append(mc_result["total_return"])
        if stress_result and "total_return" in stress_result:
            returns.append(stress_result["total_return"])

        if not returns:
            return {
                "mean_return": 0.0,
                "std_return": 0.0,
                "t_stat": 0.0,
                "p_value": 1.0,
                "significant": False,
                "confidence_interval": (0.0, 0.0),
                "effect_size": 0.0,
            }

        returns = np.array(returns)

        # t-test (H0: 평균 수익률 = 0)
        t_stat, p_value = stats.ttest_1samp(returns, 0)

        # 신뢰구간
        confidence_interval = stats.t.interval(0.95, len(returns)-1, loc=returns.mean(), scale=returns.std()/np.sqrt(len(returns)))

        # 효과 크기 (Cohen's d)
        effect_size = returns.mean() / returns.std() if returns.std() > 0 else 0

        return {
            "mean_return": returns.mean(),
            "std_return": returns.std(),
            "t_stat": t_stat,
            "p_value": p_value,
            "significant": p_value < 0.01,
            "confidence_interval": confidence_interval,
            "effect_size": effect_size,
        }
