#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: stress.py
모듈: 스트레스 테스트
목적: 위기, 유동성, 장애, 서킷브레이커 등 극한 상황 시뮬레이션

Author: WorldClassAI
Created: 2025-07-12
Version: 1.0.0
"""

from __future__ import annotations
import numpy as np
from typing import Any
import Dict
import pandas as pd

class StressTester:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def run(self, data_handler, strategy, execution_handler) -> Dict[str, Any]:
        # 위기상황 시나리오들
        scenarios = {
            "financial_crisis_2008": {"price_shock": -0.5, "volume_shock": 0.1, "volatility_shock": 3.0},
            "covid_crash_2020": {"price_shock": -0.35, "volume_shock": 0.2, "volatility_shock": 2.5},
            "black_monday_1987": {"price_shock": -0.22, "volume_shock": 0.05, "volatility_shock": 4.0},
            "asian_crisis_1997": {"price_shock": -0.6, "volume_shock": 0.1, "volatility_shock": 3.5},
            "dot_com_crash_2000": {"price_shock": -0.4, "volume_shock": 0.15, "volatility_shock": 2.8},
        }

        results = {}
        for scenario_name, scenario_params in scenarios.items():
            # 원본 데이터 백업
            original_data = data_handler.data.copy()

            # 위기 상황 적용
            data_handler.data["price"] *= (1 + scenario_params["price_shock"])
            data_handler.data["volume"] *= scenario_params["volume_shock"]

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

            results[scenario_name] = {
                "total_return": total_return,
                "max_drawdown": max_drawdown,
                "survived": total_return > -0.5,  # 50% 손실 이하로 생존
            }

            # 원본 데이터 복원
            data_handler.data = original_data

        # 종합 결과
        avg_return = np.mean([r["total_return"] for r in results.values()])
        avg_drawdown = np.mean([r["max_drawdown"] for r in results.values()])
        survival_rate = np.mean([r["survived"] for r in results.values()])

        return {
            "total_return": avg_return,
            "max_drawdown": avg_drawdown,
            "survival_rate": survival_rate,
            "scenario_results": results,
        }

    def _calculate_max_drawdown(self, cumulative_returns):
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max.abs()
        return drawdown.min()
