#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: engine.py
모듈: 백테스팅 메인 엔진
목적: 현실적 거래환경, 다차원 검증, 통계적 유의성, 위기상황, 실전 시뮬레이션 통합

Author: WorldClassAI
Created: 2025-07-12
Version: 1.0.0
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import datetime
import random
import logging
from typing import Any, Dict, List, Optional, Tuple, Callable, Union, Literal
from .data_handler import DataHandler
from .execution_handler import ExecutionHandler
from .strategy import Strategy
from .performance import PerformanceAnalyzer
from .stat_test import StatisticalTester
from .stress import StressTester
from .walkforward import WalkForwardAnalyzer
from .montecarlo import MonteCarloSimulator

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class BacktestEngine:
    def __init__(:
        self,:
        data_handler: DataHandler,
        execution_handler: ExecutionHandler,
        strategy: Strategy,
        performance_analyzer: PerformanceAnalyzer,
        stat_tester: StatisticalTester,
        stress_tester: StressTester,
        walkforward_analyzer: WalkForwardAnalyzer,
        montecarlo_simulator: MonteCarloSimulator,
        config: Dict[str, Any],
    ):
        self.data_handler = data_handler
        self.execution_handler = execution_handler
        self.strategy = strategy
        self.performance_analyzer = performance_analyzer
        self.stat_tester = stat_tester
        self.stress_tester = stress_tester
        self.walkforward_analyzer = walkforward_analyzer
        self.montecarlo_simulator = montecarlo_simulator
        self.config = config

    def run(self) -> Dict[str, Any]:
        logger.info("=== [1/5] 기본 백테스트 시작 ===")
        base_result = self._run_base_backtest()
        logger.info("=== [2/5] Walk-Forward Analysis 시작 ===")
        walk_result = self.walkforward_analyzer.run(self.data_handler, self.strategy, self.execution_handler)
        logger.info("=== [3/5] Monte Carlo Simulation 시작 ===")
        mc_result = self.montecarlo_simulator.run(self.data_handler, self.strategy, self.execution_handler)
        logger.info("=== [4/5] Stress Test 시작 ===")
        stress_result = self.stress_tester.run(self.data_handler, self.strategy, self.execution_handler)
        logger.info("=== [5/5] 통계적 유의성 검증 시작 ===")
        stat_result = self.stat_tester.run(base_result, walk_result, mc_result, stress_result)
        logger.info("=== [완료] 전체 백테스트 파이프라인 종료 ===")
        return {
            "base": base_result,
            "walkforward": walk_result,
            "montecarlo": mc_result,
            "stress": stress_result,
            "stat": stat_result,
        }

    def _run_base_backtest(self) -> Dict[str, Any]:
        self.data_handler.reset()
        self.execution_handler.reset()
        self.strategy.reset()
        while not self.data_handler.is_end():
            market_event = self.data_handler.get_next_event()
            signal = self.strategy.generate_signal(market_event)
            order = self.execution_handler.create_order(signal, market_event)
            self.execution_handler.execute_order(order, market_event)
        return self.performance_analyzer.analyze(self.execution_handler.get_trades())
