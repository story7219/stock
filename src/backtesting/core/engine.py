#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: src/backtesting/core/engine.py
"""

from __future__ import annotations
import logging
from typing import Any, Dict
from collections import deque

from .data_handler import DataHandler
from .execution_handler import ExecutionHandler
from .strategy import Strategy
from ..analysis.performance_analyzer import PerformanceAnalyzer
from ..analysis.statistical_tester import StatisticalTester
from ..analysis.stress_tester import StressTester
from ..analysis.walkforward_analyzer import WalkForwardAnalyzer
from ..analysis.montecarlo_analyzer import MonteCarloAnalyzer
from ..events import MarketEvent, SignalEvent, OrderEvent, FillEvent

logger = logging.getLogger(__name__)

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
        montecarlo_analyzer: MonteCarloAnalyzer,
        config: Dict[str, Any],
    ):
        self.data_handler = data_handler
        self.execution_handler = execution_handler
        self.strategy = strategy
        self.performance_analyzer = performance_analyzer
        self.stat_tester = stat_tester
        self.stress_tester = stress_tester
        self.walkforward_analyzer = walkforward_analyzer
        self.montecarlo_analyzer = montecarlo_analyzer
        self.config = config
        self.events = deque()

    def run(self) -> Dict[str, Any]:
        logger.info("=== [1/5] 기본 백테스트 시작 ===")
        base_result = self._run_base_backtest()

        # TODO: Refactor these analyzers to use the new event-driven engine
        logger.info("=== [2/5] Walk-Forward Analysis 시작 ===")
        walk_result = {} # self.walkforward_analyzer.run(self.data_handler, self.strategy, self.execution_handler)
        logger.info("=== [3/5] Monte Carlo Simulation 시작 ===")
        mc_result = {} # self.montecarlo_analyzer.run(self.data_handler, self.strategy, self.execution_handler)
        logger.info("=== [4/5] Stress Test 시작 ===")
        stress_result = {} # self.stress_tester.run(self.data_handler, self.strategy, self.execution_handler)

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

        # Start by loading historical data bars
        self.data_handler.load_events_into_queue(self.events)

        while True:
            try:
                event = self.events.popleft()
            except IndexError:
                # No more events to process
                break
            else:
                if event:
                    if isinstance(event, MarketEvent):
                        self.strategy.on_market_event(event, self.events)

                    elif isinstance(event, SignalEvent):
                        self.execution_handler.on_signal_event(event, self.events)

                    elif isinstance(event, OrderEvent):
                        self.execution_handler.on_order_event(event, self.events)

                    elif isinstance(event, FillEvent):
                        self.performance_analyzer.on_fill_event(event) # TODO: Update performance analyzer

        return self.performance_analyzer.analyze(self.execution_handler.get_trades())
