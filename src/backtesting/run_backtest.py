#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: src/backtesting/run_backtest.py
"""
from __future__ import annotations
import json
import logging
import time
from typing import Dict, Any, Optional

from .core.engine import BacktestEngine
from .core.data_handler import DataHandler
from .core.execution_handler import ExecutionHandler
from .core.strategy import Strategy
from .analysis.performance_analyzer import PerformanceAnalyzer
from .analysis.statistical_tester import StatisticalTester
from .analysis.stress_tester import StressTester
from .analysis.walkforward_analyzer import WalkForwardAnalyzer
from .analysis.montecarlo_analyzer import MonteCarloAnalyzer
from .utils.config_manager import get_config
from .reporting.report_generator import ReportGenerator

def setup_logging(config: Dict[str, Any]):
    log_config = config.get("logging", {})
    logging.basicConfig(
        level=log_config.get("level", "INFO"),
        format=log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
        handlers=[
            logging.FileHandler(log_config.get("file", "logs/backtest.log"), encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

def main(config_path: Optional[str] = None):
    config = get_config(config_path)
    setup_logging(config)
    logger = logging.getLogger(__name__)

    logger.info("🚀 백테스팅 시스템 시작")

    try:
        data_handler = DataHandler(config["data_path"], config)
        execution_handler = ExecutionHandler(config)
        strategy = Strategy(config)
        performance_analyzer = PerformanceAnalyzer(config)
        stat_tester = StatisticalTester(config)
        stress_tester = StressTester(config)
        walkforward_analyzer = WalkForwardAnalyzer(config)
        montecarlo_analyzer = MonteCarloAnalyzer(config)

        engine = BacktestEngine(
            data_handler, execution_handler, strategy,
            performance_analyzer, stat_tester, stress_tester,
            walkforward_analyzer, montecarlo_analyzer, config
        )

        start_time = time.time()
        result = engine.run()
        logger.info(f"⏱️ 백테스팅 완료 (소요시간: {time.time() - start_time:.2f}초)")

        report_generator = ReportGenerator(config)
        summary = report_generator.generate_full_report(result)
        print(summary)

    except Exception as e:
        logger.error(f"❌ 백테스팅 실행 중 오류 발생: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
