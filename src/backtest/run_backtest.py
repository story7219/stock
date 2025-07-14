#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: run_backtest.py
목적: 완전 자동화 백테스팅 마스터 시스템 실행

Author: WorldClassAI
Created: 2025-07-12
Version: 1.0.0
"""

from __future__ import annotations
import json
import logging
import time
from datetime import datetime
from typing import Dict
import Any

from .engine import BacktestEngine
from .data_handler import DataHandler
from .execution_handler import ExecutionHandler
from .strategy import Strategy
from .performance import PerformanceAnalyzer
from .stat_test import StatisticalTester
from .stress import StressTester
from .walkforward import WalkForwardAnalyzer
from .montecarlo import MonteCarloSimulator
from .config import get_config
from .report import ReportGenerator

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backtest_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """완전 자동화 백테스팅 시스템 메인 실행"""

    # 설정 로드
    config = get_config()

    logger.info("🚀 완전 자동화 백테스팅 시스템 시작")
    logger.info(f"설정: {config}")

    try:
        # 컴포넌트 초기화
        logger.info("📦 백테스팅 컴포넌트 초기화 중...")

        data_handler = DataHandler(config["data_path"], config)
        execution_handler = ExecutionHandler(config)
        strategy = Strategy(config)
        performance_analyzer = PerformanceAnalyzer(config)
        stat_tester = StatisticalTester(config)
        stress_tester = StressTester(config)
        walkforward_analyzer = WalkForwardAnalyzer(config)
        montecarlo_simulator = MonteCarloSimulator(config)

        # 백테스팅 엔진 생성
        engine = BacktestEngine(
            data_handler=data_handler,
            execution_handler=execution_handler,
            strategy=strategy,
            performance_analyzer=performance_analyzer,
            stat_tester=stat_tester,
            stress_tester=stress_tester,
            walkforward_analyzer=walkforward_analyzer,
            montecarlo_simulator=montecarlo_simulator,
            config=config,
        )

        # 백테스팅 실행
        logger.info("🔬 백테스팅 파이프라인 실행 시작")
        start_time = time.time()

        result = engine.run()

        execution_time = time.time() - start_time
        logger.info(f"⏱️ 백테스팅 완료 (소요시간: {execution_time:.2f}초)")

        # 리포트 생성
        logger.info("📊 백테스팅 리포트 생성 중...")
        report_generator = ReportGenerator(config)
        full_report = report_generator.generate_full_report(result)

        # 요약 결과 출력
        print("\n" + "="*80)
        print("🏆 완전 자동화 백테스팅 시스템 결과 요약")
        print("="*80)

        if "base" in result and result["base"]:
            base = result["base"]
            print(f"📈 기본 백테스트:")
            print(f"   총 수익률: {base.get('total_return', 0):.2%}")
            print(f"   샤프 비율: {base.get('sharpe', 0):.2f}")
            print(f"   최대 낙폭: {base.get('max_drawdown', 0):.2%}")
            print(f"   승률: {base.get('win_rate', 0):.2%}")

        if "stat" in result and result["stat"]:
            stat = result["stat"]
            print(f"\n🔬 통계적 검증:")
            print(f"   p-value: {stat.get('p_value', 1):.4f}")
            print(f"   통계적 유의성: {'✅' if stat.get('significant', False) else '❌'}")
            print(f"   효과 크기: {stat.get('effect_size', 0):.2f}")

        if "stress" in result and result["stress"]:
            stress = result["stress"]
            print(f"\n🛡️ 스트레스 테스트:")
            print(f"   평균 수익률: {stress.get('total_return', 0):.2%}")
            print(f"   생존률: {stress.get('survival_rate', 0):.2%}")

        print("\n" + "="*80)
        print("✅ 백테스팅 시스템 실행 완료!")
        print("📄 상세 보고서: backtest_report_*.json")
        print("="*80)

        return result

    except Exception as e:
        logger.error(f"❌ 백테스팅 시스템 실행 중 오류 발생: {e}")
        raise

if __name__ == "__main__":
    main()
