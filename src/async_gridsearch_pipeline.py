#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: async_gridsearch_pipeline.py
목적: 초고속 비동기/병렬 백테스트/ML/DL 최적화 파이프라인 (Cursor Rule 100%)
"""

from __future__ import annotations
import asyncio
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List
from pathlib import Path
import pandas as pd
import numpy as np
import diskcache
import logging
from tqdm.auto import tqdm
from dataclasses import dataclass, asdict
import time

# 캐시/체크포인트 디렉토리
CACHE_DIR = Path("cache/async_gridsearch")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
cache = diskcache.Cache(str(CACHE_DIR))

# 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

@dataclass
class ParamSet:
    market: str
    w_short: float
    w_swing: float
    w_mid: float
    w_cash: float
    stop: float
    take: float
    ml_n_estimators: int
    ml_max_depth: int
    dl_layers: int
    dl_neurons: int
    dl_lr: float
    dl_batch_size: int

@dataclass
class Result:
    params: ParamSet
    total_return: float
    sharpe: float
    mdd: float
    win_rate: float

# 실전 신호/ML/DL/성과지표 연결부 (여기선 더미)
def run_single_backtest(params: ParamSet) -> Result:
    """단일 파라미터 조합에 대한 백테스트/ML/DL 실행 (실전 로직 연결)"""
    cache_key = str(asdict(params))
    if cache_key in cache:
        return cache[cache_key]
    # --- 실제 백테스트/ML/DL 로직 연결 (여기선 더미) ---
    np.random.seed(hash(cache_key) % 2**32)
    total_return = np.random.uniform(-0.2, 0.5)
    sharpe = np.random.uniform(0.5, 2.0)
    mdd = np.random.uniform(0.05, 0.2)
    win_rate = np.random.uniform(0.4, 0.8)
    result = Result(params, total_return, sharpe, mdd, win_rate)
    cache[cache_key] = result
    return result

async def async_gridsearch(param_grid: List[ParamSet], max_workers: int = 8) -> List[Result]:
    """비동기+병렬 그리드서치"""
    loop = asyncio.get_running_loop()
    results: List[Result] = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        tasks = [
            loop.run_in_executor(executor, run_single_backtest, params)
            for params in param_grid
        ]
        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="GridSearch"):
            res = await f
            results.append(res)
    return results

def build_param_grid() -> List[ParamSet]:
    """파라미터 그리드 생성 (범위/조합 최대치)"""
    market_conditions = ['bull', 'bear', 'sideways']
    weight_grid = np.arange(0, 1.1, 0.2)
    stop_grid = np.arange(0.01, 0.11, 0.03)
    take_grid = np.arange(0.05, 0.31, 0.05)
    ml_n_estimators = [50, 100, 200]
    ml_max_depth = [3, 5, 7]
    dl_layers = [1, 2, 3]
    dl_neurons = [32, 64, 128]
    dl_lr = [0.001, 0.01]
    dl_batch_size = [16, 32]
    grid = []
    for market in market_conditions:
        for w_short in weight_grid:
            for w_swing in weight_grid:
                for w_mid in weight_grid:
                    w_cash = 1.0 - (w_short + w_swing + w_mid)
                    if w_cash < 0 or w_cash > 1: continue
                    for stop in stop_grid:
                        for take in take_grid:
                            for n_est in ml_n_estimators:
                                for max_depth in ml_max_depth:
                                    for layers in dl_layers:
                                        for neurons in dl_neurons:
                                            for lr in dl_lr:
                                                for bs in dl_batch_size:
                                                    grid.append(ParamSet(
                                                        market, w_short, w_swing, w_mid, w_cash,
                                                        stop, take, n_est, max_depth, layers, neurons, lr, bs
                                                    ))
    return grid

async def main():
    logger.info("파라미터 그리드 생성 중...")
    param_grid = build_param_grid()
    logger.info(f"총 조합 수: {len(param_grid):,}")
    logger.info("비동기 병렬 그리드서치 시작")
    t0 = time.time()
    results = await async_gridsearch(param_grid, max_workers=16)
    t1 = time.time()
    logger.info(f"그리드서치 완료 (소요시간: {t1-t0:.1f}초)")
    # 결과 저장 (Parquet)
    df = pd.DataFrame([asdict(r.params) | asdict(r) for r in results])
    save_path = Path("reports/backtest_results/async_gridsearch_results.parquet")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(save_path, index=False)
    logger.info(f"결과 저장 완료: {save_path} (행: {len(df)})")

if __name__ == "__main__":
    asyncio.run(main()) 