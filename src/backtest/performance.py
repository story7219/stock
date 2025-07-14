#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: performance.py
모듈: 성능 분석
목적: 자동매매 백테스트 성능지표(수익률, 샤프, MDD 등) 분석

Author: World-Class Python Engineer
Created: 2025-07-13
Version: 1.0.0
"""
from __future__ import annotations
from typing import Any, Dict
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class PerformanceAnalyzer:
    """성능 분석 담당 클래스"""
    def __init__(self):
        pass

    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """주요 성능지표 분석: 총수익률, 연환산수익률, 샤프, MDD, 변동성 등"""
        import numpy as np
        try:
            returns = df['return'].to_numpy(dtype=float)
            total_return = np.prod(1 + returns) - 1
            annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
            sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
            cum = np.cumprod(1 + returns)
            high = np.maximum.accumulate(cum)
            mdd = np.min(cum / high - 1)
            volatility = np.std(returns) * np.sqrt(252)
            result = {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'sharpe': sharpe,
                'max_drawdown': mdd,
                'volatility': volatility
            }
            logger.info(f"Performance: {result}")
            return result
        except Exception as e:
            logger.error(f"Performance analyze error: {e}")
            raise 