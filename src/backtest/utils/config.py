#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: config.py
모듈: 백테스팅 시스템 설정
목적: 모든 백테스팅 설정 중앙 관리

Author: WorldClassAI
Created: 2025-07-12
Version: 1.0.0
"""

from __future__ import annotations
from typing import Dict
import Any

# 기본 백테스팅 설정
DEFAULT_CONFIG = {
    # 데이터 설정
    "data_path": "collected_data/kospi/krx_40years.parquet",
    "holiday_path": "config/krx_holidays.csv",
    "remove_survivorship_bias": True,

    # 시간 설정
    "train_window": 24,      # Walk-Forward 학습 기간 (개월)
    "test_window": 3,        # Walk-Forward 테스트 기간 (개월)
    "embargo": 1,            # Walk-Forward 엠바고 기간 (개월)

    # Monte Carlo 설정
    "n_simulations": 1000,   # Monte Carlo 시뮬레이션 횟수

    # 거래 설정
    "commission_rate": 0.00015,  # 위탁수수료 0.015%
    "tax_rate": 0.0025,         # 제세공과금 0.25%
    "min_commission": 1000,      # 최소 수수료 1,000원

    # 호가단위 설정
    "tick_sizes": {
        "under_1000": 1,     # 1,000원 미만: 1원
        "under_5000": 5,     # 5,000원 미만: 5원
        "under_10000": 10,   # 10,000원 미만: 10원
        "under_50000": 50,   # 50,000원 미만: 50원
        "under_100000": 100, # 100,000원 미만: 100원
        "under_500000": 500, # 500,000원 미만: 500원
        "over_500000": 1000, # 500,000원 이상: 1,000원
    },

    # 거래시간 설정
    "trading_hours": {
        "morning_start": "09:00",
        "morning_end": "11:30",
        "afternoon_start": "12:30",
        "afternoon_end": "15:30",
        "pre_market_start": "08:30",
        "pre_market_end": "09:00",
        "post_market_start": "15:30",
        "post_market_end": "15:40",
    },

    # 위기 상황 설정
    "stress_scenarios": {
        "financial_crisis_2008": {
            "price_shock": -0.5,
            "volume_shock": 0.1,
            "volatility_shock": 3.0,
            "duration": 18,  # 개월
        },
        "covid_crash_2020": {
            "price_shock": -0.35,
            "volume_shock": 0.2,
            "volatility_shock": 2.5,
            "duration": 1,   # 개월
        },
        "black_monday_1987": {
            "price_shock": -0.22,
            "volume_shock": 0.05,
            "volatility_shock": 4.0,
            "duration": 1,   # 일
        },
        "asian_crisis_1997": {
            "price_shock": -0.6,
            "volume_shock": 0.1,
            "volatility_shock": 3.5,
            "duration": 6,   # 개월
        },
        "dot_com_crash_2000": {
            "price_shock": -0.4,
            "volume_shock": 0.15,
            "volatility_shock": 2.8,
            "duration": 24,  # 개월
        },
    },

    # 성과 기준
    "performance_thresholds": {
        "min_sharpe_ratio": 1.5,
        "max_drawdown": -0.15,
        "min_win_rate": 0.55,
        "min_profit_factor": 1.2,
        "min_annual_return": 0.15,
    },

    # 통계 검증 기준
    "statistical_thresholds": {
        "significance_level": 0.01,
        "confidence_level": 0.95,
        "min_effect_size": 0.8,
    },

    # 로깅 설정
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file": "backtest_system.log",
    },
}

def get_config(custom_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """설정을 반환합니다."""
    config = DEFAULT_CONFIG.copy()
    if custom_config:
        config.update(custom_config)
    return config
