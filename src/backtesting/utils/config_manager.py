#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: src/backtesting/utils/config_manager.py
"""
from __future__ import annotations
from typing import Dict
import Any
import json

DEFAULT_CONFIG = {
    "initial_capital": 1_000_000,
    "data_path": "data/01_raw/krx_sample_data.parquet",
    "holiday_path": "config/krx_holidays.csv",
    "remove_survivorship_bias": True,
    "train_window": 24, "test_window": 3, "embargo": 1,
    "n_simulations": 100, # Reduced for faster demo
    "commission_rate": 0.00015, "tax_rate": 0.0025, "min_commission": 1000,
    "tick_sizes": {
        "under_1000": 1, "under_5000": 5, "under_10000": 10, "under_50000": 50,
        "under_100000": 100, "under_500000": 500, "over_500000": 1000,
    },
    "trading_hours": {
        "morning_start": "09:00", "morning_end": "11:30",
        "afternoon_start": "12:30", "afternoon_end": "15:30",
    },
    "stress_scenarios": {
        "financial_crisis_2008": {"price_shock": -0.5, "volume_shock": 0.1},
        "covid_crash_2020": {"price_shock": -0.35, "volume_shock": 0.2},
    },
    "performance_thresholds": {
        "min_sharpe_ratio": 1.5, "max_drawdown": -0.15,
    },
    "statistical_thresholds": {
        "significance_level": 0.05, "confidence_level": 0.95,
    },
    "logging": {
        "level": "INFO", "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file": "logs/backtest_system.log",
    },
}

def get_config(custom_config_path: str = None) -> Dict[str, Any]:
    config = DEFAULT_CONFIG.copy()
    if custom_config_path:
        with open(custom_config_path, 'r', encoding='utf-8') as f:
            custom_config = json.load(f)
            config.update(custom_config)
    return config
