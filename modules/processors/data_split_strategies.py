#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
파일명: data_split_strategies.py
모듈: 고급 데이터 분할 전략
목적: 시계열 데이터의 다양한 분할 방법과 검증 전략

Author: AI Assistant
Created: 2025-01-27
Modified: 2025-01-27
Version: 1.0.0

Dependencies:
    - Python 3.11+
    - pandas, numpy, scikit-learn
    - matplotlib, seaborn (시각화)
    - optuna (하이퍼파라미터 최적화)
"""

from abc import ABC
import abstractmethod
from dataclasses import dataclass
import field
from datetime import datetime
import timedelta
from enum import Enum
from pathlib import Path
from typing import Any
import Dict
import List, Optional, Tuple, Union
import logging
import random

try:
    from sklearn.model_selection import TimeSeriesSplit
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import asyncio
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
    NP_AVAILABLE = True
    PD_AVAILABLE = True
    SEABORN_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    NP_AVAILABLE = False
    PD_AVAILABLE = False
    SEABORN_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_split.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SplitType(Enum):
    """분할 타입"""
    TIME_SERIES = "time_series"
    RANDOM = "random"
    STRATIFIED = "stratified"
    BLOCK = "block"
    EXPANDING = "expanding"
    ROLLING = "rolling"
    CUSTOM = "custom"


@dataclass
class SplitConfig:
    """분할 설정"""
    split_type: SplitType = SplitType.TIME_SERIES
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    n_splits: int = 5
    gap: int = 0
    test_size: Optional[int] = None
    random_state: int = 42
    shuffle: bool = False
    stratify: Optional[str] = None

    def __post_init__(self) -> None:
        # 비율 검증
        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"분할 비율의 합이 1이어야 합니다. 현재: {total_ratio}")

        # 비율 정규화
        if total_ratio != 1.0:
            self.train_ratio /= total_ratio
            self.val_ratio /= total_ratio
            self.test_ratio /= total_ratio


class DataSplitter(ABC):
    """데이터 분할기 추상 클래스"""

    def __init__(self, config: SplitConfig) -> None:
        self.config = config
        self.splits: List[Dict[str, Any]] = []

    @abstractmethod
    def split(self, data) -> List[Dict[str, Any]]:
        """데이터 분할"""
        pass

    def validate_split(self, data, split: Dict[str, Any]) -> bool:
        """분할 결과 검증"""
        try:
            # 기본 검증
            if not all(key in split for key in ['train', 'val', 'test']):
                return False

            # 데이터 크기 검증
            total_size = len(split['train']) + len(split['val']) + len(split['test'])
            if total_size != len(data):
                return False

            # 중복 검증 (pandas DataFrame인 경우에만)
            if hasattr(split['train'], 'index'):
                train_indices = set(split['train'].index)
                val_indices = set(split['val'].index)
                test_indices = set(split['test'].index)

                if train_indices & val_indices or train_indices & test_indices or val_indices & test_indices:
                    return False

            return True

        except Exception as e:
            logger.error(f"분할 검증 실패: {e}")
            return False

    def get_split_info(self, split: Dict[str, Any]) -> Dict[str, Any]:
        """분할 정보 반환"""
        total_size = len(split['train']) + len(split['val']) + len(split['test'])
        return {
            'train_size': len(split['train']),
            'val_size': len(split['val']),
            'test_size': len(split['test']),
            'total_size': total_size,
            'train_ratio': len(split['train']) / total_size,
            'val_ratio': len(split['val']) / total_size,
            'test_ratio': len(split['test']) / total_size
        }


class DataSplitStrategies:
    """데이터 분할 전략 클래스"""

    def __init__(self) -> None:
        self.config = SplitConfig()

    def time_series_split(self, data, config: Optional[SplitConfig] = None):
        """시계열 분할"""
        if config:
            self.config = config

        if not PD_AVAILABLE:
            logger.warning("pandas를 사용할 수 없습니다. 기본 분할을 수행합니다.")
            return self._basic_split(data)

        logger.info("시계열 데이터 분할 시작")

        # 데이터가 pandas DataFrame인 경우 정렬
        if hasattr(data, 'sort_values'):
            if 'timestamp' in data.columns:
                data = data.sort_values('timestamp').reset_index(drop=True)
            elif 'date' in data.columns:
                data = data.sort_values('date').reset_index(drop=True)

        total_size = len(data)
        train_size = int(total_size * self.config.train_ratio)
        val_size = int(total_size * self.config.val_ratio)

        if hasattr(data, 'iloc'):
            # pandas DataFrame
            split = {
                'train': data.iloc[:train_size],
                'val': data.iloc[train_size:train_size + val_size],
                'test': data.iloc[train_size + val_size:]
            }
        else:
            # 일반 리스트 또는 배열
            split = {
                'train': data[:train_size],
                'val': data[train_size:train_size + val_size],
                'test': data[train_size + val_size:]
            }

        logger.info("시계열 분할 완료")
        return [split]

    def _basic_split(self, data):
        """기본 분할"""
        total_size = len(data)
        train_size = int(total_size * self.config.train_ratio)
        val_size = int(total_size * self.config.val_ratio)

        split = {
            'train': data[:train_size],
            'val': data[train_size:train_size + val_size],
            'test': data[train_size + val_size:]
        }

        return [split]


async def main() -> None:
    """메인 함수"""
    logger.info("데이터 분할 전략 테스트 시작")

    # 테스트 데이터 생성
    if PD_AVAILABLE:
        test_data = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=1000, freq='D'),
            'value': range(1000)
        })
    else:
        test_data = list(range(1000))

    # 분할 전략 테스트
    splitter = DataSplitStrategies()
    splits = splitter.time_series_split(test_data)

    logger.info(f"분할 완료: {len(splits)}개 분할")
    for i, split in enumerate(splits):
        info = splitter.get_split_info(split) if hasattr(splitter, 'get_split_info') else {}
        logger.info(f"분할 {i+1}: {info}")


if __name__ == "__main__":
    asyncio.run(main())
