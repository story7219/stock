#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from sklearn.model_selection import TimeSeriesSplit
    MATPLOTLIB_AVAILABLE = True
    NP_AVAILABLE = True
    PD_AVAILABLE = True
    SEABORN_AVAILABLE = True
    SKLEARN_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    NP_AVAILABLE = False
    PD_AVAILABLE = False
    SEABORN_AVAILABLE = False
    SKLEARN_AVAILABLE = False

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
    def split(self, data: pd.DataFrame) -> List[Dict[str, pd.DataFrame]]:
        """데이터 분할"""
        pass

    def validate_split(self, data: pd.DataFrame, split: Dict[str, pd.DataFrame]) -> bool:
        """분할 결과 검증"""
        try:
            # 기본 검증
            if not all(key in split for key in ['train', 'val', 'test']):
                return False

            # 데이터 크기 검증
            total_size = len(split['train']) + len(split['val']) + len(split['test'])
            if total_size != len(data):
                return False

            # 중복 검증
            train_indices = set(split['train'].index)
            val_indices = set(split['val'].index)
            test_indices = set(split['test'].index)

            if train_indices & val_indices or train_indices & test_indices or val_indices & test_indices:
                return False

            return True

        except Exception as e:
            logger.error(f"분할 검증 실패: {e}")
            return False

    def get_split_info(self, split: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """분할 정보 반환"""
        return {
            'train_size': len(split['train']),
            'val_size': len(split['val']),
            'test_size': len(split['test']),
            'total_size': len(split['train']) + len(split['val']) + len(split['test']),
            'train_ratio': len(split['train']) / (len(split['train']) + len(split['val']) + len(split['test'])),
            'val_ratio': len(split['val']) / (len(split['train']) + len(split['val']) + len(split['test'])),
            'test_ratio': len(split['test']) / (len(split['train']) + len(split['val']) + len(split['test']))
        }


class TimeSeriesSplitter(DataSplitter):
    """시계열 데이터 분할기"""

    def __init__(self, config: SplitConfig) -> None:
        super().__init__(config)
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn이 설치되지 않았습니다.")

    def split(self, data: pd.DataFrame) -> List[Dict[str, pd.DataFrame]]:
        """시계열 분할"""
        if not PD_AVAILABLE:
            raise ImportError("pandas가 설치되지 않았습니다.")

        logger.info("시계열 데이터 분할 시작")

        # 데이터 정렬 (시간순)
        if 'timestamp' in data.columns:
            data = data.sort_values('timestamp').reset_index(drop=True)
        elif 'date' in data.columns:
            data = data.sort_values('date').reset_index(drop=True)

        splits = []

        if self.config.n_splits > 1:
            # 교차 검증 분할
            tscv = TimeSeriesSplit(
                n_splits=self.config.n_splits,
                gap=self.config.gap,
                test_size=self.config.test_size
            )

            for train_idx, test_idx in tscv.split(data):
                # 검증 세트 분할
                val_size = int(len(train_idx) * self.config.val_ratio / (1 - self.config.test_ratio))
                val_idx = train_idx[-val_size:]
                train_idx = train_idx[:-val_size]

                split = {
                    'train': data.iloc[train_idx],
                    'val': data.iloc[val_idx],
                    'test': data.iloc[test_idx]
                }

                if self.validate_split(data, split):
                    splits.append(split)
                else:
                    logger.warning("시계열 분할 검증 실패")

        else:
            # 단일 분할
            total_size = len(data)
            train_size = int(total_size * self.config.train_ratio)
            val_size = int(total_size * self.config.val_ratio)

            split = {
                'train': data.iloc[:train_size],
                'val': data.iloc[train_size:train_size + val_size],
                'test': data.iloc[train_size + val_size:]
            }

            if self.validate_split(data, split):
                splits.append(split)

        logger.info(f"시계열 분할 완료: {len(splits)}개 분할")
        return splits


class RandomSplitter(DataSplitter):
    """랜덤 데이터 분할기"""

    def split(self, data: pd.DataFrame) -> List[Dict[str, pd.DataFrame]]:
        """랜덤 분할"""
        if not PD_AVAILABLE:
            raise ImportError("pandas가 설치되지 않았습니다.")

        logger.info("랜덤 데이터 분할 시작")

        # 랜덤 시드 설정
        random.seed(self.config.random_state)
        if NP_AVAILABLE:
            np.random.seed(self.config.random_state)

        # 인덱스 셔플
        indices = list(range(len(data)))
        if self.config.shuffle:
            random.shuffle(indices)

        # 분할 크기 계산
        total_size = len(data)
        train_size = int(total_size * self.config.train_ratio)
        val_size = int(total_size * self.config.val_ratio)

        # 분할
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]

        split = {
            'train': data.iloc[train_indices],
            'val': data.iloc[val_indices],
            'test': data.iloc[test_indices]
        }

        splits = []
        if self.validate_split(data, split):
            splits.append(split)

        logger.info("랜덤 분할 완료")
        return splits


class StratifiedSplitter(DataSplitter):
    """층화 데이터 분할기"""

    def split(self, data: pd.DataFrame) -> List[Dict[str, pd.DataFrame]]:
        """층화 분할"""
        if not PD_AVAILABLE:
            raise ImportError("pandas가 설치되지 않았습니다.")

        if not self.config.stratify:
            raise ValueError("층화 분할을 위해서는 stratify 컬럼이 필요합니다.")

        logger.info("층화 데이터 분할 시작")

        splits = []
        unique_values = data[self.config.stratify].unique()

        train_data = []
        val_data = []
        test_data = []

        for value in unique_values:
            subset = data[data[self.config.stratify] == value]
            subset_size = len(subset)

            train_size = int(subset_size * self.config.train_ratio)
            val_size = int(subset_size * self.config.val_ratio)

            train_data.append(subset.iloc[:train_size])
            val_data.append(subset.iloc[train_size:train_size + val_size])
            test_data.append(subset.iloc[train_size + val_size:])

        split = {
            'train': pd.concat(train_data, ignore_index=True),
            'val': pd.concat(val_data, ignore_index=True),
            'test': pd.concat(test_data, ignore_index=True)
        }

        if self.validate_split(data, split):
            splits.append(split)

        logger.info("층화 분할 완료")
        return splits


class BlockSplitter(DataSplitter):
    """블록 데이터 분할기"""

    def split(self, data: pd.DataFrame) -> List[Dict[str, pd.DataFrame]]:
        """블록 분할"""
        if not PD_AVAILABLE:
            raise ImportError("pandas가 설치되지 않았습니다.")

        logger.info("블록 데이터 분할 시작")

        total_size = len(data)
        block_size = total_size // self.config.n_splits

        splits = []

        for i in range(self.config.n_splits):
            start_idx = i * block_size
            end_idx = start_idx + block_size if i < self.config.n_splits - 1 else total_size

            # 블록 내에서 분할
            block_data = data.iloc[start_idx:end_idx]
            block_size_actual = len(block_data)

            train_size = int(block_size_actual * self.config.train_ratio)
            val_size = int(block_size_actual * self.config.val_ratio)

            split = {
                'train': block_data.iloc[:train_size],
                'val': block_data.iloc[train_size:train_size + val_size],
                'test': block_data.iloc[train_size + val_size:]
            }

            if self.validate_split(block_data, split):
                splits.append(split)

        logger.info(f"블록 분할 완료: {len(splits)}개 분할")
        return splits


class ExpandingSplitter(DataSplitter):
    """확장 데이터 분할기"""

    def split(self, data: pd.DataFrame) -> List[Dict[str, pd.DataFrame]]:
        """확장 분할"""
        if not PD_AVAILABLE:
            raise ImportError("pandas가 설치되지 않았습니다.")

        logger.info("확장 데이터 분할 시작")

        total_size = len(data)
        min_train_size = int(total_size * 0.3)  # 최소 훈련 크기
        step_size = int((total_size - min_train_size) / self.config.n_splits)

        splits = []

        for i in range(self.config.n_splits):
            train_end = min_train_size + i * step_size
            val_end = train_end + int(step_size * self.config.val_ratio / (1 - self.config.test_ratio))

            if val_end >= total_size:
                break

            split = {
                'train': data.iloc[:train_end],
                'val': data.iloc[train_end:val_end],
                'test': data.iloc[val_end:min(val_end + int(step_size * self.config.test_ratio / (1 - self.config.test_ratio)), total_size)]
            }

            if self.validate_split(data.iloc[:min(val_end + int(step_size * self.config.test_ratio / (1 - self.config.test_ratio)), total_size)], split):
                splits.append(split)

        logger.info(f"확장 분할 완료: {len(splits)}개 분할")
        return splits


class RollingSplitter(DataSplitter):
    """롤링 데이터 분할기"""

    def __init__(self, config: SplitConfig, window_size: int = 100) -> None:
        super().__init__(config)
        self.window_size = window_size

    def split(self, data: pd.DataFrame) -> List[Dict[str, pd.DataFrame]]:
        """롤링 분할"""
        if not PD_AVAILABLE:
            raise ImportError("pandas가 설치되지 않았습니다.")

        logger.info("롤링 데이터 분할 시작")

        total_size = len(data)
        splits = []

        for i in range(0, total_size - self.window_size, self.window_size // 2):
            window_data = data.iloc[i:i + self.window_size]
            window_size_actual = len(window_data)

            if window_size_actual < self.window_size * 0.8:  # 최소 80% 크기
                continue

            train_size = int(window_size_actual * self.config.train_ratio)
            val_size = int(window_size_actual * self.config.val_ratio)

            split = {
                'train': window_data.iloc[:train_size],
                'val': window_data.iloc[train_size:train_size + val_size],
                'test': window_data.iloc[train_size + val_size:]
            }

            if self.validate_split(window_data, split):
                splits.append(split)

        logger.info(f"롤링 분할 완료: {len(splits)}개 분할")
        return splits


class CustomSplitter(DataSplitter):
    """커스텀 데이터 분할기"""

    def __init__(self, config: SplitConfig, split_function: Optional[callable] = None) -> None:
        super().__init__(config)
        self.split_function = split_function

    def split(self, data: pd.DataFrame) -> List[Dict[str, pd.DataFrame]]:
        """커스텀 분할"""
        if not PD_AVAILABLE:
            raise ImportError("pandas가 설치되지 않았습니다.")

        logger.info("커스텀 데이터 분할 시작")

        if self.split_function:
            # 사용자 정의 분할 함수 사용
            splits = self.split_function(data, self.config)
        else:
            # 기본 분할 (시간 기반)
            if 'timestamp' in data.columns:
                data = data.sort_values('timestamp').reset_index(drop=True)

            total_size = len(data)
            train_size = int(total_size * self.config.train_ratio)
            val_size = int(total_size * self.config.val_ratio)

            split = {
                'train': data.iloc[:train_size],
                'val': data.iloc[train_size:train_size + val_size],
                'test': data.iloc[train_size + val_size:]
            }

            splits = [split] if self.validate_split(data, split) else []

        logger.info(f"커스텀 분할 완료: {len(splits)}개 분할")
        return splits


class DataSplitManager:
    """데이터 분할 관리자"""

    def __init__(self) -> None:
        self.splitters: Dict[SplitType, DataSplitter] = {}

    def register_splitter(self, split_type: SplitType, splitter: DataSplitter) -> None:
        """분할기 등록"""
        self.splitters[split_type] = splitter

    def get_splitter(self, split_type: SplitType) -> DataSplitter:
        """분할기 반환"""
        if split_type not in self.splitters:
            raise ValueError(f"등록되지 않은 분할 타입: {split_type}")
        return self.splitters[split_type]

    def split_data(self, data: pd.DataFrame, config: SplitConfig) -> List[Dict[str, pd.DataFrame]]:
        """데이터 분할"""
        splitter = self._create_splitter(config)
        return splitter.split(data)

    def _create_splitter(self, config: SplitConfig) -> DataSplitter:
        """분할기 생성"""
        if config.split_type == SplitType.TIME_SERIES:
            return TimeSeriesSplitter(config)
        elif config.split_type == SplitType.RANDOM:
            return RandomSplitter(config)
        elif config.split_type == SplitType.STRATIFIED:
            return StratifiedSplitter(config)
        elif config.split_type == SplitType.BLOCK:
            return BlockSplitter(config)
        elif config.split_type == SplitType.EXPANDING:
            return ExpandingSplitter(config)
        elif config.split_type == SplitType.ROLLING:
            return RollingSplitter(config)
        elif config.split_type == SplitType.CUSTOM:
            return CustomSplitter(config)
        else:
            raise ValueError(f"지원하지 않는 분할 타입: {config.split_type}")

    def evaluate_splits(self, data: pd.DataFrame, splits: List[Dict[str, pd.DataFrame]]) -> Dict[str, Any]:
        """분할 결과 평가"""
        if not PD_AVAILABLE:
            raise ImportError("pandas가 설치되지 않았습니다.")

        evaluation = {
            'total_splits': len(splits),
            'split_info': [],
            'data_distribution': {},
            'overlap_analysis': {}
        }

        for i, split in enumerate(splits):
            # 분할 정보
            split_info = self._get_split_info(split)
            split_info['split_index'] = i
            evaluation['split_info'].append(split_info)

            # 데이터 분포 분석
            if i == 0:  # 첫 번째 분할만 분석
                evaluation['data_distribution'] = self._analyze_distribution(data, split)

        # 중복 분석
        evaluation['overlap_analysis'] = self._analyze_overlaps(splits)

        return evaluation

    def _get_split_info(self, split: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """분할 정보 반환"""
        return {
            'train_size': len(split['train']),
            'val_size': len(split['val']),
            'test_size': len(split['test']),
            'total_size': len(split['train']) + len(split['val']) + len(split['test']),
            'train_ratio': len(split['train']) / (len(split['train']) + len(split['val']) + len(split['test'])),
            'val_ratio': len(split['val']) / (len(split['train']) + len(split['val']) + len(split['test'])),
            'test_ratio': len(split['test']) / (len(split['train']) + len(split['val']) + len(split['test']))
        }

    def _analyze_distribution(self, data: pd.DataFrame, split: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """데이터 분포 분석"""
        analysis = {}

        # 수치형 컬럼 분석
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            analysis[col] = {
                'original': {
                    'mean': data[col].mean(),
                    'std': data[col].std(),
                    'min': data[col].min(),
                    'max': data[col].max()
                },
                'train': {
                    'mean': split['train'][col].mean(),
                    'std': split['train'][col].std(),
                    'min': split['train'][col].min(),
                    'max': split['train'][col].max()
                },
                'val': {
                    'mean': split['val'][col].mean(),
                    'std': split['val'][col].std(),
                    'min': split['val'][col].min(),
                    'max': split['val'][col].max()
                },
                'test': {
                    'mean': split['test'][col].mean(),
                    'std': split['test'][col].std(),
                    'min': split['test'][col].min(),
                    'max': split['test'][col].max()
                }
            }

        return analysis

    def _analyze_overlaps(self, splits: List[Dict[str, pd.DataFrame]]) -> Dict[str, Any]:
        """중복 분석"""
        if len(splits) < 2:
            return {'overlaps': []}

        overlaps = []
        for i in range(len(splits)):
            for j in range(i + 1, len(splits)):
                train_overlap = len(set(splits[i]['train'].index) & set(splits[j]['train'].index))
                val_overlap = len(set(splits[i]['val'].index) & set(splits[j]['val'].index))
                test_overlap = len(set(splits[i]['test'].index) & set(splits[j]['test'].index))

                overlaps.append({
                    'split_pair': (i, j),
                    'train_overlap': train_overlap,
                    'val_overlap': val_overlap,
                    'test_overlap': test_overlap,
                    'total_overlap': train_overlap + val_overlap + test_overlap
                })

        return {'overlaps': overlaps}

    def visualize_splits(self, splits: List[Dict[str, pd.DataFrame]], save_path: Optional[str] = None) -> None:
        """분할 결과 시각화"""
        if not MATPLOTLIB_AVAILABLE or not SEABORN_AVAILABLE:
            logger.warning("시각화를 위해 matplotlib과 seaborn이 필요합니다.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Data Split Analysis', fontsize=16)

        # 1. 분할 크기 비교
        split_sizes = []
        for i, split in enumerate(splits):
            split_sizes.append({
                'split': i,
                'train': len(split['train']),
                'val': len(split['val']),
                'test': len(split['test'])
            })

        df_sizes = pd.DataFrame(split_sizes)
        df_sizes.plot(x='split', y=['train', 'val', 'test'], kind='bar', ax=axes[0, 0])
        axes[0, 0].set_title('Split Sizes')
        axes[0, 0].set_ylabel('Number of Samples')

        # 2. 분할 비율
        ratios = []
        for i, split in enumerate(splits):
            total = len(split['train']) + len(split['val']) + len(split['test'])
            ratios.append({
                'split': i,
                'train_ratio': len(split['train']) / total,
                'val_ratio': len(split['val']) / total,
                'test_ratio': len(split['test']) / total
            })

        df_ratios = pd.DataFrame(ratios)
        df_ratios.plot(x='split', y=['train_ratio', 'val_ratio', 'test_ratio'], kind='bar', ax=axes[0, 1])
        axes[0, 1].set_title('Split Ratios')
        axes[0, 1].set_ylabel('Ratio')

        # 3. 데이터 분포 (첫 번째 분할 기준)
        if splits and len(splits[0]['train'].columns) > 0:
            numeric_cols = splits[0]['train'].select_dtypes(include=[np.number]).columns[:3]
            for i, col in enumerate(numeric_cols):
                if i >= 2:
                    break
                axes[1, 0].hist(splits[0]['train'][col], alpha=0.7, label=f'Train {col}', bins=20)
                axes[1, 0].hist(splits[0]['val'][col], alpha=0.7, label=f'Val {col}', bins=20)
                axes[1, 0].hist(splits[0]['test'][col], alpha=0.7, label=f'Test {col}', bins=20)
            axes[1, 0].set_title('Data Distribution (First Split)')
            axes[1, 0].legend()

        # 4. 중복 분석
        if len(splits) > 1:
            overlaps = []
            for i in range(len(splits)):
                for j in range(i + 1, len(splits)):
                    overlap = len(set(splits[i]['train'].index) & set(splits[j]['train'].index))
                    overlaps.append(overlap)

            if overlaps:
                axes[1, 1].hist(overlaps, bins=10, alpha=0.7)
                axes[1, 1].set_title('Train Set Overlaps')
                axes[1, 1].set_xlabel('Overlap Size')
                axes[1, 1].set_ylabel('Frequency')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"시각화 저장: {save_path}")

        plt.show()


def optimize_split_config(data: pd.DataFrame, target_metric: str = 'distribution_similarity') -> SplitConfig:
    """분할 설정 최적화"""
    if not OPTUNA_AVAILABLE:
        logger.warning("optuna가 설치되지 않아 최적화를 건너뜁니다.")
        return SplitConfig()

    def objective(trial: optuna.Trial) -> float:
        # 하이퍼파라미터 샘플링
        train_ratio = trial.suggest_float('train_ratio', 0.5, 0.8)
        val_ratio = trial.suggest_float('val_ratio', 0.1, 0.3)
        test_ratio = 1.0 - train_ratio - val_ratio

        if test_ratio < 0.1:
            return float('-inf')

        config = SplitConfig(
            split_type=SplitType.TIME_SERIES,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            n_splits=trial.suggest_int('n_splits', 3, 10)
        )

        try:
            manager = DataSplitManager()
            splits = manager.split_data(data, config)
            evaluation = manager.evaluate_splits(data, splits)

            # 목표 메트릭 계산
            if target_metric == 'distribution_similarity':
                return _calculate_distribution_similarity(evaluation)
            elif target_metric == 'split_balance':
                return _calculate_split_balance(evaluation)
            else:
                return 0.0

        except Exception as e:
            logger.error(f"최적화 중 오류: {e}")
            return float('-inf')

    # 최적화 실행
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)

    # 최적 설정 반환
    best_params = study.best_params
    return SplitConfig(
        split_type=SplitType.TIME_SERIES,
        train_ratio=best_params['train_ratio'],
        val_ratio=best_params['val_ratio'],
        test_ratio=1.0 - best_params['train_ratio'] - best_params['val_ratio'],
        n_splits=best_params['n_splits']
    )


def _calculate_distribution_similarity(evaluation: Dict[str, Any]) -> float:
    """분포 유사도 계산"""
    if not evaluation.get('data_distribution'):
        return 0.0

    similarities = []
    for col, dist_info in evaluation['data_distribution'].items():
        if 'mean' in dist_info['original']:
            # 평균 유사도
            train_sim = 1.0 / (1.0 + abs(dist_info['train']['mean'] - dist_info['original']['mean']))
            val_sim = 1.0 / (1.0 + abs(dist_info['val']['mean'] - dist_info['original']['mean']))
            test_sim = 1.0 / (1.0 + abs(dist_info['test']['mean'] - dist_info['original']['mean']))
            similarities.append((train_sim + val_sim + test_sim) / 3)

    return np.mean(similarities) if similarities else 0.0


def _calculate_split_balance(evaluation: Dict[str, Any]) -> float:
    """분할 균형 계산"""
    if not evaluation.get('split_info'):
        return 0.0

    balances = []
    for split_info in evaluation['split_info']:
        # 목표 비율과의 차이
        target_ratios = [0.7, 0.15, 0.15]  # train, val, test
        actual_ratios = [split_info['train_ratio'], split_info['val_ratio'], split_info['test_ratio']]
        
        balance = 1.0 - np.mean(np.abs(np.array(actual_ratios) - np.array(target_ratios)))
        balances.append(balance)

    return np.mean(balances)


async def main() -> None:
    """메인 함수"""
    print("🚀 데이터 분할 전략 시스템 시작")
    print("=" * 60)

    # 샘플 데이터 생성
    if not PD_AVAILABLE or not NP_AVAILABLE:
        print("❌ pandas 또는 numpy가 설치되지 않았습니다.")
        return

    # 샘플 데이터 생성
    dates = pd.date_range('2020-01-01', '2024-12-31', freq='D')
    data = pd.DataFrame({
        'timestamp': dates,
        'price': np.random.randn(len(dates)).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, len(dates)),
        'category': np.random.choice(['A', 'B', 'C'], len(dates))
    })

    print(f"📊 샘플 데이터 생성: {len(data)}개 행")

    # 분할 관리자 생성
    manager = DataSplitManager()

    # 다양한 분할 전략 테스트
    split_configs = [
        SplitConfig(split_type=SplitType.TIME_SERIES, n_splits=3),
        SplitConfig(split_type=SplitType.RANDOM, shuffle=True),
        SplitConfig(split_type=SplitType.STRATIFIED, stratify='category'),
        SplitConfig(split_type=SplitType.BLOCK, n_splits=4),
        SplitConfig(split_type=SplitType.EXPANDING, n_splits=3)
    ]

    for config in split_configs:
        try:
            print(f"\n🔧 {config.split_type.value} 분할 테스트")
            splits = manager.split_data(data, config)
            evaluation = manager.evaluate_splits(data, splits)
            
            print(f"   분할 수: {evaluation['total_splits']}")
            if evaluation['split_info']:
                first_split = evaluation['split_info'][0]
                print(f"   첫 번째 분할 - 훈련: {first_split['train_size']}, 검증: {first_split['val_size']}, 테스트: {first_split['test_size']}")

        except Exception as e:
            print(f"   ❌ 오류: {e}")

    # 시각화
    try:
        config = SplitConfig(split_type=SplitType.TIME_SERIES, n_splits=5)
        splits = manager.split_data(data, config)
        manager.visualize_splits(splits, 'data_split_analysis.png')
        print("\n📈 시각화 완료")
    except Exception as e:
        print(f"\n❌ 시각화 오류: {e}")

    print("\n✅ 데이터 분할 전략 시스템 완료")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
