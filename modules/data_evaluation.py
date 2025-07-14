#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: data_evaluation.py
모듈: 데이터 품질 평가
목적: 결측/이상치/중복/스케일링/인코딩 품질지표 산출

Author: User
Created: 2025-07-06
Version: 1.0.0

Dependencies:
    - Python 3.11+
    - pandas>=2.0.0
    - numpy>=1.24.0
    - structlog>=24.1.0

Performance:
    - O(n) for all operations

Security:
    - 입력 검증: 경로, 파일 존재성
    - 에러 로깅: structlog

License: MIT
"""

from __future__ import annotations

import os
from typing import Dict
import pandas as pd
import numpy as np
import structlog

logger = structlog.get_logger(__name__)

def evaluate_data_quality(
    input_path: str
) -> float:
    """데이터 품질 평가 함수

    Args:
        input_path: 전처리 데이터 파일 경로 (csv/parquet)

    Returns:
        품질점수 (0~100)

    Raises:
        FileNotFoundError: 입력 파일이 없을 때
        ValueError: 데이터프레임이 비어있을 때
    """
    logger.info("Start data quality evaluation", input_path=input_path)
    if not os.path.exists(input_path):
        logger.error("Input file not found", input_path=input_path)
        raise FileNotFoundError(f"Input file not found: {input_path}")
    df = pd.read_csv(input_path) if input_path.endswith(".csv") else pd.read_parquet(input_path)
    if df.empty:
        logger.error("Input data is empty", input_path=input_path)
        raise ValueError("Input data is empty")
    n = len(df)
    score = 100.0
    # 결측치 비율
    na_ratio = df.isna().sum().sum() / (df.size + 1e-9)
    score -= na_ratio * 50
    # 중복 비율
    dup_ratio = 1.0 - (df.drop_duplicates().shape[0] / n)
    score -= dup_ratio * 20
    # 이상치 비율 (z-score 4.0 초과)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outlier_count = 0
    for col in numeric_cols:
        z = np.abs((df[col] - df[col].mean()) / (df[col].std() + 1e-9))
        outlier_count += (z > 4.0).sum()
    outlier_ratio = outlier_count / (n * len(numeric_cols) + 1e-9) if len(numeric_cols) > 0 else 0.0
    score -= outlier_ratio * 20
    # 스케일링/인코딩 품질 (간단히: 모든 수치형 평균 0, std 1이면 가산점)
    if len(numeric_cols) > 0:
        mean_deviation = np.abs(df[numeric_cols].mean()).mean()
        std_deviation = np.abs(df[numeric_cols].std() - 1).mean()
        if mean_deviation < 0.1 and std_deviation < 0.1:
            score += 5
    # 범주형 인코딩 여부 (0/1만 있으면 가산점)
    cat_cols = [c for c in df.columns if set(df[c].unique()).issubset({0, 1})]
    if len(cat_cols) > 0:
        score += 5
    score = max(0.0, min(100.0, score))
    logger.info("Data quality score", input_path=input_path, score=score)
    return float(score)


def _test_evaluate_data_quality() -> None:
    """단위 테스트: evaluate_data_quality 함수"""
    import tempfile
    import pandas as pd
    # 테스트 데이터 생성
    df = pd.DataFrame({
        "a": [0, 0, 0, 0],
        "b": [1, 1, 1, 1],
        "c": [0, 1, 0, 1]
    })
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "test.csv")
        df.to_csv(input_path, index=False)
        score = evaluate_data_quality(input_path)
        assert 90 <= score <= 100
        print(f"[PASS] evaluate_data_quality unit test: score={score}")

if __name__ == "__main__":
    _test_evaluate_data_quality() 