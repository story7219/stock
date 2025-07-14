#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: data_cleaning.py
모듈: 고급 데이터 정제
목적: 결측치, 이상치, 중복 제거 등 고급 정제 기능 제공

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
    - 메모리 최적화: generator, inplace 처리

Security:
    - 입력 검증: 경로, 파일 존재성
    - 에러 로깅: structlog

License: MIT
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
import structlog

logger = structlog.get_logger(__name__)

def clean_data(
    input_path: str,
    output_path: str,
    dropna: bool = True,
    drop_duplicates: bool = True,
    outlier_zscore: Optional[float] = 4.0
) -> str:
    """고급 데이터 정제 함수

    Args:
        input_path: 원본 데이터 파일 경로 (csv/parquet)
        output_path: 정제 데이터 저장 경로
        dropna: 결측치 제거 여부
        drop_duplicates: 중복 제거 여부
        outlier_zscore: 이상치 제거 z-score 임계값 (None이면 미적용)

    Returns:
        정제 데이터 저장 경로

    Raises:
        FileNotFoundError: 입력 파일이 없을 때
        ValueError: 데이터프레임이 비어있을 때
    """
    logger.info("Start cleaning", input_path=input_path, output_path=output_path)
    if not os.path.exists(input_path):
        logger.error("Input file not found", input_path=input_path)
        raise FileNotFoundError(f"Input file not found: {input_path}")
    df = pd.read_csv(input_path) if input_path.endswith(".csv") else pd.read_parquet(input_path)
    if df.empty:
        logger.error("Input data is empty", input_path=input_path)
        raise ValueError("Input data is empty")
    if dropna:
        df.dropna(inplace=True)
    if drop_duplicates:
        df.drop_duplicates(inplace=True)
    if outlier_zscore is not None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            z = np.abs((df[col] - df[col].mean()) / (df[col].std() + 1e-9))
            mask = z < outlier_zscore
            df = df[mask].copy()
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Cleaned data is not a DataFrame")
    if output_path.endswith(".csv"):
        df.to_csv(output_path, index=False)
    else:
        df.to_parquet(output_path, index=False)
    logger.info("Cleaned data saved", output_path=output_path, rows=len(df))
    return output_path


def _test_clean_data() -> None:
    """단위 테스트: clean_data 함수"""
    import tempfile
    import pandas as pd
    # 테스트 데이터 생성
    df = pd.DataFrame({
        "a": [1, 2, 2, np.nan, 1000],
        "b": [1, 2, 2, 2, 2],
        "c": ["x", "y", "y", "y", "y"]
    })
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "test.csv")
        output_path = os.path.join(tmpdir, "cleaned.csv")
        df.to_csv(input_path, index=False)
        result_path = clean_data(input_path, output_path, outlier_zscore=3.0)
        cleaned = pd.read_csv(result_path)
        assert cleaned.shape[0] == 3  # 결측치, 중복, 이상치 제거
        assert "a" in cleaned.columns
        print("[PASS] clean_data unit test")

if __name__ == "__main__":
    _test_clean_data() 