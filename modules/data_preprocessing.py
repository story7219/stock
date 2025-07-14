#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: data_preprocessing.py
모듈: 고급 데이터 전처리
목적: 스케일링, 인코딩 등 고급 전처리 기능 제공

Author: User
Created: 2025-07-06
Version: 1.0.0

Dependencies:
    - Python 3.11+
    - pandas>=2.0.0
    - numpy>=1.24.0
    - scikit-learn>=1.3.0
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
from typing import Optional, List
import pandas as pd
import numpy as np
import structlog
from sklearn.preprocessing import StandardScaler, OneHotEncoder

logger = structlog.get_logger(__name__)

def preprocess_data(
    input_path: str,
    output_path: str,
    scale_numeric: bool = True,
    encode_categorical: bool = True,
    categorical_cols: Optional[List[str]] = None
) -> str:
    """고급 데이터 전처리 함수

    Args:
        input_path: 정제 데이터 파일 경로 (csv/parquet)
        output_path: 전처리 데이터 저장 경로
        scale_numeric: 수치형 스케일링 적용 여부
        encode_categorical: 범주형 인코딩 적용 여부
        categorical_cols: 인코딩할 컬럼 리스트 (None이면 자동 탐지)

    Returns:
        전처리 데이터 저장 경로

    Raises:
        FileNotFoundError: 입력 파일이 없을 때
        ValueError: 데이터프레임이 비어있을 때
    """
    logger.info("Start preprocessing", input_path=input_path, output_path=output_path)
    if not os.path.exists(input_path):
        logger.error("Input file not found", input_path=input_path)
        raise FileNotFoundError(f"Input file not found: {input_path}")
    df = pd.read_csv(input_path) if input_path.endswith(".csv") else pd.read_parquet(input_path)
    if df.empty:
        logger.error("Input data is empty", input_path=input_path)
        raise ValueError("Input data is empty")
    if scale_numeric:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            scaler = StandardScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    if encode_categorical:
        if categorical_cols is None:
            categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if categorical_cols is not None and len(categorical_cols) > 0:
            from sklearn.preprocessing import OneHotEncoder
            import inspect
            params = {}
            if 'sparse_output' in inspect.signature(OneHotEncoder).parameters:
                params['sparse_output'] = False
            elif 'sparse' in inspect.signature(OneHotEncoder).parameters:
                params['sparse'] = False
            encoder = OneHotEncoder(handle_unknown="ignore", **params)
            encoded = encoder.fit_transform(df[categorical_cols])
            encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols), index=df.index)
            df = pd.concat([df.drop(columns=categorical_cols), encoded_df], axis=1)
    df.to_csv(output_path, index=False) if output_path.endswith(".csv") else df.to_parquet(output_path, index=False)
    logger.info("Preprocessed data saved", output_path=output_path, rows=len(df))
    return output_path


def _test_preprocess_data() -> None:
    """단위 테스트: preprocess_data 함수"""
    import tempfile
    import pandas as pd
    # 테스트 데이터 생성
    df = pd.DataFrame({
        "a": [1, 2, 3, 4],
        "b": ["x", "y", "x", "z"]
    })
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "test.csv")
        output_path = os.path.join(tmpdir, "preprocessed.csv")
        df.to_csv(input_path, index=False)
        result_path = preprocess_data(input_path, output_path)
        preprocessed = pd.read_csv(result_path)
        assert "b_x" in preprocessed.columns
        assert "b_y" in preprocessed.columns
        assert "b_z" in preprocessed.columns
        print("[PASS] preprocess_data unit test")

if __name__ == "__main__":
    _test_preprocess_data() 