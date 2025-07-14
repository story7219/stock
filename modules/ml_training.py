#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: ml_training.py
모듈: ML/DL 학습
목적: 랜덤포레스트 기반 분류/회귀 학습/예측/평가

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
    - O(n log n) for training

Security:
    - 입력 검증: 경로, 파일 존재성, 타겟 컬럼
    - 에러 로깅: structlog

License: MIT
"""

from __future__ import annotations

import os
from typing import Dict, Any
import pandas as pd
import numpy as np
import structlog
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

logger = structlog.get_logger(__name__)

def train_ml_model(
    input_path: str,
    target_col: str,
    task: str = "auto",
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Any]:
    """ML/DL 학습 및 평가 함수

    Args:
        input_path: 전처리 데이터 파일 경로 (csv/parquet)
        target_col: 예측 타겟 컬럼명
        task: "auto"(자동), "classification", "regression"
        test_size: 테스트셋 비율
        random_state: 랜덤 시드

    Returns:
        dict: {"model": model, "score": float, "y_pred": np.ndarray, "y_test": np.ndarray}

    Raises:
        FileNotFoundError: 입력 파일이 없을 때
        ValueError: 데이터프레임/타겟 컬럼 오류
    """
    logger.info("Start ML training", input_path=input_path, target_col=target_col)
    if not os.path.exists(input_path):
        logger.error("Input file not found", input_path=input_path)
        raise FileNotFoundError(f"Input file not found: {input_path}")
    df = pd.read_csv(input_path) if input_path.endswith(".csv") else pd.read_parquet(input_path)
    if df.empty or target_col not in df.columns:
        logger.error("Invalid input data or target_col", input_path=input_path, target_col=target_col)
        raise ValueError("Invalid input data or target_col")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    # task 자동 판별
    if task == "auto":
        task = "classification" if y.nunique() <= 20 and y.dtype in [np.int64, np.int32, np.int16, np.int8] else "regression"
    if task == "classification":
        model = RandomForestClassifier(random_state=random_state)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = accuracy_score(y_test, y_pred)
    else:
        model = RandomForestRegressor(random_state=random_state)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        try:
            score = mean_squared_error(y_test, y_pred, squared=False)  # RMSE
        except TypeError:
            score = np.sqrt(mean_squared_error(y_test, y_pred))
    logger.info("ML training complete", task=task, score=score)
    return {"model": model, "score": score, "y_pred": y_pred, "y_test": getattr(y_test, 'values', y_test)}


def _test_train_ml_model() -> None:
    """단위 테스트: train_ml_model 함수"""
    import tempfile
    import pandas as pd
    # 분류 테스트
    df = pd.DataFrame({
        "a": [1, 2, 3, 4, 5, 6],
        "b": [0, 1, 0, 1, 0, 1],
        "target": [0, 1, 0, 1, 0, 1]
    })
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "test.csv")
        df.to_csv(input_path, index=False)
        result = train_ml_model(input_path, "target")
        assert "score" in result
        print(f"[PASS] train_ml_model unit test: score={result['score']}")

if __name__ == "__main__":
    _test_train_ml_model() 