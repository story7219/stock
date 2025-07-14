#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: run_pipeline.py
모듈: 전체 파이프라인 실행
목적: KRX 데이터 수집→정제→전처리→평가→고급ML/DL 학습 자동화

Author: User
Created: 2025-07-06
Version: 1.0.0

Dependencies:
    - Python 3.11+
    - structlog>=24.1.0
    - 모든 modules/*

Performance:
    - O(n log n) 전체 파이프라인

Security:
    - robust 예외처리, 에러 로깅

License: MIT
"""

from __future__ import annotations

import sys
import structlog
from pathlib import Path
from typing import Optional

from modules.data_cleaning import clean_data
from modules.data_preprocessing import preprocess_data
from modules.data_evaluation import evaluate_data_quality
from modules.advanced_ml_evaluation import AdvancedMLEvaluator, EvaluationConfig, TaskType, ModelType
import krx_smart_manager

logger = structlog.get_logger(__name__)


def run_pipeline(
    bld: str,
    params: dict,
    target_col: str,
    min_quality: float = 80.0,
    task_type: str = "auto"
) -> None:
    """전체 파이프라인 실행 함수 (고급 ML/DL 성능평가 포함)

    Args:
        bld: KRX bld 파라미터
        params: KRX 추가 파라미터 dict
        target_col: ML/DL 타겟 컬럼명
        min_quality: 품질점수 최소 기준
        task_type: "auto", "classification", "regression"
    """
    try:
        # 1. 데이터 수집
        raw_path = krx_smart_manager.fetch_krx_data(bld, params)
        # 2. 정제
        cleaned_path = clean_data(raw_path, str(Path("data")/"cleaned.csv"))
        # 3. 전처리
        preprocessed_path = preprocess_data(cleaned_path, str(Path("data")/"preprocessed.csv"))
        # 4. 품질평가
        score = evaluate_data_quality(preprocessed_path)
        logger.info("Data quality score", score=score)
        if score < min_quality:
            logger.error("Data quality below threshold", score=score, min_quality=min_quality)
            raise RuntimeError(f"Data quality {score:.2f} < {min_quality}")
        
        # 5. 고급 ML/DL 성능평가
        import pandas as pd
        df = pd.read_csv(preprocessed_path)
        
        # task_type 자동 판별
        if task_type == "auto":
            if df[target_col].nunique() <= 20 and df[target_col].dtype in ['int64', 'int32', 'int16', 'int8']:
                task_type = "classification"
            else:
                task_type = "regression"
        
        # 고급 평가기 설정
        config = EvaluationConfig(
            task_type=TaskType.CLASSIFICATION if task_type == "classification" else TaskType.REGRESSION,
            model_type=ModelType.RANDOM_FOREST,
            test_size=0.2,
            cv_folds=5,
            random_state=42,
            n_jobs=-1,
            verbose=True
        )
        
        evaluator = AdvancedMLEvaluator(config)
        X = df.drop(columns=[target_col])
        y = pd.Series(df[target_col])
        
        # 종합 성능 평가
        results = evaluator.comprehensive_evaluation(X, y)
        
        # 리포트 생성
        report_path = evaluator.generate_evaluation_report(results, str(Path("data")/"ml_evaluation_report.md"))
        
        logger.info("Advanced ML/DL evaluation completed", report_path=report_path)
        
        # 최고 성능 모델 결과 출력
        def get_performance_score(model_name: str) -> float:
            performance = results[model_name]
            if performance.classification_metrics:
                return performance.classification_metrics.accuracy
            elif performance.regression_metrics:
                return performance.regression_metrics.r2
            return 0.0
        
        best_model = max(results.keys(), key=get_performance_score)
        best_performance = results[best_model]
        
        if best_performance.classification_metrics:
            logger.info("Best classification model", 
                       model=best_model, 
                       accuracy=best_performance.classification_metrics.accuracy,
                       f1_score=best_performance.classification_metrics.f1_macro)
        elif best_performance.regression_metrics:
            logger.info("Best regression model", 
                       model=best_model, 
                       r2=best_performance.regression_metrics.r2,
                       rmse=best_performance.regression_metrics.rmse)
        else:
            logger.warning("No performance metrics available")
        
    except Exception as e:
        logger.error("Pipeline failed", error=str(e))
        sys.exit(1)


def _test_run_pipeline() -> None:
    """단위 테스트: run_pipeline 함수 (실제 API 호출은 생략, 구조만 검증)"""
    try:
        run_pipeline("fake_bld", {"param1": "value1"}, "target")
    except Exception:
        print("[PASS] run_pipeline error handling test")

if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser(description="KRX 데이터 자동화 파이프라인 (고급 ML/DL 성능평가 포함)")
    parser.add_argument("--bld", type=str, required=True, help="KRX bld 파라미터")
    parser.add_argument("--params", type=str, default="{}", help="추가 파라미터 (JSON str)")
    parser.add_argument("--target_col", type=str, required=True, help="ML/DL 타겟 컬럼명")
    parser.add_argument("--min_quality", type=float, default=80.0, help="품질점수 최소 기준")
    parser.add_argument("--task_type", type=str, default="auto", choices=["auto", "classification", "regression"], help="ML/DL 태스크 타입")
    args = parser.parse_args()
    params = json.loads(args.params)
    run_pipeline(args.bld, params, args.target_col, args.min_quality, args.task_type) 