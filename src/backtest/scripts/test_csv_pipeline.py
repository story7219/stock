import traceback
from csv_trading_ai import TradingAIPipeline
import CSVDataLoader
from pathlib import Path
import numpy as np
import os
import pandas as pd
import sys

# -*- coding: utf-8 -*-
"""
파일명: test_csv_pipeline.py
목적: CSV 기반 주식 트레이딩 AI 파이프라인 테스트
작성일: 2025-07-08
"""


def test_basic_pipeline():
    pipeline = TradingAIPipeline(
        csv_path="sample_stock_data.csv",
        output_dir="./test_results",
        model_dir="./test_models"
    )
    results = pipeline.run_pipeline(
        target_col="Close",
        sequence_length=10,
        model_type="keras",
        epochs=10
    )
    print(f"✅ 기본 테스트 완료 - 정확도: {results['accuracy']:.4f}")
    return results


def test_different_sequence_lengths():
    sequence_lengths = [5, 10, 15]
    results_comparison = []
    for seq_len in sequence_lengths:
        pipeline = TradingAIPipeline(
            csv_path="sample_stock_data.csv",
            output_dir=f"./test_results_seq_{seq_len}",
            model_dir=f"./test_models_seq_{seq_len}"
        )
        results = pipeline.run_pipeline(
            target_col="Close",
            sequence_length=seq_len,
            model_type="keras",
            epochs=5
        )
        results_comparison.append({
            'sequence_length': seq_len,
            'accuracy': results['accuracy'],
            'precision': results['precision'],
            'recall': results['recall'],
            'f1_score': results['f1_score']
        })
        print(f"  정확도: {results['accuracy']:.4f}")
    comparison_df = pd.DataFrame(results_comparison)
    print(f"\n📈 성능 비교 결과:")
    print(comparison_df.to_string(index=False))
    return comparison_df


def test_feature_selection():
    loader = CSVDataLoader("sample_stock_data.csv")
    df = loader.load_csv()
    df_clean = loader.preprocess_data()
    columns = loader.detect_columns()
    print("📋 감지된 컬럼:")
    for category, cols in columns.items():
        if cols:
            print(f"  {category}: {cols}")
    basic_features = ['Open', 'High', 'Low', 'Close', 'Volume']
    pipeline = TradingAIPipeline(
        csv_path="sample_stock_data.csv",
        output_dir="./test_results_basic",
        model_dir="./test_models_basic"
    )
    results_basic = pipeline.run_pipeline(
        target_col="Close",
        sequence_length=10,
        model_type="keras",
        epochs=5
    )
    print(f"  기본 피처 정확도: {results_basic['accuracy']:.4f}")
    return results_basic


def test_data_quality():
    loader = CSVDataLoader("sample_stock_data.csv")
    df_original = loader.load_csv()
    print(f"📊 원본 데이터 크기: {df_original.shape}")
    print(f"📋 컬럼 수: {len(df_original.columns)}")
    missing_data = df_original.isnull().sum()
    print(f"\n🔍 결측치 분석:")
    for col, missing_count in missing_data.items():
        if missing_count > 0:
            print(f"  {col}: {missing_count}개")
    df_clean = loader.preprocess_data()
    print(f"\n📊 전처리 후 데이터 크기: {df_clean.shape}")
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    print(f"\n🔍 수치형 컬럼: {list(numeric_cols)}")
    return df_original, df_clean


def test_prediction_analysis():
    predictions_file = "./results/predictions.csv"
    if os.path.exists(predictions_file):
        predictions_df = pd.read_csv(predictions_file)
        print(f"📊 예측 결과 요약:")
        print(f"  총 예측 수: {len(predictions_df)}")
        print(f"  실제 상승: {(predictions_df['y_true'] == 1).sum()}")
        print(f"  예측 상승: {(predictions_df['y_pred'] == 1).sum()}")
        print(f"\n📈 예측 확률 분석:")
        print(f"  평균 확률: {predictions_df['y_pred_prob'].mean():.4f}")
        print(f"  최대 확률: {predictions_df['y_pred_prob'].max():.4f}")
        print(f"  최소 확률: {predictions_df['y_pred_prob'].min():.4f}")
        correct_predictions = predictions_df[predictions_df['y_true'] == predictions_df['y_pred']]
        wrong_predictions = predictions_df[predictions_df['y_true'] != predictions_df['y_pred']]
        print(f"\n✅ 정확한 예측: {len(correct_predictions)}개")
        print(f"❌ 잘못된 예측: {len(wrong_predictions)}개")
        if len(wrong_predictions) > 0:
            print(f"\n🔍 잘못된 예측 상세:")
            print(wrong_predictions.to_string(index=False))
        return predictions_df
    else:
        print("❌ 예측 결과 파일을 찾을 수 없습니다.")
        return None


def main():
    print("🚀 CSV 기반 주식 트레이딩 AI 파이프라인 테스트 시작")
    print("=" * 60)
    try:
        basic_results = test_basic_pipeline()
        seq_results = test_different_sequence_lengths()
        feature_results = test_feature_selection()
        original_df, clean_df = test_data_quality()
        predictions_df = test_prediction_analysis()
        print("\n🎉 모든 테스트 완료!")
        print("=" * 60)
        print("📊 테스트 결과 요약:")
        print(f"  기본 파이프라인 정확도: {basic_results['accuracy']:.4f}")
        print(f"  최고 시퀀스 길이 성능: {seq_results.loc[seq_results['accuracy'].idxmax(), 'sequence_length']}")
        print(f"  데이터 품질: 원본 {original_df.shape} → 전처리 {clean_df.shape}")
        if predictions_df is not None:
            accuracy = (predictions_df['y_true'] == predictions_df['y_pred']).mean()
            print(f"  예측 정확도: {accuracy:.4f}")
    except Exception as e:
        print(f"❌ 테스트 중 오류 발생: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
