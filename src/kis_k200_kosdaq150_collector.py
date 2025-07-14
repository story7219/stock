#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: kis_k200_kosdaq150_collector.py
목적: KOSPI200/KOSDAQ150 전체 종목의 과거 일별 시세(OHLCV) 자동수집 (KIS OpenAPI, 병렬, World-Class)
Author: Auto Trading System
Created: 2025-07-13
Version: 1.0.0

- detailed_report_20250712_054858_backup_backup.json에서 종목코드 자동 추출
- KIS OpenAPI 실계정으로 2020~2024년 일별 시세 parquet 저장
- 병렬처리, 에러/로깅, 재시도, 성능 최적화
"""

from __future__ import annotations
import os
import json
import time
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

import requests
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# 환경변수 로드
KIS_APP_KEY = os.getenv("LIVE_KIS_APP_KEY")
KIS_APP_SECRET = os.getenv("LIVE_KIS_APP_SECRET")
KIS_ACC_NO = os.getenv("LIVE_KIS_ACCOUNT_NUMBER")

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("logs/kis_k200_kosdaq150_collector.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
Path("logs").mkdir(exist_ok=True)

# 상수
DETAILED_REPORT_PATH = "backup/krx_k200_kosdaq50/krx_backup_20250712_054858/detailed_report_20250712_054858_backup_backup.json"
SAVE_DIR = Path("datasets/kis_k200_kosdaq150")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
START_DATE = "2020-01-01"
END_DATE = "2024-12-31"
MAX_WORKERS = 8
REQUEST_INTERVAL = 0.25
MAX_RETRY = 3

RETRY_WAIT_BASE = 0.5  # Adaptive Throttling base
MAX_TOTAL_RETRY = 3

# 품질 리포트 저장 경로
QUALITY_REPORT_PATH = SAVE_DIR / "quality_report.csv"
ERROR_REPORT_PATH = SAVE_DIR / "error_report.csv"
MERGED_CSV_PATH = SAVE_DIR / "merged_all.csv"

# 1분봉 설정 추가
MINUTE_DATA_ENABLED = True
MINUTE_DAYS_BACK = 30  # 최근 30일 1분봉 수집

class KISAPIClient:
    def __init__(self):
        self.access_token: Optional[str] = None
        self.token_expires_at: Optional[datetime] = None
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
    def get_access_token(self) -> str:
        if self.access_token and self.token_expires_at and datetime.now() < self.token_expires_at:
            return self.access_token
        url = "https://openapi.koreainvestment.com:9443/oauth2/tokenP"
        headers = {"content-type": "application/json"}
        payload = {
            "grant_type": "client_credentials",
            "appkey": KIS_APP_KEY,
            "appsecret": KIS_APP_SECRET
        }
        try:
            response = self.session.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            token = data["access_token"]
            self.access_token = token
            self.token_expires_at = datetime.now() + timedelta(hours=23)
            logger.info("KIS API 토큰 발급 성공")
            return token
        except Exception as e:
            logger.error(f"토큰 발급 실패: {e}")
            raise
    def make_request(self, url: str, params: Dict[str, Any], tr_id: str, retry_count: int = 0) -> Dict[str, Any]:
        if retry_count >= MAX_RETRY:
            raise Exception(f"최대 재시도 횟수 초과: {url}")
        headers = {
            "authorization": f"Bearer {self.get_access_token()}",
            "appkey": KIS_APP_KEY,
            "appsecret": KIS_APP_SECRET,
            "tr_id": tr_id,
        }
        try:
            response = self.session.get(url, headers=headers, params=params, timeout=30)
            if response.status_code == 401:
                logger.warning("토큰 만료, 재발급 후 재시도")
                self.access_token = None
                return self.make_request(url, params, tr_id, retry_count + 1)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API 요청 실패 (재시도 {retry_count + 1}/{MAX_RETRY}): {e}")
            time.sleep(REQUEST_INTERVAL * (retry_count + 1))
            return self.make_request(url, params, tr_id, retry_count + 1)

def extract_stock_codes_from_report(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    codes = [item["stock_code"] for item in data["quality_reports"] if "stock_code" in item]
    logger.info(f"종목코드 {len(codes)}개 추출 완료")
    return codes

def get_stock_ohlcv(api: KISAPIClient, symbol: str, start: str, end: str) -> pd.DataFrame:
    url = "https://openapi.koreainvestment.com:9443/uapi/domestic-stock/v1/quotations/inquire-daily-price"
    tr_id = "FHKST01010400"
    params = {
        "fid_cond_mrkt_div_code": "J",
        "fid_input_iscd": symbol,
        "fid_org_adj_prc": "0",
        "fid_period_div_code": "D",
        "fid_input_date_1": start.replace("-", ""),
        "fid_input_date_2": end.replace("-", ""),
        "fid_vol_cond_code": "0"
    }
    data = api.make_request(url, params, tr_id)
    rows = data.get("output", [])
    df = pd.DataFrame(rows)
    if not df.empty:
        df.columns = [col.lower() for col in df.columns]
        if 'stck_bsop_dt' in df.columns:
            df['date'] = pd.to_datetime(df['stck_bsop_dt'], format='%Y%m%d')
        numeric_columns = ['stck_prpr', 'stck_oprc', 'stck_hgpr', 'stck_lwpr', 'acml_vol']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def save_to_parquet(df: pd.DataFrame, symbol: str, start: str, end: str):
    fname = f"{symbol}_daily_{start.replace('-', '')}_{end.replace('-', '')}.parquet"
    df.to_parquet(SAVE_DIR / fname, index=False)
    logger.info(f"{symbol} 저장 완료: {fname}")

def collect_one(symbol: str, api: KISAPIClient, start: str, end: str, retry: int = 0, error_log: Optional[dict] = None) -> str:
    try:
        df = get_stock_ohlcv(api, symbol, start, end)
        if not df.empty:
            save_to_parquet(df, symbol, start, end)
            return f"{symbol}: OK"
        else:
            logger.warning(f"{symbol}: 데이터 없음")
            if error_log is not None:
                error_log[symbol].append("NO DATA")
            return f"{symbol}: NO DATA"
    except Exception as e:
        logger.error(f"{symbol} 수집 실패: {e}")
        if error_log is not None:
            error_log[symbol].append(str(e))
        if retry < MAX_TOTAL_RETRY:
            wait = RETRY_WAIT_BASE * (2 ** retry)
            logger.info(f"{symbol} 재시도 {retry+1}/{MAX_TOTAL_RETRY} (대기 {wait:.1f}s)")
            time.sleep(wait)
            return collect_one(symbol, api, start, end, retry+1, error_log)
        return f"{symbol}: FAIL"

def quality_report():
    """수집된 parquet 전체 품질/결측/이상치/중복 리포트 생성 및 시각화"""
    rows = []
    for f in SAVE_DIR.glob("*_daily_*.parquet"):
        try:
            df = pd.read_parquet(f)
            n = len(df)
            n_null = df.isnull().sum().sum()
            n_dup = df.duplicated().sum()
            n_outlier = 0
            for col in [c for c in df.columns if df[c].dtype in [np.float64, np.int64]]:
                q1, q3 = df[col].quantile([0.25, 0.75])
                iqr = q3 - q1
                out = ((df[col] < (q1 - 1.5*iqr)) | (df[col] > (q3 + 1.5*iqr))).sum()
                n_outlier += out
            rows.append({
                "file": f.name, "rows": n, "nulls": n_null, "dups": n_dup, "outliers": n_outlier
            })
        except Exception as e:
            logger.error(f"품질 리포트 오류: {f.name}: {e}")
    
    quality_df = pd.DataFrame(rows)
    quality_df.to_csv(QUALITY_REPORT_PATH, index=False)
    logger.info(f"품질 리포트 저장: {QUALITY_REPORT_PATH}")
    
    # 시각화 생성
    create_quality_visualization(quality_df)

def create_quality_visualization(quality_df: pd.DataFrame):
    """품질 리포트 시각화 대시보드 생성"""
    if quality_df.empty:
        logger.warning("품질 리포트 데이터가 없어 시각화를 건너뜁니다.")
        return
    
    # 시각화 설정
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('데이터 품질 리포트 대시보드', fontsize=16, fontweight='bold')
    
    # 1. 데이터 행 수 분포
    axes[0, 0].hist(quality_df['rows'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('데이터 행 수 분포')
    axes[0, 0].set_xlabel('행 수')
    axes[0, 0].set_ylabel('빈도')
    
    # 2. 결측값 분포
    axes[0, 1].scatter(quality_df['rows'], quality_df['nulls'], alpha=0.6, color='red')
    axes[0, 1].set_title('결측값 vs 데이터 행 수')
    axes[0, 1].set_xlabel('행 수')
    axes[0, 1].set_ylabel('결측값 수')
    
    # 3. 이상치 분포
    axes[1, 0].boxplot([quality_df['outliers']], labels=['이상치'])
    axes[1, 0].set_title('이상치 분포')
    axes[1, 0].set_ylabel('이상치 수')
    
    # 4. 중복값 분포
    axes[1, 1].hist(quality_df['dups'], bins=15, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[1, 1].set_title('중복값 분포')
    axes[1, 1].set_xlabel('중복값 수')
    axes[1, 1].set_ylabel('빈도')
    
    plt.tight_layout()
    viz_path = SAVE_DIR / "quality_dashboard.png"
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"품질 리포트 시각화 저장: {viz_path}")
    
    # 요약 통계 출력
    print("\n" + "="*50)
    print("📊 데이터 품질 요약 통계")
    print("="*50)
    print(f"총 파일 수: {len(quality_df)}")
    print(f"평균 행 수: {quality_df['rows'].mean():.1f}")
    print(f"총 결측값: {quality_df['nulls'].sum()}")
    print(f"총 이상치: {quality_df['outliers'].sum()}")
    print(f"총 중복값: {quality_df['dups'].sum()}")
    print("="*50)

def merge_all_to_csv():
    """모든 parquet → 통합 CSV/ML 데이터셋 변환"""
    dfs = []
    for f in SAVE_DIR.glob("*_daily_*.parquet"):
        try:
            df = pd.read_parquet(f)
            df['symbol'] = f.name.split('_')[0]
            dfs.append(df)
        except Exception as e:
            logger.error(f"CSV 병합 오류: {f.name}: {e}")
    if dfs:
        pd.concat(dfs, ignore_index=True).to_csv(MERGED_CSV_PATH, index=False)
        logger.info(f"통합 CSV 저장: {MERGED_CSV_PATH}")

def error_report(error_log: dict):
    """실패 종목별 상세 에러 리포트 자동 생성"""
    rows = [{"symbol": k, "errors": '|'.join(v)} for k, v in error_log.items() if v]
    pd.DataFrame(rows).to_csv(ERROR_REPORT_PATH, index=False)
    logger.info(f"에러 리포트 저장: {ERROR_REPORT_PATH}")

def get_stock_minute_data(api: KISAPIClient, symbol: str, days_back: int = 30) -> pd.DataFrame:
    """1분봉 데이터 수집"""
    url = "https://openapi.koreainvestment.com:9443/uapi/domestic-stock/v1/quotations/inquire-time-price"
    tr_id = "FHKST01010200"
    
    # 최근 N일 전부터 오늘까지
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    params = {
        "fid_etc_cls_code": "",
        "fid_cond_mrkt_div_code": "J",
        "fid_input_iscd": symbol,
        "fid_input_hour_1": start_date.strftime("%Y%m%d"),
        "fid_pw_data_incu_yn": "Y"
    }
    
    try:
        data = api.make_request(url, params, tr_id)
        rows = data.get("output2", [])
        df = pd.DataFrame(rows)
        
        if not df.empty:
            df.columns = [col.lower() for col in df.columns]
            if 'stck_cntg_hour' in df.columns:
                df['datetime'] = pd.to_datetime(df['stck_cntg_hour'], format='%Y%m%d%H%M%S')
            numeric_columns = ['stck_prpr', 'stck_oprc', 'stck_hgpr', 'stck_lwpr', 'cntg_vol']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    except Exception as e:
        logger.error(f"1분봉 수집 실패 {symbol}: {e}")
        return pd.DataFrame()

def save_minute_data_to_parquet(df: pd.DataFrame, symbol: str):
    """1분봉 데이터를 parquet로 저장"""
    if not df.empty:
        fname = f"{symbol}_minute_{datetime.now().strftime('%Y%m%d')}.parquet"
        df.to_parquet(SAVE_DIR / fname, index=False)
        logger.info(f"{symbol} 1분봉 저장 완료: {fname}")

def collect_minute_data(symbol: str, api: KISAPIClient) -> str:
    """1분봉 데이터 수집"""
    try:
        df = get_stock_minute_data(api, symbol, MINUTE_DAYS_BACK)
        if not df.empty:
            save_minute_data_to_parquet(df, symbol)
            return f"{symbol}: MINUTE OK"
        else:
            return f"{symbol}: MINUTE NO DATA"
    except Exception as e:
        logger.error(f"{symbol} 1분봉 수집 실패: {e}")
        return f"{symbol}: MINUTE FAIL"

def ml_auto_training():
    """ML 학습 자동화 - 가격 예측 모델"""
    try:
        # 통합 데이터 로드
        if not MERGED_CSV_PATH.exists():
            logger.warning("통합 CSV 파일이 없어 ML 학습을 건너뜁니다.")
            return
        
        df = pd.read_csv(MERGED_CSV_PATH)
        logger.info(f"ML 학습 시작: {len(df)} 행, {len(df.columns)} 컬럼")
        
        # 데이터 전처리
        df_processed = preprocess_for_ml(df)
        if df_processed.empty:
            logger.warning("전처리 후 데이터가 없어 ML 학습을 건너뜁니다.")
            return
        
        # 모델 학습 및 평가
        results = train_price_prediction_model(df_processed)
        
        # 결과 저장
        save_ml_results(results)
        
        logger.info("ML 학습 자동화 완료!")
        
    except Exception as e:
        logger.error(f"ML 학습 자동화 실패: {e}")

def preprocess_for_ml(df: pd.DataFrame) -> pd.DataFrame:
    """ML 학습을 위한 데이터 전처리"""
    try:
        # 필요한 컬럼만 선택
        required_cols = ['symbol', 'date', 'stck_prpr', 'stck_oprc', 'stck_hgpr', 'stck_lwpr', 'acml_vol']
        available_cols = [col for col in required_cols if col in df.columns]
        
        if len(available_cols) < 4:
            logger.warning("필요한 컬럼이 부족합니다.")
            return pd.DataFrame()
        
        df_subset = df[available_cols].copy()
        
        # 결측값 처리
        df_subset = df_subset.dropna()
        
        # 날짜 변환
        if 'date' in df_subset.columns:
            df_subset['date'] = pd.to_datetime(df_subset['date'])
            df_subset['year'] = df_subset['date'].dt.year
            df_subset['month'] = df_subset['date'].dt.month
            df_subset['day'] = df_subset['date'].dt.day
        
        # 기술적 지표 추가
        if 'stck_prpr' in df_subset.columns:
            df_subset['price_change'] = df_subset.groupby('symbol')['stck_prpr'].diff()
            df_subset['price_change_pct'] = df_subset.groupby('symbol')['stck_prpr'].pct_change()
            
            # 이동평균
            df_subset['ma_5'] = df_subset.groupby('symbol')['stck_prpr'].rolling(5).mean().reset_index(0, drop=True)
            df_subset['ma_20'] = df_subset.groupby('symbol')['stck_prpr'].rolling(20).mean().reset_index(0, drop=True)
        
        # 범주형 변수 인코딩
        if 'symbol' in df_subset.columns:
            df_subset['symbol_encoded'] = df_subset['symbol'].astype('category').cat.codes
        
        # 최종 정리
        numeric_cols = df_subset.select_dtypes(include=[np.number]).columns
        df_final = df_subset[numeric_cols].dropna()
        
        logger.info(f"전처리 완료: {len(df_final)} 행, {len(df_final.columns)} 컬럼")
        return df_final
        
    except Exception as e:
        logger.error(f"데이터 전처리 실패: {e}")
        return pd.DataFrame()

def train_price_prediction_model(df: pd.DataFrame) -> dict:
    """가격 예측 모델 학습 및 평가"""
    try:
        # 타겟 변수 설정 (다음날 종가 예측)
        if 'stck_prpr' not in df.columns:
            logger.warning("가격 컬럼이 없어 모델 학습을 건너뜁니다.")
            return {}
        
        # 특성과 타겟 분리
        feature_cols = [col for col in df.columns if col not in ['stck_prpr']]
        X = df[feature_cols]
        y = df['stck_prpr']
        
        # 훈련/테스트 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 모델 학습
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        # 예측
        y_pred = model.predict(X_test)
        
        # 평가
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        # 특성 중요도
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        results = {
            'model': model,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'feature_importance': feature_importance,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred
        }
        
        logger.info(f"모델 학습 완료 - R²: {r2:.4f}, RMSE: {rmse:.2f}")
        return results
        
    except Exception as e:
        logger.error(f"모델 학습 실패: {e}")
        return {}

def save_ml_results(results: dict):
    """ML 결과 저장 및 시각화"""
    if not results:
        return
    
    # 결과 저장
    ml_results_path = SAVE_DIR / "ml_results.json"
    ml_results = {
        'mse': results['mse'],
        'rmse': results['rmse'],
        'r2': results['r2'],
        'feature_importance': results['feature_importance'].to_dict('records')
    }
    
    with open(ml_results_path, 'w') as f:
        json.dump(ml_results, f, indent=2)
    
    logger.info(f"ML 결과 저장: {ml_results_path}")
    
    # 시각화
    create_ml_visualization(results)

def create_ml_visualization(results: dict):
    """ML 결과 시각화"""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('ML 모델 성능 대시보드', fontsize=16, fontweight='bold')
        
        # 1. 실제 vs 예측
        axes[0, 0].scatter(results['y_test'], results['y_pred'], alpha=0.6, color='blue')
        axes[0, 0].plot([results['y_test'].min(), results['y_test'].max()], 
                        [results['y_test'].min(), results['y_test'].max()], 'r--', lw=2)
        axes[0, 0].set_title(f'실제 vs 예측 (R² = {results["r2"]:.4f})')
        axes[0, 0].set_xlabel('실제 가격')
        axes[0, 0].set_ylabel('예측 가격')
        
        # 2. 특성 중요도
        top_features = results['feature_importance'].head(10)
        axes[0, 1].barh(range(len(top_features)), top_features['importance'])
        axes[0, 1].set_yticks(range(len(top_features)))
        axes[0, 1].set_yticklabels(top_features['feature'])
        axes[0, 1].set_title('상위 10개 특성 중요도')
        axes[0, 1].set_xlabel('중요도')
        
        # 3. 예측 오차 분포
        residuals = results['y_test'] - results['y_pred']
        axes[1, 0].hist(residuals, bins=30, alpha=0.7, color='green', edgecolor='black')
        axes[1, 0].set_title(f'예측 오차 분포 (RMSE = {results["rmse"]:.2f})')
        axes[1, 0].set_xlabel('오차')
        axes[1, 0].set_ylabel('빈도')
        
        # 4. 성능 메트릭
        metrics = ['MSE', 'RMSE', 'R²']
        values = [results['mse'], results['rmse'], results['r2']]
        axes[1, 1].bar(metrics, values, color=['red', 'orange', 'green'])
        axes[1, 1].set_title('모델 성능 메트릭')
        axes[1, 1].set_ylabel('값')
        
        plt.tight_layout()
        ml_viz_path = SAVE_DIR / "ml_dashboard.png"
        plt.savefig(ml_viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"ML 시각화 저장: {ml_viz_path}")
        
        # 성능 출력
        print("\n" + "="*50)
        print("🤖 ML 모델 성능 결과")
        print("="*50)
        print(f"R² Score: {results['r2']:.4f}")
        print(f"RMSE: {results['rmse']:.2f}")
        print(f"MSE: {results['mse']:.2f}")
        print("\n상위 5개 중요 특성:")
        for i, row in results['feature_importance'].head().iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"ML 시각화 실패: {e}")

def main():
    if not all([KIS_APP_KEY, KIS_APP_SECRET, KIS_ACC_NO]):
        raise ValueError("KIS API 환경변수가 설정되지 않았습니다")
    
    codes = extract_stock_codes_from_report(DETAILED_REPORT_PATH)
    api = KISAPIClient()
    results = []
    error_log = defaultdict(list)
    
    # 1. 일별 데이터 수집
    logger.info("=== 1단계: 일별 데이터 수집 시작 ===")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(collect_one, code, api, START_DATE, END_DATE, 0, error_log) for code in codes]
        for i, f in enumerate(as_completed(futures), 1):
            result = f.result()
            logger.info(f"[{i}/{len(codes)}] {result}")
            results.append(result)
    
    # 실패 종목만 재시도 (최대 3회)
    failed = [r.split(':')[0] for r in results if ("FAIL" in r or "NO DATA" in r)]
    for retry in range(1, MAX_TOTAL_RETRY):
        if not failed:
            break
        logger.info(f"실패 종목 {len(failed)}개 재시도 {retry}/{MAX_TOTAL_RETRY}")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(collect_one, code, api, START_DATE, END_DATE, retry, error_log) for code in failed]
            failed = []
            for i, f in enumerate(as_completed(futures), 1):
                result = f.result()
                logger.info(f"[재시도 {retry}] {result}")
                if ("FAIL" in result or "NO DATA" in result):
                    failed.append(result.split(':')[0])
    
    # 2. 1분봉 데이터 수집 (옵션)
    if MINUTE_DATA_ENABLED:
        logger.info("=== 2단계: 1분봉 데이터 수집 시작 ===")
        minute_results = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(collect_minute_data, code, api) for code in codes[:50]]  # 상위 50개만
            for i, f in enumerate(as_completed(futures), 1):
                result = f.result()
                logger.info(f"[1분봉 {i}/50] {result}")
                minute_results.append(result)
    
    # 3. 품질/결측/이상치 리포트
    logger.info("=== 3단계: 품질 리포트 생성 ===")
    quality_report()
    
    # 4. 통합 CSV/ML 데이터셋
    logger.info("=== 4단계: 통합 CSV 생성 ===")
    merge_all_to_csv()
    
    # 5. 에러 상세 리포트
    logger.info("=== 5단계: 에러 리포트 생성 ===")
    error_report(error_log)
    
    # 6. ML 학습 자동화
    logger.info("=== 6단계: ML 학습 자동화 ===")
    ml_auto_training()
    
    logger.info("🎉 전체 파이프라인 완료! 모든 확장 기능이 적용되었습니다.")

if __name__ == "__main__":
    main() 