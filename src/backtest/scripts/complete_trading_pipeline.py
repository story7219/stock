from pathlib import Path
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
import ModelCheckpoint
from tensorflow.keras.layers import LSTM
import Dense
import Dropout
from tensorflow.keras.models import Sequential
from typing import Tuple
import List
import Dict, Any, Optional
import numpy as np
import os
import pandas as pd
import tensorflow as tf
import warnings
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: complete_trading_pipeline.py
목적: 실제 CSV 데이터 기반 완전한 주식 트레이딩 AI 파이프라인
작성일: 2025-07-08
Author: AI Assistant
"""


warnings.filterwarnings('ignore')

# 경로 설정
DATA_PATH = r"C:\data\stock_data.csv"
OUTPUT_DIR = r"C:\results"
MODEL_DIR = r"C:\models"

# 경로 자동 생성
for p in [Path(OUTPUT_DIR), Path(MODEL_DIR)]:
    p.mkdir(parents=True, exist_ok=True)

class DataQualityChecker:
    """데이터 품질 검사 및 전처리"""
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.quality_report = {}
    def check_data_quality(self) -> Dict[str, Any]:
        """데이터 품질 종합 검사"""
        report = {
            'shape': self.df.shape,
            'missing_values': self.df.isnull().sum().to_dict(),
            'duplicates': self.df.duplicated().sum(),
            'data_types': self.df.dtypes.to_dict(),
            'numeric_columns': self.df.select_dtypes(include=[np.number]).columns.tolist(),
            'date_columns': self.df.select_dtypes(include=['datetime64']).columns.tolist()
        }
        # 이상치 검사 (IQR 방법)
        outliers = {}
        for col in report['numeric_columns']:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outlier_count = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
            outliers[col] = outlier_count
        report['outliers'] = outliers
        self.quality_report = report
        return report
    def clean_data(self) -> pd.DataFrame:
        """데이터 정제"""
        df_clean = self.df.copy()
        # 1. 날짜 컬럼 처리
        date_cols = ['date', 'datetime', 'time']
        for col in date_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
                df_clean = df_clean.sort_values(col).reset_index(drop=True)
                break
        # 2. 결측치 처리
        ohlc_cols = ['Open', 'High', 'Low', 'Close']
        for col in ohlc_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(method='ffill')
        if 'Volume' in df_clean.columns:
            df_clean['Volume'] = df_clean['Volume'].fillna(0)
        # 3. 이상치 처리 (IQR 방법)
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
        # 4. 중복 제거
        df_clean = df_clean.drop_duplicates()
        # 5. 음수 값 처리 (가격은 양수여야 함)
        price_cols = ['Open', 'High', 'Low', 'Close']
        for col in price_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].abs()
        return df_clean
    def print_quality_report(self):
        """품질 보고서 출력"""
        print("=" * 60)
        print("📊 데이터 품질 보고서")
        print("=" * 60)
        print(f"데이터 크기: {self.quality_report['shape']}")
        print(f"중복 행: {self.quality_report['duplicates']}개")
        print("\n결측치 현황:")
        for col, count in self.quality_report['missing_values'].items():
            if count > 0:
                print(f"  {col}: {count}개")
        print("\n이상치 현황:")
        for col, count in self.quality_report['outliers'].items():
            if count > 0:
                print(f"  {col}: {count}개")


class TechnicalIndicators:
    """기술적 지표 계산"""

    @staticmethod
    def add_moving_averages(df: pd.DataFrame, periods: List[int] = [5, 10, 20, 60]) -> pd.DataFrame:
        """이동평균선 추가"""
        df_tech = df.copy()
        for period in periods:
            df_tech[f'MA{period}'] = df_tech['Close'].rolling(window=period).mean()
        return df_tech

    @staticmethod
    def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """RSI 추가"""
        df_tech = df.copy()
        delta = df_tech['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        df_tech['RSI'] = 100 - (100 / (1 + rs))
        return df_tech

    @staticmethod
    def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """MACD 추가"""
        df_tech = df.copy()
        exp1 = df_tech['Close'].ewm(span=fast).mean()
        exp2 = df_tech['Close'].ewm(span=slow).mean()
        df_tech['MACD'] = exp1 - exp2
        df_tech['MACD_Signal'] = df_tech['MACD'].ewm(span=signal).mean()
        df_tech['MACD_Histogram'] = df_tech['MACD'] - df_tech['MACD_Signal']
        return df_tech

    @staticmethod
    def add_bollinger_bands(df: pd.DataFrame, period: int = 20, std: int = 2) -> pd.DataFrame:
        """볼린저 밴드 추가"""
        df_tech = df.copy()
        df_tech['BB_Middle'] = df_tech['Close'].rolling(window=period).mean()
        bb_std = df_tech['Close'].rolling(window=period).std()
        df_tech['BB_Upper'] = df_tech['BB_Middle'] + (bb_std * std)
        df_tech['BB_Lower'] = df_tech['BB_Middle'] - (bb_std * std)
        df_tech['BB_Width'] = (df_tech['BB_Upper'] - df_tech['BB_Lower']) / df_tech['BB_Middle']
        return df_tech

    @staticmethod
    def add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """거래량 지표 추가"""
        df_tech = df.copy()
        df_tech['Volume_MA'] = df_tech['Volume'].rolling(window=20).mean()
        df_tech['Volume_Ratio'] = df_tech['Volume'] / df_tech['Volume_MA']
        return df_tech

    @staticmethod
    def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """모든 기술적 지표 추가"""
        df_tech = df.copy()
        df_tech = TechnicalIndicators.add_moving_averages(df_tech)
        df_tech = TechnicalIndicators.add_rsi(df_tech)
        df_tech = TechnicalIndicators.add_macd(df_tech)
        df_tech = TechnicalIndicators.add_bollinger_bands(df_tech)
        df_tech = TechnicalIndicators.add_volume_indicators(df_tech)
        return df_tech


class LabelGenerator:
    """전략별 레이블 생성"""

    @staticmethod
    def create_daytrading_labels(df: pd.DataFrame, lookahead: int = 1, threshold: float = 0.01) -> pd.Series:
        """데이트레이딩 레이블: 다음 캔들 상승/하락"""
        labels = pd.Series(index=df.index, dtype=int)
        for i in range(len(df) - lookahead):
            current_price = df['Close'].iloc[i]
            future_price = df['Close'].iloc[i + lookahead]
            change_rate = (future_price - current_price) / current_price

            if change_rate > threshold:
                labels.iloc[i] = 1  # 상승
            elif change_rate < -threshold:
                labels.iloc[i] = 0  # 하락
            else:
                labels.iloc[i] = -1  # 중립 (학습 제외)

        return labels

    @staticmethod
    def create_swing_labels(df: pd.DataFrame, lookahead: int = 5,
                          profit_threshold: float = 0.10, loss_threshold: float = -0.05) -> pd.Series:
        """스윙매매 레이블: n일 내 목표수익/손실"""
        labels = pd.Series(index=df.index, dtype=int)
        for i in range(len(df) - lookahead):
            base_price = df['Close'].iloc[i]
            future_prices = df['Close'].iloc[i+1:i+lookahead+1]

            max_profit = (future_prices.max() - base_price) / base_price
            max_loss = (future_prices.min() - base_price) / base_price

            if max_profit >= profit_threshold:
                labels.iloc[i] = 1  # 매수 신호
            elif max_loss <= loss_threshold:
                labels.iloc[i] = 0  # 매도 신호
            else:
                labels.iloc[i] = -1  # 중립 (학습 제외)

        return labels


class FeatureEngineer:
    """피처 엔지니어링"""

    @staticmethod
    def create_price_features(df: pd.DataFrame) -> pd.DataFrame:
        """가격 관련 피처 생성"""
        df_feat = df.copy()

        # 가격 변화율
        df_feat['Price_Change'] = df_feat['Close'].pct_change()
        df_feat['Price_Change_5'] = df_feat['Close'].pct_change(5)
        df_feat['Price_Change_20'] = df_feat['Close'].pct_change(20)

        # 고가/저가 비율
        df_feat['High_Low_Ratio'] = df_feat['High'] / df_feat['Low']

        # 시가/종가 비율
        df_feat['Open_Close_Ratio'] = df_feat['Open'] / df_feat['Close']

        # 변동성 (ATR 대신 단순화)
        df_feat['Volatility'] = (df_feat['High'] - df_feat['Low']) / df_feat['Close']

        return df_feat

    @staticmethod
    def create_technical_features(df: pd.DataFrame) -> pd.DataFrame:
        """기술적 지표 피처 생성"""
        df_feat = df.copy()

        # 이동평균 대비 위치
        for ma in [5, 20, 60]:
            if f'MA{ma}' in df_feat.columns:
                df_feat[f'Price_vs_MA{ma}'] = df_feat['Close'] / df_feat[f'MA{ma}'] - 1

        # RSI 구간별 인코딩
        if 'RSI' in df_feat.columns:
            df_feat['RSI_Overbought'] = (df_feat['RSI'] > 70).astype(int)
            df_feat['RSI_Oversold'] = (df_feat['RSI'] < 30).astype(int)

        # 볼린저 밴드 위치
        if 'BB_Upper' in df_feat.columns and 'BB_Lower' in df_feat.columns:
            df_feat['BB_Position'] = (df_feat['Close'] - df_feat['BB_Lower']) / (df_feat['BB_Upper'] - df_feat['BB_Lower'])

        return df_feat

    @staticmethod
    def normalize_features(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, Any]:
        """피처 정규화"""

        scaler = StandardScaler()
        df_norm = df.copy()
        df_norm[feature_cols] = scaler.fit_transform(df[feature_cols])

        return df_norm, scaler


class ModelTrainer:
    """모델 훈련"""

    def __init__(self, model_type: str = 'lstm'):
        self.model_type = model_type
        self.model = None
        self.scaler = None

    def prepare_sequences(self, df: pd.DataFrame, feature_cols: List[str],
                         label_col: str, sequence_length: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """시계열 시퀀스 준비"""
        X, y = [], []

        for i in range(len(df) - sequence_length):
            # 피처 시퀀스
            seq_features = df[feature_cols].iloc[i:i+sequence_length].values
            X.append(seq_features)

            # 레이블 (시퀀스 마지막 시점의 레이블)
            label = df[label_col].iloc[i+sequence_length-1]
            if label != -1:  # 중립이 아닌 경우만
                y.append(label)
            else:
                y.append(0)  # 중립은 0으로 처리

        return np.array(X), np.array(y)

    def train_lstm(self, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray) -> Any:
        """LSTM 모델 훈련"""

        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint(f'{MODEL_DIR}/best_lstm.h5', monitor='val_loss', save_best_only=True)
        ]

        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )

        self.model = model
        return history


def main():
    """메인 파이프라인"""
    print("🚀 주식 트레이딩 AI 파이프라인 시작")
    print("=" * 60)

    # 1. 데이터 로드
    print("📊 1단계: 데이터 로드")
    try:
        df = pd.read_csv(DATA_PATH)
        print(f"데이터 로드 완료: {df.shape}")
    except FileNotFoundError:
        print(f"❌ 데이터 파일을 찾을 수 없습니다: {DATA_PATH}")
        return

    # 2. 데이터 품질 검사 및 정제
    print("\n🔍 2단계: 데이터 품질 검사 및 정제")
    checker = DataQualityChecker(df)
    quality_report = checker.check_data_quality()
    checker.print_quality_report()

    df_clean = checker.clean_data()
    print(f"데이터 정제 완료: {df_clean.shape}")

    # 3. 기술적 지표 추가
    print("\n📈 3단계: 기술적 지표 추가")
    df_tech = TechnicalIndicators.add_all_indicators(df_clean)
    print("기술적 지표 추가 완료")

    # 4. 피처 엔지니어링
    print("\n⚙️ 4단계: 피처 엔지니어링")
    df_feat = FeatureEngineer.create_price_features(df_tech)
    df_feat = FeatureEngineer.create_technical_features(df_feat)

    # 사용할 피처 선택
    feature_cols = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'MA5', 'MA20', 'MA60', 'RSI', 'MACD', 'MACD_Signal',
        'BB_Position', 'BB_Width', 'Volume_Ratio',
        'Price_Change', 'Price_Change_5', 'Price_Change_20',
        'High_Low_Ratio', 'Open_Close_Ratio', 'Volatility',
        'Price_vs_MA5', 'Price_vs_MA20', 'Price_vs_MA60',
        'RSI_Overbought', 'RSI_Oversold'
    ]

    # 결측치 제거
    df_feat = df_feat.dropna()
    print(f"피처 엔지니어링 완료: {df_feat.shape}")

    # 5. 레이블 생성
    print("\n🏷️ 5단계: 레이블 생성")

    # 데이트레이딩 레이블
    df_feat['daytrading_label'] = LabelGenerator.create_daytrading_labels(df_feat)
    daytrading_samples = (df_feat['daytrading_label'] != -1).sum()
    print(f"데이트레이딩 레이블 생성: {daytrading_samples}개 유효 샘플")

    # 스윙매매 레이블
    df_feat['swing_label'] = LabelGenerator.create_swing_labels(df_feat)
    swing_samples = (df_feat['swing_label'] != -1).sum()
    print(f"스윙매매 레이블 생성: {swing_samples}개 유효 샘플")

    # 6. 피처 정규화
    print("\n📏 6단계: 피처 정규화")
    df_norm, scaler = FeatureEngineer.normalize_features(df_feat, feature_cols)

    # 7. 모델 훈련
    print("\n🤖 7단계: 모델 훈련")

    # 시퀀스 준비
    trainer = ModelTrainer('lstm')
    X, y = trainer.prepare_sequences(df_norm, feature_cols, 'daytrading_label', sequence_length=30)

    # 훈련/검증 분할
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    print(f"훈련 데이터: {X_train.shape}, 검증 데이터: {X_val.shape}")

    # LSTM 모델 훈련
    history = trainer.train_lstm(X_train, y_train, X_val, y_val)

    # 8. 결과 저장
    print("\n💾 8단계: 결과 저장")

    # 전처리된 데이터 저장
    df_norm.to_csv(f'{OUTPUT_DIR}/processed_data.csv', index=False)
    print(f"전처리된 데이터 저장: {OUTPUT_DIR}/processed_data.csv")

    # 품질 보고서 저장
    quality_df = pd.DataFrame([quality_report])
    quality_df.to_csv(f'{OUTPUT_DIR}/quality_report.csv', index=False)
    print(f"품질 보고서 저장: {OUTPUT_DIR}/quality_report.csv")

    print("\n✅ 파이프라인 완료!")
    print(f"모델 저장: {MODEL_DIR}/best_lstm.h5")
    print(f"결과 저장: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

