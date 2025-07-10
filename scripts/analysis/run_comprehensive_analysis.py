#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
종합 시장 분석 및 학습 시스템
한투 API를 이용한 모든 데이터 수집 및 머신러닝/딥러닝 학습
"""

import os
import sys
import asyncio
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.kis_data_collector import KISDataCollector
from models.market_predictor import MarketPredictor

class ComprehensiveMarketAnalyzer:
    """종합 시장 분석기"""
    
    def __init__(self):
        """초기화"""
        self.logger = logging.getLogger(__name__)
        self.data_collector = KISDataCollector()
        self.market_predictor = MarketPredictor()
        
        # 로깅 설정
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/market_analysis.log'),
                logging.StreamHandler()
            ]
        )
        
        # 로그 디렉토리 생성
        Path("logs").mkdir(exist_ok=True)
    
    async def collect_comprehensive_data(self):
        """종합 데이터 수집"""
        self.logger.info("=== 종합 데이터 수집 시작 ===")
        
        try:
            # 1. 기본 주식/선물/옵션 데이터 수집
            await self.data_collector.collect_all_data()
            
            # 2. 추가 데이터 수집 (뉴스, 경제지표, 섹터 정보 등)
            await self._collect_additional_data()
            
            # 3. 데이터 통합 및 전처리
            self._integrate_and_preprocess_data()
            
            self.logger.info("=== 데이터 수집 완료 ===")
            
        except Exception as e:
            self.logger.error(f"데이터 수집 중 오류 발생: {e}")
            raise
    
    async def _collect_additional_data(self):
        """추가 데이터 수집"""
        self.logger.info("추가 데이터 수집 시작")
        
        # 1. 섹터별 주식 데이터
        await self._collect_sector_data()
        
        # 2. 지수 데이터 (KOSPI, KOSDAQ, KOSPI200 등)
        await self._collect_index_data()
        
        # 3. 외국인/기관 투자 데이터
        await self._collect_investor_data()
        
        # 4. 재무제표 데이터
        await self._collect_financial_data()
        
        # 5. 뉴스 및 감정 데이터
        await self._collect_sentiment_data()
    
    async def _collect_sector_data(self):
        """섹터별 데이터 수집"""
        sectors = {
            'technology': ['005930', '000660', '035420', '035720'],  # IT/반도체
            'finance': ['086790', '323410', '055550'],  # 금융
            'healthcare': ['207940', '068270', '051910'],  # 바이오/헬스케어
            'energy': ['006400', '051910'],  # 에너지/화학
            'consumer': ['035420', '035720']  # 소비재
        }
        
        for sector, symbols in sectors.items():
            self.logger.info(f"{sector} 섹터 데이터 수집 중...")
            # 섹터별 데이터 수집 로직
            for symbol in symbols:
                # 기존 수집 로직 활용
                pass
    
    async def _collect_index_data(self):
        """지수 데이터 수집"""
        indices = ['KS11', 'KS26', 'KS50']  # KOSPI, KOSDAQ, KOSPI200
        
        for index in indices:
            self.logger.info(f"{index} 지수 데이터 수집 중...")
            # 지수 데이터 수집 로직
    
    async def _collect_investor_data(self):
        """투자자별 데이터 수집"""
        self.logger.info("투자자별 데이터 수집 중...")
        # 외국인, 기관, 개인 투자 데이터 수집
    
    async def _collect_financial_data(self):
        """재무제표 데이터 수집"""
        self.logger.info("재무제표 데이터 수집 중...")
        # PER, PBR, ROE, 부채비율 등 재무지표 수집
    
    async def _collect_sentiment_data(self):
        """감정 데이터 수집"""
        self.logger.info("감정 데이터 수집 중...")
        # 뉴스, 소셜미디어 감정 분석 데이터 수집
    
    def _integrate_and_preprocess_data(self):
        """데이터 통합 및 전처리"""
        self.logger.info("데이터 통합 및 전처리 시작")
        
        # 1. 모든 데이터 파일 로드
        data_files = list(Path("data/market_data").glob("*.csv"))
        
        # 2. 데이터 통합
        integrated_data = {}
        for file_path in data_files:
            data_type = file_path.stem
            df = pd.read_csv(file_path)
            integrated_data[data_type] = df
        
        # 3. 특성 엔지니어링
        for data_type, df in integrated_data.items():
            df = self._enhance_features(df)
            integrated_data[data_type] = df
        
        # 4. 통합 데이터 저장
        self._save_integrated_data(integrated_data)
    
    def _enhance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """특성 강화"""
        df = df.copy()
        
        # 1. 기술적 지표 추가
        if 'close' in df.columns:
            # 이동평균
            for period in [5, 10, 20, 60, 120]:
                df[f'MA_{period}'] = df['close'].rolling(period).mean()
                df[f'EMA_{period}'] = df['close'].ewm(span=period).mean()
            
            # RSI
            df['RSI'] = self._calculate_rsi(df['close'])
            
            # MACD
            df['MACD'], df['MACD_Signal'], df['MACD_Histogram'] = self._calculate_macd(df['close'])
            
            # 볼린저 밴드
            df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = self._calculate_bollinger_bands(df['close'])
            
            # 스토캐스틱
            if 'high' in df.columns and 'low' in df.columns:
                df['Stoch_K'], df['Stoch_D'] = self._calculate_stochastic(df['high'], df['low'], df['close'])
            
            # ATR
            if 'high' in df.columns and 'low' in df.columns:
                df['ATR'] = self._calculate_atr(df['high'], df['low'], df['close'])
        
        # 2. 가격 관련 특성
        if 'close' in df.columns:
            df['price_change'] = df['close'].pct_change()
            df['price_change_5'] = df['close'].pct_change(5)
            df['price_change_10'] = df['close'].pct_change(10)
            df['price_change_20'] = df['close'].pct_change(20)
            
            # 변동성
            df['volatility_5'] = df['price_change'].rolling(5).std()
            df['volatility_10'] = df['price_change'].rolling(10).std()
            df['volatility_20'] = df['price_change'].rolling(20).std()
            
            # 모멘텀
            df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
            df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
            df['momentum_20'] = df['close'] / df['close'].shift(20) - 1
        
        # 3. 거래량 관련 특성
        if 'volume' in df.columns:
            df['volume_ma_5'] = df['volume'].rolling(5).mean()
            df['volume_ma_20'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma_20']
            df['volume_change'] = df['volume'].pct_change()
        
        # 4. OHLC 관련 특성
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            df['high_low_ratio'] = df['high'] / df['low']
            df['open_close_ratio'] = df['open'] / df['close']
            df['body_size'] = abs(df['close'] - df['open']) / df['close']
            df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
            df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']
        
        # 5. 시간 관련 특성
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df['day_of_week'] = df['date'].dt.dayofweek
            df['month'] = df['date'].dt.month
            df['quarter'] = df['date'].dt.quarter
            df['year'] = df['date'].dt.year
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        """MACD 계산"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std: int = 2):
        """볼린저 밴드 계산"""
        middle = prices.rolling(window=period).mean()
        std_dev = prices.rolling(window=period).std()
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        return upper, middle, lower
    
    def _calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3):
        """스토캐스틱 계산"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_line = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_line = k_line.rolling(window=d_period).mean()
        return k_line, d_line
    
    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
        """ATR 계산"""
        high_low = high - low
        high_close = abs(high - close.shift())
        low_close = abs(low - close.shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr
    
    def _save_integrated_data(self, integrated_data: dict):
        """통합 데이터 저장"""
        for data_type, df in integrated_data.items():
            file_path = Path(f"data/market_data/{data_type}_enhanced.csv")
            df.to_csv(file_path, index=False)
            self.logger.info(f"강화된 데이터 저장: {file_path}")
    
    def train_comprehensive_models(self):
        """종합 모델 학습"""
        self.logger.info("=== 종합 모델 학습 시작 ===")
        
        try:
            # 1. 다양한 모델 아키텍처 학습
            self._train_multiple_architectures()
            
            # 2. 앙상블 모델 학습
            self._train_ensemble_models()
            
            # 3. 특화 모델 학습 (분류, 회귀, 시계열)
            self._train_specialized_models()
            
            # 4. 모델 성능 평가 및 비교
            self._evaluate_all_models()
            
            self.logger.info("=== 모델 학습 완료 ===")
            
        except Exception as e:
            self.logger.error(f"모델 학습 중 오류 발생: {e}")
            raise
    
    def _train_multiple_architectures(self):
        """다양한 아키텍처 학습"""
        self.logger.info("다양한 아키텍처 모델 학습 중...")
        
        # 기존 MarketPredictor 활용
        self.market_predictor.train_all_models()
    
    def _train_ensemble_models(self):
        """앙상블 모델 학습"""
        self.logger.info("앙상블 모델 학습 중...")
        
        # 1. Voting Regressor
        # 2. Stacking Regressor
        # 3. Bagging Regressor
        # 구현 예정
    
    def _train_specialized_models(self):
        """특화 모델 학습"""
        self.logger.info("특화 모델 학습 중...")
        
        # 1. 분류 모델 (상승/하락 예측)
        # 2. 회귀 모델 (가격 예측)
        # 3. 시계열 모델 (LSTM, GRU, Transformer)
        # 구현 예정
    
    def _evaluate_all_models(self):
        """모든 모델 평가"""
        self.logger.info("모델 평가 중...")
        
        # 1. 성능 지표 계산
        # 2. 모델 비교
        # 3. 최적 모델 선택
        # 구현 예정
    
    async def run_complete_analysis(self):
        """완전한 분석 실행"""
        self.logger.info("=== 종합 시장 분석 시작 ===")
        
        try:
            # 1. 데이터 수집
            await self.collect_comprehensive_data()
            
            # 2. 모델 학습
            self.train_comprehensive_models()
            
            # 3. 결과 분석 및 리포트 생성
            self._generate_analysis_report()
            
            self.logger.info("=== 종합 시장 분석 완료 ===")
            
        except Exception as e:
            self.logger.error(f"분석 중 오류 발생: {e}")
            raise
    
    def _generate_analysis_report(self):
        """분석 리포트 생성"""
        self.logger.info("분석 리포트 생성 중...")
        
        # 1. 데이터 품질 리포트
        # 2. 모델 성능 리포트
        # 3. 예측 결과 리포트
        # 4. 투자 전략 제안
        # 구현 예정

async def main():
    """메인 함수"""
    analyzer = ComprehensiveMarketAnalyzer()
    await analyzer.run_complete_analysis()

if __name__ == "__main__":
    asyncio.run(main()) 