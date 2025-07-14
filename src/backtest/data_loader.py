#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: data_loader.py
모듈: 데이터 로딩/전처리/검증
목적: 실전 자동매매 백테스트용 데이터 적재/전처리/검증/캐싱

Author: World-Class Python Engineer
Created: 2025-07-13
Version: 1.0.0
"""
from __future__ import annotations
from typing import Any, Dict, Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    """실전 데이터 로딩/전처리/검증/캐싱 담당 클래스"""
    def __init__(self, data_path: str):
        self.data_path = data_path
        self._cache: Optional[pd.DataFrame] = None

    def load(self) -> pd.DataFrame:
        """데이터 로딩 및 캐싱. 파일 없으면 샘플 데이터 생성."""
        import os
        try:
            if self._cache is not None:
                logger.info("Returning cached data.")
                return self._cache
            if not os.path.exists(self.data_path):
                logger.warning(f"{self.data_path} not found. 샘플 데이터 생성.")
                df = self._generate_sample_data()
                df.to_csv(self.data_path, index=False)
                logger.info(f"샘플 데이터 저장: {self.data_path}")
            else:
                # parquet 파일인지 확인
                if self.data_path.endswith('.parquet'):
                    df = pd.read_parquet(self.data_path)
                    logger.info(f"Parquet 데이터 로드: {self.data_path}")
                else:
                    df = pd.read_csv(self.data_path)
                    logger.info(f"CSV 데이터 로드: {self.data_path}")
            self._cache = df
            logger.info(f"Data loaded from {self.data_path}")
            return df
        except Exception as e:
            logger.error(f"Data loading failed: {e}")
            raise

    def _generate_sample_data(self, n: int = 200) -> pd.DataFrame:
        """랜덤 샘플 데이터프레임 생성 (날짜, 종가, 수익률 등)"""
        import numpy as np
        import pandas as pd
        from datetime import datetime, timedelta
        np.random.seed(42)
        dates = [datetime.today() - timedelta(days=i) for i in range(n)][::-1]
        close = np.cumprod(1 + np.random.normal(0.0005, 0.01, n)) * 10000
        returns = np.insert(np.diff(close) / close[:-1], 0, 0)
        df = pd.DataFrame({
            'date': [d.strftime('%Y-%m-%d') for d in dates],
            'close': close,
            'return': returns
        })
        return df

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터 전처리: 실제 KRX 데이터에 맞게 수익률 계산"""
        try:
            # 실제 KRX 데이터 구조에 맞게 처리
            if '종가' in df.columns:
                # 종가 데이터로부터 수익률 계산
                df = df.sort_values('날짜')  # 날짜순 정렬
                df['return'] = df['종가'].pct_change()  # 수익률 계산
                df = df.dropna()  # NaN 제거
                logger.info(f"실제 KRX 데이터 전처리 완료: {len(df)}행")
            else:
                logger.info("샘플 데이터 사용")
            return df
        except Exception as e:
            logger.error(f"데이터 전처리 오류: {e}")
            raise

    def validate(self, df: pd.DataFrame) -> bool:
        """데이터 유효성 검증 (구현 예정)"""
        # TODO: 컬럼/타입/결측 등 검증
        return True 