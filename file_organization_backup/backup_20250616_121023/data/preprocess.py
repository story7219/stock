# preprocess.py
# 데이터 전처리 및 신저가/횡보 구간 탐지 함수 정의 

import pandas as pd

def preprocess_ohlcv(df):
    """
    결측치 처리, 이동평균선 등 필요한 전처리
    """
    df = df.copy()
    df['MA30'] = df['Close'].rolling(window=30).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()
    return df 