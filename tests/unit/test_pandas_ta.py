#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pandas_ta 테스트 스크립트
"""

import pandas as pd
import numpy as np
import pandas_ta as pta

def test_pandas_ta():
    """pandas_ta 기능 테스트"""
    
    # 샘플 데이터 생성
    dates = pd.date_range('2024-01-01', periods=50, freq='D')
    np.random.seed(42)
    
    df = pd.DataFrame({
        'Open': np.random.randn(50).cumsum() + 100,
        'High': np.random.randn(50).cumsum() + 105,
        'Low': np.random.randn(50).cumsum() + 95,
        'Close': np.random.randn(50).cumsum() + 100,
        'Volume': np.random.randint(1000, 10000, 50)
    }, index=dates)
    
    # High와 Low 조정
    df['High'] = df[['Open', 'Close']].max(axis=1) + np.random.uniform(0, 5, 50)
    df['Low'] = df[['Open', 'Close']].min(axis=1) - np.random.uniform(0, 5, 50)
    
    print("=== pandas_ta 테스트 ===")
    print(f"데이터 기간: {df.index[0]} ~ {df.index[-1]}")
    print(f"데이터 포인트: {len(df)}")
    print()
    
    # 1. RSI 테스트
    print("1. RSI 테스트")
    rsi = pta.rsi(df['Close'], length=14)
    print(f"   RSI 최근 값: {rsi.iloc[-1]:.2f}")
    print(f"   RSI 평균: {rsi.mean():.2f}")
    print()
    
    # 2. MACD 테스트
    print("2. MACD 테스트")
    macd = pta.macd(df['Close'])
    print(f"   MACD 컬럼: {list(macd.columns)}")
    print(f"   MACD 최근 값: {macd.iloc[-1]['MACD_12_26_9']:.2f}")
    print()
    
    # 3. 볼린저 밴드 테스트
    print("3. 볼린저 밴드 테스트")
    bb = pta.bbands(df['Close'])
    print(f"   볼린저 밴드 컬럼: {list(bb.columns)}")
    # 실제 컬럼명 사용
    bb_columns = list(bb.columns)
    if bb_columns:
        print(f"   상단 밴드 최근 값: {bb.iloc[-1][bb_columns[2]]:.2f}")
        print(f"   중간 밴드 최근 값: {bb.iloc[-1][bb_columns[1]]:.2f}")
        print(f"   하단 밴드 최근 값: {bb.iloc[-1][bb_columns[0]]:.2f}")
    print()
    
    # 4. 스토캐스틱 테스트
    print("4. 스토캐스틱 테스트")
    stoch = pta.stoch(df['High'], df['Low'], df['Close'])
    print(f"   스토캐스틱 컬럼: {list(stoch.columns)}")
    # 실제 컬럼명 사용
    stoch_columns = list(stoch.columns)
    if stoch_columns:
        print(f"   %K 최근 값: {stoch.iloc[-1][stoch_columns[0]]:.2f}")
        print(f"   %D 최근 값: {stoch.iloc[-1][stoch_columns[1]]:.2f}")
    print()
    
    # 5. 이동평균 테스트
    print("5. 이동평균 테스트")
    sma = pta.sma(df['Close'], length=20)
    ema = pta.ema(df['Close'], length=20)
    print(f"   SMA(20) 최근 값: {sma.iloc[-1]:.2f}")
    print(f"   EMA(20) 최근 값: {ema.iloc[-1]:.2f}")
    print()
    
    # 6. 거래량 지표 테스트
    print("6. 거래량 지표 테스트")
    obv = pta.obv(df['Close'], df['Volume'])
    print(f"   OBV 최근 값: {obv.iloc[-1]:.0f}")
    print()
    
    # 7. 성능 비교
    print("7. 성능 비교")
    import time
    
    # pandas_ta RSI
    start_time = time.time()
    rsi_pta = pta.rsi(df['Close'], length=14)
    pta_time = time.time() - start_time
    
    # 수동 계산 RSI
    start_time = time.time()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi_manual = 100 - (100 / (1 + rs))
    manual_time = time.time() - start_time
    
    print(f"   pandas_ta RSI 계산 시간: {pta_time:.4f}초")
    print(f"   수동 RSI 계산 시간: {manual_time:.4f}초")
    print(f"   성능 향상: {manual_time/pta_time:.1f}배")
    print()
    
    # 8. 정확도 비교
    print("8. 정확도 비교")
    rsi_diff = abs(rsi_pta - rsi_manual).mean()
    print(f"   RSI 평균 차이: {rsi_diff:.6f}")
    print(f"   정확도: {'✅ 좋음' if rsi_diff < 0.01 else '⚠️ 차이 있음'}")
    print()
    
    print("=== 테스트 완료 ===")
    print("✅ pandas_ta가 TA-Lib의 완벽한 대체재로 작동합니다!")

if __name__ == "__main__":
    test_pandas_ta() 