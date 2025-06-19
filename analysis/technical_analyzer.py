"""
📈 기술적 분석기
- 가격, 거래량 등 시세 데이터를 기반으로 기술적 지표를 계산합니다.
- 이동평균, RSI, 모멘텀 등 다양한 지표를 제공합니다.
"""
import logging
from typing import Dict, Any, List
import pandas as pd

logger = logging.getLogger(__name__)

def calculate_rsi(prices: List[float], period: int = 14) -> float:
    """RSI(상대강도지수)를 계산합니다."""
    if len(prices) < period + 1:
        return 50.0
        
    try:
        series = pd.Series(prices).iloc[::-1] # 최신 데이터가 맨 뒤로 가도록 역순
        delta = series.diff()
        
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return round(rsi.iloc[-1], 2)
    except Exception as e:
        logger.warning(f"⚠️ RSI 계산 실패: {e}")
        return 50.0

def get_technical_indicators(daily_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    일봉 데이터를 기반으로 주요 기술적 지표를 계산합니다.
    :param daily_data: KIS API에서 받은 일봉 데이터 리스트
    :return: 계산된 기술적 지표 딕셔너리
    """
    if not daily_data or len(daily_data) < 20:
        return {}

    try:
        df = pd.DataFrame(daily_data)
        df['stck_clpr'] = pd.to_numeric(df['stck_clpr'])
        df['acml_vol'] = pd.to_numeric(df['acml_vol'])
        df = df.iloc[::-1] # 시간순으로 정렬 (오래된 데이터 -> 최신 데이터)

        # 이동평균
        ma5 = df['stck_clpr'].rolling(window=5).mean().iloc[-1]
        ma20 = df['stck_clpr'].rolling(window=20).mean().iloc[-1]
        ma60 = df['stck_clpr'].rolling(window=60).mean().iloc[-1]
        
        # 거래량 이동평균
        volume_ma20 = df['acml_vol'].rolling(window=20).mean().iloc[-1]

        # RSI
        # pandas Series로 전달 (순서 중요: 오래된 데이터 -> 최신)
        prices_for_rsi = df['stck_clpr'].tolist()
        rsi = calculate_rsi(prices_for_rsi, 14)

        # 골든크로스 / 데드크로스
        is_golden_cross = ma5 > ma20 and df['stck_clpr'].rolling(window=5).mean().iloc[-2] <= df['stck_clpr'].rolling(window=20).mean().iloc[-2]
        
        return {
            'ma5': round(ma5, 2),
            'ma20': round(ma20, 2),
            'ma60': round(ma60, 2),
            'rsi': rsi,
            'volume_ma20': round(volume_ma20, 2),
            'is_golden_cross': is_golden_cross
        }
    except Exception as e:
        logger.warning(f"⚠️ 기술적 지표 계산 중 오류: {e}")
        return {} 