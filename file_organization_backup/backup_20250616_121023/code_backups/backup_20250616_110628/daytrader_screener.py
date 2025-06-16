import logging
from typing import Dict, Optional
import pandas as pd
from kis_api import KIS_API

class DayTraderScreener:
    """
    래리 코너스 & 린다 라쉬케 스타일의 단기 트레이딩 분석기
    (평균 회귀, 과매도/과매수 전략)
    """
    def __init__(self, api: KIS_API):
        self.api = api
        self.logger = logging.getLogger(__name__)
        # 단기 트레이딩 기준 (래리 코너스 RSI(2) 전략 기반)
        self.criteria = {
            'rsi_period': 2,
            'rsi_oversold': 10,  # 극단적 과매도 기준
            'rsi_overbought': 90, # 극단적 과매수 기준
            'ma_period': 200,      # 장기 추세 판단선
        }

    async def analyze_stock(self, symbol: str) -> Optional[Dict]:
        """
        종목에 대해 단기 과매도/과매수 상태를 분석합니다.
        """
        self.logger.info(f"📈 {symbol}에 대한 단기 데이트레이딩 분석 시작...")
        try:
            # 1. 1년치 일봉 데이터 가져오기
            chart_data = self.api.get_daily_chart(symbol, period=250)
            if chart_data is None or len(chart_data) < self.criteria['ma_period']:
                self.logger.warning(f"{symbol}: 분석에 필요한 차트 데이터 부족.")
                return None

            # 2. RSI 및 이동평균 계산
            current_price = chart_data['close'].iloc[-1]
            ma200 = chart_data['close'].rolling(window=self.criteria['ma_period']).mean().iloc[-1]
            rsi = self._calculate_rsi(chart_data, self.criteria['rsi_period'])

            # 3. 매매 신호 판단
            buy_signal = False
            sell_signal = False
            reasoning = "중립. 특별한 과매도/과매수 신호 없음."

            # 매수 조건: 주가가 200일선 위에 있고(상승 추세) + RSI가 극단적 과매도 상태
            if current_price > ma200 and rsi < self.criteria['rsi_oversold']:
                buy_signal = True
                reasoning = f"매수 신호: 상승 추세 중 단기 과매도 상태 (RSI({self.criteria['rsi_period']}) = {rsi:.2f})"
            
            # 매도 조건: 주가가 200일선 아래에 있고(하락 추세) + RSI가 극단적 과매수 상태
            elif current_price < ma200 and rsi > self.criteria['rsi_overbought']:
                sell_signal = True
                reasoning = f"매도 신호: 하락 추세 중 단기 과매수 상태 (RSI({self.criteria['rsi_period']}) = {rsi:.2f})"

            # 4. 최종 분석 결과 생성
            analysis = {
                'symbol': symbol,
                'strategy': "Larry Connors RSI(2)",
                'rsi_value': rsi,
                'is_above_ma200': current_price > ma200,
                'buy_setup': current_price > ma200 and rsi < self.criteria['rsi_oversold'],
                'sell_setup': current_price < ma200 and rsi > self.criteria['rsi_overbought'],
                'reasoning': reasoning
            }
            self.logger.info(f"✅ {symbol} 데이트레이딩 분석 완료. {reasoning}")
            return analysis

        except Exception as e:
            self.logger.error(f"{symbol} 데이트레이딩 분석 중 오류: {e}", exc_info=True)
            return None

    def _calculate_rsi(self, df: pd.DataFrame, period: int) -> float:
        """RSI 계산"""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] 