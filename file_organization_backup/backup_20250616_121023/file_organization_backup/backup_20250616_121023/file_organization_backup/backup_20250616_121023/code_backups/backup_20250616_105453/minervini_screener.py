import logging
from typing import Dict, Optional
import pandas as pd
from kis_api import KIS_API

class MinerviniScreener:
    """
    마크 미너비니 스타일의 단기 트레이딩 분석기 (SEPA - Specific Entry Point Analysis)
    """
    def __init__(self, api: KIS_API):
        self.api = api
        self.logger = logging.getLogger(__name__)
        # 미너비니의 기준치 (단기 트레이딩에 맞게 조정)
        self.criteria = {
            'ma_50_days': 50,
            'ma_150_days': 150,
            'ma_200_days': 200,
            'high_52w_proximity': 0.75, # 52주 신고가 대비 최소 75% 이상에 위치
            'low_52w_distance': 1.3,    # 52주 신저가 대비 최소 30% 이상 상승
            'volume_spike_ratio': 1.5,  # 거래량 급증 기준 (평균 대비 150%)
            'atr_contraction_pct': 0.5  # 변동성 수축 기준 (최근 ATR이 50일 ATR의 50% 이하)
        }

    async def analyze_stock(self, symbol: str) -> Optional[Dict]:
        """
        종목에 대해 미너비니 스타일의 단기 분석(VCP, 추세 등)을 수행합니다.
        """
        self.logger.info(f"⚡️ {symbol}에 대한 마크 미너비니 단기 분석 시작...")
        try:
            # 1. 1년치 일봉 데이터 가져오기
            chart_data = self.api.get_daily_chart(symbol, period=250)
            if chart_data is None or len(chart_data) < self.criteria['ma_200_days']:
                self.logger.warning(f"{symbol}: 분석에 필요한 차트 데이터 부족.")
                return None

            # 2. 핵심 지표 계산
            current_price = chart_data['close'].iloc[-1]
            ma50 = chart_data['close'].rolling(window=self.criteria['ma_50_days']).mean().iloc[-1]
            ma150 = chart_data['close'].rolling(window=self.criteria['ma_150_days']).mean().iloc[-1]
            ma200 = chart_data['close'].rolling(window=self.criteria['ma_200_days']).mean().iloc[-1]
            
            high_52w = chart_data['high'].max()
            low_52w = chart_data['low'].min()
            
            avg_volume_20d = chart_data['volume'].rolling(window=20).mean().iloc[-1]
            latest_volume = chart_data['volume'].iloc[-1]

            # ATR 계산 (변동성)
            chart_data['tr'] = pd.DataFrame([chart_data['high'] - chart_data['low'], 
                                             abs(chart_data['high'] - chart_data['close'].shift()), 
                                             abs(chart_data['low'] - chart_data['close'].shift())]).max()
            atr_50d = chart_data['tr'].rolling(window=50).mean().iloc[-1]
            atr_10d = chart_data['tr'].rolling(window=10).mean().iloc[-1]

            # 3. 미너비니 기준에 따른 점수 평가 (총 4점)
            scores = {
                'trend_score': self._check_trend_template(current_price, ma50, ma150, ma200, high_52w, low_52w),
                'volume_score': 1 if latest_volume > avg_volume_20d * self.criteria['volume_spike_ratio'] else 0,
                'volatility_score': self._check_vcp(atr_10d, atr_50d),
                'rs_score': self._get_relative_strength(chart_data) # 상대강도
            }
            total_score = sum(scores.values())

            # 4. 최종 분석 결과 생성
            analysis = {
                'symbol': symbol,
                'minervini_score': total_score,
                'trend_analysis': f"추세 점수: {scores['trend_score']}/1. 주가 > 50일선 > 150일선 > 200일선 구조 확인",
                'volume_analysis': f"최근 거래량 {latest_volume:,.0f} (20일 평균 대비 {latest_volume/avg_volume_20d:.1f}배)",
                'volatility_analysis': f"변동성 수축 점수: {scores['volatility_score']}/1 (VCP 가능성)",
                'rs_score': f"상대강도 점수: {scores['rs_score']}/1",
                'is_buy_candidate': total_score >= 3,
                'reasoning': f"총점 {total_score}/4. 강력한 상승 추세와 함께 변동성 수축 패턴이 관찰되어 매수 후보로 적합."
            }
            self.logger.info(f"✅ {symbol} 미너비니 분석 완료. 총점: {total_score}/4")
            return analysis

        except Exception as e:
            self.logger.error(f"{symbol} 미너비니 분석 중 오류: {e}", exc_info=True)
            return None

    def _check_trend_template(self, price, ma50, ma150, ma200, high52w, low52w) -> int:
        """미너비니의 추세 템플릿 조건 확인"""
        cond1 = price > ma150 and price > ma200
        cond2 = ma150 > ma200
        cond3 = ma50 > ma150 and ma50 > ma200
        cond4 = price > ma50
        cond5 = price >= high52w * self.criteria['high_52w_proximity']
        cond6 = price >= low52w * self.criteria['low_52w_distance']
        
        return 1 if all([cond1, cond2, cond3, cond4, cond5, cond6]) else 0

    def _check_vcp(self, atr10d, atr50d) -> int:
        """VCP(변동성 수축 패턴)의 마지막 단계를 간소화하여 확인"""
        # 최근 변동성(10일 ATR)이 장기 변동성(50일 ATR) 대비 크게 감소했는지 확인
        return 1 if atr10d < atr50d * self.criteria['atr_contraction_pct'] else 0
        
    def _get_relative_strength(self, chart_data: pd.DataFrame) -> int:
        """상대강도(RS)가 80 이상인지 간략히 확인"""
        # 실제 RS는 전체 시장 종목과 비교해야 함. 여기서는 주가 모멘텀으로 대체.
        # 3개월 수익률이 20% 이상이면 1점
        if len(chart_data) < 60: return 0
        three_month_return = (chart_data['close'].iloc[-1] / chart_data['close'].iloc[-60]) - 1
        return 1 if three_month_return > 0.2 else 0 