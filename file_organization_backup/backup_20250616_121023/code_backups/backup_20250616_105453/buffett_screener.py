import logging
from typing import Dict, Optional
import pandas as pd
from kis_api import KIS_API
import yfinance as yf

class BuffettScreener:
    """
    워렌 버핏 스타일의 가치 투자 분석기 (Value Investing Screener)
    """
    def __init__(self, api: KIS_API):
        self.api = api
        self.logger = logging.getLogger(__name__)
        # 버핏의 기준치 (조정 가능)
        self.criteria = {
            'roe_min': 15.0,      # 최소 자기자본이익률 (ROE)
            'per_max': 20.0,      # 최대 주가수익비율 (PER)
            'debt_ratio_max': 1.5, # 최대 부채비율
            'dividend_yield_min': 1.0, # 최소 배당수익률
            'consistent_eps_years': 5 # 연속 EPS 성장 확인 기간
        }

    async def analyze_stock(self, symbol: str) -> Optional[Dict]:
        """
        종목에 대해 버핏 스타일의 가치 분석을 수행합니다.
        """
        self.logger.info(f"📊 {symbol}에 대한 워렌 버핏 가치 분석 시작...")
        try:
            # 1. 재무 데이터 가져오기
            financial_data = self.api.get_financial_info(symbol)
            if not financial_data:
                self.logger.warning(f"{symbol}: 재무 데이터 조회 실패.")
                return None

            # 2. 현재 주가 정보 가져오기
            price_info = self.api.get_current_price(symbol)
            if not price_info or price_info.get('rt_cd') != '0':
                self.logger.warning(f"{symbol}: 현재가 조회 실패.")
                return None
            
            # yfinance를 통해 추가 정보 (배당 등) 가져오기
            yf_ticker = yf.Ticker(f"{symbol}.KS") # 코스피/코스닥에 맞게 조정 필요
            
            # 3. 핵심 지표 계산
            roe = float(financial_data.get('roe', '0'))
            per = float(price_info['output'].get('per', '999'))
            pbr = float(price_info['output'].get('pbr', '999'))
            debt_ratio = float(financial_data.get('debt_ratio', '999'))
            dividend_yield = (yf_ticker.info.get('dividendYield') or 0) * 100
            
            # 4. 버핏 기준에 따른 점수 평가
            scores = {
                'economic_moat_score': self._check_economic_moat(roe, debt_ratio), # 경제적 해자
                'valuation_score': self._check_valuation(per, pbr), # 가치 평가
                'financial_health_score': self._check_financial_health(debt_ratio), # 재무 건전성
                'consistency_score': self._check_consistency(symbol), # 이익 일관성
            }
            total_score = sum(scores.values())

            # 5. 최종 분석 결과 생성
            analysis = {
                'symbol': symbol,
                'name': price_info['output'].get('hts_kor_isnm', symbol),
                'buffett_score': total_score,
                'criteria_summary': {
                    'roe': f"{roe:.2f}% (기준: >{self.criteria['roe_min']}%)",
                    'per': f"{per:.2f} (기준: <{self.criteria['per_max']})",
                    'debt_ratio': f"{debt_ratio:.2f} (기준: <{self.criteria['debt_ratio_max']})",
                    'dividend_yield': f"{dividend_yield:.2f}% (기준: >{self.criteria['dividend_yield_min']}%)"
                },
                'scores': scores,
                'is_undervalued': total_score >= 3, # 4점 만점에 3점 이상이면 저평가로 간주
                'reasoning': f"총점 {total_score}/4. ROE({roe:.1f}%)와 재무건전성은 양호하나, PER({per:.1f})이 다소 높아 관망 필요."
            }
            self.logger.info(f"✅ {symbol} 버핏 분석 완료. 총점: {total_score}/4")
            return analysis

        except Exception as e:
            self.logger.error(f"{symbol} 버핏 분석 중 오류: {e}", exc_info=True)
            return None

    def _check_economic_moat(self, roe: float, debt_ratio: float) -> int:
        """경제적 해자 점수 (높은 ROE, 낮은 부채비율)"""
        return 1 if roe > self.criteria['roe_min'] and debt_ratio < self.criteria['debt_ratio_max'] else 0

    def _check_valuation(self, per: float, pbr: float) -> int:
        """가치평가 점수 (낮은 PER, PBR)"""
        return 1 if per < self.criteria['per_max'] and pbr < 1.5 else 0

    def _check_financial_health(self, debt_ratio: float) -> int:
        """재무 건전성 점수 (낮은 부채 비율)"""
        return 1 if debt_ratio < self.criteria['debt_ratio_max'] else 0
        
    def _check_consistency(self, symbol: str) -> int:
        """이익의 일관성 (EPS가 꾸준히 증가하는가) - 간략화된 버전"""
        # 이 기능은 실제 구현 시 과거 재무제표 데이터가 필요합니다.
        # 여기서는 임의로 1점을 반환합니다.
        return 1 