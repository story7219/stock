"""
벤저민 그레이엄 (Benjamin Graham) 투자 전략

가치투자의 아버지, 워렌 버핏의 스승
- 내재가치 대비 저평가된 주식 발굴
- 안전마진(Margin of Safety) 중시
- 정량적 분석 기반 투자 결정
- "지능적 투자자"의 원칙 적용
"""

import logging
from typing import Dict, List, Optional, Tuple
from ..common import BaseStrategy, StrategyResult

logger = logging.getLogger(__name__)

class GrahamStrategy(BaseStrategy):
    """벤저민 그레이엄 전략 구현"""
    
    def __init__(self):
        super().__init__()
        self.strategy_name = "벤저민 그레이엄 (Benjamin Graham)"
        self.description = "정량적 가치투자와 안전마진 전략"
        
        # 가중치 설정 (총합 100%)
        self.weights = {
            'valuation_metrics': 35,     # 밸류에이션 지표
            'financial_strength': 25,    # 재무 건전성
            'margin_of_safety': 20,      # 안전마진
            'dividend_yield': 10,        # 배당수익률
            'market_position': 10        # 시장 지위
        }
    
    def analyze_stock(self, stock) -> StrategyResult:
        """그레이엄 전략으로 주식 분석"""
        try:
            scores = {}
            analysis_details = {}
            
            # 1. 밸류에이션 지표 분석 (35%)
            valuation_score, valuation_analysis = self._analyze_valuation_metrics(stock)
            scores['valuation_metrics'] = valuation_score
            analysis_details['valuation_metrics'] = valuation_analysis
            
            # 2. 재무 건전성 분석 (25%)
            financial_score, financial_analysis = self._analyze_financial_strength(stock)
            scores['financial_strength'] = financial_score
            analysis_details['financial_strength'] = financial_analysis
            
            # 3. 안전마진 분석 (20%)
            safety_score, safety_analysis = self._analyze_margin_of_safety(stock)
            scores['margin_of_safety'] = safety_score
            analysis_details['margin_of_safety'] = safety_analysis
            
            # 4. 배당수익률 분석 (10%)
            dividend_score, dividend_analysis = self._analyze_dividend_yield(stock)
            scores['dividend_yield'] = dividend_score
            analysis_details['dividend_yield'] = dividend_analysis
            
            # 5. 시장 지위 분석 (10%)
            market_score, market_analysis = self._analyze_market_position(stock)
            scores['market_position'] = market_score
            analysis_details['market_position'] = market_analysis
            
            # 총점 계산
            total_score = sum(scores[key] * self.weights[key] / 100 for key in scores)
            
            # 투자 판단
            investment_decision = self._make_investment_decision(total_score)
            
            # 핵심 포인트 추출
            key_points = self._extract_key_points(scores, analysis_details)
            
            return StrategyResult(
                total_score=total_score,
                scores=scores,
                strategy_name=self.strategy_name,
                investment_decision=investment_decision,
                key_points=key_points,
                analysis_details=analysis_details
            )
            
        except Exception as e:
            logger.error(f"그레이엄 전략 분석 중 오류: {e}")
            return self._create_error_result()
    
    def _analyze_valuation_metrics(self, stock) -> tuple:
        """밸류에이션 지표 분석"""
        try:
            score = 50
            analysis = {}
            
            # PER 분석 (그레이엄 기준: PER < 15)
            if hasattr(stock, 'per') and stock.per > 0:
                if stock.per <= 10:
                    per_score = 30
                elif stock.per <= 15:
                    per_score = 20
                elif stock.per <= 20:
                    per_score = 10
                else:
                    per_score = 0
                score += per_score
                analysis['per'] = stock.per
                analysis['per_score'] = per_score
            
            # PBR 분석 (그레이엄 기준: PBR < 1.5)
            if hasattr(stock, 'pbr') and stock.pbr > 0:
                if stock.pbr <= 1.0:
                    pbr_score = 25
                elif stock.pbr <= 1.5:
                    pbr_score = 15
                elif stock.pbr <= 2.0:
                    pbr_score = 5
                else:
                    pbr_score = 0
                score += pbr_score
                analysis['pbr'] = stock.pbr
                analysis['pbr_score'] = pbr_score
            
            # PER × PBR < 22.5 (그레이엄의 공식)
            if hasattr(stock, 'per') and hasattr(stock, 'pbr') and stock.per > 0 and stock.pbr > 0:
                per_pbr_product = stock.per * stock.pbr
                if per_pbr_product <= 22.5:
                    product_score = 20
                elif per_pbr_product <= 30:
                    product_score = 10
                else:
                    product_score = 0
                score += product_score
                analysis['per_pbr_product'] = per_pbr_product
                analysis['product_score'] = product_score
            
            return min(max(score, 0), 100), analysis
            
        except Exception as e:
            logger.error(f"밸류에이션 지표 분석 오류: {e}")
            return 50, {"error": str(e)}
    
    def _analyze_financial_strength(self, stock) -> tuple:
        """재무 건전성 분석"""
        try:
            score = 50
            analysis = {}
            
            # 부채비율 (그레이엄 기준: 부채비율 < 50%)
            if hasattr(stock, 'debt_ratio'):
                if stock.debt_ratio <= 30:
                    debt_score = 25
                elif stock.debt_ratio <= 50:
                    debt_score = 15
                elif stock.debt_ratio <= 70:
                    debt_score = 5
                else:
                    debt_score = 0
                score += debt_score
                analysis['debt_ratio'] = stock.debt_ratio
                analysis['debt_score'] = debt_score
            
            # 유동비율 (그레이엄 기준: 유동비율 > 2.0)
            if hasattr(stock, 'current_ratio'):
                if stock.current_ratio >= 2.0:
                    current_score = 20
                elif stock.current_ratio >= 1.5:
                    current_score = 15
                elif stock.current_ratio >= 1.0:
                    current_score = 10
                else:
                    current_score = 0
                score += current_score
                analysis['current_ratio'] = stock.current_ratio
                analysis['current_score'] = current_score
            
            # 순이익 안정성 (최근 10년간 적자 없음)
            if hasattr(stock, 'earnings_stability'):
                stability_score = stock.earnings_stability * 15
                score += stability_score
                analysis['earnings_stability'] = stock.earnings_stability
            
            return min(max(score, 0), 100), analysis
            
        except Exception as e:
            logger.error(f"재무 건전성 분석 오류: {e}")
            return 50, {"error": str(e)}
    
    def _analyze_margin_of_safety(self, stock) -> tuple:
        """안전마진 분석"""
        try:
            score = 50
            analysis = {}
            
            # 내재가치 대비 현재가격
            if hasattr(stock, 'intrinsic_value') and hasattr(stock, 'current_price'):
                if stock.intrinsic_value > 0:
                    margin = (stock.intrinsic_value - stock.current_price) / stock.intrinsic_value
                    if margin >= 0.5:  # 50% 이상 할인
                        margin_score = 40
                    elif margin >= 0.3:  # 30% 이상 할인
                        margin_score = 30
                    elif margin >= 0.2:  # 20% 이상 할인
                        margin_score = 20
                    elif margin >= 0.1:  # 10% 이상 할인
                        margin_score = 10
                    else:
                        margin_score = 0
                    score += margin_score
                    analysis['safety_margin'] = margin
                    analysis['margin_score'] = margin_score
            
            # 자산가치 대비 시가총액
            if hasattr(stock, 'book_value') and hasattr(stock, 'market_cap'):
                if stock.book_value > 0:
                    asset_margin = (stock.book_value - stock.market_cap) / stock.book_value
                    if asset_margin > 0:
                        asset_score = min(asset_margin * 100, 10)
                        score += asset_score
                        analysis['asset_margin'] = asset_margin
            
            return min(max(score, 0), 100), analysis
            
        except Exception as e:
            logger.error(f"안전마진 분석 오류: {e}")
            return 50, {"error": str(e)}
    
    def _analyze_dividend_yield(self, stock) -> tuple:
        """배당수익률 분석"""
        try:
            score = 50
            analysis = {}
            
            # 배당수익률 (그레이엄 선호: 3% 이상)
            if hasattr(stock, 'dividend_yield'):
                if stock.dividend_yield >= 5:
                    div_score = 30
                elif stock.dividend_yield >= 3:
                    div_score = 20
                elif stock.dividend_yield >= 1:
                    div_score = 10
                else:
                    div_score = 0
                score += div_score
                analysis['dividend_yield'] = stock.dividend_yield
                analysis['div_score'] = div_score
            
            # 배당 지속성
            if hasattr(stock, 'dividend_consistency'):
                consistency_score = stock.dividend_consistency * 20
                score += consistency_score
                analysis['dividend_consistency'] = stock.dividend_consistency
            
            return min(max(score, 0), 100), analysis
            
        except Exception as e:
            logger.error(f"배당수익률 분석 오류: {e}")
            return 50, {"error": str(e)}
    
    def _analyze_market_position(self, stock) -> tuple:
        """시장 지위 분석"""
        try:
            score = 50
            analysis = {}
            
            # 시가총액 규모 (그레이엄 선호: 대형주)
            if hasattr(stock, 'market_cap'):
                if stock.market_cap >= 1000000:  # 1조원 이상
                    size_score = 30
                elif stock.market_cap >= 500000:  # 5천억원 이상
                    size_score = 20
                elif stock.market_cap >= 100000:  # 1천억원 이상
                    size_score = 10
                else:
                    size_score = 0
                score += size_score
                analysis['market_cap'] = stock.market_cap
                analysis['size_score'] = size_score
            
            # 업종 안정성
            if hasattr(stock, 'industry_stability'):
                industry_score = stock.industry_stability * 20
                score += industry_score
                analysis['industry_stability'] = stock.industry_stability
            
            return min(max(score, 0), 100), analysis
            
        except Exception as e:
            logger.error(f"시장 지위 분석 오류: {e}")
            return 50, {"error": str(e)}
    
    def _make_investment_decision(self, total_score):
        """투자 판단 결정"""
        if total_score >= 80:
            return "강력매수 - 그레이엄 기준 우수한 가치주"
        elif total_score >= 70:
            return "매수 - 양호한 가치투자 기회"
        elif total_score >= 60:
            return "관심 - 일부 기준 충족"
        elif total_score >= 50:
            return "중립 - 가치 판단 애매"
        else:
            return "회피 - 그레이엄 기준 미달"
    
    def _extract_key_points(self, scores, analysis_details):
        """핵심 포인트 추출"""
        points = []
        
        # 밸류에이션
        if scores['valuation_metrics'] >= 70:
            points.append("우수한 밸류에이션 지표")
        elif scores['valuation_metrics'] <= 40:
            points.append("높은 밸류에이션")
        
        # 재무 건전성
        if scores['financial_strength'] >= 70:
            points.append("견고한 재무 구조")
        elif scores['financial_strength'] <= 40:
            points.append("재무 건전성 우려")
        
        # 안전마진
        if scores['margin_of_safety'] >= 70:
            points.append("충분한 안전마진 확보")
        elif scores['margin_of_safety'] <= 40:
            points.append("안전마진 부족")
        
        # 배당
        if scores['dividend_yield'] >= 70:
            points.append("매력적인 배당수익률")
        
        # 시장 지위
        if scores['market_position'] >= 70:
            points.append("안정적인 대형주")
        
        return points[:5]  # 최대 5개 포인트
    
    def _create_error_result(self):
        """오류 발생시 기본 결과 반환"""
        return StrategyResult(
            total_score=50,
            scores={},
            strategy_name=self.strategy_name,
            investment_decision="분석 불가 - 데이터 부족",
            key_points=["데이터 부족으로 분석 제한"],
            analysis_details={}
        ) 