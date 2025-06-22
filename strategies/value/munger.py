"""
찰리 멍거 (Charlie Munger) 투자 전략

워렌 버핏의 파트너, 버크셔 해서웨이 부회장
- 다학제적 사고 모델 적용
- 우수한 비즈니스를 합리적 가격에 매수
- 인지 편향과 심리적 오류 회피
- "좋은 회사를 적절한 가격에 사는 것이 적절한 회사를 좋은 가격에 사는 것보다 낫다"
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from ..common import BaseStrategy, StrategyResult

logger = logging.getLogger(__name__)

class MungerStrategy(BaseStrategy):
    """찰리 멍거 전략 구현"""
    
    def __init__(self):
        super().__init__()
        self.strategy_name = "찰리 멍거 (Charlie Munger)"
        self.description = "다학제적 사고와 우수 기업 장기보유 전략"
        
        # 가중치 설정 (총합 100%)
        self.weights = {
            'business_quality': 35,      # 사업 품질
            'mental_models': 25,         # 사고 모델
            'management_quality': 20,    # 경영진 품질
            'rational_price': 15,        # 합리적 가격
            'long_term_moat': 5          # 장기 해자
        }
    
    def analyze_stock(self, stock) -> StrategyResult:
        """멍거 전략으로 주식 분석"""
        try:
            scores = {}
            analysis_details = {}
            
            # 1. 사업 품질 분석 (35%)
            quality_score, quality_analysis = self._analyze_business_quality(stock)
            scores['business_quality'] = quality_score
            analysis_details['business_quality'] = quality_analysis
            
            # 2. 사고 모델 분석 (25%)
            mental_score, mental_analysis = self._analyze_mental_models(stock)
            scores['mental_models'] = mental_score
            analysis_details['mental_models'] = mental_analysis
            
            # 3. 경영진 품질 분석 (20%)
            management_score, management_analysis = self._analyze_management_quality(stock)
            scores['management_quality'] = management_score
            analysis_details['management_quality'] = management_analysis
            
            # 4. 합리적 가격 분석 (15%)
            price_score, price_analysis = self._analyze_rational_price(stock)
            scores['rational_price'] = price_score
            analysis_details['rational_price'] = price_analysis
            
            # 5. 장기 해자 분석 (5%)
            moat_score, moat_analysis = self._analyze_long_term_moat(stock)
            scores['long_term_moat'] = moat_score
            analysis_details['long_term_moat'] = moat_analysis
            
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
            logger.error(f"멍거 전략 분석 중 오류: {e}")
            return self._create_error_result()
    
    def _analyze_business_quality(self, stock) -> tuple:
        """사업 품질 분석"""
        try:
            score = 50
            analysis = {}
            
            # 경쟁 우위
            if hasattr(stock, 'competitive_advantage'):
                advantage_score = stock.competitive_advantage * 30
                score += advantage_score
                analysis['competitive_advantage'] = stock.competitive_advantage
            
            # 수익성 안정성
            if hasattr(stock, 'earnings_stability'):
                stability_score = stock.earnings_stability * 25
                score += stability_score
                analysis['earnings_stability'] = stock.earnings_stability
            
            # 현금 창출 능력
            if hasattr(stock, 'cash_generation'):
                cash_score = stock.cash_generation * 20
                score += cash_score
                analysis['cash_generation'] = stock.cash_generation
            
            # 자본 효율성
            if hasattr(stock, 'capital_efficiency'):
                efficiency_score = stock.capital_efficiency * 15
                score += efficiency_score
                analysis['capital_efficiency'] = stock.capital_efficiency
            
            # 사업 모델 단순성
            if hasattr(stock, 'business_simplicity'):
                simplicity_score = stock.business_simplicity * 10
                score += simplicity_score
                analysis['business_simplicity'] = stock.business_simplicity
            
            return min(max(score, 0), 100), analysis
            
        except Exception as e:
            logger.error(f"사업 품질 분석 오류: {e}")
            return 50, {"error": str(e)}
    
    def _analyze_mental_models(self, stock) -> tuple:
        """사고 모델 분석"""
        try:
            score = 50
            analysis = {}
            
            # 다학제적 접근
            if hasattr(stock, 'multidisciplinary_score'):
                multi_score = stock.multidisciplinary_score * 30
                score += multi_score
                analysis['multidisciplinary_score'] = stock.multidisciplinary_score
            
            # 인버전 사고 (역발상)
            if hasattr(stock, 'inversion_thinking'):
                inversion_score = stock.inversion_thinking * 25
                score += inversion_score
                analysis['inversion_thinking'] = stock.inversion_thinking
            
            # 확률적 사고
            if hasattr(stock, 'probabilistic_thinking'):
                prob_score = stock.probabilistic_thinking * 20
                score += prob_score
                analysis['probabilistic_thinking'] = stock.probabilistic_thinking
            
            # 시스템 사고
            if hasattr(stock, 'systems_thinking'):
                systems_score = stock.systems_thinking * 15
                score += systems_score
                analysis['systems_thinking'] = stock.systems_thinking
            
            # 심리학적 편향 인식
            if hasattr(stock, 'bias_awareness'):
                bias_score = stock.bias_awareness * 10
                score += bias_score
                analysis['bias_awareness'] = stock.bias_awareness
            
            return min(max(score, 0), 100), analysis
            
        except Exception as e:
            logger.error(f"사고 모델 분석 오류: {e}")
            return 50, {"error": str(e)}
    
    def _analyze_management_quality(self, stock) -> tuple:
        """경영진 품질 분석"""
        try:
            score = 50
            analysis = {}
            
            # 경영진 정직성
            if hasattr(stock, 'management_integrity'):
                integrity_score = stock.management_integrity * 35
                score += integrity_score
                analysis['management_integrity'] = stock.management_integrity
            
            # 자본 배분 능력
            if hasattr(stock, 'capital_allocation_skill'):
                allocation_score = stock.capital_allocation_skill * 30
                score += allocation_score
                analysis['capital_allocation_skill'] = stock.capital_allocation_skill
            
            # 장기적 사고
            if hasattr(stock, 'long_term_thinking'):
                longterm_score = stock.long_term_thinking * 20
                score += longterm_score
                analysis['long_term_thinking'] = stock.long_term_thinking
            
            # 주주 친화적
            if hasattr(stock, 'shareholder_friendly'):
                shareholder_score = stock.shareholder_friendly * 15
                score += shareholder_score
                analysis['shareholder_friendly'] = stock.shareholder_friendly
            
            return min(max(score, 0), 100), analysis
            
        except Exception as e:
            logger.error(f"경영진 품질 분석 오류: {e}")
            return 50, {"error": str(e)}
    
    def _analyze_rational_price(self, stock) -> tuple:
        """합리적 가격 분석"""
        try:
            score = 50
            analysis = {}
            
            # 내재가치 대비 할인
            if hasattr(stock, 'intrinsic_value_discount'):
                discount_score = stock.intrinsic_value_discount * 40
                score += discount_score
                analysis['intrinsic_value_discount'] = stock.intrinsic_value_discount
            
            # 안전마진
            if hasattr(stock, 'margin_of_safety'):
                safety_score = stock.margin_of_safety * 30
                score += safety_score
                analysis['margin_of_safety'] = stock.margin_of_safety
            
            # 기회비용 고려
            if hasattr(stock, 'opportunity_cost'):
                opportunity_score = max(20 - stock.opportunity_cost * 10, 0)
                score += opportunity_score
                analysis['opportunity_cost'] = stock.opportunity_cost
            
            # 가격 합리성
            if hasattr(stock, 'price_rationality'):
                rationality_score = stock.price_rationality * 10
                score += rationality_score
                analysis['price_rationality'] = stock.price_rationality
            
            return min(max(score, 0), 100), analysis
            
        except Exception as e:
            logger.error(f"합리적 가격 분석 오류: {e}")
            return 50, {"error": str(e)}
    
    def _analyze_long_term_moat(self, stock) -> tuple:
        """장기 해자 분석"""
        try:
            score = 50
            analysis = {}
            
            # 브랜드 파워
            if hasattr(stock, 'brand_power'):
                brand_score = stock.brand_power * 40
                score += brand_score
                analysis['brand_power'] = stock.brand_power
            
            # 네트워크 효과
            if hasattr(stock, 'network_effect'):
                network_score = stock.network_effect * 30
                score += network_score
                analysis['network_effect'] = stock.network_effect
            
            # 규모의 경제
            if hasattr(stock, 'economies_of_scale'):
                scale_score = stock.economies_of_scale * 20
                score += scale_score
                analysis['economies_of_scale'] = stock.economies_of_scale
            
            # 전환 비용
            if hasattr(stock, 'switching_costs'):
                switching_score = stock.switching_costs * 10
                score += switching_score
                analysis['switching_costs'] = stock.switching_costs
            
            return min(max(score, 0), 100), analysis
            
        except Exception as e:
            logger.error(f"장기 해자 분석 오류: {e}")
            return 50, {"error": str(e)}
    
    def _make_investment_decision(self, total_score):
        """투자 판단 결정"""
        if total_score >= 80:
            return "강력매수 - 완벽한 우수 기업"
        elif total_score >= 70:
            return "매수 - 합리적 가격의 좋은 기업"
        elif total_score >= 60:
            return "관심 - 장기 관찰 대상"
        elif total_score >= 50:
            return "중립 - 더 나은 기회 대기"
        else:
            return "회피 - 투자 기준 미달"
    
    def _extract_key_points(self, scores, analysis_details):
        """핵심 포인트 추출"""
        points = []
        
        # 사업 품질
        if scores['business_quality'] >= 70:
            points.append("우수한 사업 품질")
        elif scores['business_quality'] <= 40:
            points.append("사업 품질 부족")
        
        # 사고 모델
        if scores['mental_models'] >= 70:
            points.append("다학제적 분석 우수")
        
        # 경영진 품질
        if scores['management_quality'] >= 70:
            points.append("뛰어난 경영진")
        
        # 합리적 가격
        if scores['rational_price'] >= 70:
            points.append("합리적 투자 가격")
        elif scores['rational_price'] <= 40:
            points.append("과대평가 우려")
        
        # 장기 해자
        if scores['long_term_moat'] >= 70:
            points.append("강력한 경쟁 해자")
        
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