"""
존 보글 (John Bogle) 투자 전략

뱅가드 그룹 창립자, 인덱스 펀드의 아버지
- 저비용 인덱스 펀드 투자
- 장기 보유와 분산투자
- 시장 타이밍 회피
"""

import logging
from typing import Dict, List
from ..common import BaseStrategy, StrategyResult

logger = logging.getLogger(__name__)

class BogleStrategy(BaseStrategy):
    """존 보글 전략 구현"""
    
    def __init__(self):
        super().__init__()
        self.strategy_name = "존 보글 (John Bogle)"
        self.description = "저비용 인덱스 펀드 패시브 투자 전략"
        
        # 가중치 설정 (총합 100%)
        self.weights = {
            'low_cost': 35,              # 저비용
            'broad_diversification': 25, # 광범위 분산
            'long_term_hold': 20,        # 장기 보유
            'market_efficiency': 15,     # 시장 효율성
            'simplicity': 5              # 단순성
        }
    
    def analyze_stock(self, stock) -> StrategyResult:
        """보글 전략으로 주식 분석"""
        try:
            scores = {}
            analysis_details = {}
            
            # 1. 저비용 분석 (35%)
            cost_score, cost_analysis = self._analyze_low_cost(stock)
            scores['low_cost'] = cost_score
            analysis_details['low_cost'] = cost_analysis
            
            # 2. 광범위 분산 분석 (25%)
            diversification_score, diversification_analysis = self._analyze_broad_diversification(stock)
            scores['broad_diversification'] = diversification_score
            analysis_details['broad_diversification'] = diversification_analysis
            
            # 3. 장기 보유 분석 (20%)
            long_term_score, long_term_analysis = self._analyze_long_term_hold(stock)
            scores['long_term_hold'] = long_term_score
            analysis_details['long_term_hold'] = long_term_analysis
            
            # 4. 시장 효율성 분석 (15%)
            efficiency_score, efficiency_analysis = self._analyze_market_efficiency(stock)
            scores['market_efficiency'] = efficiency_score
            analysis_details['market_efficiency'] = efficiency_analysis
            
            # 5. 단순성 분석 (5%)
            simplicity_score, simplicity_analysis = self._analyze_simplicity(stock)
            scores['simplicity'] = simplicity_score
            analysis_details['simplicity'] = simplicity_analysis
            
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
            logger.error(f"보글 전략 분석 중 오류: {e}")
            return self._create_error_result()
    
    def _analyze_low_cost(self, stock) -> tuple:
        """저비용 분석"""
        try:
            score = 50
            analysis = {}
            
            # 운용 보수
            if hasattr(stock, 'expense_ratio'):
                # 낮은 운용보수일수록 높은 점수
                if stock.expense_ratio <= 0.1:
                    expense_score = 40
                elif stock.expense_ratio <= 0.2:
                    expense_score = 30
                elif stock.expense_ratio <= 0.5:
                    expense_score = 20
                elif stock.expense_ratio <= 1.0:
                    expense_score = 10
                else:
                    expense_score = 0
                score += expense_score
                analysis['expense_ratio'] = stock.expense_ratio
            
            # 거래 비용
            if hasattr(stock, 'trading_cost'):
                # 낮은 거래비용일수록 좋음
                trading_score = max(30 - stock.trading_cost * 100, 0)
                score += trading_score
                analysis['trading_cost'] = stock.trading_cost
            
            # 세금 효율성
            if hasattr(stock, 'tax_efficiency'):
                tax_score = stock.tax_efficiency * 20
                score += tax_score
                analysis['tax_efficiency'] = stock.tax_efficiency
            
            # 추적 오차
            if hasattr(stock, 'tracking_error'):
                # 낮은 추적오차일수록 좋음
                tracking_score = max(10 - stock.tracking_error * 50, 0)
                score += tracking_score
                analysis['tracking_error'] = stock.tracking_error
            
            return min(max(score, 0), 100), analysis
            
        except Exception as e:
            logger.error(f"저비용 분석 오류: {e}")
            return 50, {"error": str(e)}
    
    def _analyze_broad_diversification(self, stock) -> tuple:
        """광범위 분산 분석"""
        try:
            score = 50
            analysis = {}
            
            # 종목 수
            if hasattr(stock, 'number_of_holdings'):
                if stock.number_of_holdings >= 500:
                    holdings_score = 30
                elif stock.number_of_holdings >= 100:
                    holdings_score = 25
                elif stock.number_of_holdings >= 50:
                    holdings_score = 20
                elif stock.number_of_holdings >= 20:
                    holdings_score = 15
                else:
                    holdings_score = 10
                score += holdings_score
                analysis['number_of_holdings'] = stock.number_of_holdings
            
            # 섹터 분산도
            if hasattr(stock, 'sector_concentration'):
                # 낮은 집중도일수록 좋음
                sector_score = max(25 - stock.sector_concentration * 50, 0)
                score += sector_score
                analysis['sector_concentration'] = stock.sector_concentration
            
            # 지역 분산도
            if hasattr(stock, 'geographic_exposure'):
                geo_score = stock.geographic_exposure * 20
                score += geo_score
                analysis['geographic_exposure'] = stock.geographic_exposure
            
            # 시가총액 분산
            if hasattr(stock, 'market_cap_diversification'):
                cap_score = stock.market_cap_diversification * 15
                score += cap_score
                analysis['market_cap_diversification'] = stock.market_cap_diversification
            
            return min(max(score, 0), 100), analysis
            
        except Exception as e:
            logger.error(f"광범위 분산 분석 오류: {e}")
            return 50, {"error": str(e)}
    
    def _analyze_long_term_hold(self, stock) -> tuple:
        """장기 보유 분석"""
        try:
            score = 50
            analysis = {}
            
            # 장기 성과 일관성
            if hasattr(stock, 'long_term_consistency'):
                consistency_score = stock.long_term_consistency * 30
                score += consistency_score
                analysis['long_term_consistency'] = stock.long_term_consistency
            
            # 배당 성장률
            if hasattr(stock, 'dividend_growth_rate'):
                dividend_score = min(stock.dividend_growth_rate * 5, 25)
                score += dividend_score
                analysis['dividend_growth_rate'] = stock.dividend_growth_rate
            
            # 복리 효과
            if hasattr(stock, 'compound_growth_rate'):
                compound_score = min(stock.compound_growth_rate * 3, 20)
                score += compound_score
                analysis['compound_growth_rate'] = stock.compound_growth_rate
            
            # 변동성 (낮을수록 좋음)
            if hasattr(stock, 'volatility'):
                volatility_score = max(15 - stock.volatility, 0)
                score += volatility_score
                analysis['volatility'] = stock.volatility
            
            return min(max(score, 0), 100), analysis
            
        except Exception as e:
            logger.error(f"장기 보유 분석 오류: {e}")
            return 50, {"error": str(e)}
    
    def _analyze_market_efficiency(self, stock) -> tuple:
        """시장 효율성 분석"""
        try:
            score = 50
            analysis = {}
            
            # 시장 대표성
            if hasattr(stock, 'market_representation'):
                representation_score = stock.market_representation * 40
                score += representation_score
                analysis['market_representation'] = stock.market_representation
            
            # 유동성
            if hasattr(stock, 'liquidity_score'):
                liquidity_score = stock.liquidity_score * 30
                score += liquidity_score
                analysis['liquidity_score'] = stock.liquidity_score
            
            # 시장 효율성 지수
            if hasattr(stock, 'market_efficiency_index'):
                efficiency_score = stock.market_efficiency_index * 20
                score += efficiency_score
                analysis['market_efficiency_index'] = stock.market_efficiency_index
            
            # 정보 투명성
            if hasattr(stock, 'information_transparency'):
                transparency_score = stock.information_transparency * 10
                score += transparency_score
                analysis['information_transparency'] = stock.information_transparency
            
            return min(max(score, 0), 100), analysis
            
        except Exception as e:
            logger.error(f"시장 효율성 분석 오류: {e}")
            return 50, {"error": str(e)}
    
    def _analyze_simplicity(self, stock) -> tuple:
        """단순성 분석"""
        try:
            score = 50
            analysis = {}
            
            # 투자 구조 단순성
            if hasattr(stock, 'structure_simplicity'):
                structure_score = stock.structure_simplicity * 40
                score += structure_score
                analysis['structure_simplicity'] = stock.structure_simplicity
            
            # 이해 용이성
            if hasattr(stock, 'understanding_ease'):
                understanding_score = stock.understanding_ease * 30
                score += understanding_score
                analysis['understanding_ease'] = stock.understanding_ease
            
            # 관리 필요성 (낮을수록 좋음)
            if hasattr(stock, 'management_requirement'):
                management_score = max(20 - stock.management_requirement * 20, 0)
                score += management_score
                analysis['management_requirement'] = stock.management_requirement
            
            return min(max(score, 0), 100), analysis
            
        except Exception as e:
            logger.error(f"단순성 분석 오류: {e}")
            return 50, {"error": str(e)}
    
    def _make_investment_decision(self, total_score):
        """투자 판단 결정"""
        if total_score >= 80:
            return "강력매수 - 완벽한 인덱스 투자 조건"
        elif total_score >= 70:
            return "매수 - 우수한 패시브 투자 대상"
        elif total_score >= 60:
            return "관심 - 장기 보유 고려"
        elif total_score >= 50:
            return "중립 - 인덱스 대안 검토"
        else:
            return "회피 - 액티브 투자 필요"
    
    def _extract_key_points(self, scores, analysis_details):
        """핵심 포인트 추출"""
        points = []
        
        # 저비용
        if scores['low_cost'] >= 70:
            points.append("매우 낮은 운용 비용")
        elif scores['low_cost'] <= 40:
            points.append("높은 비용 구조")
        
        # 광범위 분산
        if scores['broad_diversification'] >= 70:
            points.append("우수한 분산 투자 효과")
        
        # 장기 보유
        if scores['long_term_hold'] >= 70:
            points.append("장기 투자 적합성")
        
        # 시장 효율성
        if scores['market_efficiency'] >= 70:
            points.append("높은 시장 효율성")
        
        # 단순성
        if scores['simplicity'] >= 70:
            points.append("단순하고 투명한 구조")
        
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