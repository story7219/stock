"""
🌱 케시 우드 (Cathie Wood) 투자 전략
파괴적 혁신과 기하급수적 성장 투자
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from ..common import BaseStrategy, StrategyResult

logger = logging.getLogger(__name__)

class WoodStrategy(BaseStrategy):
    """케시 우드 전략 구현"""
    
    def __init__(self):
        super().__init__()
        self.strategy_name = "케시 우드 (Cathie Wood)"
        self.description = "파괴적 혁신과 기하급수적 성장 전략"
        
        # 가중치 설정 (총합 100%)
        self.weights = {
            'disruptive_innovation': 35,  # 파괴적 혁신
            'tech_inflection': 25,        # 기술 전환점
            'growth_potential': 20,       # 성장 잠재력
            'market_expansion': 15,       # 시장 확장성
            'innovation_ecosystem': 5     # 혁신 생태계
        }
    
    def analyze_stock(self, stock) -> StrategyResult:
        """우드 전략으로 주식 분석"""
        try:
            scores = {}
            analysis_details = {}
            
            # 1. 파괴적 혁신 분석 (35%)
            innovation_score, innovation_analysis = self._analyze_disruptive_innovation(stock)
            scores['disruptive_innovation'] = innovation_score
            analysis_details['disruptive_innovation'] = innovation_analysis
            
            # 2. 기술 전환점 분석 (25%)
            inflection_score, inflection_analysis = self._analyze_tech_inflection(stock)
            scores['tech_inflection'] = inflection_score
            analysis_details['tech_inflection'] = inflection_analysis
            
            # 3. 성장 잠재력 분석 (20%)
            growth_score, growth_analysis = self._analyze_growth_potential(stock)
            scores['growth_potential'] = growth_score
            analysis_details['growth_potential'] = growth_analysis
            
            # 4. 시장 확장성 분석 (15%)
            expansion_score, expansion_analysis = self._analyze_market_expansion(stock)
            scores['market_expansion'] = expansion_score
            analysis_details['market_expansion'] = expansion_analysis
            
            # 5. 혁신 생태계 분석 (5%)
            ecosystem_score, ecosystem_analysis = self._analyze_innovation_ecosystem(stock)
            scores['innovation_ecosystem'] = ecosystem_score
            analysis_details['innovation_ecosystem'] = ecosystem_analysis
            
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
            logger.error(f"우드 전략 분석 중 오류: {e}")
            return self._create_error_result()
    
    def _analyze_disruptive_innovation(self, stock) -> tuple:
        """파괴적 혁신 분석"""
        try:
            score = 50
            analysis = {}
            
            # 혁신 기술 적용도
            if hasattr(stock, 'innovation_tech_adoption'):
                tech_score = stock.innovation_tech_adoption * 30
                score += tech_score
                analysis['innovation_tech_adoption'] = stock.innovation_tech_adoption
            
            # 기존 산업 파괴력
            if hasattr(stock, 'industry_disruption_power'):
                disruption_score = stock.industry_disruption_power * 25
                score += disruption_score
                analysis['industry_disruption_power'] = stock.industry_disruption_power
            
            # 혁신 투자 비중
            if hasattr(stock, 'innovation_investment_ratio'):
                investment_score = stock.innovation_investment_ratio * 20
                score += investment_score
                analysis['innovation_investment_ratio'] = stock.innovation_investment_ratio
            
            # 특허 및 IP 강도
            if hasattr(stock, 'patent_ip_strength'):
                patent_score = stock.patent_ip_strength * 15
                score += patent_score
                analysis['patent_ip_strength'] = stock.patent_ip_strength
            
            # 혁신 속도
            if hasattr(stock, 'innovation_velocity'):
                velocity_score = stock.innovation_velocity * 10
                score += velocity_score
                analysis['innovation_velocity'] = stock.innovation_velocity
            
            return min(max(score, 0), 100), analysis
            
        except Exception as e:
            logger.error(f"파괴적 혁신 분석 오류: {e}")
            return 50, {"error": str(e)}
    
    def _analyze_tech_inflection(self, stock) -> tuple:
        """기술 전환점 분석"""
        try:
            score = 50
            analysis = {}
            
            # 기술 성숙도 곡선 위치
            if hasattr(stock, 'tech_maturity_position'):
                maturity_score = self._evaluate_tech_maturity(stock.tech_maturity_position)
                score += maturity_score * 35
                analysis['tech_maturity_position'] = stock.tech_maturity_position
            
            # 시장 채택 가속화
            if hasattr(stock, 'market_adoption_acceleration'):
                adoption_score = stock.market_adoption_acceleration * 25
                score += adoption_score
                analysis['market_adoption_acceleration'] = stock.market_adoption_acceleration
            
            # 비용 곡선 개선
            if hasattr(stock, 'cost_curve_improvement'):
                cost_score = stock.cost_curve_improvement * 20
                score += cost_score
                analysis['cost_curve_improvement'] = stock.cost_curve_improvement
            
            # 네트워크 효과 임계점
            if hasattr(stock, 'network_effect_threshold'):
                network_score = stock.network_effect_threshold * 15
                score += network_score
                analysis['network_effect_threshold'] = stock.network_effect_threshold
            
            # 규제 환경 변화
            if hasattr(stock, 'regulatory_tailwind'):
                regulatory_score = stock.regulatory_tailwind * 5
                score += regulatory_score
                analysis['regulatory_tailwind'] = stock.regulatory_tailwind
            
            return min(max(score, 0), 100), analysis
            
        except Exception as e:
            logger.error(f"기술 전환점 분석 오류: {e}")
            return 50, {"error": str(e)}
    
    def _analyze_growth_potential(self, stock) -> tuple:
        """성장 잠재력 분석"""
        try:
            score = 50
            analysis = {}
            
            # 매출 성장률
            if hasattr(stock, 'revenue_growth_rate'):
                if stock.revenue_growth_rate >= 50:
                    revenue_score = 35
                elif stock.revenue_growth_rate >= 30:
                    revenue_score = 25
                elif stock.revenue_growth_rate >= 20:
                    revenue_score = 15
                elif stock.revenue_growth_rate >= 10:
                    revenue_score = 5
                else:
                    revenue_score = 0
                score += revenue_score
                analysis['revenue_growth_rate'] = stock.revenue_growth_rate
            
            # 시장 점유율 확대 속도
            if hasattr(stock, 'market_share_expansion'):
                share_score = stock.market_share_expansion * 25
                score += share_score
                analysis['market_share_expansion'] = stock.market_share_expansion
            
            # 수익성 개선 궤적
            if hasattr(stock, 'profitability_trajectory'):
                profit_score = stock.profitability_trajectory * 20
                score += profit_score
                analysis['profitability_trajectory'] = stock.profitability_trajectory
            
            # 확장성 지표
            if hasattr(stock, 'scalability_metrics'):
                scalability_score = stock.scalability_metrics * 15
                score += scalability_score
                analysis['scalability_metrics'] = stock.scalability_metrics
            
            # 경쟁 우위 지속성
            if hasattr(stock, 'competitive_moat_durability'):
                moat_score = stock.competitive_moat_durability * 5
                score += moat_score
                analysis['competitive_moat_durability'] = stock.competitive_moat_durability
            
            return min(max(score, 0), 100), analysis
            
        except Exception as e:
            logger.error(f"성장 잠재력 분석 오류: {e}")
            return 50, {"error": str(e)}
    
    def _analyze_market_expansion(self, stock) -> tuple:
        """시장 확장성 분석"""
        try:
            score = 50
            analysis = {}
            
            # 총 주소 가능 시장 (TAM)
            if hasattr(stock, 'total_addressable_market'):
                if stock.total_addressable_market >= 1000:  # 1조 달러 이상
                    tam_score = 40
                elif stock.total_addressable_market >= 100:
                    tam_score = 30
                elif stock.total_addressable_market >= 10:
                    tam_score = 20
                else:
                    tam_score = 10
                score += tam_score
                analysis['total_addressable_market'] = stock.total_addressable_market
            
            # 글로벌 확장 가능성
            if hasattr(stock, 'global_expansion_potential'):
                global_score = stock.global_expansion_potential * 30
                score += global_score
                analysis['global_expansion_potential'] = stock.global_expansion_potential
            
            # 시장 침투율
            if hasattr(stock, 'market_penetration_rate'):
                # 낮은 침투율일수록 성장 여지 큼
                penetration_score = max(20 - stock.market_penetration_rate, 0)
                score += penetration_score
                analysis['market_penetration_rate'] = stock.market_penetration_rate
            
            # 인접 시장 진출
            if hasattr(stock, 'adjacent_market_opportunity'):
                adjacent_score = stock.adjacent_market_opportunity * 10
                score += adjacent_score
                analysis['adjacent_market_opportunity'] = stock.adjacent_market_opportunity
            
            return min(max(score, 0), 100), analysis
            
        except Exception as e:
            logger.error(f"시장 확장성 분석 오류: {e}")
            return 50, {"error": str(e)}
    
    def _analyze_innovation_ecosystem(self, stock) -> tuple:
        """혁신 생태계 분석"""
        try:
            score = 50
            analysis = {}
            
            # 파트너십 네트워크
            if hasattr(stock, 'partnership_network_strength'):
                partnership_score = stock.partnership_network_strength * 40
                score += partnership_score
                analysis['partnership_network_strength'] = stock.partnership_network_strength
            
            # 인재 확보 능력
            if hasattr(stock, 'talent_acquisition_capability'):
                talent_score = stock.talent_acquisition_capability * 30
                score += talent_score
                analysis['talent_acquisition_capability'] = stock.talent_acquisition_capability
            
            # 혁신 문화
            if hasattr(stock, 'innovation_culture_score'):
                culture_score = stock.innovation_culture_score * 20
                score += culture_score
                analysis['innovation_culture_score'] = stock.innovation_culture_score
            
            # 생태계 영향력
            if hasattr(stock, 'ecosystem_influence'):
                influence_score = stock.ecosystem_influence * 10
                score += influence_score
                analysis['ecosystem_influence'] = stock.ecosystem_influence
            
            return min(max(score, 0), 100), analysis
            
        except Exception as e:
            logger.error(f"혁신 생태계 분석 오류: {e}")
            return 50, {"error": str(e)}
    
    def _evaluate_tech_maturity(self, position):
        """기술 성숙도 곡선 평가"""
        # S-곡선에서 급성장 구간(20-80%)에 높은 점수
        if 0.2 <= position <= 0.8:
            return 1.0  # 최고점
        elif 0.1 <= position < 0.2 or 0.8 < position <= 0.9:
            return 0.7  # 높음
        else:
            return 0.3  # 낮음
    
    def _make_investment_decision(self, total_score):
        """투자 판단 결정"""
        if total_score >= 80:
            return "강력매수 - 파괴적 혁신 완벽 조건"
        elif total_score >= 70:
            return "매수 - 강한 혁신 성장 잠재력"
        elif total_score >= 60:
            return "관심 - 기술 전환점 관찰"
        elif total_score >= 50:
            return "중립 - 혁신 신호 불분명"
        else:
            return "회피 - 혁신 동력 부족"
    
    def _extract_key_points(self, scores, analysis_details):
        """핵심 포인트 추출"""
        points = []
        
        # 파괴적 혁신
        if scores['disruptive_innovation'] >= 70:
            points.append("강력한 파괴적 혁신 기술")
        elif scores['disruptive_innovation'] <= 40:
            points.append("혁신 기술 부족")
        
        # 기술 전환점
        if scores['tech_inflection'] >= 70:
            points.append("기술 전환점 도달")
        
        # 성장 잠재력
        if scores['growth_potential'] >= 70:
            points.append("기하급수적 성장 가능성")
        
        # 시장 확장성
        if scores['market_expansion'] >= 70:
            points.append("거대한 시장 확장 기회")
        
        # 혁신 생태계
        if scores['innovation_ecosystem'] >= 70:
            points.append("우수한 혁신 생태계")
        
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