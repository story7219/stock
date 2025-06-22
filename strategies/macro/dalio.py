"""
레이 달리오 (Ray Dalio) 투자 전략

브리지워터 어소시에이츠 창립자의 올웨더 포트폴리오 전략
- 경제 사이클과 인플레이션 분석
- 위험 패리티와 분산투자
- "원칙(Principles)" 기반 체계적 접근
"""

import logging
from typing import Dict, List
from ..common import BaseStrategy, StrategyResult

logger = logging.getLogger(__name__)

class DalioStrategy(BaseStrategy):
    """레이 달리오 전략 구현"""
    
    def __init__(self):
        super().__init__()
        self.strategy_name = "레이 달리오 (Ray Dalio)"
        self.description = "올웨더 포트폴리오와 경제 사이클 전략"
        
        # 가중치 설정 (총합 100%)
        self.weights = {
            'economic_cycle': 30,        # 경제 사이클
            'inflation_regime': 25,      # 인플레이션 체제
            'risk_parity': 20,           # 위험 패리티
            'diversification': 15,       # 분산 효과
            'regime_change': 10          # 체제 변화
        }
    
    def analyze_stock(self, stock) -> StrategyResult:
        """달리오 전략으로 주식 분석"""
        try:
            scores = {}
            analysis_details = {}
            
            # 1. 경제 사이클 분석 (30%)
            cycle_score, cycle_analysis = self._analyze_economic_cycle(stock)
            scores['economic_cycle'] = cycle_score
            analysis_details['economic_cycle'] = cycle_analysis
            
            # 2. 인플레이션 체제 분석 (25%)
            inflation_score, inflation_analysis = self._analyze_inflation_regime(stock)
            scores['inflation_regime'] = inflation_score
            analysis_details['inflation_regime'] = inflation_analysis
            
            # 3. 위험 패리티 분석 (20%)
            risk_parity_score, risk_parity_analysis = self._analyze_risk_parity(stock)
            scores['risk_parity'] = risk_parity_score
            analysis_details['risk_parity'] = risk_parity_analysis
            
            # 4. 분산 효과 분석 (15%)
            diversification_score, diversification_analysis = self._analyze_diversification(stock)
            scores['diversification'] = diversification_score
            analysis_details['diversification'] = diversification_analysis
            
            # 5. 체제 변화 분석 (10%)
            regime_score, regime_analysis = self._analyze_regime_change(stock)
            scores['regime_change'] = regime_score
            analysis_details['regime_change'] = regime_analysis
            
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
            logger.error(f"달리오 전략 분석 중 오류: {e}")
            return self._create_error_result()
    
    def _analyze_economic_cycle(self, stock) -> tuple:
        """경제 사이클 분석"""
        try:
            score = 50
            analysis = {}
            
            # 성장률 추세
            if hasattr(stock, 'gdp_growth_trend'):
                growth_score = self._evaluate_growth_phase(stock.gdp_growth_trend)
                score += growth_score * 25
                analysis['gdp_growth_trend'] = stock.gdp_growth_trend
            
            # 신용 사이클
            if hasattr(stock, 'credit_cycle_position'):
                credit_score = stock.credit_cycle_position * 20
                score += credit_score
                analysis['credit_cycle_position'] = stock.credit_cycle_position
            
            # 고용 상황
            if hasattr(stock, 'employment_cycle'):
                employment_score = stock.employment_cycle * 15
                score += employment_score
                analysis['employment_cycle'] = stock.employment_cycle
            
            # 기업 수익 사이클
            if hasattr(stock, 'earnings_cycle'):
                earnings_score = stock.earnings_cycle * 15
                score += earnings_score
                analysis['earnings_cycle'] = stock.earnings_cycle
            
            # 중앙은행 정책 사이클
            if hasattr(stock, 'monetary_policy_cycle'):
                policy_score = stock.monetary_policy_cycle * 10
                score += policy_score
                analysis['monetary_policy_cycle'] = stock.monetary_policy_cycle
            
            return min(max(score, 0), 100), analysis
            
        except Exception as e:
            logger.error(f"경제 사이클 분석 오류: {e}")
            return 50, {"error": str(e)}
    
    def _analyze_inflation_regime(self, stock) -> tuple:
        """인플레이션 체제 분석"""
        try:
            score = 50
            analysis = {}
            
            # 인플레이션 추세
            if hasattr(stock, 'inflation_trend'):
                inflation_score = self._evaluate_inflation_impact(stock.inflation_trend)
                score += inflation_score * 30
                analysis['inflation_trend'] = stock.inflation_trend
            
            # 실질금리 환경
            if hasattr(stock, 'real_interest_rate'):
                real_rate_score = self._evaluate_real_rate(stock.real_interest_rate)
                score += real_rate_score * 25
                analysis['real_interest_rate'] = stock.real_interest_rate
            
            # 통화 공급량 증가율
            if hasattr(stock, 'money_supply_growth'):
                money_score = stock.money_supply_growth * 20
                score += money_score
                analysis['money_supply_growth'] = stock.money_supply_growth
            
            # 원자재 가격 압력
            if hasattr(stock, 'commodity_price_pressure'):
                commodity_score = stock.commodity_price_pressure * 15
                score += commodity_score
                analysis['commodity_price_pressure'] = stock.commodity_price_pressure
            
            return min(max(score, 0), 100), analysis
            
        except Exception as e:
            logger.error(f"인플레이션 체제 분석 오류: {e}")
            return 50, {"error": str(e)}
    
    def _analyze_risk_parity(self, stock) -> tuple:
        """위험 패리티 분석"""
        try:
            score = 50
            analysis = {}
            
            # 위험 기여도 균형
            if hasattr(stock, 'risk_contribution_balance'):
                balance_score = stock.risk_contribution_balance * 30
                score += balance_score
                analysis['risk_contribution_balance'] = stock.risk_contribution_balance
            
            # 변동성 조정 수익률
            if hasattr(stock, 'volatility_adjusted_return'):
                vol_adj_score = stock.volatility_adjusted_return * 25
                score += vol_adj_score
                analysis['volatility_adjusted_return'] = stock.volatility_adjusted_return
            
            # 샤프 비율
            if hasattr(stock, 'sharpe_ratio'):
                sharpe_score = min(stock.sharpe_ratio * 20, 20)
                score += sharpe_score
                analysis['sharpe_ratio'] = stock.sharpe_ratio
            
            # 최대 손실 제한
            if hasattr(stock, 'max_drawdown'):
                drawdown_score = max(25 - abs(stock.max_drawdown), 0)
                score += drawdown_score
                analysis['max_drawdown'] = stock.max_drawdown
            
            return min(max(score, 0), 100), analysis
            
        except Exception as e:
            logger.error(f"위험 패리티 분석 오류: {e}")
            return 50, {"error": str(e)}
    
    def _analyze_diversification(self, stock) -> tuple:
        """분산 효과 분석"""
        try:
            score = 50
            analysis = {}
            
            # 자산군 간 상관관계
            if hasattr(stock, 'cross_asset_correlation'):
                # 낮은 상관관계일수록 좋음
                correlation_score = (1 - abs(stock.cross_asset_correlation)) * 30
                score += correlation_score
                analysis['cross_asset_correlation'] = stock.cross_asset_correlation
            
            # 지역별 분산
            if hasattr(stock, 'geographic_diversification'):
                geo_score = stock.geographic_diversification * 25
                score += geo_score
                analysis['geographic_diversification'] = stock.geographic_diversification
            
            # 섹터별 분산
            if hasattr(stock, 'sector_diversification'):
                sector_score = stock.sector_diversification * 20
                score += sector_score
                analysis['sector_diversification'] = stock.sector_diversification
            
            # 시간 분산 (리밸런싱)
            if hasattr(stock, 'rebalancing_benefit'):
                rebalance_score = stock.rebalancing_benefit * 15
                score += rebalance_score
                analysis['rebalancing_benefit'] = stock.rebalancing_benefit
            
            return min(max(score, 0), 100), analysis
            
        except Exception as e:
            logger.error(f"분산 효과 분석 오류: {e}")
            return 50, {"error": str(e)}
    
    def _analyze_regime_change(self, stock) -> tuple:
        """체제 변화 분석"""
        try:
            score = 50
            analysis = {}
            
            # 정책 체제 변화
            if hasattr(stock, 'policy_regime_shift'):
                policy_score = abs(stock.policy_regime_shift) * 40
                score += policy_score
                analysis['policy_regime_shift'] = stock.policy_regime_shift
            
            # 시장 구조 변화
            if hasattr(stock, 'market_structure_change'):
                structure_score = abs(stock.market_structure_change) * 30
                score += structure_score
                analysis['market_structure_change'] = stock.market_structure_change
            
            # 기술적 패러다임 변화
            if hasattr(stock, 'tech_paradigm_shift'):
                tech_score = abs(stock.tech_paradigm_shift) * 20
                score += tech_score
                analysis['tech_paradigm_shift'] = stock.tech_paradigm_shift
            
            return min(max(score, 0), 100), analysis
            
        except Exception as e:
            logger.error(f"체제 변화 분석 오류: {e}")
            return 50, {"error": str(e)}
    
    def _evaluate_growth_phase(self, gdp_growth):
        """성장 국면 평가"""
        if gdp_growth > 3:
            return 1  # 강한 성장
        elif gdp_growth > 2:
            return 0.7  # 보통 성장
        elif gdp_growth > 0:
            return 0.3  # 약한 성장
        else:
            return -0.5  # 침체
    
    def _evaluate_inflation_impact(self, inflation_trend):
        """인플레이션 영향 평가"""
        # 달리오의 4가지 경제 환경: 성장↑인플레↑, 성장↑인플레↓, 성장↓인플레↑, 성장↓인플레↓
        if 2 <= inflation_trend <= 4:
            return 1  # 적정 인플레이션
        elif inflation_trend > 4:
            return 0.5  # 높은 인플레이션
        elif inflation_trend < 0:
            return 0.3  # 디플레이션
        else:
            return 0.7  # 낮은 인플레이션
    
    def _evaluate_real_rate(self, real_rate):
        """실질금리 평가"""
        if -2 <= real_rate <= 2:
            return 1  # 적정 실질금리
        elif real_rate > 2:
            return 0.5  # 높은 실질금리
        else:
            return 0.3  # 매우 낮은 실질금리
    
    def _make_investment_decision(self, total_score):
        """투자 판단 결정"""
        if total_score >= 80:
            return "강력매수 - 올웨더 조건 완벽"
        elif total_score >= 70:
            return "매수 - 경제 사이클 유리"
        elif total_score >= 60:
            return "관심 - 분산 투자 고려"
        elif total_score >= 50:
            return "중립 - 균형 포트폴리오 유지"
        else:
            return "회피 - 경제 사이클 불리"
    
    def _extract_key_points(self, scores, analysis_details):
        """핵심 포인트 추출"""
        points = []
        
        # 경제 사이클
        if scores['economic_cycle'] >= 70:
            points.append("유리한 경제 사이클 국면")
        elif scores['economic_cycle'] <= 40:
            points.append("불리한 경제 사이클")
        
        # 인플레이션 체제
        if scores['inflation_regime'] >= 70:
            points.append("적정 인플레이션 환경")
        elif scores['inflation_regime'] <= 40:
            points.append("인플레이션 리스크")
        
        # 위험 패리티
        if scores['risk_parity'] >= 70:
            points.append("우수한 위험 조정 수익률")
        
        # 분산 효과
        if scores['diversification'] >= 70:
            points.append("효과적 분산 투자")
        
        # 체제 변화
        if scores['regime_change'] >= 70:
            points.append("중요한 체제 변화 감지")
        
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