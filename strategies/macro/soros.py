"""
조지 소로스 (George Soros) 통화/금융위기 투기 전략

"시장을 부수는 남자"로 불리는 조지 소로스의 반사성 이론 기반 투자
- 시장 불균형과 금융위기 상황 활용
- 통화 투기와 거시경제 분석
- 반사성 이론(Reflexivity Theory) 적용
"""

import logging
from typing import Dict, List
from ..common import BaseStrategy, StrategyResult

logger = logging.getLogger(__name__)

class SorosStrategy(BaseStrategy):
    """조지 소로스 전략 구현"""
    
    def __init__(self):
        super().__init__()
        self.strategy_name = "조지 소로스 (George Soros)"
        self.description = "반사성 이론 기반 매크로 투기 전략"
        
        # 가중치 설정 (총합 100%)
        self.weights = {
            'market_reflexivity': 30,    # 시장 반사성
            'macro_imbalance': 25,       # 거시경제 불균형
            'currency_dynamics': 20,     # 통화 역학
            'crisis_opportunity': 15,    # 위기 기회
            'sentiment_extreme': 10      # 극단적 심리
        }
    
    def analyze_stock(self, stock) -> StrategyResult:
        """소로스 전략으로 주식 분석"""
        try:
            scores = {}
            analysis_details = {}
            
            # 1. 시장 반사성 분석 (30%)
            reflexivity_score, reflexivity_analysis = self._analyze_market_reflexivity(stock)
            scores['market_reflexivity'] = reflexivity_score
            analysis_details['market_reflexivity'] = reflexivity_analysis
            
            # 2. 거시경제 불균형 분석 (25%)
            macro_score, macro_analysis = self._analyze_macro_imbalance(stock)
            scores['macro_imbalance'] = macro_score
            analysis_details['macro_imbalance'] = macro_analysis
            
            # 3. 통화 역학 분석 (20%)
            currency_score, currency_analysis = self._analyze_currency_dynamics(stock)
            scores['currency_dynamics'] = currency_score
            analysis_details['currency_dynamics'] = currency_analysis
            
            # 4. 위기 기회 분석 (15%)
            crisis_score, crisis_analysis = self._analyze_crisis_opportunity(stock)
            scores['crisis_opportunity'] = crisis_score
            analysis_details['crisis_opportunity'] = crisis_analysis
            
            # 5. 극단적 심리 분석 (10%)
            sentiment_score, sentiment_analysis = self._analyze_sentiment_extreme(stock)
            scores['sentiment_extreme'] = sentiment_score
            analysis_details['sentiment_extreme'] = sentiment_analysis
            
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
            logger.error(f"소로스 전략 분석 중 오류: {e}")
            return self._create_error_result()
    
    def _analyze_market_reflexivity(self, stock) -> tuple:
        """시장 반사성 분석"""
        try:
            score = 50
            analysis = {}
            
            # 인식과 현실의 괴리
            if hasattr(stock, 'perception_reality_gap'):
                gap_score = abs(stock.perception_reality_gap) * 30
                score += gap_score
                analysis['perception_reality_gap'] = stock.perception_reality_gap
            
            # 자기강화 순환
            if hasattr(stock, 'self_reinforcing_cycle'):
                cycle_score = stock.self_reinforcing_cycle * 25
                score += cycle_score
                analysis['self_reinforcing_cycle'] = stock.self_reinforcing_cycle
            
            # 시장 피드백 강도
            if hasattr(stock, 'market_feedback_intensity'):
                feedback_score = stock.market_feedback_intensity * 20
                score += feedback_score
                analysis['market_feedback_intensity'] = stock.market_feedback_intensity
            
            # 버블 형성 징후
            if hasattr(stock, 'bubble_indicators'):
                bubble_score = stock.bubble_indicators * 15
                score += bubble_score
                analysis['bubble_indicators'] = stock.bubble_indicators
            
            return min(max(score, 0), 100), analysis
            
        except Exception as e:
            logger.error(f"시장 반사성 분석 오류: {e}")
            return 50, {"error": str(e)}
    
    def _analyze_macro_imbalance(self, stock) -> tuple:
        """거시경제 불균형 분석"""
        try:
            score = 50
            analysis = {}
            
            # 금리 불균형
            if hasattr(stock, 'interest_rate_imbalance'):
                rate_score = abs(stock.interest_rate_imbalance) * 25
                score += rate_score
                analysis['interest_rate_imbalance'] = stock.interest_rate_imbalance
            
            # 경상수지 불균형
            if hasattr(stock, 'current_account_imbalance'):
                ca_score = abs(stock.current_account_imbalance) * 20
                score += ca_score
                analysis['current_account_imbalance'] = stock.current_account_imbalance
            
            # 재정 불균형
            if hasattr(stock, 'fiscal_imbalance'):
                fiscal_score = abs(stock.fiscal_imbalance) * 15
                score += fiscal_score
                analysis['fiscal_imbalance'] = stock.fiscal_imbalance
            
            # 중앙은행 정책 변화
            if hasattr(stock, 'central_bank_policy_shift'):
                policy_score = abs(stock.central_bank_policy_shift) * 15
                score += policy_score
                analysis['central_bank_policy_shift'] = stock.central_bank_policy_shift
            
            return min(max(score, 0), 100), analysis
            
        except Exception as e:
            logger.error(f"거시경제 불균형 분석 오류: {e}")
            return 50, {"error": str(e)}
    
    def _analyze_currency_dynamics(self, stock) -> tuple:
        """통화 역학 분석"""
        try:
            score = 50
            analysis = {}
            
            # 환율 변동성
            if hasattr(stock, 'currency_volatility'):
                volatility_score = stock.currency_volatility * 30
                score += volatility_score
                analysis['currency_volatility'] = stock.currency_volatility
            
            # 통화 정책 분기
            if hasattr(stock, 'monetary_policy_divergence'):
                divergence_score = abs(stock.monetary_policy_divergence) * 25
                score += divergence_score
                analysis['monetary_policy_divergence'] = stock.monetary_policy_divergence
            
            # 자본 흐름 변화
            if hasattr(stock, 'capital_flow_change'):
                flow_score = abs(stock.capital_flow_change) * 20
                score += flow_score
                analysis['capital_flow_change'] = stock.capital_flow_change
            
            # 통화 위기 신호
            if hasattr(stock, 'currency_crisis_signals'):
                crisis_score = stock.currency_crisis_signals * 15
                score += crisis_score
                analysis['currency_crisis_signals'] = stock.currency_crisis_signals
            
            return min(max(score, 0), 100), analysis
            
        except Exception as e:
            logger.error(f"통화 역학 분석 오류: {e}")
            return 50, {"error": str(e)}
    
    def _analyze_crisis_opportunity(self, stock) -> tuple:
        """위기 기회 분석"""
        try:
            score = 50
            analysis = {}
            
            # 시스템 위험 수준
            if hasattr(stock, 'systemic_risk_level'):
                risk_score = stock.systemic_risk_level * 30
                score += risk_score
                analysis['systemic_risk_level'] = stock.systemic_risk_level
            
            # 구조적 취약점
            if hasattr(stock, 'structural_vulnerability'):
                vulnerability_score = stock.structural_vulnerability * 25
                score += vulnerability_score
                analysis['structural_vulnerability'] = stock.structural_vulnerability
            
            # 정책 대응 여력
            if hasattr(stock, 'policy_response_capacity'):
                response_score = (1 - stock.policy_response_capacity) * 20
                score += response_score
                analysis['policy_response_capacity'] = stock.policy_response_capacity
            
            # 전염 효과 가능성
            if hasattr(stock, 'contagion_potential'):
                contagion_score = stock.contagion_potential * 15
                score += contagion_score
                analysis['contagion_potential'] = stock.contagion_potential
            
            return min(max(score, 0), 100), analysis
            
        except Exception as e:
            logger.error(f"위기 기회 분석 오류: {e}")
            return 50, {"error": str(e)}
    
    def _analyze_sentiment_extreme(self, stock) -> tuple:
        """극단적 심리 분석"""
        try:
            score = 50
            analysis = {}
            
            # 시장 심리 극단성
            if hasattr(stock, 'market_sentiment'):
                # 극단적 낙관(-1) 또는 극단적 비관(1)일 때 높은 점수
                extreme_score = abs(stock.market_sentiment) * 40
                score += extreme_score
                analysis['market_sentiment'] = stock.market_sentiment
            
            # 공포/탐욕 지수
            if hasattr(stock, 'fear_greed_index'):
                # 극단적 공포(0-20) 또는 극단적 탐욕(80-100)
                if stock.fear_greed_index <= 20 or stock.fear_greed_index >= 80:
                    fg_score = 30
                elif stock.fear_greed_index <= 30 or stock.fear_greed_index >= 70:
                    fg_score = 20
                else:
                    fg_score = 0
                score += fg_score
                analysis['fear_greed_index'] = stock.fear_greed_index
            
            # 미디어 센티멘트
            if hasattr(stock, 'media_sentiment_extreme'):
                media_score = abs(stock.media_sentiment_extreme) * 20
                score += media_score
                analysis['media_sentiment_extreme'] = stock.media_sentiment_extreme
            
            return min(max(score, 0), 100), analysis
            
        except Exception as e:
            logger.error(f"극단적 심리 분석 오류: {e}")
            return 50, {"error": str(e)}
    
    def _make_investment_decision(self, total_score):
        """투자 판단 결정"""
        if total_score >= 80:
            return "강력매수 - 매크로 불균형 극대화 기회"
        elif total_score >= 70:
            return "매수 - 반사성 투기 기회 포착"
        elif total_score >= 60:
            return "관심 - 불균형 확대 관찰"
        elif total_score >= 50:
            return "중립 - 매크로 신호 불분명"
        else:
            return "회피 - 안정적 균형 상태"
    
    def _extract_key_points(self, scores, analysis_details):
        """핵심 포인트 추출"""
        points = []
        
        # 시장 반사성
        if scores['market_reflexivity'] >= 70:
            points.append("강한 시장 반사성 확인")
        elif scores['market_reflexivity'] <= 40:
            points.append("시장 반사성 부족")
        
        # 거시경제 불균형
        if scores['macro_imbalance'] >= 70:
            points.append("심각한 거시경제 불균형")
        
        # 통화 역학
        if scores['currency_dynamics'] >= 70:
            points.append("통화 불안정성 증가")
        
        # 위기 기회
        if scores['crisis_opportunity'] >= 70:
            points.append("시스템 위기 기회 포착")
        
        # 극단적 심리
        if scores['sentiment_extreme'] >= 70:
            points.append("극단적 시장 심리")
        
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