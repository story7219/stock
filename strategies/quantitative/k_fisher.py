"""
켄 피셔 (Ken Fisher) 투자 전략

피셔 인베스트먼트 창립자, 행동경제학과 퀀트 분석의 융합
- 시장 효율성의 불완전성 활용
- 행동 편향과 정량적 지표 결합
- 글로벌 분산 투자
"""

import logging
from typing import Dict, List
from ..common import BaseStrategy, StrategyResult

logger = logging.getLogger(__name__)

class KFisherStrategy(BaseStrategy):
    """켄 피셔 전략 구현"""
    
    def __init__(self):
        super().__init__()
        self.strategy_name = "켄 피셔 (Ken Fisher)"
        self.description = "행동경제학과 퀀트 분석 융합 전략"
        
        # 가중치 설정 (총합 100%)
        self.weights = {
            'market_inefficiency': 30,    # 시장 효율성 불완전성
            'behavioral_bias': 25,        # 행동 편향
            'quantitative_metrics': 20,   # 정량적 지표
            'global_diversification': 15, # 글로벌 분산
            'market_timing': 10          # 시장 타이밍
        }
    
    def analyze_stock(self, stock) -> StrategyResult:
        """켄 피셔 전략으로 주식 분석"""
        try:
            scores = {}
            analysis_details = {}
            
            # 1. 시장 효율성 불완전성 분석 (30%)
            inefficiency_score, inefficiency_analysis = self._analyze_market_inefficiency(stock)
            scores['market_inefficiency'] = inefficiency_score
            analysis_details['market_inefficiency'] = inefficiency_analysis
            
            # 2. 행동 편향 분석 (25%)
            bias_score, bias_analysis = self._analyze_behavioral_bias(stock)
            scores['behavioral_bias'] = bias_score
            analysis_details['behavioral_bias'] = bias_analysis
            
            # 3. 정량적 지표 분석 (20%)
            quant_score, quant_analysis = self._analyze_quantitative_metrics(stock)
            scores['quantitative_metrics'] = quant_score
            analysis_details['quantitative_metrics'] = quant_analysis
            
            # 4. 글로벌 분산 분석 (15%)
            global_score, global_analysis = self._analyze_global_diversification(stock)
            scores['global_diversification'] = global_score
            analysis_details['global_diversification'] = global_analysis
            
            # 5. 시장 타이밍 분석 (10%)
            timing_score, timing_analysis = self._analyze_market_timing(stock)
            scores['market_timing'] = timing_score
            analysis_details['market_timing'] = timing_analysis
            
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
            logger.error(f"켄 피셔 전략 분석 중 오류: {e}")
            return self._create_error_result()
    
    def _analyze_market_inefficiency(self, stock) -> tuple:
        """시장 효율성 불완전성 분석"""
        try:
            score = 50
            analysis = {}
            
            # 가격 이상 현상
            if hasattr(stock, 'price_anomaly_score'):
                anomaly_score = stock.price_anomaly_score * 35
                score += anomaly_score
                analysis['price_anomaly_score'] = stock.price_anomaly_score
            
            # 정보 비대칭성
            if hasattr(stock, 'information_asymmetry'):
                info_score = stock.information_asymmetry * 25
                score += info_score
                analysis['information_asymmetry'] = stock.information_asymmetry
            
            # 시장 과반응/과소반응
            if hasattr(stock, 'market_overreaction'):
                reaction_score = abs(stock.market_overreaction) * 20
                score += reaction_score
                analysis['market_overreaction'] = stock.market_overreaction
            
            # 유동성 프리미엄
            if hasattr(stock, 'liquidity_premium'):
                liquidity_score = stock.liquidity_premium * 15
                score += liquidity_score
                analysis['liquidity_premium'] = stock.liquidity_premium
            
            # 규모 효과
            if hasattr(stock, 'size_effect'):
                size_score = stock.size_effect * 5
                score += size_score
                analysis['size_effect'] = stock.size_effect
            
            return min(max(score, 0), 100), analysis
            
        except Exception as e:
            logger.error(f"시장 효율성 불완전성 분석 오류: {e}")
            return 50, {"error": str(e)}
    
    def _analyze_behavioral_bias(self, stock) -> tuple:
        """행동 편향 분석"""
        try:
            score = 50
            analysis = {}
            
            # 군중심리 역이용
            if hasattr(stock, 'crowd_psychology_contrarian'):
                crowd_score = stock.crowd_psychology_contrarian * 30
                score += crowd_score
                analysis['crowd_psychology_contrarian'] = stock.crowd_psychology_contrarian
            
            # 확증편향 활용
            if hasattr(stock, 'confirmation_bias_exploitation'):
                confirm_score = stock.confirmation_bias_exploitation * 25
                score += confirm_score
                analysis['confirmation_bias_exploitation'] = stock.confirmation_bias_exploitation
            
            # 손실회피 편향
            if hasattr(stock, 'loss_aversion_bias'):
                loss_score = stock.loss_aversion_bias * 20
                score += loss_score
                analysis['loss_aversion_bias'] = stock.loss_aversion_bias
            
            # 앵커링 효과
            if hasattr(stock, 'anchoring_effect'):
                anchor_score = stock.anchoring_effect * 15
                score += anchor_score
                analysis['anchoring_effect'] = stock.anchoring_effect
            
            # 과신 편향
            if hasattr(stock, 'overconfidence_bias'):
                confidence_score = stock.overconfidence_bias * 10
                score += confidence_score
                analysis['overconfidence_bias'] = stock.overconfidence_bias
            
            return min(max(score, 0), 100), analysis
            
        except Exception as e:
            logger.error(f"행동 편향 분석 오류: {e}")
            return 50, {"error": str(e)}
    
    def _analyze_quantitative_metrics(self, stock) -> tuple:
        """정량적 지표 분석"""
        try:
            score = 50
            analysis = {}
            
            # PSR (Price-to-Sales Ratio)
            if hasattr(stock, 'price_sales_ratio'):
                psr_score = self._evaluate_psr(stock.price_sales_ratio)
                score += psr_score * 30
                analysis['price_sales_ratio'] = stock.price_sales_ratio
            
            # PEG 비율
            if hasattr(stock, 'peg_ratio'):
                peg_score = self._evaluate_peg(stock.peg_ratio)
                score += peg_score * 25
                analysis['peg_ratio'] = stock.peg_ratio
            
            # 수익률 모멘텀
            if hasattr(stock, 'earnings_momentum'):
                momentum_score = stock.earnings_momentum * 20
                score += momentum_score
                analysis['earnings_momentum'] = stock.earnings_momentum
            
            # 재무 품질 점수
            if hasattr(stock, 'financial_quality_score'):
                quality_score = stock.financial_quality_score * 15
                score += quality_score
                analysis['financial_quality_score'] = stock.financial_quality_score
            
            # 상대적 강도
            if hasattr(stock, 'relative_strength'):
                rs_score = stock.relative_strength * 10
                score += rs_score
                analysis['relative_strength'] = stock.relative_strength
            
            return min(max(score, 0), 100), analysis
            
        except Exception as e:
            logger.error(f"정량적 지표 분석 오류: {e}")
            return 50, {"error": str(e)}
    
    def _analyze_global_diversification(self, stock) -> tuple:
        """글로벌 분산 분석"""
        try:
            score = 50
            analysis = {}
            
            # 지역 분산 효과
            if hasattr(stock, 'regional_diversification'):
                regional_score = stock.regional_diversification * 40
                score += regional_score
                analysis['regional_diversification'] = stock.regional_diversification
            
            # 통화 헤지 효과
            if hasattr(stock, 'currency_hedge_benefit'):
                hedge_score = stock.currency_hedge_benefit * 25
                score += hedge_score
                analysis['currency_hedge_benefit'] = stock.currency_hedge_benefit
            
            # 섹터 분산
            if hasattr(stock, 'sector_diversification'):
                sector_score = stock.sector_diversification * 20
                score += sector_score
                analysis['sector_diversification'] = stock.sector_diversification
            
            # 글로벌 경제 노출도
            if hasattr(stock, 'global_economic_exposure'):
                global_score = stock.global_economic_exposure * 15
                score += global_score
                analysis['global_economic_exposure'] = stock.global_economic_exposure
            
            return min(max(score, 0), 100), analysis
            
        except Exception as e:
            logger.error(f"글로벌 분산 분석 오류: {e}")
            return 50, {"error": str(e)}
    
    def _analyze_market_timing(self, stock) -> tuple:
        """시장 타이밍 분석"""
        try:
            score = 50
            analysis = {}
            
            # 경제 선행지표
            if hasattr(stock, 'leading_economic_indicators'):
                leading_score = stock.leading_economic_indicators * 40
                score += leading_score
                analysis['leading_economic_indicators'] = stock.leading_economic_indicators
            
            # 시장 밸류에이션 레벨
            if hasattr(stock, 'market_valuation_level'):
                valuation_score = self._evaluate_market_valuation(stock.market_valuation_level)
                score += valuation_score * 30
                analysis['market_valuation_level'] = stock.market_valuation_level
            
            # 투자자 심리 지표
            if hasattr(stock, 'investor_sentiment_indicator'):
                sentiment_score = abs(stock.investor_sentiment_indicator - 0.5) * 2 * 20
                score += sentiment_score
                analysis['investor_sentiment_indicator'] = stock.investor_sentiment_indicator
            
            # 계절성 패턴
            if hasattr(stock, 'seasonal_pattern_strength'):
                seasonal_score = stock.seasonal_pattern_strength * 10
                score += seasonal_score
                analysis['seasonal_pattern_strength'] = stock.seasonal_pattern_strength
            
            return min(max(score, 0), 100), analysis
            
        except Exception as e:
            logger.error(f"시장 타이밍 분석 오류: {e}")
            return 50, {"error": str(e)}
    
    def _evaluate_psr(self, psr):
        """PSR 평가"""
        if psr <= 1:
            return 1.0  # 매우 좋음
        elif psr <= 2:
            return 0.8  # 좋음
        elif psr <= 3:
            return 0.6  # 보통
        elif psr <= 5:
            return 0.4  # 높음
        else:
            return 0.2  # 매우 높음
    
    def _evaluate_peg(self, peg):
        """PEG 비율 평가"""
        if peg <= 0.5:
            return 1.0  # 매우 저평가
        elif peg <= 1:
            return 0.8  # 적정
        elif peg <= 1.5:
            return 0.6  # 약간 고평가
        elif peg <= 2:
            return 0.4  # 고평가
        else:
            return 0.2  # 매우 고평가
    
    def _evaluate_market_valuation(self, valuation_level):
        """시장 밸류에이션 평가"""
        # 극단적 저평가나 고평가에서 높은 점수
        if valuation_level <= 0.2 or valuation_level >= 0.8:
            return 1.0
        elif valuation_level <= 0.3 or valuation_level >= 0.7:
            return 0.7
        else:
            return 0.3
    
    def _make_investment_decision(self, total_score):
        """투자 판단 결정"""
        if total_score >= 80:
            return "강력매수 - 완벽한 행동-퀀트 신호"
        elif total_score >= 70:
            return "매수 - 우수한 비효율성 활용 기회"
        elif total_score >= 60:
            return "관심 - 행동 편향 관찰"
        elif total_score >= 50:
            return "중립 - 정량적 신호 불분명"
        else:
            return "회피 - 효율성 가정 하에 투자 부적절"
    
    def _extract_key_points(self, scores, analysis_details):
        """핵심 포인트 추출"""
        points = []
        
        # 시장 효율성 불완전성
        if scores['market_inefficiency'] >= 70:
            points.append("시장 비효율성 활용 기회")
        elif scores['market_inefficiency'] <= 40:
            points.append("효율적 시장 가정")
        
        # 행동 편향
        if scores['behavioral_bias'] >= 70:
            points.append("행동 편향 역이용 가능")
        
        # 정량적 지표
        if scores['quantitative_metrics'] >= 70:
            points.append("우수한 정량적 지표")
        
        # 글로벌 분산
        if scores['global_diversification'] >= 70:
            points.append("글로벌 분산 효과")
        
        # 시장 타이밍
        if scores['market_timing'] >= 70:
            points.append("유리한 시장 타이밍")
        
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