"""
폴 튜더 존스 (Paul Tudor Jones) 투자 전략

튜더 투자 코퍼레이션 창립자, 매크로 헤지펀드의 전설
- 리스크 우선 접근법
- 매크로 경제와 기술적 분석 결합
- 5:1 리스크 리워드 비율 준수
"""

import logging
from typing import Dict, List
from ..common import BaseStrategy, StrategyResult

logger = logging.getLogger(__name__)

class TudorJonesStrategy(BaseStrategy):
    """폴 튜더 존스 전략 구현"""
    
    def __init__(self):
        super().__init__()
        self.strategy_name = "폴 튜더 존스 (Paul Tudor Jones)"
        self.description = "리스크 우선 매크로 기술적 분석 전략"
        
        # 가중치 설정 (총합 100%)
        self.weights = {
            'risk_management': 35,       # 리스크 관리
            'macro_trend': 25,           # 매크로 추세
            'technical_pattern': 20,     # 기술적 패턴
            'market_psychology': 15,     # 시장 심리
            'timing': 5                  # 타이밍
        }
    
    def analyze_stock(self, stock) -> StrategyResult:
        """튜더 존스 전략으로 주식 분석"""
        try:
            scores = {}
            analysis_details = {}
            
            # 1. 리스크 관리 분석 (35%)
            risk_score, risk_analysis = self._analyze_risk_management(stock)
            scores['risk_management'] = risk_score
            analysis_details['risk_management'] = risk_analysis
            
            # 2. 매크로 추세 분석 (25%)
            macro_score, macro_analysis = self._analyze_macro_trend(stock)
            scores['macro_trend'] = macro_score
            analysis_details['macro_trend'] = macro_analysis
            
            # 3. 기술적 패턴 분석 (20%)
            pattern_score, pattern_analysis = self._analyze_technical_pattern(stock)
            scores['technical_pattern'] = pattern_score
            analysis_details['technical_pattern'] = pattern_analysis
            
            # 4. 시장 심리 분석 (15%)
            psychology_score, psychology_analysis = self._analyze_market_psychology(stock)
            scores['market_psychology'] = psychology_score
            analysis_details['market_psychology'] = psychology_analysis
            
            # 5. 타이밍 분석 (5%)
            timing_score, timing_analysis = self._analyze_timing(stock)
            scores['timing'] = timing_score
            analysis_details['timing'] = timing_analysis
            
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
            logger.error(f"튜더 존스 전략 분석 중 오류: {e}")
            return self._create_error_result()
    
    def _analyze_risk_management(self, stock) -> tuple:
        """리스크 관리 분석"""
        try:
            score = 50
            analysis = {}
            
            # 리스크-리워드 비율
            if hasattr(stock, 'risk_reward_ratio'):
                # 5:1 이상 선호
                if stock.risk_reward_ratio >= 5:
                    rr_score = 40
                elif stock.risk_reward_ratio >= 3:
                    rr_score = 30
                elif stock.risk_reward_ratio >= 2:
                    rr_score = 20
                elif stock.risk_reward_ratio >= 1:
                    rr_score = 10
                else:
                    rr_score = 0
                score += rr_score
                analysis['risk_reward_ratio'] = stock.risk_reward_ratio
            
            # 최대 손실 한도
            if hasattr(stock, 'max_drawdown_limit'):
                # 낮은 최대손실일수록 좋음
                drawdown_score = max(25 - stock.max_drawdown_limit * 50, 0)
                score += drawdown_score
                analysis['max_drawdown_limit'] = stock.max_drawdown_limit
            
            # 포지션 사이징
            if hasattr(stock, 'position_sizing_discipline'):
                sizing_score = stock.position_sizing_discipline * 20
                score += sizing_score
                analysis['position_sizing_discipline'] = stock.position_sizing_discipline
            
            # 손절매 규율
            if hasattr(stock, 'stop_loss_discipline'):
                stop_score = stock.stop_loss_discipline * 10
                score += stop_score
                analysis['stop_loss_discipline'] = stock.stop_loss_discipline
            
            # 변동성 조정
            if hasattr(stock, 'volatility_adjustment'):
                vol_score = stock.volatility_adjustment * 5
                score += vol_score
                analysis['volatility_adjustment'] = stock.volatility_adjustment
            
            return min(max(score, 0), 100), analysis
            
        except Exception as e:
            logger.error(f"리스크 관리 분석 오류: {e}")
            return 50, {"error": str(e)}
    
    def _analyze_macro_trend(self, stock) -> tuple:
        """매크로 추세 분석"""
        try:
            score = 50
            analysis = {}
            
            # 경제 사이클 위치
            if hasattr(stock, 'economic_cycle_position'):
                cycle_score = self._evaluate_cycle_position(stock.economic_cycle_position)
                score += cycle_score * 30
                analysis['economic_cycle_position'] = stock.economic_cycle_position
            
            # 통화정책 방향
            if hasattr(stock, 'monetary_policy_direction'):
                policy_score = abs(stock.monetary_policy_direction) * 25
                score += policy_score
                analysis['monetary_policy_direction'] = stock.monetary_policy_direction
            
            # 인플레이션 추세
            if hasattr(stock, 'inflation_trend'):
                inflation_score = self._evaluate_inflation_impact(stock.inflation_trend)
                score += inflation_score * 20
                analysis['inflation_trend'] = stock.inflation_trend
            
            # 글로벌 리스크 온/오프
            if hasattr(stock, 'global_risk_sentiment'):
                risk_score = stock.global_risk_sentiment * 15
                score += risk_score
                analysis['global_risk_sentiment'] = stock.global_risk_sentiment
            
            # 유동성 환경
            if hasattr(stock, 'liquidity_environment'):
                liquidity_score = stock.liquidity_environment * 10
                score += liquidity_score
                analysis['liquidity_environment'] = stock.liquidity_environment
            
            return min(max(score, 0), 100), analysis
            
        except Exception as e:
            logger.error(f"매크로 추세 분석 오류: {e}")
            return 50, {"error": str(e)}
    
    def _analyze_technical_pattern(self, stock) -> tuple:
        """기술적 패턴 분석"""
        try:
            score = 50
            analysis = {}
            
            # 차트 패턴 강도
            if hasattr(stock, 'chart_pattern_strength'):
                pattern_score = stock.chart_pattern_strength * 35
                score += pattern_score
                analysis['chart_pattern_strength'] = stock.chart_pattern_strength
            
            # 추세 확인
            if hasattr(stock, 'trend_confirmation'):
                trend_score = stock.trend_confirmation * 25
                score += trend_score
                analysis['trend_confirmation'] = stock.trend_confirmation
            
            # 지지/저항 레벨
            if hasattr(stock, 'support_resistance_level'):
                sr_score = stock.support_resistance_level * 20
                score += sr_score
                analysis['support_resistance_level'] = stock.support_resistance_level
            
            # 거래량 확인
            if hasattr(stock, 'volume_confirmation'):
                volume_score = stock.volume_confirmation * 15
                score += volume_score
                analysis['volume_confirmation'] = stock.volume_confirmation
            
            # 모멘텀 지표
            if hasattr(stock, 'momentum_indicators'):
                momentum_score = stock.momentum_indicators * 5
                score += momentum_score
                analysis['momentum_indicators'] = stock.momentum_indicators
            
            return min(max(score, 0), 100), analysis
            
        except Exception as e:
            logger.error(f"기술적 패턴 분석 오류: {e}")
            return 50, {"error": str(e)}
    
    def _analyze_market_psychology(self, stock) -> tuple:
        """시장 심리 분석"""
        try:
            score = 50
            analysis = {}
            
            # 투자자 심리 극단성
            if hasattr(stock, 'investor_sentiment_extreme'):
                # 극단적 심리일 때 역방향 베팅 기회
                sentiment_score = abs(stock.investor_sentiment_extreme) * 40
                score += sentiment_score
                analysis['investor_sentiment_extreme'] = stock.investor_sentiment_extreme
            
            # 공포/탐욕 지수
            if hasattr(stock, 'fear_greed_index'):
                fg_score = self._evaluate_fear_greed(stock.fear_greed_index)
                score += fg_score * 30
                analysis['fear_greed_index'] = stock.fear_greed_index
            
            # 시장 참여자 포지셔닝
            if hasattr(stock, 'market_positioning'):
                positioning_score = abs(stock.market_positioning) * 20
                score += positioning_score
                analysis['market_positioning'] = stock.market_positioning
            
            # 뉴스 센티멘트
            if hasattr(stock, 'news_sentiment'):
                news_score = abs(stock.news_sentiment) * 10
                score += news_score
                analysis['news_sentiment'] = stock.news_sentiment
            
            return min(max(score, 0), 100), analysis
            
        except Exception as e:
            logger.error(f"시장 심리 분석 오류: {e}")
            return 50, {"error": str(e)}
    
    def _analyze_timing(self, stock) -> tuple:
        """타이밍 분석"""
        try:
            score = 50
            analysis = {}
            
            # 진입 신호 강도
            if hasattr(stock, 'entry_signal_strength'):
                entry_score = stock.entry_signal_strength * 50
                score += entry_score
                analysis['entry_signal_strength'] = stock.entry_signal_strength
            
            # 시장 타이밍
            if hasattr(stock, 'market_timing_score'):
                timing_score = stock.market_timing_score * 30
                score += timing_score
                analysis['market_timing_score'] = stock.market_timing_score
            
            # 촉매 이벤트
            if hasattr(stock, 'catalyst_event_proximity'):
                catalyst_score = stock.catalyst_event_proximity * 20
                score += catalyst_score
                analysis['catalyst_event_proximity'] = stock.catalyst_event_proximity
            
            return min(max(score, 0), 100), analysis
            
        except Exception as e:
            logger.error(f"타이밍 분석 오류: {e}")
            return 50, {"error": str(e)}
    
    def _evaluate_cycle_position(self, position):
        """경제 사이클 위치 평가"""
        # 확장기 초기와 회복기에 높은 점수
        cycle_scores = {
            'recession': 0.2,
            'recovery': 0.8,
            'expansion_early': 1.0,
            'expansion_late': 0.6,
            'peak': 0.3,
            'contraction': 0.1
        }
        return cycle_scores.get(position, 0.5)
    
    def _evaluate_inflation_impact(self, inflation_rate):
        """인플레이션 영향 평가"""
        if 2 <= inflation_rate <= 3:
            return 1.0  # 적정 인플레이션
        elif inflation_rate > 5:
            return 0.3  # 높은 인플레이션
        elif inflation_rate < 0:
            return 0.2  # 디플레이션
        else:
            return 0.7  # 보통
    
    def _evaluate_fear_greed(self, fg_index):
        """공포/탐욕 지수 평가"""
        # 극단적 공포(0-20)나 극단적 탐욕(80-100)에서 높은 점수
        if fg_index <= 20 or fg_index >= 80:
            return 1.0
        elif fg_index <= 30 or fg_index >= 70:
            return 0.7
        else:
            return 0.3
    
    def _make_investment_decision(self, total_score):
        """투자 판단 결정"""
        if total_score >= 80:
            return "강력매수 - 완벽한 리스크-리워드 기회"
        elif total_score >= 70:
            return "매수 - 우수한 매크로 기술적 신호"
        elif total_score >= 60:
            return "관심 - 리스크 관리하며 관찰"
        elif total_score >= 50:
            return "중립 - 명확한 신호 대기"
        else:
            return "회피 - 리스크 대비 보상 부족"
    
    def _extract_key_points(self, scores, analysis_details):
        """핵심 포인트 추출"""
        points = []
        
        # 리스크 관리
        if scores['risk_management'] >= 70:
            points.append("우수한 리스크-리워드 비율")
        elif scores['risk_management'] <= 40:
            points.append("리스크 관리 부족")
        
        # 매크로 추세
        if scores['macro_trend'] >= 70:
            points.append("유리한 매크로 환경")
        
        # 기술적 패턴
        if scores['technical_pattern'] >= 70:
            points.append("강력한 기술적 신호")
        
        # 시장 심리
        if scores['market_psychology'] >= 70:
            points.append("극단적 시장 심리 활용")
        
        # 타이밍
        if scores['timing'] >= 70:
            points.append("최적 진입 타이밍")
        
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