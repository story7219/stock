"""
스탠리 드러켄밀러 (Stanley Druckenmiller) 투자 전략

듀케인 캐피털의 창립자, 소로스와 함께 영란은행을 공격한 전설적 투자자
- 탑다운 매크로 경제 분석
- 고집중 포트폴리오와 레버리지 활용
- 변곡점(Inflection Point) 포착
"""

import logging
from typing import Dict, List
from ..common import BaseStrategy, StrategyResult

logger = logging.getLogger(__name__)

class DruckenmillerStrategy(BaseStrategy):
    """스탠리 드러켄밀러 전략 구현"""
    
    def __init__(self):
        super().__init__()
        self.strategy_name = "스탠리 드러켄밀러 (Stanley Druckenmiller)"
        self.description = "탑다운 매크로 분석과 변곡점 포착 전략"
        
        # 가중치 설정 (총합 100%)
        self.weights = {
            'macro_theme': 35,           # 매크로 테마
            'inflection_point': 25,      # 변곡점 포착
            'concentration': 20,         # 집중투자
            'risk_reward': 15,           # 위험대비수익률
            'timing': 5                  # 타이밍
        }
    
    def analyze_stock(self, stock) -> StrategyResult:
        """드러켄밀러 전략으로 주식 분석"""
        try:
            scores = {}
            analysis_details = {}
            
            # 1. 매크로 테마 분석 (35%)
            macro_score, macro_analysis = self._analyze_macro_theme(stock)
            scores['macro_theme'] = macro_score
            analysis_details['macro_theme'] = macro_analysis
            
            # 2. 변곡점 포착 분석 (25%)
            inflection_score, inflection_analysis = self._analyze_inflection_point(stock)
            scores['inflection_point'] = inflection_score
            analysis_details['inflection_point'] = inflection_analysis
            
            # 3. 집중투자 분석 (20%)
            concentration_score, concentration_analysis = self._analyze_concentration(stock)
            scores['concentration'] = concentration_score
            analysis_details['concentration'] = concentration_analysis
            
            # 4. 위험대비수익률 분석 (15%)
            risk_reward_score, risk_reward_analysis = self._analyze_risk_reward(stock)
            scores['risk_reward'] = risk_reward_score
            analysis_details['risk_reward'] = risk_reward_analysis
            
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
            logger.error(f"드러켄밀러 전략 분석 중 오류: {e}")
            return self._create_error_result()
    
    def _analyze_macro_theme(self, stock) -> tuple:
        """매크로 테마 분석"""
        try:
            score = 50
            analysis = {}
            
            # 주요 매크로 트렌드
            if hasattr(stock, 'macro_trend_strength'):
                trend_score = stock.macro_trend_strength * 30
                score += trend_score
                analysis['macro_trend_strength'] = stock.macro_trend_strength
            
            # 통화정책 영향
            if hasattr(stock, 'monetary_policy_impact'):
                policy_score = abs(stock.monetary_policy_impact) * 25
                score += policy_score
                analysis['monetary_policy_impact'] = stock.monetary_policy_impact
            
            # 지정학적 리스크
            if hasattr(stock, 'geopolitical_risk'):
                geo_score = stock.geopolitical_risk * 20
                score += geo_score
                analysis['geopolitical_risk'] = stock.geopolitical_risk
            
            # 구조적 변화
            if hasattr(stock, 'structural_change'):
                structural_score = stock.structural_change * 15
                score += structural_score
                analysis['structural_change'] = stock.structural_change
            
            # 글로벌 자본 흐름
            if hasattr(stock, 'global_capital_flow'):
                capital_score = abs(stock.global_capital_flow) * 10
                score += capital_score
                analysis['global_capital_flow'] = stock.global_capital_flow
            
            return min(max(score, 0), 100), analysis
            
        except Exception as e:
            logger.error(f"매크로 테마 분석 오류: {e}")
            return 50, {"error": str(e)}
    
    def _analyze_inflection_point(self, stock) -> tuple:
        """변곡점 포착 분석"""
        try:
            score = 50
            analysis = {}
            
            # 트렌드 전환 신호
            if hasattr(stock, 'trend_reversal_signal'):
                reversal_score = stock.trend_reversal_signal * 35
                score += reversal_score
                analysis['trend_reversal_signal'] = stock.trend_reversal_signal
            
            # 모멘텀 변화
            if hasattr(stock, 'momentum_shift'):
                momentum_score = abs(stock.momentum_shift) * 25
                score += momentum_score
                analysis['momentum_shift'] = stock.momentum_shift
            
            # 시장 구조 변화
            if hasattr(stock, 'market_structure_shift'):
                structure_score = stock.market_structure_shift * 20
                score += structure_score
                analysis['market_structure_shift'] = stock.market_structure_shift
            
            # 펀더멘털 변곡점
            if hasattr(stock, 'fundamental_inflection'):
                fundamental_score = stock.fundamental_inflection * 15
                score += fundamental_score
                analysis['fundamental_inflection'] = stock.fundamental_inflection
            
            # 센티멘트 극단성
            if hasattr(stock, 'sentiment_extreme'):
                sentiment_score = abs(stock.sentiment_extreme) * 5
                score += sentiment_score
                analysis['sentiment_extreme'] = stock.sentiment_extreme
            
            return min(max(score, 0), 100), analysis
            
        except Exception as e:
            logger.error(f"변곡점 포착 분석 오류: {e}")
            return 50, {"error": str(e)}
    
    def _analyze_concentration(self, stock) -> tuple:
        """집중투자 분석"""
        try:
            score = 50
            analysis = {}
            
            # 확신도
            if hasattr(stock, 'conviction_level'):
                conviction_score = stock.conviction_level * 40
                score += conviction_score
                analysis['conviction_level'] = stock.conviction_level
            
            # 정보 우위
            if hasattr(stock, 'information_edge'):
                info_score = stock.information_edge * 30
                score += info_score
                analysis['information_edge'] = stock.information_edge
            
            # 포지션 사이징
            if hasattr(stock, 'position_sizing_score'):
                sizing_score = stock.position_sizing_score * 20
                score += sizing_score
                analysis['position_sizing_score'] = stock.position_sizing_score
            
            # 집중도 리스크
            if hasattr(stock, 'concentration_risk'):
                # 높은 집중도는 리스크이지만 드러켄밀러 스타일
                risk_score = stock.concentration_risk * 10
                score += risk_score
                analysis['concentration_risk'] = stock.concentration_risk
            
            return min(max(score, 0), 100), analysis
            
        except Exception as e:
            logger.error(f"집중투자 분석 오류: {e}")
            return 50, {"error": str(e)}
    
    def _analyze_risk_reward(self, stock) -> tuple:
        """위험대비수익률 분석"""
        try:
            score = 50
            analysis = {}
            
            # 비대칭 수익 기회
            if hasattr(stock, 'asymmetric_opportunity'):
                asymmetric_score = stock.asymmetric_opportunity * 40
                score += asymmetric_score
                analysis['asymmetric_opportunity'] = stock.asymmetric_opportunity
            
            # 하방 리스크 제한
            if hasattr(stock, 'downside_protection'):
                downside_score = stock.downside_protection * 30
                score += downside_score
                analysis['downside_protection'] = stock.downside_protection
            
            # 상방 잠재력
            if hasattr(stock, 'upside_potential'):
                upside_score = stock.upside_potential * 20
                score += upside_score
                analysis['upside_potential'] = stock.upside_potential
            
            # 리스크 조정 수익률
            if hasattr(stock, 'risk_adjusted_return'):
                risk_adj_score = stock.risk_adjusted_return * 10
                score += risk_adj_score
                analysis['risk_adjusted_return'] = stock.risk_adjusted_return
            
            return min(max(score, 0), 100), analysis
            
        except Exception as e:
            logger.error(f"위험대비수익률 분석 오류: {e}")
            return 50, {"error": str(e)}
    
    def _analyze_timing(self, stock) -> tuple:
        """타이밍 분석"""
        try:
            score = 50
            analysis = {}
            
            # 시장 타이밍
            if hasattr(stock, 'market_timing_signal'):
                timing_score = stock.market_timing_signal * 50
                score += timing_score
                analysis['market_timing_signal'] = stock.market_timing_signal
            
            # 진입/청산 신호
            if hasattr(stock, 'entry_exit_signal'):
                signal_score = stock.entry_exit_signal * 30
                score += signal_score
                analysis['entry_exit_signal'] = stock.entry_exit_signal
            
            # 시장 사이클 위치
            if hasattr(stock, 'market_cycle_position'):
                cycle_score = stock.market_cycle_position * 20
                score += cycle_score
                analysis['market_cycle_position'] = stock.market_cycle_position
            
            return min(max(score, 0), 100), analysis
            
        except Exception as e:
            logger.error(f"타이밍 분석 오류: {e}")
            return 50, {"error": str(e)}
    
    def _make_investment_decision(self, total_score):
        """투자 판단 결정"""
        if total_score >= 80:
            return "강력매수 - 매크로 변곡점 완벽 포착"
        elif total_score >= 70:
            return "매수 - 강한 매크로 테마 확인"
        elif total_score >= 60:
            return "관심 - 변곡점 신호 관찰"
        elif total_score >= 50:
            return "중립 - 매크로 신호 불분명"
        else:
            return "회피 - 매크로 환경 불리"
    
    def _extract_key_points(self, scores, analysis_details):
        """핵심 포인트 추출"""
        points = []
        
        # 매크로 테마
        if scores['macro_theme'] >= 70:
            points.append("강력한 매크로 테마 부상")
        elif scores['macro_theme'] <= 40:
            points.append("매크로 테마 부재")
        
        # 변곡점 포착
        if scores['inflection_point'] >= 70:
            points.append("중요한 변곡점 포착")
        
        # 집중투자
        if scores['concentration'] >= 70:
            points.append("높은 확신도 집중투자 기회")
        
        # 위험대비수익률
        if scores['risk_reward'] >= 70:
            points.append("우수한 비대칭 수익 기회")
        
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