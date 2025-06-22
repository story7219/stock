"""
에드 세이코타 (Ed Seykota) 투자 전략

자동매매의 선구자이자 트레이딩 심리학의 대가
- "트레이딩은 심리 싸움"의 철학
- 시스템적 접근과 심리적 통제의 조화
- 추세 추종과 위험 관리의 균형
"""

import logging
from typing import Dict, List
from ..common import BaseStrategy, StrategyResult

logger = logging.getLogger(__name__)

class SeykotaStrategy(BaseStrategy):
    """에드 세이코타 전략 구현"""
    
    def __init__(self):
        super().__init__()
        self.strategy_name = "에드 세이코타 (Ed Seykota)"
        self.description = "자동매매 시스템과 심리적 트레이딩 전략"
        
        # 가중치 설정 (총합 100%)
        self.weights = {
            'system_discipline': 30,     # 시스템 규율
            'trend_following': 25,       # 추세 추종
            'risk_control': 20,          # 위험 통제
            'psychological_edge': 15,    # 심리적 우위
            'market_timing': 10          # 시장 타이밍
        }
    
    def analyze_stock(self, stock) -> StrategyResult:
        """세이코타 전략으로 주식 분석"""
        try:
            scores = {}
            analysis_details = {}
            
            # 1. 시스템 규율 분석 (30%)
            system_score, system_analysis = self._analyze_system_discipline(stock)
            scores['system_discipline'] = system_score
            analysis_details['system_discipline'] = system_analysis
            
            # 2. 추세 추종 분석 (25%)
            trend_score, trend_analysis = self._analyze_trend_following(stock)
            scores['trend_following'] = trend_score
            analysis_details['trend_following'] = trend_analysis
            
            # 3. 위험 통제 분석 (20%)
            risk_score, risk_analysis = self._analyze_risk_control(stock)
            scores['risk_control'] = risk_score
            analysis_details['risk_control'] = risk_analysis
            
            # 4. 심리적 우위 분석 (15%)
            psychology_score, psychology_analysis = self._analyze_psychological_edge(stock)
            scores['psychological_edge'] = psychology_score
            analysis_details['psychological_edge'] = psychology_analysis
            
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
            logger.error(f"세이코타 전략 분석 중 오류: {e}")
            return self._create_error_result()
    
    def _analyze_system_discipline(self, stock) -> tuple:
        """시스템 규율 분석"""
        try:
            score = 50
            analysis = {}
            
            # 시스템 신호 일관성
            if hasattr(stock, 'system_consistency'):
                consistency_score = stock.system_consistency * 25
                score += consistency_score
                analysis['system_consistency'] = stock.system_consistency
            
            # 규칙 준수도
            if hasattr(stock, 'rule_adherence'):
                adherence_score = stock.rule_adherence * 20
                score += adherence_score
                analysis['rule_adherence'] = stock.rule_adherence
            
            # 자동화 수준
            if hasattr(stock, 'automation_level'):
                automation_score = stock.automation_level * 15
                score += automation_score
                analysis['automation_level'] = stock.automation_level
            
            # 백테스트 신뢰도
            if hasattr(stock, 'backtest_reliability'):
                backtest_score = stock.backtest_reliability * 15
                score += backtest_score
                analysis['backtest_reliability'] = stock.backtest_reliability
            
            # 시스템 안정성
            if hasattr(stock, 'system_stability'):
                stability_score = stock.system_stability * 10
                score += stability_score
                analysis['system_stability'] = stock.system_stability
            
            return min(max(score, 0), 100), analysis
            
        except Exception as e:
            logger.error(f"시스템 규율 분석 오류: {e}")
            return 50, {"error": str(e)}
    
    def _analyze_trend_following(self, stock) -> tuple:
        """추세 추종 분석"""
        try:
            score = 50
            analysis = {}
            
            # 추세 강도
            if hasattr(stock, 'prices') and len(stock.prices) >= 50:
                trend_strength = self._calculate_trend_strength(stock.prices)
                score += trend_strength * 30
                analysis['trend_strength'] = trend_strength
            
            # 추세 지속성
            if hasattr(stock, 'trend_persistence'):
                persistence_score = stock.trend_persistence * 25
                score += persistence_score
                analysis['trend_persistence'] = stock.trend_persistence
            
            # 브레이크아웃 품질
            if hasattr(stock, 'breakout_quality'):
                breakout_score = stock.breakout_quality * 20
                score += breakout_score
                analysis['breakout_quality'] = stock.breakout_quality
            
            # 추세 확인 신호
            if hasattr(stock, 'trend_confirmation'):
                confirmation_score = stock.trend_confirmation * 15
                score += confirmation_score
                analysis['trend_confirmation'] = stock.trend_confirmation
            
            return min(max(score, 0), 100), analysis
            
        except Exception as e:
            logger.error(f"추세 추종 분석 오류: {e}")
            return 50, {"error": str(e)}
    
    def _analyze_risk_control(self, stock) -> tuple:
        """위험 통제 분석"""
        try:
            score = 50
            analysis = {}
            
            # 손절 규율
            if hasattr(stock, 'stop_loss_discipline'):
                stop_loss_score = stock.stop_loss_discipline * 30
                score += stop_loss_score
                analysis['stop_loss_discipline'] = stock.stop_loss_discipline
            
            # 포지션 사이징
            if hasattr(stock, 'position_sizing_quality'):
                position_score = stock.position_sizing_quality * 25
                score += position_score
                analysis['position_sizing_quality'] = stock.position_sizing_quality
            
            # 리스크 대비 수익률
            if hasattr(stock, 'risk_reward_ratio'):
                rr_score = min(stock.risk_reward_ratio * 15, 20)
                score += rr_score
                analysis['risk_reward_ratio'] = stock.risk_reward_ratio
            
            # 최대 손실 제한
            if hasattr(stock, 'max_loss_control'):
                max_loss_score = stock.max_loss_control * 15
                score += max_loss_score
                analysis['max_loss_control'] = stock.max_loss_control
            
            return min(max(score, 0), 100), analysis
            
        except Exception as e:
            logger.error(f"위험 통제 분석 오류: {e}")
            return 50, {"error": str(e)}
    
    def _analyze_psychological_edge(self, stock) -> tuple:
        """심리적 우위 분석"""
        try:
            score = 50
            analysis = {}
            
            # 시장 심리 역행
            if hasattr(stock, 'market_sentiment'):
                # 시장 심리와 반대 방향일 때 높은 점수
                sentiment_score = (1 - abs(stock.market_sentiment)) * 25
                score += sentiment_score
                analysis['market_sentiment'] = stock.market_sentiment
            
            # 감정적 안정성
            if hasattr(stock, 'emotional_stability'):
                stability_score = stock.emotional_stability * 20
                score += stability_score
                analysis['emotional_stability'] = stock.emotional_stability
            
            # 군중 심리 분석
            if hasattr(stock, 'crowd_psychology'):
                crowd_score = stock.crowd_psychology * 15
                score += crowd_score
                analysis['crowd_psychology'] = stock.crowd_psychology
            
            # 두려움/탐욕 지수
            if hasattr(stock, 'fear_greed_index'):
                fg_score = self._evaluate_fear_greed(stock.fear_greed_index)
                score += fg_score * 15
                analysis['fear_greed_index'] = stock.fear_greed_index
            
            # 자기 통제력
            if hasattr(stock, 'self_control_score'):
                control_score = stock.self_control_score * 10
                score += control_score
                analysis['self_control_score'] = stock.self_control_score
            
            return min(max(score, 0), 100), analysis
            
        except Exception as e:
            logger.error(f"심리적 우위 분석 오류: {e}")
            return 50, {"error": str(e)}
    
    def _analyze_market_timing(self, stock) -> tuple:
        """시장 타이밍 분석"""
        try:
            score = 50
            analysis = {}
            
            # 진입 타이밍
            if hasattr(stock, 'entry_timing_quality'):
                entry_score = stock.entry_timing_quality * 40
                score += entry_score
                analysis['entry_timing_quality'] = stock.entry_timing_quality
            
            # 청산 타이밍
            if hasattr(stock, 'exit_timing_quality'):
                exit_score = stock.exit_timing_quality * 35
                score += exit_score
                analysis['exit_timing_quality'] = stock.exit_timing_quality
            
            # 시장 사이클
            if hasattr(stock, 'market_cycle_position'):
                cycle_score = stock.market_cycle_position * 25
                score += cycle_score
                analysis['market_cycle_position'] = stock.market_cycle_position
            
            return min(max(score, 0), 100), analysis
            
        except Exception as e:
            logger.error(f"시장 타이밍 분석 오류: {e}")
            return 50, {"error": str(e)}
    
    def _calculate_trend_strength(self, prices):
        """추세 강도 계산"""
        if len(prices) < 50:
            return 0
        
        # ADX 유사 계산
        highs = prices
        lows = prices
        closes = prices
        
        # 간단한 추세 강도 계산
        short_trend = sum(closes[-10:]) / 10
        long_trend = sum(closes[-50:]) / 50
        
        trend_strength = abs(short_trend - long_trend) / long_trend
        return min(trend_strength * 100, 1)
    
    def _evaluate_fear_greed(self, fear_greed_index):
        """두려움/탐욕 지수 평가"""
        # 극단적 두려움(0-20)이나 극단적 탐욕(80-100)에서 역방향 기회
        if fear_greed_index <= 20:
            return 1  # 극단적 두려움 - 매수 기회
        elif fear_greed_index >= 80:
            return -1  # 극단적 탐욕 - 매도 기회
        elif 40 <= fear_greed_index <= 60:
            return 0  # 중립
        else:
            return 0.5  # 보통
    
    def _make_investment_decision(self, total_score):
        """투자 판단 결정"""
        if total_score >= 80:
            return "강력매수 - 시스템과 심리 모두 유리"
        elif total_score >= 70:
            return "매수 - 시스템 신호 양호"
        elif total_score >= 60:
            return "관심 - 시스템 준비 중"
        elif total_score >= 50:
            return "중립 - 명확한 신호 부재"
        else:
            return "회피 - 시스템 규율 위반"
    
    def _extract_key_points(self, scores, analysis_details):
        """핵심 포인트 추출"""
        points = []
        
        # 시스템 규율
        if scores['system_discipline'] >= 70:
            points.append("시스템 규율 우수")
        elif scores['system_discipline'] <= 40:
            points.append("시스템 규율 부족")
        
        # 추세 추종
        if scores['trend_following'] >= 70:
            points.append("강력한 추세 신호")
        elif scores['trend_following'] <= 40:
            points.append("약한 추세 또는 횡보")
        
        # 위험 통제
        if scores['risk_control'] >= 70:
            points.append("우수한 위험 관리")
        elif scores['risk_control'] <= 40:
            points.append("위험 관리 부족")
        
        # 심리적 우위
        if scores['psychological_edge'] >= 70:
            points.append("심리적 우위 확보")
        elif scores['psychological_edge'] <= 40:
            points.append("심리적 불리함")
        
        # 시장 타이밍
        if scores['market_timing'] >= 70:
            points.append("적절한 타이밍")
        
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