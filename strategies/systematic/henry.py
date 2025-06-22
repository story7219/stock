"""
존 헨리 (John W. Henry) 투자 전략

CTA(Commodity Trading Advisor) 및 추세 추종 전략의 대가
- 시스템 매매와 추세 추종에 특화
- 다양한 시장에서 체계적 접근
- 보스턴 레드삭스 구단주로도 유명
"""

import logging
from typing import Dict, List
from ..common import BaseStrategy, StrategyResult

logger = logging.getLogger(__name__)

class HenryStrategy(BaseStrategy):
    """존 헨리 전략 구현"""
    
    def __init__(self):
        super().__init__()
        self.strategy_name = "존 헨리 (John W. Henry)"
        self.description = "CTA 추세 추종 시스템 매매 전략"
        
        # 가중치 설정 (총합 100%)
        self.weights = {
            'trend_following': 35,       # 추세 추종 신호
            'system_signals': 25,        # 시스템 매매 신호
            'multi_market': 20,          # 다중 시장 분석
            'risk_management': 15,       # 위험 관리
            'momentum_strength': 5       # 모멘텀 강도
        }
    
    def analyze_stock(self, stock) -> StrategyResult:
        """헨리 전략으로 주식 분석"""
        try:
            scores = {}
            analysis_details = {}
            
            # 1. 추세 추종 신호 분석 (35%)
            trend_score, trend_analysis = self._analyze_trend_following(stock)
            scores['trend_following'] = trend_score
            analysis_details['trend_following'] = trend_analysis
            
            # 2. 시스템 매매 신호 분석 (25%)
            system_score, system_analysis = self._analyze_system_signals(stock)
            scores['system_signals'] = system_score
            analysis_details['system_signals'] = system_analysis
            
            # 3. 다중 시장 분석 (20%)
            multi_market_score, multi_market_analysis = self._analyze_multi_market(stock)
            scores['multi_market'] = multi_market_score
            analysis_details['multi_market'] = multi_market_analysis
            
            # 4. 위험 관리 분석 (15%)
            risk_score, risk_analysis = self._analyze_risk_management(stock)
            scores['risk_management'] = risk_score
            analysis_details['risk_management'] = risk_analysis
            
            # 5. 모멘텀 강도 분석 (5%)
            momentum_score, momentum_analysis = self._analyze_momentum_strength(stock)
            scores['momentum_strength'] = momentum_score
            analysis_details['momentum_strength'] = momentum_analysis
            
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
            logger.error(f"헨리 전략 분석 중 오류: {e}")
            return self._create_error_result()
    
    def _analyze_trend_following(self, stock) -> tuple:
        """추세 추종 신호 분석"""
        try:
            score = 50
            analysis = {}
            
            # 이동평균 추세
            if hasattr(stock, 'prices') and len(stock.prices) >= 50:
                ma_trend = self._calculate_moving_average_trend(stock.prices)
                score += ma_trend * 30
                analysis['ma_trend'] = ma_trend
            
            # 추세 강도
            if hasattr(stock, 'trend_strength'):
                trend_strength = min(stock.trend_strength * 20, 20)
                score += trend_strength
                analysis['trend_strength'] = stock.trend_strength
            
            # 추세 지속성
            if hasattr(stock, 'prices') and len(stock.prices) >= 20:
                trend_persistence = self._calculate_trend_persistence(stock.prices)
                score += trend_persistence * 15
                analysis['trend_persistence'] = trend_persistence
            
            # 브레이크아웃 강도
            if hasattr(stock, 'breakout_strength'):
                breakout_score = stock.breakout_strength * 15
                score += breakout_score
                analysis['breakout_strength'] = stock.breakout_strength
            
            return min(max(score, 0), 100), analysis
            
        except Exception as e:
            logger.error(f"추세 추종 분석 오류: {e}")
            return 50, {"error": str(e)}
    
    def _analyze_system_signals(self, stock) -> tuple:
        """시스템 매매 신호 분석"""
        try:
            score = 50
            analysis = {}
            
            # 다중 시간대 신호
            if hasattr(stock, 'multi_timeframe_signals'):
                mtf_score = stock.multi_timeframe_signals * 25
                score += mtf_score
                analysis['multi_timeframe'] = stock.multi_timeframe_signals
            
            # 시스템 일치도
            if hasattr(stock, 'system_consensus'):
                consensus_score = stock.system_consensus * 20
                score += consensus_score
                analysis['system_consensus'] = stock.system_consensus
            
            # 신호 강도
            if hasattr(stock, 'signal_strength'):
                signal_score = stock.signal_strength * 15
                score += signal_score
                analysis['signal_strength'] = stock.signal_strength
            
            # 백테스트 성과
            if hasattr(stock, 'backtest_performance'):
                backtest_score = min(stock.backtest_performance * 10, 10)
                score += backtest_score
                analysis['backtest_performance'] = stock.backtest_performance
            
            return min(max(score, 0), 100), analysis
            
        except Exception as e:
            logger.error(f"시스템 신호 분석 오류: {e}")
            return 50, {"error": str(e)}
    
    def _analyze_multi_market(self, stock) -> tuple:
        """다중 시장 분석"""
        try:
            score = 50
            analysis = {}
            
            # 섹터 상관관계
            if hasattr(stock, 'sector_correlation'):
                sector_score = (1 - abs(stock.sector_correlation)) * 20
                score += sector_score
                analysis['sector_correlation'] = stock.sector_correlation
            
            # 시장 간 강도
            if hasattr(stock, 'inter_market_strength'):
                inter_market_score = stock.inter_market_strength * 15
                score += inter_market_score
                analysis['inter_market_strength'] = stock.inter_market_strength
            
            # 글로벌 추세 일치
            if hasattr(stock, 'global_trend_alignment'):
                global_score = stock.global_trend_alignment * 15
                score += global_score
                analysis['global_trend_alignment'] = stock.global_trend_alignment
            
            # 원자재 연관성
            if hasattr(stock, 'commodity_correlation'):
                commodity_score = abs(stock.commodity_correlation) * 10
                score += commodity_score
                analysis['commodity_correlation'] = stock.commodity_correlation
            
            return min(max(score, 0), 100), analysis
            
        except Exception as e:
            logger.error(f"다중 시장 분석 오류: {e}")
            return 50, {"error": str(e)}
    
    def _analyze_risk_management(self, stock) -> tuple:
        """위험 관리 분석"""
        try:
            score = 50
            analysis = {}
            
            # 변동성 조정 수익률
            if hasattr(stock, 'volatility_adjusted_return'):
                vol_adj_score = min(stock.volatility_adjusted_return * 25, 25)
                score += vol_adj_score
                analysis['volatility_adjusted_return'] = stock.volatility_adjusted_return
            
            # 최대 손실 제한
            if hasattr(stock, 'max_drawdown'):
                drawdown_score = max(25 - abs(stock.max_drawdown), 0)
                score += drawdown_score
                analysis['max_drawdown'] = stock.max_drawdown
            
            # 포지션 사이징
            if hasattr(stock, 'position_sizing_score'):
                position_score = stock.position_sizing_score * 15
                score += position_score
                analysis['position_sizing'] = stock.position_sizing_score
            
            # 상관관계 분산
            if hasattr(stock, 'correlation_diversification'):
                corr_score = stock.correlation_diversification * 10
                score += corr_score
                analysis['correlation_diversification'] = stock.correlation_diversification
            
            return min(max(score, 0), 100), analysis
            
        except Exception as e:
            logger.error(f"위험 관리 분석 오류: {e}")
            return 50, {"error": str(e)}
    
    def _analyze_momentum_strength(self, stock) -> tuple:
        """모멘텀 강도 분석"""
        try:
            score = 50
            analysis = {}
            
            # 가격 모멘텀
            if hasattr(stock, 'price_momentum'):
                price_momentum_score = stock.price_momentum * 30
                score += price_momentum_score
                analysis['price_momentum'] = stock.price_momentum
            
            # 거래량 모멘텀
            if hasattr(stock, 'volume_momentum'):
                volume_momentum_score = stock.volume_momentum * 20
                score += volume_momentum_score
                analysis['volume_momentum'] = stock.volume_momentum
            
            return min(max(score, 0), 100), analysis
            
        except Exception as e:
            logger.error(f"모멘텀 강도 분석 오류: {e}")
            return 50, {"error": str(e)}
    
    def _calculate_moving_average_trend(self, prices):
        """이동평균 추세 계산"""
        if len(prices) < 50:
            return 0
        
        # 단기, 중기, 장기 이동평균
        short_ma = sum(prices[-10:]) / 10
        medium_ma = sum(prices[-20:]) / 20
        long_ma = sum(prices[-50:]) / 50
        
        current_price = prices[-1]
        
        # 추세 방향 판단
        if current_price > short_ma > medium_ma > long_ma:
            return 1  # 강한 상승 추세
        elif current_price > short_ma > medium_ma:
            return 0.7  # 중간 상승 추세
        elif current_price > short_ma:
            return 0.3  # 약한 상승 추세
        elif current_price < short_ma < medium_ma < long_ma:
            return -1  # 강한 하락 추세
        elif current_price < short_ma < medium_ma:
            return -0.7  # 중간 하락 추세
        elif current_price < short_ma:
            return -0.3  # 약한 하락 추세
        else:
            return 0  # 횡보
    
    def _calculate_trend_persistence(self, prices):
        """추세 지속성 계산"""
        if len(prices) < 20:
            return 0
        
        # 최근 20일간 추세 방향 일관성
        daily_changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        recent_changes = daily_changes[-20:]
        
        positive_days = sum(1 for change in recent_changes if change > 0)
        negative_days = sum(1 for change in recent_changes if change < 0)
        
        # 추세 일관성 점수
        consistency = abs(positive_days - negative_days) / len(recent_changes)
        return consistency
    
    def _make_investment_decision(self, total_score):
        """투자 판단 결정"""
        if total_score >= 80:
            return "강력매수 - 강한 추세와 시스템 신호 일치"
        elif total_score >= 70:
            return "매수 - 추세 추종 신호 양호"
        elif total_score >= 60:
            return "관심 - 추세 형성 중 관찰"
        elif total_score >= 50:
            return "중립 - 명확한 추세 부재"
        else:
            return "회피 - 역추세 또는 불안정"
    
    def _extract_key_points(self, scores, analysis_details):
        """핵심 포인트 추출"""
        points = []
        
        # 추세 추종
        if scores['trend_following'] >= 70:
            points.append("강력한 추세 추종 신호")
        elif scores['trend_following'] <= 40:
            points.append("약한 추세 또는 역추세")
        
        # 시스템 신호
        if scores['system_signals'] >= 70:
            points.append("시스템 매매 신호 양호")
        elif scores['system_signals'] <= 40:
            points.append("시스템 신호 불일치")
        
        # 다중 시장
        if scores['multi_market'] >= 70:
            points.append("다중 시장 분석 긍정적")
        
        # 위험 관리
        if scores['risk_management'] >= 70:
            points.append("우수한 위험 관리 지표")
        elif scores['risk_management'] <= 40:
            points.append("높은 위험 수준")
        
        # 모멘텀
        if scores['momentum_strength'] >= 70:
            points.append("강한 모멘텀 확인")
        
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