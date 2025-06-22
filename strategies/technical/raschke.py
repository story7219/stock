"""
린다 래쉬키 (Linda Raschke) 투자 전략

단타 전문 트레이더로 유명한 린다 래쉬키의 패턴 기반 매매 전략
- 기술적 분석과 차트 패턴에 특화
- 단기 매매와 리스크 관리 중심
- "Street Smarts" 저자
"""

import logging
from typing import Dict, List
from ..common import BaseStrategy, StrategyResult

logger = logging.getLogger(__name__)

class RaschkeStrategy(BaseStrategy):
    """린다 래쉬키 전략 구현"""
    
    def __init__(self):
        super().__init__()
        self.strategy_name = "린다 래쉬키 (Linda Raschke)"
        self.description = "패턴 기반 단타 매매 전략"
        
        # 가중치 설정 (총합 100%)
        self.weights = {
            'pattern_strength': 30,      # 차트 패턴 강도
            'momentum_signals': 25,      # 모멘텀 신호
            'volatility_edge': 20,       # 변동성 우위
            'risk_reward': 15,           # 위험대비수익률
            'market_timing': 10          # 시장 타이밍
        }
    
    def analyze_stock(self, stock) -> StrategyResult:
        """래쉬키 전략으로 주식 분석"""
        try:
            scores = {}
            analysis_details = {}
            
            # 1. 차트 패턴 강도 분석 (30%)
            pattern_score, pattern_analysis = self._analyze_pattern_strength(stock)
            scores['pattern_strength'] = pattern_score
            analysis_details['pattern_strength'] = pattern_analysis
            
            # 2. 모멘텀 신호 분석 (25%)
            momentum_score, momentum_analysis = self._analyze_momentum_signals(stock)
            scores['momentum_signals'] = momentum_score
            analysis_details['momentum_signals'] = momentum_analysis
            
            # 3. 변동성 우위 분석 (20%)
            volatility_score, volatility_analysis = self._analyze_volatility_edge(stock)
            scores['volatility_edge'] = volatility_score
            analysis_details['volatility_edge'] = volatility_analysis
            
            # 4. 위험대비수익률 분석 (15%)
            risk_reward_score, risk_reward_analysis = self._analyze_risk_reward(stock)
            scores['risk_reward'] = risk_reward_score
            analysis_details['risk_reward'] = risk_reward_analysis
            
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
            logger.error(f"래쉬키 전략 분석 중 오류: {e}")
            return self._create_error_result()
    
    def _analyze_pattern_strength(self, stock) -> tuple:
        """차트 패턴 강도 분석"""
        try:
            score = 50
            analysis = {}
            
            # 가격 패턴 분석
            if hasattr(stock, 'prices') and len(stock.prices) >= 20:
                recent_prices = stock.prices[-20:]
                
                # 지지/저항선 강도
                support_resistance = self._calculate_support_resistance(recent_prices)
                score += min(support_resistance * 10, 20)
                analysis['support_resistance'] = support_resistance
                
                # 브레이크아웃 패턴
                breakout_strength = self._detect_breakout_pattern(recent_prices)
                score += min(breakout_strength * 15, 15)
                analysis['breakout_pattern'] = breakout_strength
                
                # 삼각형 패턴
                triangle_pattern = self._detect_triangle_pattern(recent_prices)
                score += min(triangle_pattern * 10, 10)
                analysis['triangle_pattern'] = triangle_pattern
            
            # 거래량 패턴
            if hasattr(stock, 'volumes') and len(stock.volumes) >= 10:
                volume_pattern = self._analyze_volume_pattern(stock.volumes)
                score += min(volume_pattern * 5, 5)
                analysis['volume_pattern'] = volume_pattern
            
            return min(max(score, 0), 100), analysis
            
        except Exception as e:
            logger.error(f"패턴 강도 분석 오류: {e}")
            return 50, {"error": str(e)}
    
    def _analyze_momentum_signals(self, stock) -> tuple:
        """모멘텀 신호 분석"""
        try:
            score = 50
            analysis = {}
            
            # RSI 분석
            if hasattr(stock, 'rsi'):
                rsi_signal = self._evaluate_rsi_signal(stock.rsi)
                score += rsi_signal * 20
                analysis['rsi_signal'] = rsi_signal
            
            # MACD 분석
            if hasattr(stock, 'macd'):
                macd_signal = self._evaluate_macd_signal(stock.macd)
                score += macd_signal * 15
                analysis['macd_signal'] = macd_signal
            
            # 스토캐스틱 분석
            if hasattr(stock, 'stochastic'):
                stoch_signal = self._evaluate_stochastic_signal(stock.stochastic)
                score += stoch_signal * 15
                analysis['stochastic_signal'] = stoch_signal
            
            return min(max(score, 0), 100), analysis
            
        except Exception as e:
            logger.error(f"모멘텀 신호 분석 오류: {e}")
            return 50, {"error": str(e)}
    
    def _analyze_volatility_edge(self, stock) -> tuple:
        """변동성 우위 분석"""
        try:
            score = 50
            analysis = {}
            
            # 변동성 지표
            if hasattr(stock, 'volatility'):
                volatility_score = self._evaluate_volatility(stock.volatility)
                score += volatility_score * 25
                analysis['volatility_score'] = volatility_score
            
            # 볼린저 밴드
            if hasattr(stock, 'bollinger_bands'):
                bb_signal = self._evaluate_bollinger_bands(stock.bollinger_bands)
                score += bb_signal * 15
                analysis['bollinger_signal'] = bb_signal
            
            # ATR (Average True Range)
            if hasattr(stock, 'atr'):
                atr_signal = self._evaluate_atr(stock.atr)
                score += atr_signal * 10
                analysis['atr_signal'] = atr_signal
            
            return min(max(score, 0), 100), analysis
            
        except Exception as e:
            logger.error(f"변동성 우위 분석 오류: {e}")
            return 50, {"error": str(e)}
    
    def _analyze_risk_reward(self, stock) -> tuple:
        """위험대비수익률 분석"""
        try:
            score = 50
            analysis = {}
            
            # 샤프 비율
            if hasattr(stock, 'sharpe_ratio'):
                sharpe_score = min(stock.sharpe_ratio * 20, 30)
                score += sharpe_score
                analysis['sharpe_ratio'] = stock.sharpe_ratio
            
            # 최대 손실 대비 수익
            if hasattr(stock, 'max_drawdown') and stock.max_drawdown != 0:
                drawdown_score = max(20 - abs(stock.max_drawdown), 0)
                score += drawdown_score
                analysis['max_drawdown'] = stock.max_drawdown
            
            return min(max(score, 0), 100), analysis
            
        except Exception as e:
            logger.error(f"위험대비수익률 분석 오류: {e}")
            return 50, {"error": str(e)}
    
    def _analyze_market_timing(self, stock) -> tuple:
        """시장 타이밍 분석"""
        try:
            score = 50
            analysis = {}
            
            # 시장 대비 상대 강도
            if hasattr(stock, 'relative_strength'):
                rs_score = stock.relative_strength * 30
                score += rs_score
                analysis['relative_strength'] = stock.relative_strength
            
            # 섹터 모멘텀
            if hasattr(stock, 'sector_momentum'):
                sector_score = stock.sector_momentum * 20
                score += sector_score
                analysis['sector_momentum'] = stock.sector_momentum
            
            return min(max(score, 0), 100), analysis
            
        except Exception as e:
            logger.error(f"시장 타이밍 분석 오류: {e}")
            return 50, {"error": str(e)}
    
    def _calculate_support_resistance(self, prices):
        """지지/저항선 강도 계산"""
        if len(prices) < 10:
            return 0
        
        # 간단한 지지/저항 계산
        highs = [p for p in prices if p > sum(prices)/len(prices)]
        lows = [p for p in prices if p < sum(prices)/len(prices)]
        
        resistance_strength = len(highs) / len(prices)
        support_strength = len(lows) / len(prices)
        
        return abs(resistance_strength - support_strength)
    
    def _detect_breakout_pattern(self, prices):
        """브레이크아웃 패턴 감지"""
        if len(prices) < 10:
            return 0
        
        recent_high = max(prices[-5:])
        previous_high = max(prices[-15:-5])
        
        if recent_high > previous_high * 1.05:
            return 1
        return 0
    
    def _detect_triangle_pattern(self, prices):
        """삼각형 패턴 감지"""
        if len(prices) < 15:
            return 0
        
        # 간단한 삼각형 패턴 감지 로직
        highs = prices[::2]  # 홀수 인덱스
        lows = prices[1::2]  # 짝수 인덱스
        
        if len(highs) >= 3 and len(lows) >= 3:
            high_trend = (highs[-1] - highs[0]) / len(highs)
            low_trend = (lows[-1] - lows[0]) / len(lows)
            
            if abs(high_trend) < 0.01 and abs(low_trend) < 0.01:
                return 1
        
        return 0
    
    def _analyze_volume_pattern(self, volumes):
        """거래량 패턴 분석"""
        if len(volumes) < 5:
            return 0
        
        avg_volume = sum(volumes) / len(volumes)
        recent_volume = volumes[-1]
        
        if recent_volume > avg_volume * 1.5:
            return 1
        return 0.5
    
    def _evaluate_rsi_signal(self, rsi):
        """RSI 신호 평가"""
        if rsi < 30:
            return 1  # 과매도
        elif rsi > 70:
            return -1  # 과매수
        return 0
    
    def _evaluate_macd_signal(self, macd):
        """MACD 신호 평가"""
        if hasattr(macd, 'signal') and hasattr(macd, 'histogram'):
            if macd.histogram > 0:
                return 1
            else:
                return -1
        return 0
    
    def _evaluate_stochastic_signal(self, stochastic):
        """스토캐스틱 신호 평가"""
        if hasattr(stochastic, 'k') and hasattr(stochastic, 'd'):
            if stochastic.k > stochastic.d and stochastic.k < 80:
                return 1
            elif stochastic.k < stochastic.d and stochastic.k > 20:
                return -1
        return 0
    
    def _evaluate_volatility(self, volatility):
        """변동성 평가"""
        if volatility > 0.3:
            return 1  # 높은 변동성 선호
        elif volatility < 0.1:
            return -1  # 낮은 변동성 회피
        return 0.5
    
    def _evaluate_bollinger_bands(self, bb):
        """볼린저 밴드 평가"""
        if hasattr(bb, 'position'):
            if bb.position < 0.2:
                return 1  # 하단 근처
            elif bb.position > 0.8:
                return -1  # 상단 근처
        return 0
    
    def _evaluate_atr(self, atr):
        """ATR 평가"""
        if atr > 0.02:  # 2% 이상 변동성
            return 1
        return 0.5
    
    def _make_investment_decision(self, total_score):
        """투자 판단 결정"""
        if total_score >= 80:
            return "강력매수 - 패턴과 모멘텀이 매우 유리"
        elif total_score >= 70:
            return "매수 - 단타 매매 기회 포착"
        elif total_score >= 60:
            return "관심 - 패턴 형성 중 관찰 필요"
        elif total_score >= 50:
            return "중립 - 명확한 신호 부재"
        else:
            return "회피 - 불리한 기술적 패턴"
    
    def _extract_key_points(self, scores, analysis_details):
        """핵심 포인트 추출"""
        points = []
        
        # 패턴 강도
        if scores['pattern_strength'] >= 70:
            points.append("강력한 차트 패턴 형성")
        elif scores['pattern_strength'] <= 40:
            points.append("불안정한 차트 패턴")
        
        # 모멘텀 신호
        if scores['momentum_signals'] >= 70:
            points.append("긍정적 모멘텀 신호")
        elif scores['momentum_signals'] <= 40:
            points.append("약한 모멘텀 신호")
        
        # 변동성 우위
        if scores['volatility_edge'] >= 70:
            points.append("변동성 활용 가능")
        
        # 위험대비수익률
        if scores['risk_reward'] >= 70:
            points.append("우수한 위험대비수익률")
        elif scores['risk_reward'] <= 40:
            points.append("높은 리스크 대비 낮은 수익")
        
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