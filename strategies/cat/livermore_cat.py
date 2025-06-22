#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 제시 리버모어 CAT 전략 - 투기의 왕 실전 적용
AI 종목분석기 연동 버전
"""

import logging
from typing import Dict, Any, Optional, List
from ..common import BaseStrategy, StrategyResult, get_stock_value, get_financial_metrics

logger = logging.getLogger(__name__)

class LivermoreCatStrategy(BaseStrategy):
    """제시 리버모어 CAT 전략 - 실전 투기 최적화"""
    
    def __init__(self):
        super().__init__()
        self.strategy_name = "제시 리버모어 CAT"
        self.description = "투기의 왕 실전 적용 - 추세 추종과 타이밍 중심"
        
        # 실전 적용 가중치
        self.weights = {
            'trend_following': 0.30,     # 추세 추종
            'timing_entry': 0.25,        # 진입 타이밍
            'volume_confirmation': 0.20, # 거래량 확인
            'momentum_strength': 0.15,   # 모멘텀 강도
            'risk_management': 0.10      # 위험 관리
        }
        
        # 리버모어 실전 기준
        self.criteria = {
            'strong_trend_threshold': 15,    # 강한 추세 기준 15%
            'volume_surge_ratio': 1.5,       # 거래량 급증 1.5배
            'momentum_acceleration': 10,     # 모멘텀 가속 10%
            'stop_loss_ratio': 0.08         # 손절 기준 8%
        }
    
    def analyze_stock(self, stock) -> StrategyResult:
        """리버모어 CAT 전략 분석"""
        try:
            metrics = get_financial_metrics(stock)
            scores = {}
            analysis_details = {}
            
            # 추세 추종 분석
            scores['trend_following'] = self._analyze_trend_following(metrics)
            
            # 진입 타이밍 분석
            scores['timing_entry'] = self._analyze_timing_entry(metrics)
            
            # 거래량 확인 분석
            scores['volume_confirmation'] = self._analyze_volume_confirmation(metrics)
            
            # 모멘텀 강도 분석
            scores['momentum_strength'] = self._analyze_momentum_strength(metrics)
            
            # 위험 관리 분석
            scores['risk_management'] = self._analyze_risk_management(metrics)
            
            # 가중 평균 계산
            total_score = sum(scores[key] * self.weights[key] for key in scores)
            total_score = min(max(total_score, 0), 100)
            
            # 리버모어 실전 신호 생성
            trading_signals = self._generate_trading_signals(metrics, scores)
            analysis_details['trading_signals'] = trading_signals
            
            # 투자 판단
            investment_decision = self._make_investment_decision(total_score)
            
            # 핵심 포인트
            key_points = self._extract_key_points(metrics, scores)
            
            return StrategyResult(
                total_score=total_score,
                scores=scores,
                strategy_name=self.strategy_name,
                investment_decision=investment_decision,
                key_points=key_points,
                analysis_details=analysis_details
            )
            
        except Exception as e:
            logger.error(f"리버모어 CAT 전략 분석 오류: {e}")
            return self._create_error_result()
    
    def _analyze_trend_following(self, metrics: Dict) -> float:
        """추세 추종 분석 - 리버모어의 핵심"""
        price_momentum_3m = metrics.get('price_momentum_3m', 0)
        price_momentum_1y = metrics.get('price_momentum_1y', 0)
        
        score = 0
        
        # 단기 추세 (3개월)
        if price_momentum_3m >= 30:
            score += 40  # 매우 강한 상승 추세
        elif price_momentum_3m >= 20:
            score += 35
        elif price_momentum_3m >= 15:
            score += 30
        elif price_momentum_3m >= 10:
            score += 25
        elif price_momentum_3m >= 5:
            score += 15
        elif price_momentum_3m < -10:
            score -= 20  # 하락 추세는 감점
        
        # 장기 추세 확인 (1년)
        if price_momentum_1y >= 20:
            score += 30  # 장기 상승 추세 확인
        elif price_momentum_1y >= 10:
            score += 20
        elif price_momentum_1y < -20:
            score -= 30  # 장기 하락 추세
        
        # 추세 일관성 (단기와 장기 방향 일치)
        if price_momentum_3m > 0 and price_momentum_1y > 0:
            score += 20  # 추세 일관성
        elif price_momentum_3m < 0 and price_momentum_1y < 0:
            score -= 20  # 하락 일관성
        
        return min(max(score, 0), 100)
    
    def _analyze_timing_entry(self, metrics: Dict) -> float:
        """진입 타이밍 분석"""
        price_momentum_3m = metrics.get('price_momentum_3m', 0)
        volatility = metrics.get('volatility', 0)
        
        score = 50
        
        # 모멘텀 기반 타이밍
        if price_momentum_3m >= 20:
            score += 30  # 강한 상승 모멘텀 = 좋은 진입 타이밍
        elif price_momentum_3m >= 15:
            score += 25
        elif price_momentum_3m >= 10:
            score += 20
        elif price_momentum_3m < 0:
            score -= 25  # 하락 모멘텀은 나쁜 타이밍
        
        # 변동성 고려 (적당한 변동성 선호)
        if volatility:
            if 15 <= volatility <= 35:
                score += 20  # 적정 변동성
            elif 10 <= volatility <= 45:
                score += 15
            elif volatility > 50:
                score -= 15  # 과도한 변동성
        
        # 섹터 모멘텀 (성장 섹터 선호)
        sector = metrics.get('sector', '')
        hot_sectors = ['Technology', 'Healthcare', 'IT', '바이오', '반도체', '게임', '엔터테인먼트']
        if any(hs in sector for hs in hot_sectors):
            score += 15
        
        return min(max(score, 0), 100)
    
    def _analyze_volume_confirmation(self, metrics: Dict) -> float:
        """거래량 확인 분석"""
        price_momentum_3m = metrics.get('price_momentum_3m', 0)
        market_cap = metrics.get('market_cap', 0)
        
        score = 50
        
        # 가격 상승과 거래량 증가 연관성 (가격 모멘텀으로 추정)
        if price_momentum_3m >= 25:
            score += 30  # 강한 가격 상승 = 거래량 증가 추정
        elif price_momentum_3m >= 15:
            score += 25
        elif price_momentum_3m >= 10:
            score += 20
        elif price_momentum_3m < 0:
            score -= 20
        
        # 유동성 (시가총액 기준)
        market_cap_billion = market_cap / 100000000 if market_cap else 0
        if market_cap_billion >= 1000:
            score += 25  # 대형주 = 충분한 유동성
        elif market_cap_billion >= 500:
            score += 20
        elif market_cap_billion >= 100:
            score += 15
        elif market_cap_billion < 50:
            score -= 15  # 소형주 = 유동성 부족
        
        return min(max(score, 0), 100)
    
    def _analyze_momentum_strength(self, metrics: Dict) -> float:
        """모멘텀 강도 분석"""
        price_momentum_3m = metrics.get('price_momentum_3m', 0)
        profit_growth = metrics.get('profit_growth', 0)
        revenue_growth = metrics.get('revenue_growth', 0)
        
        score = 0
        
        # 가격 모멘텀 강도
        if price_momentum_3m >= 40:
            score += 40  # 매우 강한 모멘텀
        elif price_momentum_3m >= 30:
            score += 35
        elif price_momentum_3m >= 20:
            score += 30
        elif price_momentum_3m >= 15:
            score += 25
        elif price_momentum_3m >= 10:
            score += 20
        
        # 실적 모멘텀 (가격 모멘텀 뒷받침)
        if profit_growth >= 30:
            score += 30
        elif profit_growth >= 20:
            score += 25
        elif profit_growth >= 10:
            score += 20
        elif profit_growth < 0:
            score -= 20
        
        # 매출 모멘텀
        if revenue_growth >= 20:
            score += 20
        elif revenue_growth >= 10:
            score += 15
        elif revenue_growth >= 5:
            score += 10
        
        return min(score, 100)
    
    def _analyze_risk_management(self, metrics: Dict) -> float:
        """위험 관리 분석"""
        volatility = metrics.get('volatility', 0)
        debt_ratio = metrics.get('debt_ratio', 0)
        market_cap = metrics.get('market_cap', 0)
        
        score = 50
        
        # 변동성 위험
        if volatility:
            if volatility <= 20:
                score += 25  # 낮은 위험
            elif volatility <= 30:
                score += 20
            elif volatility <= 40:
                score += 15
            elif volatility > 60:
                score -= 25  # 높은 위험
        
        # 재무 위험
        if debt_ratio <= 30:
            score += 20  # 낮은 부채
        elif debt_ratio <= 50:
            score += 15
        elif debt_ratio > 100:
            score -= 20  # 높은 부채 위험
        
        # 유동성 위험
        market_cap_billion = market_cap / 100000000 if market_cap else 0
        if market_cap_billion >= 1000:
            score += 15  # 높은 유동성
        elif market_cap_billion >= 500:
            score += 10
        elif market_cap_billion < 100:
            score -= 10  # 낮은 유동성
        
        return min(max(score, 0), 100)
    
    def _generate_trading_signals(self, metrics: Dict, scores: Dict) -> List[str]:
        """리버모어 실전 매매 신호 생성"""
        signals = []
        
        price_momentum_3m = metrics.get('price_momentum_3m', 0)
        
        # 매수 신호
        if price_momentum_3m >= 20 and scores['trend_following'] >= 70:
            signals.append("🟢 강력 매수 신호 - 추세 돌파 확인")
        elif price_momentum_3m >= 15 and scores['volume_confirmation'] >= 70:
            signals.append("🔵 매수 신호 - 거래량 뒷받침")
        
        # 손절 신호
        if price_momentum_3m < -8:
            signals.append("🔴 손절 신호 - 8% 하락 기준")
        elif scores['trend_following'] < 40:
            signals.append("⚠️ 추세 약화 - 포지션 축소 고려")
        
        # 관망 신호
        if 40 <= scores['trend_following'] <= 60:
            signals.append("⚪ 관망 - 명확한 신호 대기")
        
        # 실전 팁
        if scores['momentum_strength'] >= 80:
            signals.append("💡 모멘텀 강함 - 피라미딩 매수 고려")
        
        return signals
    
    def _make_investment_decision(self, total_score):
        """투자 판단"""
        if total_score >= 85:
            return "🟢 강력 매수 - 완벽한 투기 조건"
        elif total_score >= 75:
            return "🔵 매수 - 좋은 추세 신호"
        elif total_score >= 65:
            return "🟡 관심 - 추세 확인 필요"
        elif total_score >= 55:
            return "⚪ 중립 - 신호 대기"
        else:
            return "🔴 회피 - 투기 조건 부적합"
    
    def _extract_key_points(self, metrics: Dict, scores: Dict) -> List[str]:
        """핵심 포인트 추출"""
        points = []
        
        price_momentum_3m = metrics.get('price_momentum_3m', 0)
        
        if price_momentum_3m >= 20:
            points.append(f"✅ 강한 가격 모멘텀: {price_momentum_3m:.1f}%")
        
        if scores.get('trend_following', 0) >= 70:
            points.append("✅ 명확한 상승 추세")
        
        if scores.get('volume_confirmation', 0) >= 70:
            points.append("✅ 거래량 뒷받침")
        
        if scores.get('momentum_strength', 0) >= 70:
            points.append("✅ 모멘텀 강도 우수")
        
        if scores.get('risk_management', 0) >= 70:
            points.append("✅ 위험 관리 양호")
        
        # 위험 신호
        if price_momentum_3m < -5:
            points.append("⚠️ 하락 모멘텀 주의")
        
        if scores.get('risk_management', 0) < 50:
            points.append("⚠️ 높은 위험도")
        
        return points[:5]
    
    def _create_error_result(self):
        """오류 결과 생성"""
        return StrategyResult(
            total_score=0.0,
            scores={key: 0.0 for key in self.weights.keys()},
            strategy_name=self.strategy_name,
            investment_decision="분석 불가",
            key_points=["분석 중 오류 발생"],
            analysis_details={"error": "분석 실패"}
        )
    
    def get_ai_analyzer_integration(self) -> Dict[str, str]:
        """AI 종목분석기 연동 가이드"""
        return {
            "매매_신호": "20일 고점 돌파 매수, 8% 하락 시 손절",
            "활용_도구": "트레이딩뷰, 키움 HTS, AI 차트 분석",
            "필터링": "3개월 수익률 ≥ 15%, 거래량 급증 종목",
            "포지션_관리": "초기 5% 투자, 수익 시 피라미딩",
            "위험_관리": "8% 손절선 엄수, 추세 약화 시 즉시 매도"
        } 