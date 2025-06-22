#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🐢 리처드 데니스 CAT 전략 - 터틀 트레이딩 실전 적용
AI 종목분석기 연동 버전
"""

import logging
from typing import Dict, Any, Optional, List
from ..common import BaseStrategy, StrategyResult, get_stock_value, get_financial_metrics

logger = logging.getLogger(__name__)

class DennisCatStrategy(BaseStrategy):
    """리처드 데니스 CAT 전략 - 터틀 트레이딩 실전 최적화"""
    
    def __init__(self):
        super().__init__()
        self.strategy_name = "리처드 데니스 CAT"
        self.description = "터틀 트레이딩 실전 적용 - 20일 고점 돌파 매수, 10일 저점 이탈 매도"
        
        # 실전 적용 가중치
        self.weights = {
            'breakout_signal': 0.30,      # 돌파 신호
            'trend_strength': 0.25,       # 추세 강도
            'volatility_filter': 0.20,    # 변동성 필터
            'risk_management': 0.15,      # 위험 관리
            'position_sizing': 0.10       # 포지션 사이징
        }
        
        # 터틀 트레이딩 기준
        self.criteria = {
            'breakout_period': 20,        # 돌파 기간 20일
            'exit_period': 10,            # 청산 기간 10일
            'atr_multiplier': 2.0,        # ATR 배수
            'max_risk_per_trade': 0.02,   # 거래당 최대 위험 2%
            'min_liquidity': 1000         # 최소 시가총액 1000억
        }
    
    def analyze_stock(self, stock) -> StrategyResult:
        """데니스 CAT 전략 분석"""
        try:
            metrics = get_financial_metrics(stock)
            scores = {}
            analysis_details = {}
            
            # 돌파 신호 분석
            scores['breakout_signal'] = self._analyze_breakout_signal(metrics)
            
            # 추세 강도 분석
            scores['trend_strength'] = self._analyze_trend_strength(metrics)
            
            # 변동성 필터 분석
            scores['volatility_filter'] = self._analyze_volatility_filter(metrics)
            
            # 위험 관리 분석
            scores['risk_management'] = self._analyze_risk_management(metrics)
            
            # 포지션 사이징 분석
            scores['position_sizing'] = self._analyze_position_sizing(metrics)
            
            # 가중 평균 계산
            total_score = sum(scores[key] * self.weights[key] for key in scores)
            total_score = min(max(total_score, 0), 100)
            
            # 터틀 실전 신호 생성
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
            logger.error(f"데니스 CAT 전략 분석 오류: {e}")
            return self._create_error_result()
    
    def _analyze_breakout_signal(self, metrics: Dict) -> float:
        """돌파 신호 분석"""
        price_momentum_3m = metrics.get('price_momentum_3m', 0)
        price_momentum_1m = metrics.get('price_momentum_1m', 0)
        
        score = 0
        
        # 3개월 모멘텀 (20일 돌파 추정)
        if price_momentum_3m >= 25:
            score += 40  # 강한 돌파 신호
        elif price_momentum_3m >= 20:
            score += 35
        elif price_momentum_3m >= 15:
            score += 30
        elif price_momentum_3m >= 10:
            score += 25
        elif price_momentum_3m >= 5:
            score += 15
        elif price_momentum_3m < 0:
            score -= 20  # 하락 추세
        
        # 1개월 모멘텀 (최근 돌파 확인)
        if price_momentum_1m >= 15:
            score += 30  # 최근 강한 돌파
        elif price_momentum_1m >= 10:
            score += 25
        elif price_momentum_1m >= 5:
            score += 20
        elif price_momentum_1m < -5:
            score -= 25  # 최근 하락
        
        # 모멘텀 가속도 (1개월이 3개월보다 강하면 가속)
        if price_momentum_1m > price_momentum_3m and price_momentum_1m > 10:
            score += 20  # 모멘텀 가속
        elif price_momentum_1m < price_momentum_3m - 10:
            score -= 15  # 모멘텀 둔화
        
        return min(max(score, 0), 100)
    
    def _analyze_trend_strength(self, metrics: Dict) -> float:
        """추세 강도 분석"""
        price_momentum_3m = metrics.get('price_momentum_3m', 0)
        price_momentum_1y = metrics.get('price_momentum_1y', 0)
        volatility = metrics.get('volatility', 0)
        
        score = 0
        
        # 단기 추세 강도
        if price_momentum_3m >= 30:
            score += 35
        elif price_momentum_3m >= 20:
            score += 30
        elif price_momentum_3m >= 15:
            score += 25
        elif price_momentum_3m >= 10:
            score += 20
        elif price_momentum_3m < 0:
            score -= 20
        
        # 장기 추세 일관성
        if price_momentum_1y >= 20:
            score += 25  # 장기 상승 추세
        elif price_momentum_1y >= 10:
            score += 20
        elif price_momentum_1y < -10:
            score -= 25  # 장기 하락 추세
        
        # 추세 지속성 (변동성 고려)
        if volatility and 20 <= volatility <= 40:
            score += 20  # 적정 변동성으로 건전한 추세
        elif volatility and volatility > 60:
            score -= 15  # 과도한 변동성
        
        # 추세 방향 일관성
        if price_momentum_3m > 0 and price_momentum_1y > 0:
            score += 20  # 단기/장기 추세 일치
        elif price_momentum_3m < 0 and price_momentum_1y < 0:
            score -= 20  # 하락 추세 일치
        
        return min(max(score, 0), 100)
    
    def _analyze_volatility_filter(self, metrics: Dict) -> float:
        """변동성 필터 분석"""
        volatility = metrics.get('volatility', 0)
        price_momentum_3m = metrics.get('price_momentum_3m', 0)
        
        score = 50
        
        # 변동성 수준 평가
        if volatility:
            if 15 <= volatility <= 35:
                score += 30  # 이상적인 변동성
            elif 10 <= volatility <= 45:
                score += 25
            elif 5 <= volatility <= 50:
                score += 20
            elif volatility > 60:
                score -= 25  # 과도한 변동성
            elif volatility < 5:
                score -= 15  # 너무 낮은 변동성
        
        # 변동성과 추세의 조화
        if volatility and price_momentum_3m > 0:
            if 20 <= volatility <= 40:
                score += 20  # 상승 추세의 건전한 변동성
            elif volatility > 50:
                score -= 10  # 상승 추세의 과도한 변동성
        
        return min(max(score, 0), 100)
    
    def _analyze_risk_management(self, metrics: Dict) -> float:
        """위험 관리 분석"""
        volatility = metrics.get('volatility', 0)
        market_cap = metrics.get('market_cap', 0)
        debt_ratio = metrics.get('debt_ratio', 0)
        
        score = 50
        
        # 유동성 위험 (시가총액)
        market_cap_billion = market_cap / 100000000 if market_cap else 0
        if market_cap_billion >= 5000:
            score += 25  # 높은 유동성
        elif market_cap_billion >= 1000:
            score += 20
        elif market_cap_billion >= 500:
            score += 15
        elif market_cap_billion < 100:
            score -= 20  # 낮은 유동성
        
        # 변동성 위험
        if volatility:
            if volatility <= 30:
                score += 20  # 관리 가능한 위험
            elif volatility <= 40:
                score += 15
            elif volatility <= 50:
                score += 10
            elif volatility > 70:
                score -= 25  # 높은 위험
        
        # 재무 위험
        if debt_ratio <= 50:
            score += 15  # 낮은 재무 위험
        elif debt_ratio <= 100:
            score += 10
        elif debt_ratio > 150:
            score -= 15  # 높은 재무 위험
        
        return min(max(score, 0), 100)
    
    def _analyze_position_sizing(self, metrics: Dict) -> float:
        """포지션 사이징 분석"""
        volatility = metrics.get('volatility', 0)
        market_cap = metrics.get('market_cap', 0)
        
        score = 50
        
        # 변동성 기반 포지션 사이징
        if volatility:
            if volatility <= 20:
                score += 30  # 낮은 변동성 = 큰 포지션 가능
            elif volatility <= 30:
                score += 25
            elif volatility <= 40:
                score += 20
            elif volatility <= 50:
                score += 15
            elif volatility > 60:
                score += 5   # 높은 변동성 = 작은 포지션
        
        # 유동성 기반 포지션 사이징
        market_cap_billion = market_cap / 100000000 if market_cap else 0
        if market_cap_billion >= 10000:
            score += 20  # 대형주 = 큰 포지션 가능
        elif market_cap_billion >= 5000:
            score += 15
        elif market_cap_billion >= 1000:
            score += 10
        elif market_cap_billion < 500:
            score -= 10  # 소형주 = 작은 포지션
        
        return min(max(score, 0), 100)
    
    def _generate_trading_signals(self, metrics: Dict, scores: Dict) -> List[str]:
        """터틀 실전 매매 신호 생성"""
        signals = []
        
        price_momentum_3m = metrics.get('price_momentum_3m', 0)
        volatility = metrics.get('volatility', 0)
        
        # 매수 신호
        if scores['breakout_signal'] >= 70 and scores['trend_strength'] >= 70:
            signals.append("🟢 터틀 매수 신호 - 20일 고점 돌파 확인")
        elif price_momentum_3m >= 15 and scores['volatility_filter'] >= 70:
            signals.append("🔵 매수 고려 - 돌파 후 추세 확인")
        
        # 매도 신호
        if price_momentum_3m < -10:
            signals.append("🔴 터틀 매도 신호 - 10일 저점 이탈")
        elif scores['trend_strength'] < 40:
            signals.append("⚠️ 추세 약화 - 포지션 축소 고려")
        
        # 포지션 관리 신호
        if volatility and volatility > 50:
            signals.append("📉 높은 변동성 - 포지션 크기 축소")
        elif volatility and volatility < 20:
            signals.append("📈 낮은 변동성 - 포지션 크기 확대 가능")
        
        # 실전 팁
        if scores['risk_management'] >= 80:
            signals.append("💡 우수한 위험 관리 - 표준 포지션 사이징 적용")
        
        return signals
    
    def _make_investment_decision(self, total_score):
        """투자 판단"""
        if total_score >= 85:
            return "🟢 강력 매수 - 완벽한 터틀 신호"
        elif total_score >= 75:
            return "🔵 매수 - 좋은 돌파 신호"
        elif total_score >= 65:
            return "🟡 관심 - 돌파 확인 필요"
        elif total_score >= 55:
            return "⚪ 중립 - 신호 대기"
        else:
            return "🔴 회피 - 터틀 조건 부적합"
    
    def _extract_key_points(self, metrics: Dict, scores: Dict) -> List[str]:
        """핵심 포인트 추출"""
        points = []
        
        price_momentum_3m = metrics.get('price_momentum_3m', 0)
        volatility = metrics.get('volatility', 0)
        
        if scores.get('breakout_signal', 0) >= 70:
            points.append("✅ 강한 돌파 신호")
        
        if price_momentum_3m >= 20:
            points.append(f"✅ 우수한 모멘텀: {price_momentum_3m:.1f}%")
        
        if scores.get('trend_strength', 0) >= 70:
            points.append("✅ 강한 추세 지속")
        
        if volatility and 15 <= volatility <= 35:
            points.append(f"✅ 적정 변동성: {volatility:.1f}%")
        
        if scores.get('risk_management', 0) >= 70:
            points.append("✅ 우수한 위험 관리")
        
        # 위험 신호
        if volatility and volatility > 50:
            points.append(f"⚠️ 높은 변동성: {volatility:.1f}%")
        
        if price_momentum_3m < -5:
            points.append("⚠️ 하락 모멘텀 주의")
        
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
            "매매_신호": "20일 고점 돌파 매수, 10일 저점 이탈 매도",
            "활용_도구": "트레이딩뷰 + 자동화 봇 or 수동 가능",
            "필터링": "시가총액 > 1000억, 변동성 15-35%",
            "포지션_관리": "변동성 기반 사이징, 거래당 위험 2%",
            "자동화": "HTS API 연동 자동매매 시스템 구축 가능"
        }
    
    def get_turtle_rules(self) -> Dict[str, Any]:
        """터틀 트레이딩 규칙"""
        return {
            "진입_규칙": {
                "신호": "20일 최고가 돌파",
                "확인": "거래량 증가 동반",
                "필터": "55일 최고가 돌파 우선"
            },
            "청산_규칙": {
                "손절": "10일 최저가 이탈",
                "익절": "추세 지속 시 보유",
                "ATR": "2ATR 손절선 설정"
            },
            "포지션_관리": {
                "초기_사이징": "계좌의 1-2%",
                "피라미딩": "0.5ATR 간격으로 추가 매수",
                "최대_포지션": "계좌의 10%"
            }
        } 