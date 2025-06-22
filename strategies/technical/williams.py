#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📊 래리 윌리엄스 (Larry Williams) 투자 전략
단기 트레이딩과 기술적 분석의 달인
"""

import logging
from typing import Dict, Any, Optional, List
from ..common import BaseStrategy, StrategyResult

logger = logging.getLogger(__name__)

class WilliamsStrategy(BaseStrategy):
    """래리 윌리엄스의 단기 트레이딩 전략"""
    
    def __init__(self):
        super().__init__()
        self.strategy_name = "래리 윌리엄스 (Larry Williams)"
        self.description = "기술적 분석과 단기 모멘텀 기반 트레이딩"
        
        # 래리 윌리엄스 전략 가중치
        self.weights = {
            'momentum_analysis': 0.30,   # 모멘텀 분석
            'volatility_trading': 0.25,  # 변동성 트레이딩
            'market_timing': 0.20,       # 시장 타이밍
            'technical_signals': 0.15,   # 기술적 신호
            'short_term_edge': 0.10      # 단기 우위
        }
    
    def analyze_stock(self, stock) -> StrategyResult:
        """래리 윌리엄스 트레이딩 분석"""
        try:
            scores = {}
            analysis_details = {}
            
            # 모멘텀 분석
            momentum_score = self._analyze_momentum(stock)
            scores['momentum_analysis'] = momentum_score
            analysis_details['momentum_analysis'] = momentum_score
            
            # 변동성 트레이딩 분석
            volatility_score = self._analyze_volatility_trading(stock)
            scores['volatility_trading'] = volatility_score
            analysis_details['volatility_trading'] = volatility_score
            
            # 시장 타이밍 분석
            timing_score = self._analyze_market_timing(stock)
            scores['market_timing'] = timing_score
            analysis_details['market_timing'] = timing_score
            
            # 기술적 신호 분석
            technical_score = self._analyze_technical_signals(stock)
            scores['technical_signals'] = technical_score
            analysis_details['technical_signals'] = technical_score
            
            # 단기 우위 분석
            edge_score = self._analyze_short_term_edge(stock)
            scores['short_term_edge'] = edge_score
            analysis_details['short_term_edge'] = edge_score
            
            # 가중 평균 계산
            total_score = sum(
                scores[key] * self.weights[key] 
                for key in scores
            )
            total_score = min(max(total_score, 0), 100)
            
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
            logger.error(f"윌리엄스 전략 분석 중 오류: {e}")
            return self._create_error_result()
    
    def _analyze_momentum(self, stock) -> float:
        """모멘텀 분석 - 윌리엄스의 핵심"""
        score = 0.0
        
        # 단기 모멘텀 (3개월)
        price_momentum = getattr(stock, 'price_momentum_3m', None) or getattr(stock, 'price_momentum', 0)
        if price_momentum:
            if price_momentum >= 30:
                score += 35  # 매우 강한 모멘텀
            elif price_momentum >= 20:
                score += 30
            elif price_momentum >= 15:
                score += 25
            elif price_momentum >= 10:
                score += 20
            elif price_momentum >= 5:
                score += 15
            elif price_momentum >= 0:
                score += 5
            else:
                # 하락 모멘텀도 트레이딩 기회
                if price_momentum <= -20:
                    score += 15  # 강한 하락 반전 기회
                elif price_momentum <= -10:
                    score += 10
                else:
                    score -= 10
        
        # 이익 모멘텀
        profit_growth = getattr(stock, 'profit_growth', None) or getattr(stock, 'earnings_growth', 0)
        if profit_growth:
            if profit_growth >= 25:
                score += 25
            elif profit_growth >= 15:
                score += 20
            elif profit_growth >= 10:
                score += 15
            elif profit_growth >= 0:
                score += 10
        
        # 매출 모멘텀
        revenue_growth = getattr(stock, 'revenue_growth', 0)
        if revenue_growth:
            if revenue_growth >= 20:
                score += 20
            elif revenue_growth >= 10:
                score += 15
            elif revenue_growth >= 5:
                score += 10
        
        # 모멘텀 가속도 (변동성 대비 수익률)
        volatility = getattr(stock, 'volatility', 0)
        if volatility and price_momentum:
            momentum_ratio = abs(price_momentum) / max(volatility, 1)
            if momentum_ratio >= 1.2:
                score += 20  # 강한 방향성
            elif momentum_ratio >= 0.8:
                score += 15
            elif momentum_ratio >= 0.5:
                score += 10
        
        return min(score, 100)
    
    def _analyze_volatility_trading(self, stock) -> float:
        """변동성 트레이딩 분석"""
        score = 50.0
        
        # 적정 변동성 (트레이딩 기회)
        volatility = getattr(stock, 'volatility', 0)
        if volatility:
            # 윌리엄스는 적당한 변동성을 선호
            if 20 <= volatility <= 40:
                score += 30  # 최적 변동성
            elif 15 <= volatility <= 50:
                score += 25
            elif 10 <= volatility <= 60:
                score += 20
            elif volatility < 10:
                score -= 20  # 너무 낮은 변동성
            elif volatility > 60:
                score -= 15  # 너무 높은 변동성
        
        # 변동성 대비 수익률
        price_momentum = getattr(stock, 'price_momentum_3m', None) or getattr(stock, 'price_momentum', 0)
        if volatility and price_momentum:
            volatility_efficiency = abs(price_momentum) / max(volatility, 1)
            if volatility_efficiency >= 1.0:
                score += 25  # 효율적 변동성
            elif volatility_efficiency >= 0.7:
                score += 20
            elif volatility_efficiency >= 0.5:
                score += 15
        
        # 유동성 (변동성 활용 가능성)
        market_cap = getattr(stock, 'market_cap', 0)
        if market_cap:
            market_cap_billion = market_cap / 100000000
            if market_cap_billion >= 100:
                score += 15  # 충분한 유동성
            elif market_cap_billion >= 50:
                score += 10
            elif market_cap_billion >= 10:
                score += 5
        
        return min(max(score, 0), 100)
    
    def _analyze_market_timing(self, stock) -> float:
        """시장 타이밍 분석"""
        score = 50.0
        
        # 단기 추세 변화
        price_momentum = getattr(stock, 'price_momentum_3m', None) or getattr(stock, 'price_momentum', 0)
        if price_momentum:
            # 강한 방향성 (상승 또는 하락)
            momentum_abs = abs(price_momentum)
            if momentum_abs >= 20:
                score += 25  # 명확한 방향성
            elif momentum_abs >= 15:
                score += 20
            elif momentum_abs >= 10:
                score += 15
            elif momentum_abs >= 5:
                score += 10
            else:
                score -= 10  # 모호한 신호
        
        # 밸류에이션 기반 타이밍
        pe_ratio = getattr(stock, 'pe_ratio', None) or getattr(stock, 'per', 0)
        if pe_ratio:
            if pe_ratio <= 10:
                score += 20  # 저평가 타이밍
            elif pe_ratio <= 15:
                score += 15
            elif pe_ratio <= 25:
                score += 10
            elif pe_ratio >= 40:
                score -= 15  # 고평가 위험
        
        # 실적 발표 시즌 고려
        profit_growth = getattr(stock, 'profit_growth', None) or getattr(stock, 'earnings_growth', 0)
        if profit_growth:
            if profit_growth >= 15:
                score += 15  # 좋은 실적 기대
            elif profit_growth >= 5:
                score += 10
            elif profit_growth < -10:
                score -= 15  # 실적 악화 위험
        
        return min(max(score, 0), 100)
    
    def _analyze_technical_signals(self, stock) -> float:
        """기술적 신호 분석"""
        score = 50.0
        
        # RSI 기반 과매수/과매도
        rsi = getattr(stock, 'rsi', 50)
        if rsi:
            if 30 <= rsi <= 70:
                score += 20  # 적정 범위
            elif 20 <= rsi < 30:
                score += 15  # 과매도 (매수 기회)
            elif 70 < rsi <= 80:
                score += 10  # 과매수 (매도 고려)
            elif rsi < 20:
                score += 25  # 강한 과매도
            elif rsi > 80:
                score -= 10  # 강한 과매수
        
        # 거래량 분석
        volume_ratio = getattr(stock, 'volume_ratio', 1.0)
        if volume_ratio:
            if volume_ratio >= 2.0:
                score += 20  # 높은 관심
            elif volume_ratio >= 1.5:
                score += 15
            elif volume_ratio >= 1.2:
                score += 10
            elif volume_ratio < 0.5:
                score -= 10  # 낮은 관심
        
        # 이동평균 돌파
        ma_signal = getattr(stock, 'ma_signal', 0)  # 1: 골든크로스, -1: 데드크로스
        if ma_signal == 1:
            score += 20
        elif ma_signal == -1:
            score -= 15
        
        return min(max(score, 0), 100)
    
    def _analyze_short_term_edge(self, stock) -> float:
        """단기 우위 분석"""
        score = 50.0
        
        # 시장 대비 성과
        market_beta = getattr(stock, 'beta', 1.0)
        if market_beta:
            if 0.8 <= market_beta <= 1.2:
                score += 15  # 적정 베타
            elif 0.5 <= market_beta < 0.8:
                score += 10  # 낮은 베타 (안정성)
            elif market_beta > 1.5:
                score += 20  # 높은 베타 (변동성 활용)
        
        # 단기 수익률 패턴
        price_momentum = getattr(stock, 'price_momentum_3m', None) or getattr(stock, 'price_momentum', 0)
        volatility = getattr(stock, 'volatility', 0)
        if price_momentum and volatility:
            # 샤프 비율 추정
            sharpe_estimate = price_momentum / max(volatility, 1)
            if sharpe_estimate >= 1.0:
                score += 25
            elif sharpe_estimate >= 0.5:
                score += 20
            elif sharpe_estimate >= 0.2:
                score += 15
            elif sharpe_estimate < -0.5:
                score -= 15
        
        # 섹터 강도
        sector_momentum = getattr(stock, 'sector_momentum', 0)
        if sector_momentum:
            if sector_momentum >= 10:
                score += 15  # 강한 섹터
            elif sector_momentum >= 5:
                score += 10
            elif sector_momentum < -10:
                score -= 10  # 약한 섹터
        
        return min(max(score, 0), 100)
    
    def _make_investment_decision(self, total_score):
        """투자 판단"""
        if total_score >= 80:
            return "강력 매수"
        elif total_score >= 70:
            return "매수"
        elif total_score >= 60:
            return "보유"
        elif total_score >= 50:
            return "관망"
        else:
            return "매도"
    
    def _extract_key_points(self, scores, analysis_details):
        """핵심 포인트 추출"""
        key_points = []
        
        if scores.get('momentum_analysis', 0) >= 80:
            key_points.append("강한 모멘텀 확인")
        
        if scores.get('volatility_trading', 0) >= 80:
            key_points.append("최적 변동성 구간")
        
        if scores.get('market_timing', 0) >= 80:
            key_points.append("좋은 진입 타이밍")
        
        if scores.get('technical_signals', 0) >= 80:
            key_points.append("긍정적 기술적 신호")
        
        if scores.get('short_term_edge', 0) >= 80:
            key_points.append("단기 우위 보유")
        
        return key_points
    
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