#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📊 제시 리버모어 (Jesse Livermore) 투자 전략
시장 타이밍과 모멘텀 기반 투기 전략
"""

import logging
from typing import Dict, Any, Optional, List
from ..common import BaseStrategy, StrategyResult
from data.processed.data_cleaner import CleanedStockData

logger = logging.getLogger(__name__)

class LivermoreStrategy(BaseStrategy):
    """제시 리버모어의 투기 전략"""
    
    def __init__(self):
        super().__init__()
        self.strategy_name = "Jesse Livermore Speculation"
        self.description = "시장 타이밍과 가격 움직임 기반 투기 전략"
        
        # 리버모어 전략 가중치
        self.weights = {
            'price_momentum': 0.30,      # 가격 모멘텀
            'volume_pattern': 0.20,      # 거래량 패턴
            'market_leadership': 0.20,   # 시장 선도성
            'timing_signals': 0.15,      # 타이밍 신호
            'speculation_safety': 0.15   # 투기 안전성
        }
    
    def analyze_stock(self, stock: CleanedStockData) -> float:
        """리버모어 투기 분석"""
        try:
            scores = {}
            
            # 가격 모멘텀 분석
            scores['price_momentum'] = self._analyze_price_momentum(stock)
            
            # 거래량 패턴 분석
            scores['volume_pattern'] = self._analyze_volume_pattern(stock)
            
            # 시장 선도성 분석
            scores['market_leadership'] = self._analyze_market_leadership(stock)
            
            # 타이밍 신호 분석
            scores['timing_signals'] = self._analyze_timing_signals(stock)
            
            # 투기 안전성 분석
            scores['speculation_safety'] = self._analyze_speculation_safety(stock)
            
            # 가중 평균 계산
            total_score = sum(
                scores[key] * self.weights[key] 
                for key in scores
            )
            
            logger.debug(f"Livermore 투기 분석 ({stock.symbol}): {total_score:.1f}")
            return min(max(total_score, 0), 100)
            
        except Exception as e:
            logger.error(f"Livermore 전략 분석 오류 ({stock.symbol}): {e}")
            return 0.0
    
    def _analyze_price_momentum(self, stock: CleanedStockData) -> float:
        """가격 모멘텀 분석 - 리버모어의 핵심"""
        score = 0.0
        
        # 강한 상승 모멘텀 (리버모어는 추세를 따름)
        if stock.price_momentum_3m:
            if stock.price_momentum_3m >= 30:
                score += 40  # 매우 강한 상승
            elif stock.price_momentum_3m >= 20:
                score += 35
            elif stock.price_momentum_3m >= 15:
                score += 30
            elif stock.price_momentum_3m >= 10:
                score += 25
            elif stock.price_momentum_3m >= 5:
                score += 20
            elif stock.price_momentum_3m >= 0:
                score += 10
            else:
                score -= 20  # 하락 추세는 매우 부정적
        
        # 가격대 분석 (고가 근처 선호)
        if stock.price:
            # 가격이 높을수록 관심 (상승 추세의 증거)
            if stock.price >= 100000:  # 10만원 이상
                score += 20
            elif stock.price >= 50000:  # 5만원 이상
                score += 15
            elif stock.price >= 20000:  # 2만원 이상
                score += 10
        
        # 시가총액 기준 (대형주 선호 - 유동성)
        if stock.market_cap:
            market_cap_billion = stock.market_cap / 100000000
            if market_cap_billion >= 1000:
                score += 20
            elif market_cap_billion >= 500:
                score += 15
            elif market_cap_billion >= 100:
                score += 10
        
        return min(score, 100)
    
    def _analyze_volume_pattern(self, stock: CleanedStockData) -> float:
        """거래량 패턴 분석"""
        score = 50.0  # 기본 점수
        
        # 시가총액으로 유동성 판단
        if stock.market_cap:
            market_cap_billion = stock.market_cap / 100000000
            
            # 대형주는 높은 유동성
            if market_cap_billion >= 1000:
                score += 30  # 매우 높은 유동성
            elif market_cap_billion >= 500:
                score += 25
            elif market_cap_billion >= 100:
                score += 20
            elif market_cap_billion >= 50:
                score += 15
            else:
                score -= 10  # 소형주는 유동성 위험
        
        # 가격 모멘텀과 연계 (상승시 거래량 증가 가정)
        if stock.price_momentum_3m:
            if stock.price_momentum_3m >= 15:
                score += 20  # 상승시 거래량 증가 가정
            elif stock.price_momentum_3m >= 5:
                score += 10
        
        return min(max(score, 0), 100)
    
    def _analyze_market_leadership(self, stock: CleanedStockData) -> float:
        """시장 선도성 분석"""
        score = 50.0
        
        # 시가총액 기반 시장 지위
        if stock.market_cap:
            market_cap_billion = stock.market_cap / 100000000
            
            if market_cap_billion >= 1000:
                score += 30  # 시장 선도주
            elif market_cap_billion >= 500:
                score += 25
            elif market_cap_billion >= 100:
                score += 20
        
        # 섹터 리더십 (주요 섹터)
        leader_sectors = ['Technology', 'Healthcare', 'Finance', 'IT', '반도체', '바이오', '금융']
        if stock.sector and any(sector in stock.sector for sector in leader_sectors):
            score += 20
        
        # 수익성 기반 경쟁력
        if stock.roe:
            if stock.roe >= 20:
                score += 15
            elif stock.roe >= 15:
                score += 10
        
        # 성장성 기반 미래 리더십
        if stock.profit_growth:
            if stock.profit_growth >= 20:
                score += 15
            elif stock.profit_growth >= 10:
                score += 10
        
        return min(max(score, 0), 100)
    
    def _analyze_timing_signals(self, stock: CleanedStockData) -> float:
        """타이밍 신호 분석"""
        score = 50.0
        
        # 모멘텀 기반 타이밍
        if stock.price_momentum_3m:
            if stock.price_momentum_3m >= 20:
                score += 25  # 강한 상승 신호
            elif stock.price_momentum_3m >= 10:
                score += 20
            elif stock.price_momentum_3m >= 5:
                score += 15
            elif stock.price_momentum_3m >= 0:
                score += 10
            else:
                score -= 30  # 하락 신호는 매우 부정적
        
        # 밸류에이션 타이밍 (과도한 고평가 회피)
        if stock.pe_ratio:
            if stock.pe_ratio <= 15:
                score += 15  # 적정 밸류에이션
            elif stock.pe_ratio <= 25:
                score += 10
            elif stock.pe_ratio <= 40:
                score += 5
            else:
                score -= 10  # 과도한 고평가
        
        # 성장성 타이밍
        if stock.revenue_growth and stock.profit_growth:
            if stock.profit_growth >= 15 and stock.revenue_growth >= 10:
                score += 15  # 성장 가속화
            elif stock.profit_growth >= 5:
                score += 10
        
        return min(max(score, 0), 100)
    
    def _analyze_speculation_safety(self, stock: CleanedStockData) -> float:
        """투기 안전성 분석"""
        score = 50.0
        
        # 재무 건전성 (투기에서도 기본은 중요)
        if stock.debt_ratio:
            if stock.debt_ratio <= 30:
                score += 20  # 낮은 부채
            elif stock.debt_ratio <= 50:
                score += 15
            elif stock.debt_ratio <= 100:
                score += 10
            else:
                score -= 15  # 높은 부채는 위험
        
        # 유동성 (시가총액)
        if stock.market_cap:
            market_cap_billion = stock.market_cap / 100000000
            if market_cap_billion >= 500:
                score += 20  # 높은 유동성
            elif market_cap_billion >= 100:
                score += 15
            elif market_cap_billion >= 50:
                score += 10
            else:
                score -= 10  # 낮은 유동성 위험
        
        # 수익성 안전성
        if stock.roe:
            if stock.roe >= 15:
                score += 15
            elif stock.roe >= 10:
                score += 10
            elif stock.roe < 0:
                score -= 20  # 손실 기업은 위험
        
        # 현금 흐름 (유동비율로 대체)
        if stock.current_ratio:
            if stock.current_ratio >= 2.0:
                score += 15
            elif stock.current_ratio >= 1.5:
                score += 10
            elif stock.current_ratio >= 1.0:
                score += 5
            else:
                score -= 15
        
        return min(max(score, 0), 100)
    
    def get_strategy_summary(self, stock: CleanedStockData) -> Dict[str, Any]:
        """전략 요약 정보"""
        analysis = self.analyze_stock_detailed(stock)
        
        return {
            "전략명": self.strategy_name,
            "총점": f"{analysis['total_score']:.1f}/100",
            "투기분석점수": {
                "가격모멘텀": f"{analysis['scores']['price_momentum']:.1f}",
                "거래량패턴": f"{analysis['scores']['volume_pattern']:.1f}",
                "시장선도성": f"{analysis['scores']['market_leadership']:.1f}",
                "타이밍신호": f"{analysis['scores']['timing_signals']:.1f}",
                "투기안전성": f"{analysis['scores']['speculation_safety']:.1f}"
            },
            "투자판단": self._get_investment_decision(analysis['total_score']),
            "핵심포인트": self._get_key_points(stock, analysis)
        }
    
    def _get_investment_decision(self, score: float) -> str:
        """투자 판단"""
        if score >= 80:
            return "🟢 강력매수 - 완벽한 투기 기회"
        elif score >= 70:
            return "🔵 매수 - 좋은 모멘텀"
        elif score >= 60:
            return "🟡 관심 - 타이밍 대기"
        elif score >= 50:
            return "⚪ 중립 - 신호 불분명"
        else:
            return "🔴 회피 - 투기 부적합"
    
    def _get_key_points(self, stock: CleanedStockData, analysis: Dict[str, Any]) -> List[str]:
        """핵심 포인트"""
        points = []
        scores = analysis['scores']
        
        if scores['price_momentum'] >= 70:
            points.append("✅ 강한 가격 모멘텀")
        if scores['market_leadership'] >= 70:
            points.append("✅ 시장 선도주")
        if scores['timing_signals'] >= 70:
            points.append("✅ 좋은 진입 타이밍")
        if stock.price_momentum_3m and stock.price_momentum_3m >= 20:
            points.append("✅ 상승 추세 확실")
        if stock.market_cap and stock.market_cap >= 50000000000:  # 500억 이상
            points.append("✅ 충분한 유동성")
        
        if scores['price_momentum'] < 50:
            points.append("⚠️ 약한 모멘텀")
        if stock.price_momentum_3m and stock.price_momentum_3m < 0:
            points.append("⚠️ 하락 추세 위험")
        if scores['speculation_safety'] < 50:
            points.append("⚠️ 투기 위험도 높음")
        
        return points[:5]  # 최대 5개 포인트 