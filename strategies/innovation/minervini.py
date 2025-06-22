#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌱 마크 미너빈 (Mark Minervini) 투자 전략
혁신적 성장주 발굴과 정밀한 타이밍 매매
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from ..common import BaseStrategy, StrategyResult
from data.processed.data_cleaner import CleanedStockData

logger = logging.getLogger(__name__)

class MinerviniStrategy(BaseStrategy):
    """마크 미너빈의 SEPA 모멘텀 성장주 전략"""
    
    def __init__(self):
        super().__init__()
        self.strategy_name = "Mark Minervini SEPA"
        self.description = "SEPA 기반 고성장 모멘텀 주식 발굴"
        
        # 미너빈 SEPA 가중치
        self.weights = {
            'earnings_growth': 0.25,     # 실적 성장
            'price_momentum': 0.25,      # 가격 모멘텀
            'market_leadership': 0.20,   # 시장 리더십
            'institutional_support': 0.15, # 기관 지지
            'risk_management': 0.15      # 위험 관리
        }
    
    def analyze_stock(self, stock: CleanedStockData) -> float:
        """미너빈 SEPA 분석"""
        try:
            scores = {}
            
            # 실적 성장 분석
            scores['earnings_growth'] = self._analyze_earnings_growth(stock)
            
            # 가격 모멘텀 분석
            scores['price_momentum'] = self._analyze_price_momentum(stock)
            
            # 시장 리더십 분석
            scores['market_leadership'] = self._analyze_market_leadership(stock)
            
            # 기관 지지 분석
            scores['institutional_support'] = self._analyze_institutional_support(stock)
            
            # 위험 관리 분석
            scores['risk_management'] = self._analyze_risk_management(stock)
            
            # 가중 평균 계산
            total_score = sum(
                scores[key] * self.weights[key] 
                for key in scores
            )
            
            logger.debug(f"Minervini SEPA 분석 ({stock.symbol}): {total_score:.1f}")
            return min(max(total_score, 0), 100)
            
        except Exception as e:
            logger.error(f"Minervini 전략 분석 오류 ({stock.symbol}): {e}")
            return 0.0
    
    def _analyze_earnings_growth(self, stock: CleanedStockData) -> float:
        """실적 성장 분석 - 미너빈의 핵심"""
        score = 0.0
        
        # 이익 성장률 (25% 이상 선호)
        if stock.profit_growth:
            if stock.profit_growth >= 50:
                score += 40  # 폭발적 성장
            elif stock.profit_growth >= 30:
                score += 35
            elif stock.profit_growth >= 25:
                score += 30  # 미너빈 기준
            elif stock.profit_growth >= 20:
                score += 25
            elif stock.profit_growth >= 15:
                score += 20
            elif stock.profit_growth >= 10:
                score += 15
            elif stock.profit_growth >= 5:
                score += 10
            elif stock.profit_growth >= 0:
                score += 5
            else:
                score -= 20  # 실적 하락
        
        # 매출 성장률
        if stock.revenue_growth:
            if stock.revenue_growth >= 25:
                score += 25
            elif stock.revenue_growth >= 20:
                score += 20
            elif stock.revenue_growth >= 15:
                score += 15
            elif stock.revenue_growth >= 10:
                score += 10
            elif stock.revenue_growth >= 5:
                score += 5
            else:
                score -= 10  # 매출 감소
        
        # ROE (자본 효율성)
        if stock.roe:
            if stock.roe >= 25:
                score += 20  # 뛰어난 효율성
            elif stock.roe >= 20:
                score += 15
            elif stock.roe >= 15:
                score += 10
            elif stock.roe >= 10:
                score += 5
            elif stock.roe < 5:
                score -= 10
        
        # 성장 가속도 (이익 > 매출 성장)
        if stock.profit_growth and stock.revenue_growth:
            if stock.profit_growth > stock.revenue_growth * 1.5:
                score += 15  # 레버리지 효과
            elif stock.profit_growth > stock.revenue_growth:
                score += 10
        
        return min(score, 100)
    
    def _analyze_price_momentum(self, stock: CleanedStockData) -> float:
        """가격 모멘텀 분석"""
        score = 0.0
        
        # 3개월 모멘텀 (신고가 근처)
        if stock.price_momentum_3m:
            if stock.price_momentum_3m >= 30:
                score += 35  # 강력한 상승세
            elif stock.price_momentum_3m >= 25:
                score += 30
            elif stock.price_momentum_3m >= 20:
                score += 25
            elif stock.price_momentum_3m >= 15:
                score += 20
            elif stock.price_momentum_3m >= 10:
                score += 15
            elif stock.price_momentum_3m >= 5:
                score += 10
            elif stock.price_momentum_3m >= 0:
                score += 5
            else:
                score -= 25  # 하락세
        
        # 장기 추세 (6개월 또는 1년)
        if hasattr(stock, 'price_momentum_6m') and stock.price_momentum_6m:
            if stock.price_momentum_6m >= 50:
                score += 25  # 장기 강세
            elif stock.price_momentum_6m >= 30:
                score += 20
            elif stock.price_momentum_6m >= 20:
                score += 15
            elif stock.price_momentum_6m >= 10:
                score += 10
            elif stock.price_momentum_6m < 0:
                score -= 15
        
        # 모멘텀 일관성 (변동성 대비)
        if stock.volatility and stock.price_momentum_3m:
            momentum_strength = stock.price_momentum_3m / max(stock.volatility, 1)
            if momentum_strength >= 1.0:
                score += 20  # 강한 추세
            elif momentum_strength >= 0.7:
                score += 15
            elif momentum_strength >= 0.5:
                score += 10
        
        # 상대 강도 (섹터 내 우위)
        if stock.price_momentum_3m and stock.price_momentum_3m >= 20:
            score += 20  # 상대적 강세
        elif stock.price_momentum_3m and stock.price_momentum_3m >= 15:
            score += 15
        elif stock.price_momentum_3m and stock.price_momentum_3m >= 10:
            score += 10
        
        return min(score, 100)
    
    def _analyze_market_leadership(self, stock: CleanedStockData) -> float:
        """시장 리더십 분석"""
        score = 50.0
        
        # 시가총액 (리더십 지위)
        if stock.market_cap:
            market_cap_billion = stock.market_cap / 100000000
            if market_cap_billion >= 1000:
                score += 25  # 대형주 리더
            elif market_cap_billion >= 500:
                score += 20
            elif market_cap_billion >= 100:
                score += 15  # 중견주 리더
            elif market_cap_billion >= 50:
                score += 10
            else:
                score -= 5  # 소형주는 리더십 제한
        
        # 수익성 우위 (ROE)
        if stock.roe:
            if stock.roe >= 25:
                score += 20  # 업계 최고 수준
            elif stock.roe >= 20:
                score += 15
            elif stock.roe >= 15:
                score += 10
            elif stock.roe >= 10:
                score += 5
        
        # 성장성 우위
        if stock.profit_growth:
            if stock.profit_growth >= 30:
                score += 15  # 성장 리더
            elif stock.profit_growth >= 20:
                score += 10
            elif stock.profit_growth >= 15:
                score += 5
        
        # 섹터 리더십 (성장 섹터)
        leader_sectors = ['Technology', 'Healthcare', 'Innovation', 
                         '반도체', '바이오', '혁신기술']
        if stock.sector and any(sector in stock.sector for sector in leader_sectors):
            score += 15  # 리더십 섹터
        
        return min(max(score, 0), 100)
    
    def _analyze_institutional_support(self, stock: CleanedStockData) -> float:
        """기관 지지 분석"""
        score = 50.0
        
        # 시가총액 기반 기관 관심도
        if stock.market_cap:
            market_cap_billion = stock.market_cap / 100000000
            if market_cap_billion >= 500:
                score += 25  # 높은 기관 관심
            elif market_cap_billion >= 200:
                score += 20
            elif market_cap_billion >= 100:
                score += 15
            elif market_cap_billion >= 50:
                score += 10
            else:
                score -= 10  # 기관 관심 제한
        
        # 실적 품질 (기관 선호)
        if stock.roe and stock.profit_growth:
            if stock.roe >= 20 and stock.profit_growth >= 20:
                score += 20  # 기관 선호 조건
            elif stock.roe >= 15 and stock.profit_growth >= 15:
                score += 15
            elif stock.roe >= 10 and stock.profit_growth >= 10:
                score += 10
        
        # 안정성 (기관 위험 관리)
        if stock.debt_ratio:
            if stock.debt_ratio <= 30:
                score += 15  # 안전한 재무구조
            elif stock.debt_ratio <= 50:
                score += 10
            elif stock.debt_ratio <= 100:
                score += 5
            else:
                score -= 15  # 높은 부채
        
        # 유동성 (기관 거래 용이성)
        if stock.market_cap:
            market_cap_billion = stock.market_cap / 100000000
            if market_cap_billion >= 1000:
                score += 10  # 높은 유동성
            elif market_cap_billion >= 500:
                score += 8
            elif market_cap_billion >= 100:
                score += 5
        
        return min(max(score, 0), 100)
    
    def _analyze_risk_management(self, stock: CleanedStockData) -> float:
        """위험 관리 분석"""
        score = 50.0
        
        # 변동성 위험
        if stock.volatility:
            if stock.volatility <= 20:
                score += 20  # 낮은 위험
            elif stock.volatility <= 30:
                score += 15
            elif stock.volatility <= 40:
                score += 10
            elif stock.volatility <= 50:
                score += 5
            else:
                score -= 15  # 높은 위험
        
        # 재무 위험
        if stock.debt_ratio:
            if stock.debt_ratio <= 20:
                score += 20  # 매우 안전
            elif stock.debt_ratio <= 40:
                score += 15
            elif stock.debt_ratio <= 60:
                score += 10
            elif stock.debt_ratio <= 100:
                score += 5
            else:
                score -= 20  # 높은 부채 위험
        
        # 유동성 위험
        if stock.current_ratio:
            if stock.current_ratio >= 2.0:
                score += 15  # 충분한 유동성
            elif stock.current_ratio >= 1.5:
                score += 10
            elif stock.current_ratio >= 1.0:
                score += 5
            else:
                score -= 15  # 유동성 부족
        
        # 밸류에이션 위험
        if stock.pe_ratio:
            if stock.pe_ratio <= 20:
                score += 10  # 적정 밸류에이션
            elif stock.pe_ratio <= 30:
                score += 5
            elif stock.pe_ratio >= 50:
                score -= 15  # 고평가 위험
        
        # 실적 안정성
        if stock.profit_growth and stock.revenue_growth:
            if stock.profit_growth >= 10 and stock.revenue_growth >= 5:
                score += 10  # 안정적 성장
            elif stock.profit_growth >= 0 and stock.revenue_growth >= 0:
                score += 5
            else:
                score -= 10  # 불안정한 실적
        
        return min(max(score, 0), 100)
    
    def get_strategy_summary(self, stock: CleanedStockData) -> Dict[str, Any]:
        """전략 요약 정보"""
        analysis = self.analyze_stock_detailed(stock)
        
        return {
            "전략명": self.strategy_name,
            "총점": f"{analysis['total_score']:.1f}/100",
            "SEPA분석점수": {
                "실적성장": f"{analysis['scores']['earnings_growth']:.1f}",
                "가격모멘텀": f"{analysis['scores']['price_momentum']:.1f}",
                "시장리더십": f"{analysis['scores']['market_leadership']:.1f}",
                "기관지지": f"{analysis['scores']['institutional_support']:.1f}",
                "위험관리": f"{analysis['scores']['risk_management']:.1f}"
            },
            "투자판단": self._get_investment_decision(analysis['total_score']),
            "핵심포인트": self._get_key_points(stock, analysis)
        }
    
    def _get_investment_decision(self, score: float) -> str:
        """투자 판단"""
        if score >= 80:
            return "🟢 강력매수 - 완벽한 SEPA 조건"
        elif score >= 70:
            return "🔵 매수 - 우수한 모멘텀주"
        elif score >= 60:
            return "🟡 관심 - 성장주 후보"
        elif score >= 50:
            return "⚪ 중립 - 조건 확인 필요"
        else:
            return "🔴 회피 - SEPA 조건 부족"
    
    def _get_key_points(self, stock: CleanedStockData, analysis: Dict[str, Any]) -> List[str]:
        """핵심 포인트"""
        points = []
        scores = analysis['scores']
        
        if scores['earnings_growth'] >= 70:
            points.append("✅ 뛰어난 실적 성장")
        if scores['price_momentum'] >= 70:
            points.append("✅ 강력한 가격 모멘텀")
        if scores['market_leadership'] >= 70:
            points.append("✅ 시장 리더십 보유")
        if stock.profit_growth and stock.profit_growth >= 25:
            points.append("✅ 미너빈 성장 기준 충족")
        if stock.price_momentum_3m and stock.price_momentum_3m >= 20:
            points.append("✅ 강한 상승 추세")
        
        if scores['earnings_growth'] < 50:
            points.append("⚠️ 실적 성장 부족")
        if scores['price_momentum'] < 50:
            points.append("⚠️ 모멘텀 부족")
        if scores['risk_management'] < 50:
            points.append("⚠️ 위험 관리 우려")
        if stock.profit_growth and stock.profit_growth < 10:
            points.append("⚠️ 성장 동력 약함")
        
        return points[:5]  # 최대 5개 포인트 