#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📈 윌리엄 오닐 (William O'Neil) 투자 전략
CAN SLIM 방법론 구현
"""

import logging
from typing import Dict, Any, Optional, List
from ..common import BaseStrategy, StrategyResult, get_stock_value, get_financial_metrics
from data.processed.data_cleaner import CleanedStockData

logger = logging.getLogger(__name__)

class ONeillStrategy(BaseStrategy):
    """윌리엄 오닐의 CAN SLIM 전략"""
    
    def __init__(self):
        super().__init__()
        self.strategy_name = "William O'Neil CAN SLIM"
        self.description = "성장주 발굴을 위한 CAN SLIM 7가지 기준"
        
        # CAN SLIM 기준 가중치
        self.weights = {
            'current_earnings': 0.20,    # C: 현재 분기 실적
            'annual_earnings': 0.20,     # A: 연간 실적 성장
            'new_products': 0.10,        # N: 신제품, 신경영진, 신고가
            'supply_demand': 0.15,       # S: 주식 수급과 대형주
            'leader_laggard': 0.15,      # L: 선도주 vs 후행주
            'institutional': 0.10,       # I: 기관 후원
            'market_direction': 0.10     # M: 시장 방향
        }
    
    def analyze_stock(self, stock) -> StrategyResult:
        """CAN SLIM 분석"""
        try:
            # 재무 지표 추출
            metrics = get_financial_metrics(stock)
            
            scores = {}
            analysis_details = {}
            
            # C - Current Quarterly Earnings (현재 분기 실적)
            scores['current_earnings'] = self._analyze_current_earnings(metrics)
            analysis_details['current_earnings'] = scores['current_earnings']
            
            # A - Annual Earnings Growth (연간 실적 성장)
            scores['annual_earnings'] = self._analyze_annual_earnings(metrics)
            analysis_details['annual_earnings'] = scores['annual_earnings']
            
            # N - New Products, Management, Price Highs (신규 요소)
            scores['new_products'] = self._analyze_new_factors(metrics)
            analysis_details['new_products'] = scores['new_products']
            
            # S - Supply and Demand (수급)
            scores['supply_demand'] = self._analyze_supply_demand(metrics)
            analysis_details['supply_demand'] = scores['supply_demand']
            
            # L - Leader or Laggard (선도주 여부)
            scores['leader_laggard'] = self._analyze_leadership(metrics)
            analysis_details['leader_laggard'] = scores['leader_laggard']
            
            # I - Institutional Sponsorship (기관 투자)
            scores['institutional'] = self._analyze_institutional(metrics)
            analysis_details['institutional'] = scores['institutional']
            
            # M - Market Direction (시장 방향)
            scores['market_direction'] = self._analyze_market_direction(metrics)
            analysis_details['market_direction'] = scores['market_direction']
            
            # 가중 평균 계산
            total_score = sum(
                scores[key] * self.weights[key] 
                for key in scores
            )
            
            # 투자 판단
            investment_decision = self._get_investment_decision(total_score)
            
            # 핵심 포인트 추출
            key_points = self._get_key_points(metrics, analysis_details)
            
            logger.debug(f"O'Neil CAN SLIM 분석 ({metrics.get('symbol', 'Unknown')}): {total_score:.1f}")
            
            return StrategyResult(
                total_score=min(max(total_score, 0), 100),
                scores=scores,
                strategy_name=self.strategy_name,
                investment_decision=investment_decision,
                key_points=key_points,
                analysis_details=analysis_details
            )
            
        except Exception as e:
            logger.error(f"O'Neil 전략 분석 오류: {e}")
            return self._create_error_result()
    
    def _analyze_current_earnings(self, metrics: Dict) -> float:
        """현재 분기 실적 분석"""
        score = 0.0
        
        # 최근 수익성 증가율 (25% 이상 선호)
        profit_growth = metrics.get('profit_growth')
        if profit_growth:
            if profit_growth >= 25:
                score += 40
            elif profit_growth >= 15:
                score += 30
            elif profit_growth >= 5:
                score += 20
            elif profit_growth >= 0:
                score += 10
        
        # ROE 기준 (18% 이상 선호)
        roe = metrics.get('roe')
        if roe:
            if roe >= 18:
                score += 30
            elif roe >= 15:
                score += 25
            elif roe >= 12:
                score += 20
            elif roe >= 8:
                score += 15
        
        # 매출 성장률
        revenue_growth = metrics.get('revenue_growth')
        if revenue_growth:
            if revenue_growth >= 25:
                score += 30
            elif revenue_growth >= 15:
                score += 20
            elif revenue_growth >= 5:
                score += 10
        
        return min(score, 100)
    
    def _analyze_annual_earnings(self, metrics: Dict) -> float:
        """연간 실적 성장 분석"""
        score = 0.0
        
        # 지속적인 성장 (3년 연속 성장 선호)
        profit_growth = metrics.get('profit_growth')
        revenue_growth = metrics.get('revenue_growth')
        if profit_growth and revenue_growth:
            # 수익성 성장
            if profit_growth >= 25:
                score += 50
            elif profit_growth >= 15:
                score += 40
            elif profit_growth >= 10:
                score += 30
            
            # 매출 성장 일관성
            if revenue_growth >= 15:
                score += 30
            elif revenue_growth >= 10:
                score += 20
            elif revenue_growth >= 5:
                score += 10
        
        # ROE 트렌드 (높은 자기자본이익률)
        roe = metrics.get('roe')
        if roe and roe >= 17:
            score += 20
        
        return min(score, 100)
    
    def _analyze_new_factors(self, metrics: Dict) -> float:
        """신규 요소 분석 (신제품, 신경영진, 신고가)"""
        score = 50.0  # 기본 점수
        
        # 가격 모멘텀 (신고가 근처)
        price_momentum_3m = metrics.get('price_momentum_3m')
        if price_momentum_3m:
            if price_momentum_3m >= 20:
                score += 30  # 강한 상승 모멘텀
            elif price_momentum_3m >= 10:
                score += 20
            elif price_momentum_3m >= 5:
                score += 10
            elif price_momentum_3m < -10:
                score -= 20  # 하락 모멘텀은 감점
        
        # 성장 섹터 보너스
        sector = metrics.get('sector')
        growth_sectors = ['Technology', 'Healthcare', 'IT', '바이오', '반도체']
        if sector and any(gs in sector for gs in growth_sectors):
            score += 20
        
        return min(max(score, 0), 100)
    
    def _analyze_supply_demand(self, metrics: Dict) -> float:
        """주식 수급 분석"""
        score = 50.0
        
        # 시가총액 기준 (중대형주 선호)
        market_cap = metrics.get('market_cap')
        if market_cap:
            market_cap_billion = market_cap / 100000000  # 억원 단위
            
            if market_cap_billion >= 1000:  # 1조원 이상
                score += 30
            elif market_cap_billion >= 500:  # 5천억원 이상
                score += 25
            elif market_cap_billion >= 100:  # 1천억원 이상
                score += 20
            elif market_cap_billion >= 50:   # 500억원 이상
                score += 15
        
        # 가격 모멘텀으로 수급 판단
        price_momentum_3m = metrics.get('price_momentum_3m')
        if price_momentum_3m:
            if price_momentum_3m >= 15:
                score += 20  # 강한 매수세
            elif price_momentum_3m >= 5:
                score += 10
        
        return min(max(score, 0), 100)
    
    def _analyze_leadership(self, metrics: Dict) -> float:
        """업종 선도주 여부 분석"""
        score = 50.0
        
        # 시장 점유율 대용 지표 (시가총액 기준)
        market_cap = metrics.get('market_cap')
        if market_cap:
            market_cap_billion = market_cap / 100000000
            
            # 대형주는 업종 리더 가능성 높음
            if market_cap_billion >= 1000:
                score += 30
            elif market_cap_billion >= 500:
                score += 20
            elif market_cap_billion >= 100:
                score += 10
        
        # 수익성 우수성 (업종 리더의 특징)
        roe = metrics.get('roe')
        if roe and roe >= 20:
            score += 20
        elif roe and roe >= 15:
            score += 15
        
        # 성장률 우수성
        profit_growth = metrics.get('profit_growth')
        if profit_growth and profit_growth >= 20:
            score += 20
        elif profit_growth and profit_growth >= 15:
            score += 15
        
        # 가격 강도 (상대적 강도)
        price_momentum_3m = metrics.get('price_momentum_3m')
        if price_momentum_3m and price_momentum_3m >= 10:
            score += 10
        
        return min(max(score, 0), 100)
    
    def _analyze_institutional(self, metrics: Dict) -> float:
        """기관 투자 분석"""
        score = 50.0  # 기본 점수
        
        # 시가총액이 클수록 기관 투자 가능성 높음
        market_cap = metrics.get('market_cap')
        if market_cap:
            market_cap_billion = market_cap / 100000000
            
            if market_cap_billion >= 1000:
                score += 30  # 대형주는 기관 선호
            elif market_cap_billion >= 500:
                score += 25
            elif market_cap_billion >= 100:
                score += 20
            elif market_cap_billion >= 50:
                score += 15
        
        # 안정적인 수익성 (기관이 선호하는 특징)
        roe = metrics.get('roe')
        if roe and roe >= 15:
            score += 20
        
        return min(max(score, 0), 100)
    
    def _analyze_market_direction(self, metrics: Dict) -> float:
        """시장 방향 분석"""
        score = 60.0  # 중립적 시장 가정
        
        # 가격 모멘텀으로 시장 방향성 판단
        price_momentum_3m = metrics.get('price_momentum_3m')
        if price_momentum_3m:
            if price_momentum_3m >= 10:
                score += 20  # 상승 시장
            elif price_momentum_3m >= 0:
                score += 10
            elif price_momentum_3m < -10:
                score -= 20  # 하락 시장
        
        # 성장 섹터는 시장 방향성에 덜 민감
        sector = metrics.get('sector')
        growth_sectors = ['Technology', 'Healthcare', 'IT', '바이오']
        if sector and any(gs in sector for gs in growth_sectors):
            score += 20
        
        return min(max(score, 0), 100)
    
    def get_strategy_summary(self, stock: CleanedStockData) -> Dict[str, Any]:
        """전략 요약 정보"""
        analysis = self.analyze_stock_detailed(stock)
        
        return {
            "전략명": self.strategy_name,
            "총점": f"{analysis['total_score']:.1f}/100",
            "CAN_SLIM_점수": {
                "현재실적(C)": f"{analysis['scores']['current_earnings']:.1f}",
                "연간성장(A)": f"{analysis['scores']['annual_earnings']:.1f}",
                "신규요소(N)": f"{analysis['scores']['new_products']:.1f}",
                "수급(S)": f"{analysis['scores']['supply_demand']:.1f}",
                "선도성(L)": f"{analysis['scores']['leader_laggard']:.1f}",
                "기관투자(I)": f"{analysis['scores']['institutional']:.1f}",
                "시장방향(M)": f"{analysis['scores']['market_direction']:.1f}"
            },
            "투자판단": self._get_investment_decision(analysis['total_score']),
            "핵심포인트": self._get_key_points(stock, analysis)
        }
    
    def _get_investment_decision(self, score: float) -> str:
        """투자 판단"""
        if score >= 80:
            return "🟢 강력매수 - CAN SLIM 기준 충족"
        elif score >= 70:
            return "🔵 매수 - 양호한 성장주"
        elif score >= 60:
            return "🟡 관심 - 일부 기준 충족"
        elif score >= 50:
            return "⚪ 중립 - 추가 관찰 필요"
        else:
            return "🔴 회피 - CAN SLIM 기준 미달"
    
    def _get_key_points(self, metrics: Dict, analysis: Dict[str, Any]) -> List[str]:
        """핵심 포인트"""
        points = []
        scores = analysis['scores']
        
        if scores['current_earnings'] >= 70:
            points.append("✅ 우수한 현재 실적")
        if scores['annual_earnings'] >= 70:
            points.append("✅ 지속적인 성장 트렌드")
        if scores['leader_laggard'] >= 70:
            points.append("✅ 업종 선도주 특성")
        
        price_momentum_3m = metrics.get('price_momentum_3m')
        if price_momentum_3m and price_momentum_3m >= 15:
            points.append("✅ 강한 가격 모멘텀")
            
        roe = metrics.get('roe')
        if roe and roe >= 18:
            points.append("✅ 높은 자기자본이익률")
        
        if scores['current_earnings'] < 50:
            points.append("⚠️ 실적 개선 필요")
        if price_momentum_3m and price_momentum_3m < -10:
            points.append("⚠️ 약한 가격 움직임")
        
        return points[:5]  # 최대 5개 포인트
    
    def _create_error_result(self) -> StrategyResult:
        """오류 발생 시 기본 결과 반환"""
        return StrategyResult(
            total_score=0.0,
            scores={},
            strategy_name=self.strategy_name,
            investment_decision="🔴 분석 오류",
            key_points=["⚠️ 분석 중 오류 발생"],
            analysis_details={}
        ) 