#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📈 일목산인 (호소다 고이치) 투자 전략
일목균형표(Ichimoku Kinko Hyo) 기반 기술적 분석
"""

import logging
from typing import Dict, Any, Optional, List
from ..common import BaseStrategy, StrategyResult
from data.processed.data_cleaner import CleanedStockData

logger = logging.getLogger(__name__)

class IchimokuStrategy(BaseStrategy):
    """일목산인의 일목균형표 전략"""
    
    def __init__(self):
        super().__init__()
        self.strategy_name = "Ichimoku Kinko Hyo"
        self.description = "일목균형표 기반 시간과 가격의 균형 분석"
        
        # 일목균형표 가중치
        self.weights = {
            'trend_direction': 0.25,     # 추세 방향
            'cloud_analysis': 0.25,      # 구름대 분석
            'momentum_signals': 0.20,    # 모멘텀 신호
            'time_cycles': 0.15,         # 시간 주기
            'balance_harmony': 0.15      # 균형과 조화
        }
    
    def analyze_stock(self, stock: CleanedStockData) -> float:
        """일목균형표 분석"""
        try:
            scores = {}
            
            # 추세 방향 분석
            scores['trend_direction'] = self._analyze_trend_direction(stock)
            
            # 구름대 분석
            scores['cloud_analysis'] = self._analyze_cloud_position(stock)
            
            # 모멘텀 신호 분석
            scores['momentum_signals'] = self._analyze_momentum_signals(stock)
            
            # 시간 주기 분석
            scores['time_cycles'] = self._analyze_time_cycles(stock)
            
            # 균형과 조화 분석
            scores['balance_harmony'] = self._analyze_balance_harmony(stock)
            
            # 가중 평균 계산
            total_score = sum(
                scores[key] * self.weights[key] 
                for key in scores
            )
            
            logger.debug(f"Ichimoku 분석 ({stock.symbol}): {total_score:.1f}")
            return min(max(total_score, 0), 100)
            
        except Exception as e:
            logger.error(f"Ichimoku 전략 분석 오류 ({stock.symbol}): {e}")
            return 0.0
    
    def _analyze_trend_direction(self, stock: CleanedStockData) -> float:
        """추세 방향 분석 - 전환선과 기준선"""
        score = 50.0  # 중립 기본값
        
        # 가격 모멘텀으로 추세 판단
        if stock.price_momentum_3m:
            # 강한 상승 추세
            if stock.price_momentum_3m >= 20:
                score += 35  # 매우 강한 상승
            elif stock.price_momentum_3m >= 15:
                score += 30
            elif stock.price_momentum_3m >= 10:
                score += 25
            elif stock.price_momentum_3m >= 5:
                score += 20
            elif stock.price_momentum_3m >= 0:
                score += 10
            else:
                # 하락 추세
                if stock.price_momentum_3m <= -20:
                    score -= 35
                elif stock.price_momentum_3m <= -15:
                    score -= 30
                elif stock.price_momentum_3m <= -10:
                    score -= 25
                else:
                    score -= 15
        
        # 장기 추세 확인 (연간 성과)
        if hasattr(stock, 'price_momentum_1y') and stock.price_momentum_1y:
            if stock.price_momentum_1y >= 30:
                score += 15  # 장기 상승 추세
            elif stock.price_momentum_1y >= 10:
                score += 10
            elif stock.price_momentum_1y < -20:
                score -= 15  # 장기 하락 추세
        
        return min(max(score, 0), 100)
    
    def _analyze_cloud_position(self, stock: CleanedStockData) -> float:
        """구름대 분석 - 지지와 저항"""
        score = 50.0
        
        # 시가총액으로 안정성 판단 (구름대 두께)
        if stock.market_cap:
            market_cap_billion = stock.market_cap / 100000000
            
            # 대형주는 두꺼운 구름대 (강한 지지/저항)
            if market_cap_billion >= 1000:
                score += 25  # 매우 안정적
            elif market_cap_billion >= 500:
                score += 20
            elif market_cap_billion >= 100:
                score += 15
            elif market_cap_billion >= 50:
                score += 10
            else:
                score -= 10  # 얇은 구름대 (불안정)
        
        # 가격 위치 분석 (구름대 위/아래)
        if stock.price_momentum_3m:
            if stock.price_momentum_3m >= 10:
                score += 20  # 구름대 위 (강세)
            elif stock.price_momentum_3m >= 0:
                score += 10  # 구름대 근처
            else:
                score -= 20  # 구름대 아래 (약세)
        
        # 변동성으로 구름대 두께 판단
        if stock.volatility:
            if stock.volatility <= 20:
                score += 15  # 낮은 변동성 = 두꺼운 구름대
            elif stock.volatility <= 30:
                score += 10
            elif stock.volatility >= 50:
                score -= 15  # 높은 변동성 = 얇은 구름대
        
        return min(max(score, 0), 100)
    
    def _analyze_momentum_signals(self, stock: CleanedStockData) -> float:
        """모멘텀 신호 분석"""
        score = 50.0
        
        # 단기 모멘텀 (전환선)
        if stock.price_momentum_3m:
            if stock.price_momentum_3m >= 15:
                score += 25
            elif stock.price_momentum_3m >= 5:
                score += 15
            elif stock.price_momentum_3m >= 0:
                score += 5
            else:
                score -= 20
        
        # 수익성 모멘텀
        if stock.profit_growth:
            if stock.profit_growth >= 20:
                score += 20
            elif stock.profit_growth >= 10:
                score += 15
            elif stock.profit_growth >= 0:
                score += 10
            else:
                score -= 15
        
        # 매출 모멘텀
        if stock.revenue_growth:
            if stock.revenue_growth >= 15:
                score += 15
            elif stock.revenue_growth >= 5:
                score += 10
            elif stock.revenue_growth >= 0:
                score += 5
            else:
                score -= 10
        
        return min(max(score, 0), 100)
    
    def _analyze_time_cycles(self, stock: CleanedStockData) -> float:
        """시간 주기 분석 - 일목균형표의 핵심"""
        score = 50.0
        
        # 기본 수치 (9, 26, 52)를 기업 성장 주기로 해석
        
        # 단기 주기 (9일 = 분기 실적)
        if stock.profit_growth:
            if stock.profit_growth >= 15:
                score += 20  # 단기 성장 가속
            elif stock.profit_growth >= 5:
                score += 15
            elif stock.profit_growth < -10:
                score -= 20
        
        # 중기 주기 (26일 = 반기/연간)
        if stock.roe:
            if stock.roe >= 20:
                score += 15  # 중기 수익성 우수
            elif stock.roe >= 15:
                score += 10
            elif stock.roe >= 10:
                score += 5
            elif stock.roe < 0:
                score -= 15
        
        # 장기 주기 (52일 = 장기 트렌드)
        if stock.debt_ratio:
            if stock.debt_ratio <= 30:
                score += 15  # 장기 안정성
            elif stock.debt_ratio <= 50:
                score += 10
            elif stock.debt_ratio >= 100:
                score -= 15
        
        return min(max(score, 0), 100)
    
    def _analyze_balance_harmony(self, stock: CleanedStockData) -> float:
        """균형과 조화 분석 - 일목균형표의 철학"""
        score = 50.0
        
        # 성장과 안정성의 균형
        growth_score = 0
        stability_score = 0
        
        # 성장성 평가
        if stock.profit_growth:
            if stock.profit_growth >= 15:
                growth_score += 30
            elif stock.profit_growth >= 5:
                growth_score += 20
            elif stock.profit_growth >= 0:
                growth_score += 10
        
        if stock.revenue_growth:
            if stock.revenue_growth >= 10:
                growth_score += 20
            elif stock.revenue_growth >= 0:
                growth_score += 10
        
        # 안정성 평가
        if stock.debt_ratio:
            if stock.debt_ratio <= 30:
                stability_score += 25
            elif stock.debt_ratio <= 50:
                stability_score += 15
            elif stock.debt_ratio <= 100:
                stability_score += 5
        
        if stock.current_ratio:
            if stock.current_ratio >= 2.0:
                stability_score += 25
            elif stock.current_ratio >= 1.5:
                stability_score += 15
            elif stock.current_ratio >= 1.0:
                stability_score += 10
        
        # 균형 점수 계산
        balance_penalty = abs(growth_score - stability_score) * 0.3
        harmony_score = (growth_score + stability_score) / 2 - balance_penalty
        
        score = max(harmony_score, 0)
        
        # 밸류에이션 균형
        if stock.pe_ratio:
            if 10 <= stock.pe_ratio <= 25:
                score += 15  # 적정 밸류에이션
            elif 5 <= stock.pe_ratio <= 40:
                score += 10
            elif stock.pe_ratio > 50:
                score -= 10  # 과도한 고평가
        
        return min(max(score, 0), 100)
    
    def get_strategy_summary(self, stock: CleanedStockData) -> Dict[str, Any]:
        """전략 요약 정보"""
        analysis = self.analyze_stock_detailed(stock)
        
        return {
            "전략명": self.strategy_name,
            "총점": f"{analysis['total_score']:.1f}/100",
            "일목분석점수": {
                "추세방향": f"{analysis['scores']['trend_direction']:.1f}",
                "구름대분석": f"{analysis['scores']['cloud_analysis']:.1f}",
                "모멘텀신호": f"{analysis['scores']['momentum_signals']:.1f}",
                "시간주기": f"{analysis['scores']['time_cycles']:.1f}",
                "균형조화": f"{analysis['scores']['balance_harmony']:.1f}"
            },
            "투자판단": self._get_investment_decision(analysis['total_score']),
            "핵심포인트": self._get_key_points(stock, analysis)
        }
    
    def _get_investment_decision(self, score: float) -> str:
        """투자 판단"""
        if score >= 80:
            return "🟢 강력매수 - 완벽한 균형"
        elif score >= 70:
            return "🔵 매수 - 좋은 조화"
        elif score >= 60:
            return "🟡 관심 - 균형 확인 필요"
        elif score >= 50:
            return "⚪ 중립 - 시간 대기"
        else:
            return "🔴 회피 - 불균형 상태"
    
    def _get_key_points(self, stock: CleanedStockData, analysis: Dict[str, Any]) -> List[str]:
        """핵심 포인트"""
        points = []
        scores = analysis['scores']
        
        if scores['trend_direction'] >= 70:
            points.append("✅ 강한 상승 추세")
        if scores['cloud_analysis'] >= 70:
            points.append("✅ 안정적 지지선")
        if scores['momentum_signals'] >= 70:
            points.append("✅ 모멘텀 신호 양호")
        if scores['balance_harmony'] >= 70:
            points.append("✅ 성장-안정성 균형")
        if stock.price_momentum_3m and stock.price_momentum_3m >= 15:
            points.append("✅ 구름대 위 강세")
        
        if scores['trend_direction'] < 50:
            points.append("⚠️ 추세 불분명")
        if scores['cloud_analysis'] < 50:
            points.append("⚠️ 지지선 약화")
        if scores['balance_harmony'] < 50:
            points.append("⚠️ 균형 깨짐")
        if stock.price_momentum_3m and stock.price_momentum_3m < -10:
            points.append("⚠️ 구름대 아래 약세")
        
        return points[:5]  # 최대 5개 포인트 