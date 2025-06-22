#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🐢 리처드 데니스 (Richard Dennis) 투자 전략
터틀 트레이딩 시스템 - 추세 추종 전략
"""

import logging
from typing import Dict, Any, Optional, List
from ..common import BaseStrategy, StrategyResult
from data.processed.data_cleaner import CleanedStockData

logger = logging.getLogger(__name__)

class DennisStrategy(BaseStrategy):
    """리처드 데니스의 터틀 트레이딩 전략"""
    
    def __init__(self):
        super().__init__()
        self.strategy_name = "리처드 데니스 (Richard Dennis)"
        self.description = "터틀 트레이딩 - 추세 추종과 위험 관리 기반 시스템적 매매"
        
        # 터틀 트레이딩 가중치
        self.weights = {
            'trend_following': 0.30,     # 추세 추종
            'breakout_signals': 0.25,    # 돌파 신호
            'risk_management': 0.20,     # 위험 관리
            'position_sizing': 0.15,     # 포지션 사이징
            'system_discipline': 0.10    # 시스템 준수
        }
    
    def analyze_stock(self, stock) -> StrategyResult:
        """터틀 트레이딩 분석"""
        try:
            scores = {}
            analysis_details = {}
            
            # 추세 추종 분석
            trend_score = self._analyze_trend_following(stock)
            scores['trend_following'] = trend_score
            analysis_details['trend_following'] = trend_score
            
            # 돌파 신호 분석
            breakout_score = self._analyze_breakout_signals(stock)
            scores['breakout_signals'] = breakout_score
            analysis_details['breakout_signals'] = breakout_score
            
            # 위험 관리 분석
            risk_score = self._analyze_risk_management(stock)
            scores['risk_management'] = risk_score
            analysis_details['risk_management'] = risk_score
            
            # 포지션 사이징 분석
            position_score = self._analyze_position_sizing(stock)
            scores['position_sizing'] = position_score
            analysis_details['position_sizing'] = position_score
            
            # 시스템 준수 분석
            discipline_score = self._analyze_system_discipline(stock)
            scores['system_discipline'] = discipline_score
            analysis_details['system_discipline'] = discipline_score
            
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
            logger.error(f"데니스 전략 분석 중 오류: {e}")
            return self._create_error_result()
    
    def _analyze_trend_following(self, stock) -> float:
        """추세 추종 분석 - 터틀의 핵심"""
        score = 0.0
        
        # 강한 추세 (20일/55일 돌파 시뮬레이션)
        price_momentum = getattr(stock, 'price_momentum_3m', None) or getattr(stock, 'price_momentum', 0)
        if price_momentum:
            if price_momentum >= 25:
                score += 40  # 매우 강한 상승 추세
            elif price_momentum >= 20:
                score += 35
            elif price_momentum >= 15:
                score += 30
            elif price_momentum >= 10:
                score += 25
            elif price_momentum >= 5:
                score += 20
            elif price_momentum >= 0:
                score += 10
            else:
                # 하락 추세는 매우 부정적 (터틀은 롱 포지션 위주)
                score -= 30
        
        # 장기 추세 확인
        price_momentum_6m = getattr(stock, 'price_momentum_6m', None) or getattr(stock, 'price_momentum_1y', 0)
        if price_momentum_6m:
            if price_momentum_6m >= 20:
                score += 20  # 장기 상승 추세 확인
            elif price_momentum_6m >= 10:
                score += 15
            elif price_momentum_6m < -10:
                score -= 20
        
        # 추세 일관성 (변동성 대비 수익률)
        volatility = getattr(stock, 'volatility', 0)
        if volatility and price_momentum:
            trend_strength = abs(price_momentum) / max(volatility, 1)
            if trend_strength >= 1.0:
                score += 20  # 강한 추세
            elif trend_strength >= 0.7:
                score += 15
            elif trend_strength >= 0.5:
                score += 10
        
        return min(score, 100)
    
    def _analyze_breakout_signals(self, stock) -> float:
        """돌파 신호 분석"""
        score = 50.0
        
        # 가격 모멘텀 기반 돌파 신호
        price_momentum = getattr(stock, 'price_momentum_3m', None) or getattr(stock, 'price_momentum', 0)
        if price_momentum:
            if price_momentum >= 20:
                score += 30  # 강한 돌파
            elif price_momentum >= 15:
                score += 25
            elif price_momentum >= 10:
                score += 20
            elif price_momentum >= 5:
                score += 15
            elif price_momentum < -5:
                score -= 25  # 하방 돌파
        
        # 거래량 증가 (돌파 확인)
        # 시가총액으로 유동성 판단
        market_cap = getattr(stock, 'market_cap', None)
        if market_cap:
            market_cap_billion = market_cap / 100000000
            if market_cap_billion >= 500:
                score += 20  # 충분한 유동성
            elif market_cap_billion >= 100:
                score += 15
            elif market_cap_billion >= 50:
                score += 10
            else:
                score -= 10  # 유동성 부족
        
        # 변동성 확장 (돌파 신호)
        volatility = getattr(stock, 'volatility', 0)
        if volatility:
            if 25 <= volatility <= 45:
                score += 15  # 적정 변동성 확장
            elif 15 <= volatility <= 60:
                score += 10
            elif volatility > 60:
                score -= 15  # 과도한 변동성
        
        return min(max(score, 0), 100)
    
    def _analyze_risk_management(self, stock) -> float:
        """위험 관리 분석"""
        score = 50.0
        
        # 변동성 기반 위험도
        volatility = getattr(stock, 'volatility', 0)
        if volatility:
            if volatility <= 20:
                score += 25  # 낮은 위험
            elif volatility <= 30:
                score += 20
            elif volatility <= 40:
                score += 15
            elif volatility <= 50:
                score += 10
            else:
                score -= 20  # 높은 위험
        
        # 재무 건전성 (손실 제한)
        debt_ratio = getattr(stock, 'debt_ratio', None)
        if debt_ratio:
            if debt_ratio <= 30:
                score += 20  # 낮은 파산 위험
            elif debt_ratio <= 50:
                score += 15
            elif debt_ratio <= 100:
                score += 10
            else:
                score -= 25  # 높은 파산 위험
        
        # 유동성 (손절매 용이성)
        market_cap = getattr(stock, 'market_cap', None)
        if market_cap:
            market_cap_billion = market_cap / 100000000
            if market_cap_billion >= 1000:
                score += 15  # 매우 높은 유동성
            elif market_cap_billion >= 500:
                score += 12
            elif market_cap_billion >= 100:
                score += 10
            else:
                score -= 10
        
        # 수익성 안정성
        roe = getattr(stock, 'roe', None)
        if roe:
            if roe >= 10:
                score += 10  # 안정적 수익성
            elif roe >= 5:
                score += 5
            elif roe < 0:
                score -= 15  # 손실 기업
        
        return min(max(score, 0), 100)
    
    def _analyze_position_sizing(self, stock) -> float:
        """포지션 사이징 분석"""
        score = 50.0
        
        # 변동성 기반 포지션 크기 (ATR 개념)
        volatility = getattr(stock, 'volatility', 0)
        if volatility:
            # 낮은 변동성 = 큰 포지션 가능
            if volatility <= 15:
                score += 30  # 대형 포지션 가능
            elif volatility <= 25:
                score += 25
            elif volatility <= 35:
                score += 20
            elif volatility <= 45:
                score += 15
            else:
                score -= 20  # 소형 포지션만 가능
        
        # 시가총액 기반 포지션 크기
        market_cap = getattr(stock, 'market_cap', None)
        if market_cap:
            market_cap_billion = market_cap / 100000000
            if market_cap_billion >= 1000:
                score += 20  # 대형 포지션 가능
            elif market_cap_billion >= 500:
                score += 15
            elif market_cap_billion >= 100:
                score += 10
            else:
                score -= 15  # 포지션 제한
        
        # 가격대 (N값 계산 용이성)
        price = getattr(stock, 'price', None)
        if price:
            if price >= 10000:  # 1만원 이상
                score += 15  # 계산 용이
            elif price >= 5000:
                score += 10
            elif price >= 1000:
                score += 5
        
        return min(max(score, 0), 100)
    
    def _analyze_system_discipline(self, stock) -> float:
        """시스템 준수 분석"""
        score = 50.0
        
        # 명확한 신호 (모호함 제거)
        price_momentum = getattr(stock, 'price_momentum_3m', None) or getattr(stock, 'price_momentum', 0)
        if price_momentum:
            momentum_abs = abs(price_momentum)
            if momentum_abs >= 15:
                score += 25  # 명확한 신호
            elif momentum_abs >= 10:
                score += 20
            elif momentum_abs >= 5:
                score += 15
            else:
                score -= 15  # 모호한 신호
        
        # 일관된 성과 (변동성 대비)
        volatility = getattr(stock, 'volatility', 0)
        if volatility and price_momentum:
            consistency = abs(price_momentum) / max(volatility, 1)
            if consistency >= 0.8:
                score += 20  # 일관된 성과
            elif consistency >= 0.6:
                score += 15
            elif consistency >= 0.4:
                score += 10
        
        # 섹터 명확성 (시스템 적용 용이성)
        sector = getattr(stock, 'sector', None)
        clear_sectors = ['Technology', 'Healthcare', 'Finance', 'Energy', 
                        '반도체', '바이오', '금융', '에너지']
        if sector and any(sector in sector for sector in clear_sectors):
            score += 15  # 명확한 섹터
        
        # 재무 투명성
        pe_ratio = getattr(stock, 'pe_ratio', None)
        roe = getattr(stock, 'roe', None)
        debt_ratio = getattr(stock, 'debt_ratio', None)
        if pe_ratio and roe and debt_ratio:
            score += 10  # 완전한 재무 정보
        
        return min(max(score, 0), 100)
    
    def get_strategy_summary(self, stock: CleanedStockData) -> Dict[str, Any]:
        """전략 요약 정보"""
        analysis = self.analyze_stock_detailed(stock)
        
        return {
            "전략명": self.strategy_name,
            "총점": f"{analysis['total_score']:.1f}/100",
            "터틀분석점수": {
                "추세추종": f"{analysis['scores']['trend_following']:.1f}",
                "돌파신호": f"{analysis['scores']['breakout_signals']:.1f}",
                "위험관리": f"{analysis['scores']['risk_management']:.1f}",
                "포지션사이징": f"{analysis['scores']['position_sizing']:.1f}",
                "시스템준수": f"{analysis['scores']['system_discipline']:.1f}"
            },
            "투자판단": self._get_investment_decision(analysis['total_score']),
            "핵심포인트": self._get_key_points(stock, analysis)
        }
    
    def _get_investment_decision(self, score: float) -> str:
        """투자 판단"""
        if score >= 80:
            return "🟢 강력매수 - 완벽한 터틀 신호"
        elif score >= 70:
            return "🔵 매수 - 좋은 추세 신호"
        elif score >= 60:
            return "🟡 관심 - 추세 확인 필요"
        elif score >= 50:
            return "⚪ 중립 - 신호 대기"
        else:
            return "🔴 회피 - 터틀 조건 부적합"
    
    def _get_key_points(self, stock: CleanedStockData, analysis: Dict[str, Any]) -> List[str]:
        """핵심 포인트"""
        points = []
        scores = analysis['scores']
        
        if scores['trend_following'] >= 70:
            points.append("✅ 강한 추세 추종 신호")
        if scores['breakout_signals'] >= 70:
            points.append("✅ 명확한 돌파 신호")
        if scores['risk_management'] >= 70:
            points.append("✅ 우수한 위험 관리")
        if stock.price_momentum_3m and stock.price_momentum_3m >= 20:
            points.append("✅ 강력한 상승 추세")
        if stock.volatility and stock.volatility <= 30:
            points.append("✅ 적정 변동성")
        
        if scores['trend_following'] < 50:
            points.append("⚠️ 추세 신호 약함")
        if scores['risk_management'] < 50:
            points.append("⚠️ 위험 관리 우려")
        if stock.volatility and stock.volatility > 50:
            points.append("⚠️ 높은 변동성 위험")
        if stock.price_momentum_3m and stock.price_momentum_3m < -10:
            points.append("⚠️ 하락 추세")
        
        return points[:5]  # 최대 5개 포인트 