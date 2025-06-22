"""
조엘 그린블라트 마법공식(Magic Formula) 전략 구현

핵심 원칙:
1. 수익률(Earnings Yield) = EBIT / Enterprise Value
2. 자본수익률(Return on Capital) = EBIT / (Net Working Capital + Net Fixed Assets)
3. 두 지표의 순위를 합산하여 최상위 종목 선택
4. 시가총액 5천만 달러 이상 종목 대상
5. 최소 1년 보유 원칙
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from ..common import BaseStrategy, StrategyResult

logger = logging.getLogger(__name__)

class GreenblattStrategy(BaseStrategy):
    """조엘 그린블라트 마법공식 전략 클래스"""
    
    def __init__(self):
        super().__init__()
        self.strategy_name = "조엘 그린블라트 (Joel Greenblatt)"
        self.description = "마법공식 - 수익률과 자본수익률 기반 가치투자"
        
        self.weights = {
            'earnings_yield_score': 0.4,  # 수익률 점수
            'roic_score': 0.4,           # 자본수익률 점수
            'quality_score': 0.2         # 품질 점수
        }
        self.min_market_cap = 500  # 최소 시가총액 (억원)
    
    def analyze_stock(self, stock) -> StrategyResult:
        """그린블라트 마법공식 분석"""
        try:
            scores = {}
            analysis_details = {}
            
            # 수익률 계산 및 점수화
            earnings_yield = self._calculate_earnings_yield(stock)
            earnings_yield_score = self._score_earnings_yield(earnings_yield)
            scores['earnings_yield_score'] = earnings_yield_score
            analysis_details['earnings_yield'] = earnings_yield
            
            # 자본수익률 계산 및 점수화
            roic = self._calculate_roic(stock)
            roic_score = self._score_roic(roic)
            scores['roic_score'] = roic_score
            analysis_details['roic'] = roic
            
            # 품질 점수 계산
            quality_score = self._calculate_quality_score(stock)
            scores['quality_score'] = quality_score
            analysis_details['quality_score'] = quality_score
            
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
            logger.error(f"그린블라트 전략 분석 중 오류: {e}")
            return self._create_error_result()
    
    def _calculate_earnings_yield(self, stock) -> Optional[float]:
        """수익률(Earnings Yield) 계산"""
        # EBIT 추정 (영업이익 또는 순이익 기반)
        ebit = getattr(stock, 'ebit', None) or getattr(stock, 'operating_income', None)
        if not ebit:
            net_income = getattr(stock, 'net_income', 0)
            if net_income > 0:
                ebit = net_income * 1.3  # 추정
        
        # Enterprise Value 추정
        market_cap = getattr(stock, 'market_cap', 0)
        net_debt = getattr(stock, 'net_debt', 0)
        enterprise_value = market_cap + net_debt
        
        if not ebit or not enterprise_value or enterprise_value <= 0:
            return None
        
        return (ebit / enterprise_value) * 100
    
    def _calculate_roic(self, stock) -> Optional[float]:
        """자본수익률(Return on Invested Capital) 계산"""
        # EBIT 추정
        ebit = getattr(stock, 'ebit', None) or getattr(stock, 'operating_income', None)
        if not ebit:
            net_income = getattr(stock, 'net_income', 0)
            if net_income > 0:
                ebit = net_income * 1.3  # 추정
        
        # Invested Capital 추정
        total_assets = getattr(stock, 'total_assets', 0)
        cash = getattr(stock, 'cash', 0)
        current_liabilities = getattr(stock, 'current_liabilities', 0)
        
        if total_assets > 0:
            invested_capital = total_assets - cash - (current_liabilities * 0.5)
        else:
            # 대안: 자기자본 + 부채
            equity = getattr(stock, 'equity', 0)
            total_debt = getattr(stock, 'total_debt', 0)
            invested_capital = equity + total_debt
        
        if not ebit or not invested_capital or invested_capital <= 0:
            return None
        
        return (ebit / invested_capital) * 100
    
    def _score_earnings_yield(self, earnings_yield: Optional[float]) -> float:
        """수익률 점수화"""
        if earnings_yield is None:
            return 0
        
        # 수익률 기준 점수화
        if earnings_yield >= 15:
            return 100
        elif earnings_yield >= 12:
            return 90
        elif earnings_yield >= 10:
            return 80
        elif earnings_yield >= 8:
            return 70
        elif earnings_yield >= 6:
            return 60
        elif earnings_yield >= 4:
            return 50
        elif earnings_yield >= 2:
            return 40
        elif earnings_yield > 0:
            return 30
        else:
            return 0
    
    def _score_roic(self, roic: Optional[float]) -> float:
        """자본수익률 점수화"""
        if roic is None:
            return 0
        
        # ROIC 기준 점수화
        if roic >= 25:
            return 100
        elif roic >= 20:
            return 90
        elif roic >= 15:
            return 80
        elif roic >= 12:
            return 70
        elif roic >= 10:
            return 60
        elif roic >= 8:
            return 50
        elif roic >= 5:
            return 40
        elif roic > 0:
            return 30
        else:
            return 0
    
    def _calculate_quality_score(self, stock) -> float:
        """품질 점수 계산"""
        score = 50.0
        
        # 매출 성장률
        revenue_growth = getattr(stock, 'revenue_growth', 0)
        if revenue_growth > 0:
            score += min(25, revenue_growth * 2.5)
        elif revenue_growth < 0:
            score -= min(25, abs(revenue_growth) * 2)
        
        # 부채비율
        debt_ratio = getattr(stock, 'debt_ratio', 50)
        if debt_ratio <= 30:
            score += 15
        elif debt_ratio <= 50:
            score += 10
        elif debt_ratio <= 100:
            score += 5
        else:
            score -= 15
        
        # ROE
        roe = getattr(stock, 'roe', 0)
        if roe >= 15:
            score += 10
        elif roe >= 10:
            score += 5
        elif roe < 0:
            score -= 10
        
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
        
        earnings_yield = analysis_details.get('earnings_yield')
        roic = analysis_details.get('roic')
        
        if earnings_yield and earnings_yield >= 10:
            key_points.append(f"높은 수익률: {earnings_yield:.1f}%")
        
        if roic and roic >= 15:
            key_points.append(f"우수한 자본수익률: {roic:.1f}%")
        
        if scores.get('quality_score', 0) >= 80:
            key_points.append("높은 품질 점수")
        
        if scores.get('earnings_yield_score', 0) >= 80 and scores.get('roic_score', 0) >= 80:
            key_points.append("마법공식 조건 충족")
        
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
    
    def screen_stocks(self, stocks_data: List[Dict]) -> List[Dict]:
        """종목 스크리닝 (마법공식 기준)"""
        # 1단계: 기본 필터링
        filtered_stocks = []
        
        for stock in stocks_data:
            market_cap = stock.get('market_cap', 0)
            ebit = stock.get('ebit')
            
            # 최소 시가총액 및 양수 EBIT 조건
            if (market_cap >= self.min_market_cap and 
                ebit and ebit > 0):
                filtered_stocks.append(stock)
        
        if not filtered_stocks:
            return []
        
        # 2단계: 마법공식 적용
        analyzed_stocks = []
        for stock in filtered_stocks:
            analysis = self.analyze_stock(stock)
            stock.update(analysis.scores)
            stock.update(analysis.analysis_details)
            analyzed_stocks.append(stock)
        
        # 3단계: 마법공식 점수순 정렬
        return sorted(analyzed_stocks, key=lambda x: x['earnings_yield_score'] + x['roic_score'], reverse=True)
    
    def get_portfolio_allocation(self, selected_stocks: List[Dict], portfolio_size: int = 30) -> List[Dict]:
        """포트폴리오 구성 (동일 가중)"""
        if len(selected_stocks) < portfolio_size:
            portfolio_size = len(selected_stocks)
        
        weight_per_stock = 100 / portfolio_size
        
        portfolio = []
        for i, stock in enumerate(selected_stocks[:portfolio_size]):
            stock['weight'] = weight_per_stock
            stock['rank'] = i + 1
            portfolio.append(stock)
        
        return portfolio
    
    def calculate_expected_return(self, stock_data: Dict) -> float:
        """예상 수익률 계산 (마법공식 기반)"""
        earnings_yield = stock_data.get('earnings_yield', 0)
        roic = stock_data.get('roic', 0)
        
        if not earnings_yield or not roic:
            return 0
        
        # 단순화된 예상 수익률 = (수익률 + 자본수익률) / 2
        return (earnings_yield + min(roic, 50)) / 2  # ROIC는 50%로 캡핑
    
    def get_strategy_description(self) -> str:
        """전략 설명 반환"""
        return """
        조엘 그린블라트 마법공식 전략
        
        핵심 지표:
        • 수익률(Earnings Yield) = EBIT / Enterprise Value
        • 자본수익률(ROIC) = EBIT / Invested Capital
        
        선별 기준:
        • 시가총액 500억원 이상
        • 양수 EBIT 보유
        • 두 지표 순위 합산 상위 종목
        
        투자 원칙:
        • 20-30개 종목 동일 가중 분산투자
        • 최소 1년 보유
        • 매년 리밸런싱
        
        투자 철학:
        "좋은 회사를 싼 가격에 사라"
        """
    
    def get_rebalancing_schedule(self) -> str:
        """리밸런싱 일정 안내"""
        return """
        마법공식 리밸런싱 가이드:
        
        • 매년 동일한 시기에 리밸런싱 실시
        • 보유 종목 중 1년 미만 보유 종목은 유지
        • 새로운 마법공식 순위에 따라 종목 교체
        • 세금 효율성을 위해 손실 종목 우선 매도
        • 거래비용 최소화를 위한 점진적 교체
        """ 