"""
피터 린치 성장투자 전략 구현

주요 원칙:
1. PEG 비율 중심 분석 (PEG < 1.0 선호)
2. 높은 성장률과 합리적인 밸류에이션
3. 이해하기 쉬운 비즈니스 모델
4. 강력한 브랜드와 시장 지위
5. 수익 성장의 지속가능성
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from ..common import BaseStrategy, StrategyResult, CommonIndicators

logger = logging.getLogger(__name__)

class LynchStrategy(BaseStrategy):
    """피터 린치 성장투자 전략 클래스"""
    
    def __init__(self):
        super().__init__()
        self.strategy_name = "피터 린치 (Peter Lynch)"
        self.description = "PEG 비율 중심의 성장투자 전략"
        self.indicators = CommonIndicators()
        
        # 가중치 설정
        self.weights = {
            'growth_score': 0.35,     # 성장성 점수 (가장 중요)
            'peg_score': 0.25,        # PEG 비율 점수
            'profitability_score': 0.2, # 수익성 점수
            'momentum_score': 0.2     # 모멘텀 점수
        }
    
    def analyze_stock(self, stock) -> StrategyResult:
        """린치 전략으로 주식 분석"""
        try:
            scores = {}
            analysis_details = {}
            
            # 1. 성장성 분석 (35%)
            growth_score = self.calculate_growth_score(stock)
            scores['growth_score'] = growth_score
            analysis_details['growth_score'] = growth_score
            
            # 2. PEG 비율 분석 (25%)
            peg_score = self.calculate_peg_score(stock)
            scores['peg_score'] = peg_score
            analysis_details['peg_score'] = peg_score
            
            # 3. 수익성 분석 (20%)
            profitability_score = self.calculate_profitability_score(stock)
            scores['profitability_score'] = profitability_score
            analysis_details['profitability_score'] = profitability_score
            
            # 4. 모멘텀 분석 (20%)
            momentum_score = self.calculate_momentum_score(stock)
            scores['momentum_score'] = momentum_score
            analysis_details['momentum_score'] = momentum_score
            
            # 품질 지표
            quality_indicators = self.calculate_quality_indicators(stock)
            analysis_details['quality_indicators'] = quality_indicators
            
            # 총점 계산
            total_score = sum(scores[key] * self.weights[key] for key in scores)
            
            # 품질 보너스/페널티
            quality_bonus = sum(quality_indicators.values()) / len(quality_indicators) * 0.1
            total_score += quality_bonus
            total_score = min(100, max(0, total_score))
            
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
            logger.error(f"린치 전략 분석 중 오류: {e}")
            return self._create_error_result()
    
    def calculate_peg_score(self, stock) -> float:
        """PEG 비율 점수 계산 (1.0 이하 우수)"""
        pe_ratio = getattr(stock, 'per', None) or getattr(stock, 'pe_ratio', None)
        earnings_growth = getattr(stock, 'earnings_growth_rate', None) or getattr(stock, 'earnings_growth', None)
        
        if not pe_ratio or pe_ratio <= 0 or not earnings_growth or earnings_growth <= 0:
            return 0
        
        peg_ratio = pe_ratio / earnings_growth
        
        # PEG 비율별 점수 계산
        if peg_ratio <= 0.5:
            return 100  # 매우 우수
        elif peg_ratio <= 1.0:
            return 100 - (peg_ratio - 0.5) * 100  # 우수
        elif peg_ratio <= 1.5:
            return 50 - (peg_ratio - 1.0) * 60   # 보통
        else:
            return max(0, 20 - (peg_ratio - 1.5) * 10)  # 나쁨
    
    def calculate_growth_score(self, stock) -> float:
        """성장성 점수 계산 (높은 성장률 선호)"""
        earnings_growth = getattr(stock, 'earnings_growth_rate', None) or getattr(stock, 'earnings_growth', None) or 0
        revenue_growth = getattr(stock, 'revenue_growth_rate', None) or getattr(stock, 'revenue_growth', None) or 0
        
        scores = []
        
        # 수익 성장률 점수 (15-30% 이상적)
        if earnings_growth > 0:
            if 15 <= earnings_growth <= 30:
                growth_score = 100
            elif 10 <= earnings_growth < 15:
                growth_score = 70 + (earnings_growth - 10) * 6
            elif 30 < earnings_growth <= 50:
                growth_score = 100 - (earnings_growth - 30) * 2
            elif earnings_growth > 50:
                growth_score = 60 - (earnings_growth - 50) * 0.5  # 너무 높은 성장률은 지속가능성 의심
            else:
                growth_score = earnings_growth * 7  # 10% 미만
            scores.append(min(100, max(0, growth_score)))
        
        # 매출 성장률 점수
        if revenue_growth > 0:
            rev_score = min(100, (revenue_growth / 20) * 100)  # 20% 기준
            scores.append(rev_score)
        
        return sum(scores) / len(scores) if scores else 0
    
    def calculate_profitability_score(self, stock) -> float:
        """수익성 점수 계산"""
        roe = getattr(stock, 'roe', 0)
        roa = getattr(stock, 'roa', 0)
        operating_margin = getattr(stock, 'operating_margin', 0)
        net_margin = getattr(stock, 'net_margin', 0)
        
        scores = []
        
        # ROE 점수 (20% 이상 우수)
        if roe > 0:
            roe_score = min(100, (roe / 20) * 100)
            scores.append(roe_score)
        
        # ROA 점수 (10% 이상 우수)
        if roa > 0:
            roa_score = min(100, (roa / 10) * 100)
            scores.append(roa_score)
        
        # 영업이익률 점수 (15% 이상 우수)
        if operating_margin > 0:
            op_score = min(100, (operating_margin / 15) * 100)
            scores.append(op_score)
        
        # 순이익률 점수 (10% 이상 우수)
        if net_margin > 0:
            net_score = min(100, (net_margin / 10) * 100)
            scores.append(net_score)
        
        return sum(scores) / len(scores) if scores else 0
    
    def calculate_momentum_score(self, stock) -> float:
        """모멘텀 점수 계산 (주가 및 실적 모멘텀)"""
        price_momentum = getattr(stock, 'price_momentum', 0)  # 3개월 수익률
        earnings_surprise = getattr(stock, 'earnings_surprise', 0)  # 실적 서프라이즈
        analyst_revision = getattr(stock, 'analyst_revision', 0)  # 애널리스트 전망 수정
        
        scores = []
        
        # 주가 모멘텀 점수 (10-30% 상승 이상적)
        if price_momentum is not None:
            if 10 <= price_momentum <= 30:
                momentum_score = 100
            elif 5 <= price_momentum < 10:
                momentum_score = 50 + (price_momentum - 5) * 10
            elif 30 < price_momentum <= 50:
                momentum_score = 100 - (price_momentum - 30) * 2
            elif price_momentum > 0:
                momentum_score = price_momentum * 5
            else:
                momentum_score = max(0, 50 + price_momentum * 2)  # 하락시 감점
            scores.append(min(100, max(0, momentum_score)))
        
        # 실적 서프라이즈 점수
        if earnings_surprise is not None:
            surprise_score = min(100, max(0, 50 + earnings_surprise * 5))
            scores.append(surprise_score)
        
        # 애널리스트 전망 수정 점수
        if analyst_revision is not None:
            revision_score = min(100, max(0, 50 + analyst_revision * 10))
            scores.append(revision_score)
        
        return sum(scores) / len(scores) if scores else 50
    
    def calculate_quality_indicators(self, stock) -> Dict:
        """린치가 중요시하는 품질 지표들"""
        indicators = {}
        
        # 부채비율 (50% 이하 선호)
        debt_ratio = getattr(stock, 'debt_ratio', 100)
        indicators['debt_health'] = max(0, 100 - debt_ratio * 2) if debt_ratio < 50 else 0
        
        # 현금 보유 상황
        cash_ratio = getattr(stock, 'cash_ratio', 0)
        indicators['cash_strength'] = min(100, cash_ratio * 20) if cash_ratio else 50
        
        # 시장 지위 (주관적 지표, 시가총액 기준 추정)
        market_cap = getattr(stock, 'market_cap', 0)
        if market_cap > 10000:  # 대형주
            indicators['market_position'] = 80
        elif market_cap > 1000:  # 중형주
            indicators['market_position'] = 90
        else:  # 소형주
            indicators['market_position'] = 70
        
        return indicators
    
    def categorize_stock_type(self, stock) -> str:
        """린치의 주식 분류 (성장주, 가치주 등)"""
        pe_ratio = getattr(stock, 'per', None) or getattr(stock, 'pe_ratio', None)
        earnings_growth = getattr(stock, 'earnings_growth_rate', None) or getattr(stock, 'earnings_growth', None)
        
        if pe_ratio and earnings_growth:
            peg_ratio = pe_ratio / earnings_growth if earnings_growth > 0 else None
            
            if peg_ratio and peg_ratio < 1.0 and earnings_growth > 15:
                return "성장주 (Growth Stock)"
            elif peg_ratio and peg_ratio < 1.5 and earnings_growth > 10:
                return "중간 성장주 (Moderate Growth)"
            elif pe_ratio < 15:
                return "가치주 (Value Stock)"
            else:
                return "고평가주 (Overvalued)"
        
        return "분류 불가"
    
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
        
        if scores.get('peg_score', 0) >= 80:
            key_points.append("PEG 비율이 매우 우수함")
        
        if scores.get('growth_score', 0) >= 80:
            key_points.append("높은 성장률 보유")
        
        if scores.get('profitability_score', 0) >= 80:
            key_points.append("뛰어난 수익성")
        
        if scores.get('momentum_score', 0) >= 80:
            key_points.append("강한 모멘텀")
        
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