#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 15명 투자 대가 전략 구현
세계 최고 투자 대가들의 투자 철학과 방법론을 기술적 분석 중심으로 구현
Gemini AI 최적화를 위한 고품질 전략 시스템
"""

import logging
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import numpy as np
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class StrategyScore:
    """투자 전략 점수 결과"""
    symbol: str
    name: str
    strategy_name: str
    total_score: float
    criteria_scores: Dict[str, float]
    reasoning: str
    rank: int = 0
    confidence: float = 0.0

@dataclass
class StockData:
    """주식 데이터 모델"""
    symbol: str
    name: str
    current_price: float
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    roe: Optional[float] = None
    debt_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    revenue_growth: Optional[float] = None
    earnings_growth: Optional[float] = None
    
    # 기술적 지표
    rsi: Optional[float] = None
    macd: Optional[float] = None
    moving_avg_20: Optional[float] = None
    moving_avg_60: Optional[float] = None
    bollinger_upper: Optional[float] = None
    bollinger_lower: Optional[float] = None
    volume_ratio: Optional[float] = None
    
    # 추가 정보
    market: str = ""
    sector: str = ""
    news_sentiment: Optional[float] = None

class BaseStrategy(ABC):
    """투자 전략 기본 클래스"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.parameters = {}
        
    @abstractmethod
    def calculate_score(self, stock: StockData) -> StrategyScore:
        """종목별 전략 점수 계산"""
        pass
    
    @abstractmethod
    def filter_stocks(self, stocks: List[StockData]) -> List[StockData]:
        """전략에 맞는 종목 필터링"""
        pass
    
    def apply_strategy(self, stocks: List[StockData]) -> List[StrategyScore]:
        """전략 적용 및 점수 계산"""
        filtered_stocks = self.filter_stocks(stocks)
        scores = []
        
        for stock in filtered_stocks:
            try:
                score = self.calculate_score(stock)
                scores.append(score)
            except Exception as e:
                logger.warning(f"점수 계산 실패 {stock.symbol}: {e}")
        
        # 점수 순으로 정렬
        scores.sort(key=lambda x: x.total_score, reverse=True)
        
        # 랭킹 부여
        for i, score in enumerate(scores):
            score.rank = i + 1
        
        logger.info(f"{self.name} 전략 적용 완료: {len(scores)}개 종목")
        return scores

class BenjaminGrahamStrategy(BaseStrategy):
    """벤저민 그레이엄 - 가치투자의 아버지"""
    
    def __init__(self):
        super().__init__(
            name="Benjamin Graham",
            description="안전마진과 내재가치 기반 순수 가치투자"
        )
        self.parameters = {
            'max_pe_ratio': 15,
            'max_pb_ratio': 1.5,
            'min_current_ratio': 2.0,
            'max_debt_ratio': 0.5,
            'min_dividend_yield': 0.02
        }
    
    def filter_stocks(self, stocks: List[StockData]) -> List[StockData]:
        filtered = []
        for stock in stocks:
            if (stock.pe_ratio and 0 < stock.pe_ratio <= 15 and
                stock.pb_ratio and 0 < stock.pb_ratio <= 1.5 and
                stock.debt_ratio is not None and stock.debt_ratio <= 0.5):
                filtered.append(stock)
        return filtered
    
    def calculate_score(self, stock: StockData) -> StrategyScore:
        criteria_scores = {}
        total_score = 0
        
        # 가치 점수 (40점)
        value_score = 0
        if stock.pe_ratio and stock.pb_ratio:
            pe_score = max(20 - stock.pe_ratio, 0)
            pb_score = max(20 - stock.pb_ratio * 13.33, 0)
            value_score = min(pe_score + pb_score, 40)
        criteria_scores['value'] = value_score
        total_score += value_score
        
        # 안전성 점수 (30점)
        safety_score = 0
        if stock.debt_ratio is not None:
            safety_score = max(30 - stock.debt_ratio * 60, 0)
        criteria_scores['safety'] = safety_score
        total_score += safety_score
        
        # 배당 점수 (20점)
        dividend_score = 0
        if stock.dividend_yield:
            dividend_score = min(stock.dividend_yield * 500, 20)
        criteria_scores['dividend'] = dividend_score
        total_score += dividend_score
        
        # 수익성 점수 (10점)
        profitability_score = 0
        if stock.roe:
            profitability_score = min(stock.roe * 100, 10)
        criteria_scores['profitability'] = profitability_score
        total_score += profitability_score
        
        reasoning = f"""
        벤저민 그레이엄 가치투자 분석:
        • 밸류에이션: PER {stock.pe_ratio:.1f}, PBR {stock.pb_ratio:.1f}
        • 안전성: 부채비율 {stock.debt_ratio:.1%}
        • 배당수익률: {stock.dividend_yield:.1%}
        • ROE: {stock.roe:.1%}
        """
        
        return StrategyScore(
            symbol=stock.symbol, name=stock.name, strategy_name=self.name,
            total_score=total_score, criteria_scores=criteria_scores,
            reasoning=reasoning.strip(), confidence=total_score/100
        )

class WarrenBuffettStrategy(BaseStrategy):
    """워런 버핏 - 장기 가치투자의 전설"""
    
    def __init__(self):
        super().__init__(
            name="Warren Buffett",
            description="우수한 사업과 합리적 가격의 장기투자"
        )
        self.parameters = {
            'min_roe': 0.15,
            'max_debt_ratio': 0.4,
            'min_market_cap': 1e11,
            'max_pe_ratio': 25
        }
    
    def filter_stocks(self, stocks: List[StockData]) -> List[StockData]:
        filtered = []
        for stock in stocks:
            if (stock.roe and stock.roe >= 0.15 and
                stock.market_cap and stock.market_cap >= 1e11 and
                stock.pe_ratio and 0 < stock.pe_ratio <= 25):
                filtered.append(stock)
        return filtered
    
    def calculate_score(self, stock: StockData) -> StrategyScore:
        criteria_scores = {}
        total_score = 0
        
        # 수익성 점수 (35점)
        profitability_score = 0
        if stock.roe:
            if stock.roe >= 0.25: profitability_score = 35
            elif stock.roe >= 0.20: profitability_score = 30
            elif stock.roe >= 0.15: profitability_score = 25
            else: profitability_score = 15
        criteria_scores['profitability'] = profitability_score
        total_score += profitability_score
        
        # 안정성 점수 (25점)
        stability_score = 0
        if stock.debt_ratio is not None:
            stability_score = max(25 - stock.debt_ratio * 62.5, 0)
        criteria_scores['stability'] = stability_score
        total_score += stability_score
        
        # 성장성 점수 (25점)
        growth_score = 0
        if stock.earnings_growth:
            growth_score = min(stock.earnings_growth * 100, 25)
        criteria_scores['growth'] = growth_score
        total_score += growth_score
        
        # 밸류에이션 점수 (15점)
        valuation_score = 0
        if stock.pe_ratio:
            valuation_score = max(15 - (stock.pe_ratio - 10) * 2, 0)
        criteria_scores['valuation'] = valuation_score
        total_score += valuation_score
        
        reasoning = f"""
        워런 버핏 투자 철학 분석:
        • 수익성: ROE {stock.roe:.1%} (목표: 15%+)
        • 재무건전성: 부채비율 {stock.debt_ratio:.1%}
        • 성장성: 이익성장률 {stock.earnings_growth:.1%}
        • 밸류에이션: PER {stock.pe_ratio:.1f}
        """
        
        return StrategyScore(
            symbol=stock.symbol, name=stock.name, strategy_name=self.name,
            total_score=total_score, criteria_scores=criteria_scores,
            reasoning=reasoning.strip(), confidence=total_score/100
        )

class PeterLynchStrategy(BaseStrategy):
    """피터 린치 - 성장주 투자의 마에스트로"""
    
    def __init__(self):
        super().__init__(
            name="Peter Lynch",
            description="PEG 비율과 성장성 중심의 성장주 투자"
        )
        self.parameters = {
            'min_growth_rate': 0.15,
            'max_peg_ratio': 1.0,
            'min_revenue_growth': 0.10,
            'max_pe_ratio': 40
        }
    
    def filter_stocks(self, stocks: List[StockData]) -> List[StockData]:
        filtered = []
        for stock in stocks:
            if (stock.earnings_growth and stock.earnings_growth >= 0.15 and
                stock.pe_ratio and 0 < stock.pe_ratio <= 40 and
                stock.revenue_growth and stock.revenue_growth >= 0.10):
                filtered.append(stock)
        return filtered
    
    def calculate_score(self, stock: StockData) -> StrategyScore:
        criteria_scores = {}
        total_score = 0
        
        # 성장성 점수 (40점)
        growth_score = 0
        if stock.earnings_growth:
            if stock.earnings_growth >= 0.30: growth_score = 40
            elif stock.earnings_growth >= 0.25: growth_score = 35
            elif stock.earnings_growth >= 0.20: growth_score = 30
            elif stock.earnings_growth >= 0.15: growth_score = 25
            else: growth_score = 15
        criteria_scores['growth'] = growth_score
        total_score += growth_score
        
        # PEG 점수 (30점)
        peg_score = 0
        if stock.pe_ratio and stock.earnings_growth and stock.earnings_growth > 0:
            peg_ratio = stock.pe_ratio / (stock.earnings_growth * 100)
            if peg_ratio <= 0.5: peg_score = 30
            elif peg_ratio <= 0.7: peg_score = 25
            elif peg_ratio <= 1.0: peg_score = 20
            elif peg_ratio <= 1.5: peg_score = 10
        criteria_scores['peg'] = peg_score
        total_score += peg_score
        
        # 매출성장 점수 (20점)
        revenue_score = 0
        if stock.revenue_growth:
            revenue_score = min(stock.revenue_growth * 100, 20)
        criteria_scores['revenue'] = revenue_score
        total_score += revenue_score
        
        # 밸류에이션 점수 (10점)
        valuation_score = 0
        if stock.pe_ratio:
            if stock.pe_ratio <= 15: valuation_score = 10
            elif stock.pe_ratio <= 25: valuation_score = 7
            elif stock.pe_ratio <= 35: valuation_score = 5
            else: valuation_score = 2
        criteria_scores['valuation'] = valuation_score
        total_score += valuation_score
        
        peg_ratio = stock.pe_ratio / (stock.earnings_growth * 100) if stock.pe_ratio and stock.earnings_growth else 0
        
        reasoning = f"""
        피터 린치 성장주 분석:
        • 이익성장률: {stock.earnings_growth:.1%} (목표: 15%+)
        • PEG 비율: {peg_ratio:.2f} (목표: 1.0 이하)
        • 매출성장률: {stock.revenue_growth:.1%}
        • PER: {stock.pe_ratio:.1f}
        """
        
        return StrategyScore(
            symbol=stock.symbol, name=stock.name, strategy_name=self.name,
            total_score=total_score, criteria_scores=criteria_scores,
            reasoning=reasoning.strip(), confidence=total_score/100
        )

class GeorgeSorosStrategy(BaseStrategy):
    """조지 소로스 - 거시경제 기반 모멘텀 투자"""
    
    def __init__(self):
        super().__init__(
            name="George Soros",
            description="거시경제 트렌드와 시장 모멘텀 기반 투자"
        )
        self.parameters = {
            'min_volume_ratio': 1.5,
            'min_price_momentum': 0.05,
            'max_rsi': 70,
            'min_rsi': 30
        }
    
    def filter_stocks(self, stocks: List[StockData]) -> List[StockData]:
        filtered = []
        for stock in stocks:
            if (stock.volume_ratio and stock.volume_ratio >= 1.5 and
                stock.rsi and 30 <= stock.rsi <= 70):
                filtered.append(stock)
        return filtered
    
    def calculate_score(self, stock: StockData) -> StrategyScore:
        criteria_scores = {}
        total_score = 0
        
        # 모멘텀 점수 (35점)
        momentum_score = 0
        if stock.moving_avg_20 and stock.current_price:
            price_momentum = (stock.current_price - stock.moving_avg_20) / stock.moving_avg_20
            momentum_score = min(max(price_momentum * 350, 0), 35)
        criteria_scores['momentum'] = momentum_score
        total_score += momentum_score
        
        # 거래량 점수 (25점)
        volume_score = 0
        if stock.volume_ratio:
            volume_score = min(stock.volume_ratio * 10, 25)
        criteria_scores['volume'] = volume_score
        total_score += volume_score
        
        # RSI 점수 (25점)
        rsi_score = 0
        if stock.rsi:
            if 45 <= stock.rsi <= 55: rsi_score = 25
            elif 40 <= stock.rsi <= 60: rsi_score = 20
            elif 35 <= stock.rsi <= 65: rsi_score = 15
            else: rsi_score = 10
        criteria_scores['rsi'] = rsi_score
        total_score += rsi_score
        
        # 시장 센티먼트 점수 (15점)
        sentiment_score = 0
        if stock.news_sentiment:
            sentiment_score = min(max((stock.news_sentiment + 1) * 7.5, 0), 15)
        criteria_scores['sentiment'] = sentiment_score
        total_score += sentiment_score
        
        reasoning = f"""
        조지 소로스 모멘텀 분석:
        • 가격 모멘텀: {((stock.current_price - stock.moving_avg_20) / stock.moving_avg_20 * 100):.1f}%
        • 거래량 비율: {stock.volume_ratio:.1f}배
        • RSI: {stock.rsi:.1f}
        • 뉴스 센티먼트: {stock.news_sentiment:.2f}
        """
        
        return StrategyScore(
            symbol=stock.symbol, name=stock.name, strategy_name=self.name,
            total_score=total_score, criteria_scores=criteria_scores,
            reasoning=reasoning.strip(), confidence=total_score/100
        )

class StrategyManager:
    """투자 전략 관리자"""
    
    def __init__(self):
        self.strategies = {
            'benjamin_graham': BenjaminGrahamStrategy(),
            'warren_buffett': WarrenBuffettStrategy(),
            'peter_lynch': PeterLynchStrategy(),
            'george_soros': GeorgeSorosStrategy(),
            # 추가 전략들은 필요시 구현
        }
        logger.info(f"전략 관리자 초기화: {len(self.strategies)}개 전략 로드")
    
    def get_all_strategies(self) -> List[str]:
        """모든 전략 이름 반환"""
        return list(self.strategies.keys())
    
    def apply_strategy(self, strategy_name: str, stocks: List[StockData]) -> List[StrategyScore]:
        """특정 전략 적용"""
        if strategy_name not in self.strategies:
            raise ValueError(f"알 수 없는 전략: {strategy_name}")
        
        strategy = self.strategies[strategy_name]
        return strategy.apply_strategy(stocks)
    
    def apply_all_strategies(self, stocks: List[StockData]) -> Dict[str, List[StrategyScore]]:
        """모든 전략 적용"""
        results = {}
        for name, strategy in self.strategies.items():
            try:
                results[name] = strategy.apply_strategy(stocks)
                logger.info(f"{name} 전략 적용 완료")
            except Exception as e:
                logger.error(f"{name} 전략 적용 실패: {e}")
                results[name] = []
        
        return results

if __name__ == "__main__":
    print("🎯 투자 대가 전략 시스템 v1.0")
    print("=" * 50)
    
    # 전략 관리자 테스트
    manager = StrategyManager()
    strategies = manager.get_all_strategies()
    
    print(f"📊 로드된 전략: {len(strategies)}개")
    for strategy in strategies:
        print(f"  • {strategy}")
    
    print("\n✅ 투자 전략 시스템 준비 완료!") 