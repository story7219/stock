"""
투자 대가 전략 모듈
워런 버핏, 피터 린치, 벤저민 그레이엄 전략 구현
"""

import logging
from typing import List, Dict, Optional, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
import pandas as pd
import numpy as np
from .data_collector import StockData

logger = logging.getLogger(__name__)

@dataclass
class StrategyScore:
    """전략별 점수 클래스"""
    symbol: str
    name: str
    strategy_name: str
    total_score: float
    criteria_scores: Dict[str, float]
    reasoning: str
    rank: int = 0

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

class WarrenBuffettStrategy(BaseStrategy):
    """워런 버핏 전략 - 우량주 중심"""
    
    def __init__(self):
        super().__init__(
            name="Warren Buffett Strategy",
            description="ROE, 부채비율 등을 중심으로 한 우량주 선별 전략"
        )
        self.parameters = {
            'min_market_cap': 1e12,  # 1조원 이상
            'max_pe_ratio': 20,      # PER 20 이하
            'min_roe': 0.15,         # ROE 15% 이상
            'max_debt_ratio': 0.5,   # 부채비율 50% 이하
            'min_dividend_yield': 0.02,  # 배당수익률 2% 이상
        }
    
    def filter_stocks(self, stocks: List[StockData]) -> List[StockData]:
        """버핏 전략 필터링"""
        filtered = []
        
        for stock in stocks:
            # 시가총액 조건
            if stock.market_cap and stock.market_cap < self.parameters['min_market_cap']:
                continue
                
            # PER 조건
            if stock.pe_ratio and stock.pe_ratio > self.parameters['max_pe_ratio']:
                continue
                
            # 배당수익률 조건
            if stock.dividend_yield and stock.dividend_yield < self.parameters['min_dividend_yield']:
                continue
                
            filtered.append(stock)
        
        return filtered
    
    def calculate_score(self, stock: StockData) -> StrategyScore:
        """버핏 전략 점수 계산"""
        criteria_scores = {}
        total_score = 0
        
        # 시가총액 점수 (40점)
        if stock.market_cap:
            if stock.market_cap >= 5e12:  # 5조원 이상
                market_cap_score = 40
            elif stock.market_cap >= 1e12:  # 1조원 이상
                market_cap_score = 30
            else:
                market_cap_score = 10
        else:
            market_cap_score = 0
        criteria_scores['market_cap'] = market_cap_score
        total_score += market_cap_score
        
        # PER 점수 (25점)
        if stock.pe_ratio:
            if stock.pe_ratio <= 10:
                pe_score = 25
            elif stock.pe_ratio <= 15:
                pe_score = 20
            elif stock.pe_ratio <= 20:
                pe_score = 15
            else:
                pe_score = 5
        else:
            pe_score = 0
        criteria_scores['pe_ratio'] = pe_score
        total_score += pe_score
        
        # PBR 점수 (15점)
        if stock.pb_ratio:
            if stock.pb_ratio <= 1.0:
                pb_score = 15
            elif stock.pb_ratio <= 1.5:
                pb_score = 12
            elif stock.pb_ratio <= 2.0:
                pb_score = 8
            else:
                pb_score = 3
        else:
            pb_score = 0
        criteria_scores['pb_ratio'] = pb_score
        total_score += pb_score
        
        # 배당수익률 점수 (20점)
        if stock.dividend_yield:
            if stock.dividend_yield >= 0.05:  # 5% 이상
                dividend_score = 20
            elif stock.dividend_yield >= 0.03:  # 3% 이상
                dividend_score = 15
            elif stock.dividend_yield >= 0.02:  # 2% 이상
                dividend_score = 10
            else:
                dividend_score = 5
        else:
            dividend_score = 0
        criteria_scores['dividend_yield'] = dividend_score
        total_score += dividend_score
        
        reasoning = f"""
        워런 버핏 전략 분석:
        - 시가총액: {stock.market_cap/1e12:.1f}조원 ({market_cap_score}점)
        - PER: {stock.pe_ratio:.1f} ({pe_score}점)
        - PBR: {stock.pb_ratio:.1f} ({pb_score}점)
        - 배당수익률: {stock.dividend_yield*100:.1f}% ({dividend_score}점)
        - 우량주 특성을 기반으로 한 안정적 투자 대상
        """
        
        return StrategyScore(
            symbol=stock.symbol,
            name=stock.name,
            strategy_name=self.name,
            total_score=total_score,
            criteria_scores=criteria_scores,
            reasoning=reasoning.strip()
        )

class PeterLynchStrategy(BaseStrategy):
    """피터 린치 전략 - 성장주 중심"""
    
    def __init__(self):
        super().__init__(
            name="Peter Lynch Strategy",
            description="성장률, PEG Ratio 등을 중심으로 한 성장주 선별 전략"
        )
        self.parameters = {
            'max_peg_ratio': 1.5,    # PEG 비율 1.5 이하
            'min_growth_rate': 0.15,  # 성장률 15% 이상
            'max_pe_ratio': 30,      # PER 30 이하
            'min_volume': 100000,    # 최소 거래량
        }
    
    def filter_stocks(self, stocks: List[StockData]) -> List[StockData]:
        """린치 전략 필터링"""
        filtered = []
        
        for stock in stocks:
            # 거래량 조건
            if stock.volume < self.parameters['min_volume']:
                continue
                
            # PER 조건 (성장주이므로 조금 더 관대)
            if stock.pe_ratio and stock.pe_ratio > self.parameters['max_pe_ratio']:
                continue
                
            # 기술적 분석 - 상승 추세 확인
            if stock.moving_avg_20 and stock.moving_avg_60:
                if stock.moving_avg_20 <= stock.moving_avg_60:  # 단기 < 장기 이평선
                    continue
                    
            filtered.append(stock)
        
        return filtered
    
    def calculate_score(self, stock: StockData) -> StrategyScore:
        """린치 전략 점수 계산"""
        criteria_scores = {}
        total_score = 0
        
        # 성장성 점수 (40점) - 기술적 지표로 대체
        growth_score = 0
        if stock.rsi and stock.rsi < 70:  # RSI 과매수 구간 아님
            growth_score += 15
        if stock.macd and stock.macd_signal and stock.macd > stock.macd_signal:  # MACD 골든크로스
            growth_score += 15
        if stock.moving_avg_20 and stock.moving_avg_60 and stock.moving_avg_20 > stock.moving_avg_60:
            growth_score += 10  # 상승 추세
        criteria_scores['growth_potential'] = growth_score
        total_score += growth_score
        
        # PER 점수 (25점)
        if stock.pe_ratio:
            if 10 <= stock.pe_ratio <= 20:
                pe_score = 25
            elif stock.pe_ratio <= 25:
                pe_score = 20
            elif stock.pe_ratio <= 30:
                pe_score = 15
            else:
                pe_score = 5
        else:
            pe_score = 10
        criteria_scores['pe_ratio'] = pe_score
        total_score += pe_score
        
        # 거래량 점수 (20점)
        if stock.volume >= 1000000:
            volume_score = 20
        elif stock.volume >= 500000:
            volume_score = 15
        elif stock.volume >= 100000:
            volume_score = 10
        else:
            volume_score = 5
        criteria_scores['volume'] = volume_score
        total_score += volume_score
        
        # 모멘텀 점수 (15점)
        momentum_score = 0
        if stock.price and stock.moving_avg_20:
            price_vs_ma20 = (stock.price - stock.moving_avg_20) / stock.moving_avg_20
            if price_vs_ma20 > 0.05:  # 5% 이상 상승
                momentum_score = 15
            elif price_vs_ma20 > 0:
                momentum_score = 10
            else:
                momentum_score = 5
        criteria_scores['momentum'] = momentum_score
        total_score += momentum_score
        
        reasoning = f"""
        피터 린치 전략 분석:
        - 성장 잠재력: {growth_score}점 (RSI, MACD, 추세 분석)
        - PER: {stock.pe_ratio:.1f} ({pe_score}점)
        - 거래량: {stock.volume:,}주 ({volume_score}점)
        - 모멘텀: {momentum_score}점
        - 성장 가능성이 높은 종목으로 판단
        """
        
        return StrategyScore(
            symbol=stock.symbol,
            name=stock.name,
            strategy_name=self.name,
            total_score=total_score,
            criteria_scores=criteria_scores,
            reasoning=reasoning.strip()
        )

class BenjaminGrahamStrategy(BaseStrategy):
    """벤저민 그레이엄 전략 - 가치주 중심"""
    
    def __init__(self):
        super().__init__(
            name="Benjamin Graham Strategy",
            description="저PBR, 저PER, 안전마진 등을 중심으로 한 가치주 선별 전략"
        )
        self.parameters = {
            'max_pe_ratio': 15,      # PER 15 이하
            'max_pb_ratio': 1.5,     # PBR 1.5 이하
            'min_current_ratio': 2.0, # 유동비율 2.0 이상
            'safety_margin': 0.33,   # 안전마진 33%
        }
    
    def filter_stocks(self, stocks: List[StockData]) -> List[StockData]:
        """그레이엄 전략 필터링"""
        filtered = []
        
        for stock in stocks:
            # PER 조건
            if stock.pe_ratio and stock.pe_ratio > self.parameters['max_pe_ratio']:
                continue
                
            # PBR 조건
            if stock.pb_ratio and stock.pb_ratio > self.parameters['max_pb_ratio']:
                continue
                
            # 기술적 분석 - 저평가 구간 확인
            if stock.bollinger_lower and stock.price:
                if stock.price > stock.bollinger_lower * 1.1:  # 볼린저 하단 근처가 아님
                    continue
                    
            filtered.append(stock)
        
        return filtered
    
    def calculate_score(self, stock: StockData) -> StrategyScore:
        """그레이엄 전략 점수 계산"""
        criteria_scores = {}
        total_score = 0
        
        # PER 점수 (30점)
        if stock.pe_ratio:
            if stock.pe_ratio <= 8:
                pe_score = 30
            elif stock.pe_ratio <= 12:
                pe_score = 25
            elif stock.pe_ratio <= 15:
                pe_score = 20
            else:
                pe_score = 10
        else:
            pe_score = 0
        criteria_scores['pe_ratio'] = pe_score
        total_score += pe_score
        
        # PBR 점수 (30점)
        if stock.pb_ratio:
            if stock.pb_ratio <= 0.7:
                pb_score = 30
            elif stock.pb_ratio <= 1.0:
                pb_score = 25
            elif stock.pb_ratio <= 1.3:
                pb_score = 20
            elif stock.pb_ratio <= 1.5:
                pb_score = 15
            else:
                pb_score = 5
        else:
            pb_score = 0
        criteria_scores['pb_ratio'] = pb_score
        total_score += pb_score
        
        # 안전마진 점수 (25점) - 기술적 지표로 대체
        safety_score = 0
        if stock.bollinger_lower and stock.price:
            distance_from_lower = (stock.price - stock.bollinger_lower) / stock.bollinger_lower
            if distance_from_lower <= 0.05:  # 볼린저 하단 5% 이내
                safety_score = 25
            elif distance_from_lower <= 0.10:  # 볼린저 하단 10% 이내
                safety_score = 20
            elif distance_from_lower <= 0.15:
                safety_score = 15
            else:
                safety_score = 5
        criteria_scores['safety_margin'] = safety_score
        total_score += safety_score
        
        # 배당 점수 (15점)
        if stock.dividend_yield:
            if stock.dividend_yield >= 0.04:  # 4% 이상
                dividend_score = 15
            elif stock.dividend_yield >= 0.03:  # 3% 이상
                dividend_score = 12
            elif stock.dividend_yield >= 0.02:  # 2% 이상
                dividend_score = 8
            else:
                dividend_score = 3
        else:
            dividend_score = 0
        criteria_scores['dividend_yield'] = dividend_score
        total_score += dividend_score
        
        reasoning = f"""
        벤저민 그레이엄 전략 분석:
        - PER: {stock.pe_ratio:.1f} ({pe_score}점)
        - PBR: {stock.pb_ratio:.1f} ({pb_score}점)
        - 안전마진: {safety_score}점 (기술적 분석 기반)
        - 배당수익률: {stock.dividend_yield*100:.1f}% ({dividend_score}점)
        - 내재가치 대비 저평가된 안전한 투자 대상
        """
        
        return StrategyScore(
            symbol=stock.symbol,
            name=stock.name,
            strategy_name=self.name,
            total_score=total_score,
            criteria_scores=criteria_scores,
            reasoning=reasoning.strip()
        )

class StrategyManager:
    """전략 관리자"""
    
    def __init__(self):
        self.strategies = {
            'buffett': WarrenBuffettStrategy(),
            'lynch': PeterLynchStrategy(),
            'graham': BenjaminGrahamStrategy()
        }
        
    def apply_all_strategies(self, stocks: List[StockData]) -> Dict[str, List[StrategyScore]]:
        """모든 전략 적용"""
        results = {}
        
        for strategy_key, strategy in self.strategies.items():
            try:
                logger.info(f"{strategy.name} 전략 적용 시작")
                scores = strategy.apply_strategy(stocks)
                results[strategy_key] = scores
                logger.info(f"{strategy.name} 전략 완료: {len(scores)}개 종목 평가")
            except Exception as e:
                logger.error(f"{strategy.name} 전략 적용 실패: {e}")
                results[strategy_key] = []
        
        return results
    
    def get_top_candidates(self, strategy_results: Dict[str, List[StrategyScore]], 
                          top_n: int = 20) -> Dict[str, List[StrategyScore]]:
        """각 전략별 상위 후보 추출"""
        top_candidates = {}
        
        for strategy_name, scores in strategy_results.items():
            top_candidates[strategy_name] = scores[:top_n]
            
        return top_candidates
    
    def combine_strategy_scores(self, strategy_results: Dict[str, List[StrategyScore]]) -> List[Dict]:
        """전략별 점수를 종합하여 통합 후보군 생성"""
        all_stocks = {}
        
        # 모든 전략의 점수를 수집
        for strategy_name, scores in strategy_results.items():
            for score in scores:
                symbol = score.symbol
                if symbol not in all_stocks:
                    all_stocks[symbol] = {
                        'symbol': symbol,
                        'name': score.name,
                        'strategies': {},
                        'total_combined_score': 0,
                        'strategy_count': 0
                    }
                
                all_stocks[symbol]['strategies'][strategy_name] = {
                    'score': score.total_score,
                    'rank': score.rank,
                    'reasoning': score.reasoning
                }
                all_stocks[symbol]['total_combined_score'] += score.total_score
                all_stocks[symbol]['strategy_count'] += 1
        
        # 평균 점수 계산
        for stock_data in all_stocks.values():
            if stock_data['strategy_count'] > 0:
                stock_data['average_score'] = stock_data['total_combined_score'] / stock_data['strategy_count']
            else:
                stock_data['average_score'] = 0
        
        # 평균 점수 순으로 정렬
        combined_results = list(all_stocks.values())
        combined_results.sort(key=lambda x: x['average_score'], reverse=True)
        
        logger.info(f"전략 종합 분석 완료: {len(combined_results)}개 종목")
        return combined_results 