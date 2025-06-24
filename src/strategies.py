"""
투자 대가 전략 모듈
워런 버핏, 피터 린치, 벤저민 그레이엄 전략 구현
"""

import logging
from typing import List, Dict, Optional, Tuple, Any
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
        """전략 매니저 초기화"""
        self.strategies = {
            'buffett': WarrenBuffettStrategy(),
            'lynch': PeterLynchStrategy(),
            'graham': BenjaminGrahamStrategy(),
            'oneil': WilliamOneilStrategy(),
            'livermore': JesseLivermoreStrategy(),
            'templeton': JohnTempletonStrategy(),
            'neff': JohnNeffStrategy(),
            'fisher': PhilipFisherStrategy(),
            'minervini': MarkMinerviniStrategy(),
            'slater': JimSlaterStrategy(),
            'greenblatt': JoelGreenblattStrategy(),
            'thorp': EdwardThorpStrategy(),
            'dalio': RayDalioStrategy(),
            'drucker': PeterDruckerStrategy(),
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
    
    async def analyze_all_strategies(self, market_data: Dict[str, List[StockData]]) -> Dict[str, Any]:
        """모든 시장 데이터에 대해 전체 전략 분석 수행"""
        try:
            all_results = {}
            
            for market_name, stocks in market_data.items():
                logger.info(f"🔍 {market_name} 전략 분석 시작")
                
                # 각 전략별 분석
                market_results = self.apply_all_strategies(stocks)
                all_results[market_name] = market_results
                
                logger.info(f"✅ {market_name} 전략 분석 완료")
            
            return all_results
            
        except Exception as e:
            logger.error(f"전략 분석 실패: {e}")
            raise
    
    async def analyze_strategies(self, market_data: Dict[str, List[StockData]], 
                               selected_strategies: List[str]) -> Dict[str, Any]:
        """선택된 전략들만 분석 수행 (GUI용)"""
        try:
            all_results = {}
            
            # 선택된 전략만 필터링
            selected_strategy_objects = {}
            strategy_name_map = {
                "워런 버핏": "buffett",
                "피터 린치": "lynch", 
                "벤저민 그레이엄": "graham"
            }
            
            for strategy_name in selected_strategies:
                if strategy_name in strategy_name_map:
                    key = strategy_name_map[strategy_name]
                    if key in self.strategies:
                        selected_strategy_objects[key] = self.strategies[key]
            
            for market_name, stocks in market_data.items():
                logger.info(f"🔍 {market_name} 선택된 전략 분석 시작")
                
                market_results = {}
                for strategy_key, strategy in selected_strategy_objects.items():
                    try:
                        logger.info(f"{strategy.name} 전략 적용 시작")
                        scores = strategy.apply_strategy(stocks)
                        market_results[strategy_key] = scores
                        logger.info(f"{strategy.name} 전략 완료: {len(scores)}개 종목 평가")
                    except Exception as e:
                        logger.error(f"{strategy.name} 전략 적용 실패: {e}")
                        market_results[strategy_key] = []
                
                all_results[market_name] = market_results
                logger.info(f"✅ {market_name} 선택된 전략 분석 완료")
            
            return all_results
            
        except Exception as e:
            logger.error(f"선택된 전략 분석 실패: {e}")
            raise

class WilliamOneilStrategy(BaseStrategy):
    """윌리엄 오닐 전략 - CAN SLIM"""
    
    def __init__(self):
        super().__init__(
            name="William O'Neil Strategy",
            description="CAN SLIM 방법론 기반 성장주 선별 전략"
        )
    
    def filter_stocks(self, stocks: List[StockData]) -> List[StockData]:
        return [s for s in stocks if s.rsi and s.rsi > 50]
    
    def calculate_score(self, stock: StockData) -> StrategyScore:
        total_score = 70 + (stock.rsi - 50) if stock.rsi else 50
        return StrategyScore(
            symbol=stock.symbol, name=stock.name, strategy_name=self.name,
            total_score=total_score, criteria_scores={'rsi': total_score},
            reasoning="CAN SLIM 기법 적용 - 모멘텀 중심 분석"
        )

class JesseLivermoreStrategy(BaseStrategy):
    """제시 리버모어 전략 - 트렌드 추종"""
    
    def __init__(self):
        super().__init__(
            name="Jesse Livermore Strategy",
            description="트렌드 추종 및 모멘텀 전략"
        )
    
    def filter_stocks(self, stocks: List[StockData]) -> List[StockData]:
        return [s for s in stocks if s.moving_avg_20 and s.price > s.moving_avg_20]
    
    def calculate_score(self, stock: StockData) -> StrategyScore:
        momentum_score = ((stock.price / stock.moving_avg_20) - 1) * 100 if stock.moving_avg_20 else 0
        total_score = 60 + momentum_score
        return StrategyScore(
            symbol=stock.symbol, name=stock.name, strategy_name=self.name,
            total_score=total_score, criteria_scores={'momentum': momentum_score},
            reasoning="트렌드 추종 전략 - 상승 모멘텀 중심"
        )

class JohnTempletonStrategy(BaseStrategy):
    """존 템플턴 전략 - 글로벌 가치투자"""
    
    def __init__(self):
        super().__init__(
            name="John Templeton Strategy",
            description="글로벌 가치투자 전략"
        )
    
    def filter_stocks(self, stocks: List[StockData]) -> List[StockData]:
        return [s for s in stocks if s.pb_ratio and s.pb_ratio < 2.0]
    
    def calculate_score(self, stock: StockData) -> StrategyScore:
        value_score = (2.0 - stock.pb_ratio) * 25 if stock.pb_ratio else 0
        total_score = 50 + value_score
        return StrategyScore(
            symbol=stock.symbol, name=stock.name, strategy_name=self.name,
            total_score=total_score, criteria_scores={'value': value_score},
            reasoning="글로벌 가치투자 - 저평가 종목 중심"
        )

class JohnNeffStrategy(BaseStrategy):
    """존 네프 전략 - 저PER 가치투자"""
    
    def __init__(self):
        super().__init__(
            name="John Neff Strategy",
            description="저PER 가치투자 전략"
        )
    
    def filter_stocks(self, stocks: List[StockData]) -> List[StockData]:
        return [s for s in stocks if s.pe_ratio and s.pe_ratio < 15]
    
    def calculate_score(self, stock: StockData) -> StrategyScore:
        pe_score = (15 - stock.pe_ratio) * 5 if stock.pe_ratio else 0
        total_score = 50 + pe_score
        return StrategyScore(
            symbol=stock.symbol, name=stock.name, strategy_name=self.name,
            total_score=total_score, criteria_scores={'pe': pe_score},
            reasoning="저PER 가치투자 - 저평가 우량주 중심"
        )

class PhilipFisherStrategy(BaseStrategy):
    """필립 피셔 전략 - 성장주 투자"""
    
    def __init__(self):
        super().__init__(
            name="Philip Fisher Strategy",
            description="성장주 투자 전략"
        )
    
    def filter_stocks(self, stocks: List[StockData]) -> List[StockData]:
        return [s for s in stocks if s.roe and s.roe > 0.15]
    
    def calculate_score(self, stock: StockData) -> StrategyScore:
        growth_score = stock.roe * 200 if stock.roe else 0
        total_score = 50 + growth_score
        return StrategyScore(
            symbol=stock.symbol, name=stock.name, strategy_name=self.name,
            total_score=total_score, criteria_scores={'growth': growth_score},
            reasoning="성장주 투자 - 높은 ROE 기업 중심"
        )

class MarkMinerviniStrategy(BaseStrategy):
    """마크 미너비니 전략 - 트레이드 마크"""
    
    def __init__(self):
        super().__init__(
            name="Mark Minervini Strategy",
            description="트레이드 마크 전략"
        )
    
    def filter_stocks(self, stocks: List[StockData]) -> List[StockData]:
        return [s for s in stocks if s.volume > 50000]
    
    def calculate_score(self, stock: StockData) -> StrategyScore:
        volume_score = min(stock.volume / 100000, 50) if stock.volume else 0
        total_score = 50 + volume_score
        return StrategyScore(
            symbol=stock.symbol, name=stock.name, strategy_name=self.name,
            total_score=total_score, criteria_scores={'volume': volume_score},
            reasoning="트레이드 마크 전략 - 거래량 기반 분석"
        )

class JimSlaterStrategy(BaseStrategy):
    """짐 슬레이터 전략 - PEG 투자"""
    
    def __init__(self):
        super().__init__(
            name="Jim Slater Strategy",
            description="PEG 기반 투자 전략"
        )
    
    def filter_stocks(self, stocks: List[StockData]) -> List[StockData]:
        return [s for s in stocks if s.pe_ratio and s.pe_ratio > 0]
    
    def calculate_score(self, stock: StockData) -> StrategyScore:
        peg_score = 100 / stock.pe_ratio if stock.pe_ratio and stock.pe_ratio > 0 else 0
        total_score = 50 + min(peg_score, 50)
        return StrategyScore(
            symbol=stock.symbol, name=stock.name, strategy_name=self.name,
            total_score=total_score, criteria_scores={'peg': peg_score},
            reasoning="PEG 기반 투자 - 성장 대비 저평가 종목"
        )

class JoelGreenblattStrategy(BaseStrategy):
    """조엘 그린블라트 전략 - 마법공식"""
    
    def __init__(self):
        super().__init__(
            name="Joel Greenblatt Strategy",
            description="마법공식 투자 전략"
        )
    
    def filter_stocks(self, stocks: List[StockData]) -> List[StockData]:
        return [s for s in stocks if s.roe and s.pe_ratio]
    
    def calculate_score(self, stock: StockData) -> StrategyScore:
        magic_score = (stock.roe * 100) / stock.pe_ratio if stock.roe and stock.pe_ratio else 0
        total_score = 50 + min(magic_score, 50)
        return StrategyScore(
            symbol=stock.symbol, name=stock.name, strategy_name=self.name,
            total_score=total_score, criteria_scores={'magic': magic_score},
            reasoning="마법공식 - ROE/PER 비율 기반"
        )

class EdwardThorpStrategy(BaseStrategy):
    """에드워드 소프 전략 - 수학적 투자"""
    
    def __init__(self):
        super().__init__(
            name="Edward Thorp Strategy",
            description="수학적 투자 전략"
        )
    
    def filter_stocks(self, stocks: List[StockData]) -> List[StockData]:
        return [s for s in stocks if s.volatility_20d]
    
    def calculate_score(self, stock: StockData) -> StrategyScore:
        math_score = 1 / stock.volatility_20d if stock.volatility_20d else 0
        total_score = 50 + min(math_score * 100, 50)
        return StrategyScore(
            symbol=stock.symbol, name=stock.name, strategy_name=self.name,
            total_score=total_score, criteria_scores={'math': math_score},
            reasoning="수학적 투자 - 변동성 기반 분석"
        )

class RayDalioStrategy(BaseStrategy):
    """레이 달리오 전략 - 올웨더"""
    
    def __init__(self):
        super().__init__(
            name="Ray Dalio Strategy",
            description="올웨더 포트폴리오 전략"
        )
    
    def filter_stocks(self, stocks: List[StockData]) -> List[StockData]:
        return stocks  # 모든 종목 대상
    
    def calculate_score(self, stock: StockData) -> StrategyScore:
        balance_score = 70  # 균형 포트폴리오 기본 점수
        if stock.market_beta:
            balance_score += (1 - abs(stock.market_beta - 1)) * 30
        total_score = balance_score
        return StrategyScore(
            symbol=stock.symbol, name=stock.name, strategy_name=self.name,
            total_score=total_score, criteria_scores={'balance': balance_score},
            reasoning="올웨더 전략 - 균형 포트폴리오 구성"
        )

class PeterDruckerStrategy(BaseStrategy):
    """피터 드러커 전략 - 경영 품질"""
    
    def __init__(self):
        super().__init__(
            name="Peter Drucker Strategy",
            description="경영 품질 중심 투자 전략"
        )
    
    def filter_stocks(self, stocks: List[StockData]) -> List[StockData]:
        return [s for s in stocks if s.market_cap and s.market_cap > 1e11]
    
    def calculate_score(self, stock: StockData) -> StrategyScore:
        management_score = 60  # 경영 품질 기본 점수
        if stock.roe and stock.roe > 0.1:
            management_score += 20
        if stock.debt_ratio and stock.debt_ratio < 0.5:
            management_score += 20
        total_score = management_score
        return StrategyScore(
            symbol=stock.symbol, name=stock.name, strategy_name=self.name,
            total_score=total_score, criteria_scores={'management': management_score},
            reasoning="경영 품질 중심 - 우수한 경영진 기업 선별"
        ) 