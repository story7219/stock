"""
투자 전략에서 공통으로 사용되는 재무 지표 계산 모듈 및 기본 전략 클래스
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class StrategyResult:
    """전략 분석 결과"""
    total_score: float
    scores: Dict[str, float]
    strategy_name: str
    investment_decision: str
    key_points: List[str]
    analysis_details: Dict[str, Any]
    
    def __lt__(self, other):
        """정렬을 위한 비교 메서드"""
        if isinstance(other, StrategyResult):
            return self.total_score < other.total_score
        return NotImplemented
    
    def __le__(self, other):
        """정렬을 위한 비교 메서드"""
        if isinstance(other, StrategyResult):
            return self.total_score <= other.total_score
        return NotImplemented
    
    def __gt__(self, other):
        """정렬을 위한 비교 메서드"""
        if isinstance(other, StrategyResult):
            return self.total_score > other.total_score
        return NotImplemented
    
    def __ge__(self, other):
        """정렬을 위한 비교 메서드"""
        if isinstance(other, StrategyResult):
            return self.total_score >= other.total_score
        return NotImplemented
    
    def __eq__(self, other):
        """정렬을 위한 비교 메서드"""
        if isinstance(other, StrategyResult):
            return self.total_score == other.total_score
        return NotImplemented

class BaseStrategy(ABC):
    """모든 투자 전략의 기본 클래스"""
    
    def __init__(self):
        self.strategy_name = "Base Strategy"
        self.description = "기본 전략"
        self.weights = {}
        
    @abstractmethod
    def analyze_stock(self, stock) -> float:
        """주식 분석 메서드 - 하위 클래스에서 구현"""
        pass
    
    def analyze_stock_detailed(self, stock) -> Dict[str, Any]:
        """상세 분석 결과 반환"""
        try:
            # 각 세부 분석 점수 계산
            scores = {}
            for key in self.weights.keys():
                method_name = f"_analyze_{key}"
                if hasattr(self, method_name):
                    scores[key] = getattr(self, method_name)(stock)
                else:
                    scores[key] = 50.0  # 기본값
            
            # 총점 계산
            total_score = sum(
                scores[key] * self.weights[key] 
                for key in scores
            )
            
            return {
                'total_score': min(max(total_score, 0), 100),
                'scores': scores,
                'strategy_name': self.strategy_name
            }
        except Exception as e:
            logger.error(f"전략 분석 오류 ({self.strategy_name}): {e}")
            return {
                'total_score': 0.0,
                'scores': {key: 0.0 for key in self.weights.keys()},
                'strategy_name': self.strategy_name
            }

class CommonIndicators:
    """공통 재무 지표 계산 클래스"""
    
    @staticmethod
    def calculate_pe_ratio(price: float, eps: float) -> Optional[float]:
        """PER(주가수익비율) 계산"""
        if eps <= 0:
            return None
        return price / eps
    
    @staticmethod
    def calculate_pb_ratio(price: float, bps: float) -> Optional[float]:
        """PBR(주가순자산비율) 계산"""
        if bps <= 0:
            return None
        return price / bps
    
    @staticmethod
    def calculate_roe(net_income: float, shareholders_equity: float) -> Optional[float]:
        """ROE(자기자본이익률) 계산"""
        if shareholders_equity <= 0:
            return None
        return (net_income / shareholders_equity) * 100
    
    @staticmethod
    def calculate_roa(net_income: float, total_assets: float) -> Optional[float]:
        """ROA(총자산이익률) 계산"""
        if total_assets <= 0:
            return None
        return (net_income / total_assets) * 100
    
    @staticmethod
    def calculate_debt_ratio(total_debt: float, total_assets: float) -> Optional[float]:
        """부채비율 계산"""
        if total_assets <= 0:
            return None
        return (total_debt / total_assets) * 100
    
    @staticmethod
    def calculate_current_ratio(current_assets: float, current_liabilities: float) -> Optional[float]:
        """유동비율 계산"""
        if current_liabilities <= 0:
            return None
        return current_assets / current_liabilities
    
    @staticmethod
    def calculate_quick_ratio(current_assets: float, inventory: float, current_liabilities: float) -> Optional[float]:
        """당좌비율 계산"""
        if current_liabilities <= 0:
            return None
        return (current_assets - inventory) / current_liabilities
    
    @staticmethod
    def calculate_dividend_yield(dividend_per_share: float, price: float) -> Optional[float]:
        """배당수익률 계산"""
        if price <= 0:
            return None
        return (dividend_per_share / price) * 100
    
    @staticmethod
    def calculate_peg_ratio(pe_ratio: float, growth_rate: float) -> Optional[float]:
        """PEG 비율 계산 (PE / 성장률)"""
        if growth_rate <= 0 or pe_ratio is None:
            return None
        return pe_ratio / growth_rate
    
    @staticmethod
    def calculate_earnings_growth(current_earnings: float, previous_earnings: float) -> Optional[float]:
        """수익 성장률 계산"""
        if previous_earnings <= 0:
            return None
        return ((current_earnings - previous_earnings) / previous_earnings) * 100
    
    @staticmethod
    def calculate_revenue_growth(current_revenue: float, previous_revenue: float) -> Optional[float]:
        """매출 성장률 계산"""
        if previous_revenue <= 0:
            return None
        return ((current_revenue - previous_revenue) / previous_revenue) * 100
    
    @staticmethod
    def calculate_gross_margin(gross_profit: float, revenue: float) -> Optional[float]:
        """매출총이익률 계산"""
        if revenue <= 0:
            return None
        return (gross_profit / revenue) * 100
    
    @staticmethod
    def calculate_operating_margin(operating_income: float, revenue: float) -> Optional[float]:
        """영업이익률 계산"""
        if revenue <= 0:
            return None
        return (operating_income / revenue) * 100
    
    @staticmethod
    def calculate_net_margin(net_income: float, revenue: float) -> Optional[float]:
        """순이익률 계산"""
        if revenue <= 0:
            return None
        return (net_income / revenue) * 100
    
    @staticmethod
    def normalize_score(value: float, min_val: float, max_val: float) -> float:
        """점수 정규화 (0-100 범위)"""
        if max_val == min_val:
            return 50.0
        return ((value - min_val) / (max_val - min_val)) * 100
    
    @staticmethod
    def calculate_composite_score(scores: Dict[str, float], weights: Dict[str, float]) -> float:
        """가중 평균 점수 계산"""
        total_weight = sum(weights.values())
        if total_weight == 0:
            return 0.0
        
        weighted_sum = sum(scores.get(key, 0) * weight for key, weight in weights.items())
        return weighted_sum / total_weight

def get_stock_value(stock, key: str, default=None):
    """주식 데이터에서 값을 안전하게 가져오는 헬퍼 함수"""
    if isinstance(stock, dict):
        return stock.get(key, default)
    else:
        return getattr(stock, key, default)

def get_financial_metrics(stock):
    """주식 데이터에서 재무 지표들을 추출하는 헬퍼 함수"""
    return {
        'symbol': get_stock_value(stock, 'symbol'),
        'name': get_stock_value(stock, 'name'),
        'sector': get_stock_value(stock, 'sector'),
        'market_cap': get_stock_value(stock, 'market_cap'),
        'price': get_stock_value(stock, 'price') or get_stock_value(stock, 'current_price'),
        'per': get_stock_value(stock, 'per') or get_stock_value(stock, 'pe_ratio'),
        'pbr': get_stock_value(stock, 'pbr') or get_stock_value(stock, 'pb_ratio'),
        'roe': get_stock_value(stock, 'roe'),
        'roa': get_stock_value(stock, 'roa'),
        'debt_ratio': get_stock_value(stock, 'debt_ratio'),
        'current_ratio': get_stock_value(stock, 'current_ratio'),
        'profit_growth': get_stock_value(stock, 'earnings_growth_rate') or get_stock_value(stock, 'profit_growth'),
        'revenue_growth': get_stock_value(stock, 'revenue_growth_rate') or get_stock_value(stock, 'revenue_growth'),
        'dividend_yield': get_stock_value(stock, 'dividend_yield'),
        'volatility': get_stock_value(stock, 'volatility'),
        'price_momentum_3m': get_stock_value(stock, 'price_momentum') or get_stock_value(stock, 'price_momentum_3m'),
        'pe_ratio': get_stock_value(stock, 'per') or get_stock_value(stock, 'pe_ratio'),
        'peg_ratio': get_stock_value(stock, 'peg_ratio')
    } 