"""
기본 전략 클래스 - 모든 투자 전략의 베이스 클래스
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass

from ..data.models import StockData, TechnicalIndicators


@dataclass
class StrategyResult:
    """전략 분석 결과"""
    stock_code: str
    score: float
    signals: Dict[str, Any]
    reasons: List[str]
    risk_level: str
    confidence: float
    technical_indicators: Optional[TechnicalIndicators] = None


class BaseStrategy(ABC):
    """모든 투자 전략의 기본 클래스"""
    
    def __init__(self, name: str, description: str, parameters: Optional[Dict[str, Any]] = None):
        self.name = name
        self.description = description
        self.parameters = parameters or {}
        self.weight = 1.0
        
    @abstractmethod
    async def analyze_stock(self, stock_data: StockData) -> StrategyResult:
        """개별 종목 분석"""
        pass
    
    @abstractmethod
    def get_strategy_parameters(self) -> Dict[str, Any]:
        """전략 파라미터 반환"""
        pass
    
    async def analyze_batch(self, stock_data_list: List[StockData]) -> List[StrategyResult]:
        """배치 분석"""
        results = []
        for stock_data in stock_data_list:
            try:
                result = await self.analyze_stock(stock_data)
                results.append(result)
            except Exception as e:
                print(f"Error analyzing {stock_data.symbol}: {e}")
                # 기본값으로 결과 생성
                result = StrategyResult(
                    stock_code=stock_data.symbol,
                    score=0.0,
                    signals={},
                    reasons=[f"분석 오류: {str(e)}"],
                    risk_level="UNKNOWN",
                    confidence=0.0
                )
                results.append(result)
        return results
    
    def calculate_technical_score(self, indicators: TechnicalIndicators) -> float:
        """기술적 지표 기반 점수 계산"""
        score = 0.0
        
        try:
            # RSI 점수 (30-70 범위가 좋음)
            if 30 <= indicators.rsi <= 70:
                score += 20
            elif indicators.rsi < 30:
                score += 10  # 과매도
            elif indicators.rsi > 70:
                score += 5   # 과매수
            
            # MACD 신호
            if indicators.macd > indicators.macd_signal:
                score += 15
            
            # 볼린저 밴드 위치
            bb_position = (indicators.close_price - indicators.bb_lower) / (indicators.bb_upper - indicators.bb_lower)
            if 0.2 <= bb_position <= 0.8:
                score += 15
            
            # 이동평균선 관계
            if indicators.sma_20 > indicators.sma_50:
                score += 10
            if indicators.ema_12 > indicators.ema_26:
                score += 10
            
            # 거래량 분석
            if indicators.volume_ratio > 1.2:
                score += 10
            
            # ADX (추세 강도)
            if indicators.adx > 25:
                score += 10
            
            # 스토캐스틱
            if 20 <= indicators.stoch_k <= 80:
                score += 10
                
        except Exception as e:
            print(f"기술적 분석 오류: {e}")
            
        return min(score, 100.0)
    
    def calculate_momentum_score(self, stock_data: StockData) -> float:
        """모멘텀 점수 계산"""
        try:
            df = stock_data.price_data
            if df is None or len(df) < 20:
                return 0.0
            
            # 가격 모멘텀
            price_change_1d = (df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100
            price_change_5d = (df['Close'].iloc[-1] - df['Close'].iloc[-6]) / df['Close'].iloc[-6] * 100
            price_change_20d = (df['Close'].iloc[-1] - df['Close'].iloc[-21]) / df['Close'].iloc[-21] * 100
            
            # 볼륨 모멘텀
            avg_volume_20d = df['Volume'].tail(20).mean()
            recent_volume = df['Volume'].tail(5).mean()
            volume_momentum = (recent_volume - avg_volume_20d) / avg_volume_20d * 100
            
            # 종합 모멘텀 점수
            momentum_score = (
                price_change_1d * 0.1 +
                price_change_5d * 0.3 +
                price_change_20d * 0.4 +
                min(volume_momentum, 50) * 0.2
            )
            
            return max(0, min(momentum_score + 50, 100))
            
        except Exception as e:
            print(f"모멘텀 계산 오류: {e}")
            return 0.0
    
    def get_risk_level(self, volatility: float, beta: float = None) -> str:
        """리스크 레벨 계산"""
        try:
            if volatility < 0.15:
                return "LOW"
            elif volatility < 0.25:
                return "MEDIUM"
            elif volatility < 0.35:
                return "HIGH"
            else:
                return "VERY_HIGH"
        except:
            return "UNKNOWN"
    
    def validate_stock_data(self, stock_data: StockData) -> bool:
        """주식 데이터 유효성 검증"""
        try:
            if not stock_data or not stock_data.symbol:
                return False
            
            if stock_data.price_data is None or len(stock_data.price_data) < 20:
                return False
            
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in stock_data.price_data.columns for col in required_columns):
                return False
            
            return True
        except:
            return False 