"""
스탠리 드러켄밀러 투자 전략
- 매크로 경제 분석과 기술적 분석 결합
- 추세 추종과 모멘텀 중시
- 리스크 관리와 집중 투자
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import asyncio

from .base_strategy import BaseStrategy, StrategyResult
from ..data.models import StockData, TechnicalIndicators


class DruckenmillerStrategy(BaseStrategy):
    """스탠리 드러켄밀러 투자 전략"""
    
    def __init__(self):
        super().__init__(
            name="stanley_druckenmiller",
            description="매크로 경제 분석과 기술적 분석을 결합한 추세 추종 전략",
            parameters={
                'trend_period': 50,
                'momentum_period': 20,
                'volume_threshold': 1.5,
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'macd_signal_threshold': 0.1,
                'adx_trend_threshold': 25,
                'volatility_threshold': 0.3
            }
        )
    
    async def analyze_stock(self, stock_data: StockData) -> StrategyResult:
        """드러켄밀러 전략으로 종목 분석"""
        
        if not self.validate_stock_data(stock_data):
            return StrategyResult(
                stock_code=stock_data.symbol,
                score=0.0,
                signals={},
                reasons=["데이터 부족 또는 유효하지 않음"],
                risk_level="UNKNOWN",
                confidence=0.0
            )
        
        try:
            # 기술적 지표 계산
            indicators = self._calculate_technical_indicators(stock_data)
            
            # 드러켄밀러 전략 신호 분석
            signals = await self._analyze_druckenmiller_signals(stock_data, indicators)
            
            # 종합 점수 계산
            score = self._calculate_total_score(signals, indicators)
            
            # 투자 사유 생성
            reasons = self._generate_investment_reasons(signals, indicators)
            
            # 리스크 레벨 계산
            volatility = self._calculate_volatility(stock_data)
            risk_level = self.get_risk_level(volatility)
            
            # 신뢰도 계산
            confidence = self._calculate_confidence(signals, indicators)
            
            return StrategyResult(
                stock_code=stock_data.symbol,
                score=score,
                signals=signals,
                reasons=reasons,
                risk_level=risk_level,
                confidence=confidence,
                technical_indicators=indicators
            )
            
        except Exception as e:
            return StrategyResult(
                stock_code=stock_data.symbol,
                score=0.0,
                signals={},
                reasons=[f"분석 오류: {str(e)}"],
                risk_level="UNKNOWN",
                confidence=0.0
            )
    
    def _calculate_technical_indicators(self, stock_data: StockData) -> TechnicalIndicators:
        """기술적 지표 계산"""
        df = stock_data.price_data.copy()
        
        # 이동평균선
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 볼린저 밴드
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        # ADX (단순화된 버전)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        df['ATR'] = true_range.rolling(window=14).mean()
        
        # 스토캐스틱
        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        df['Stoch_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
        df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
        
        # 거래량 비율
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        # 최신 값들 추출
        latest = df.iloc[-1]
        
        return TechnicalIndicators(
            symbol=stock_data.symbol,
            close_price=latest['Close'],
            sma_20=latest.get('SMA_20', 0),
            sma_50=latest.get('SMA_50', 0),
            ema_12=latest.get('EMA_12', 0),
            ema_26=latest.get('EMA_26', 0),
            macd=latest.get('MACD', 0),
            macd_signal=latest.get('MACD_Signal', 0),
            macd_histogram=latest.get('MACD_Histogram', 0),
            rsi=latest.get('RSI', 50),
            bb_upper=latest.get('BB_Upper', 0),
            bb_middle=latest.get('BB_Middle', 0),
            bb_lower=latest.get('BB_Lower', 0),
            atr=latest.get('ATR', 0),
            adx=25.0,  # 단순화
            stoch_k=latest.get('Stoch_K', 50),
            stoch_d=latest.get('Stoch_D', 50),
            volume_ratio=latest.get('Volume_Ratio', 1.0)
        )
    
    async def _analyze_druckenmiller_signals(self, stock_data: StockData, indicators: TechnicalIndicators) -> Dict[str, Any]:
        """드러켄밀러 전략 신호 분석"""
        signals = {}
        
        # 1. 추세 분석 (가장 중요)
        trend_score = self._analyze_trend(stock_data, indicators)
        signals['trend'] = {
            'score': trend_score,
            'direction': 'UP' if trend_score > 60 else 'DOWN' if trend_score < 40 else 'SIDEWAYS'
        }
        
        # 2. 모멘텀 분석
        momentum_score = self.calculate_momentum_score(stock_data)
        signals['momentum'] = {
            'score': momentum_score,
            'strength': 'STRONG' if momentum_score > 70 else 'WEAK' if momentum_score < 30 else 'MODERATE'
        }
        
        # 3. 거래량 분석
        volume_signal = self._analyze_volume(stock_data, indicators)
        signals['volume'] = volume_signal
        
        # 4. 기술적 지표 종합
        technical_score = self.calculate_technical_score(indicators)
        signals['technical'] = {
            'score': technical_score,
            'quality': 'EXCELLENT' if technical_score > 80 else 'GOOD' if technical_score > 60 else 'POOR'
        }
        
        # 5. 리스크/보상 비율
        risk_reward = self._calculate_risk_reward_ratio(stock_data, indicators)
        signals['risk_reward'] = risk_reward
        
        # 6. 매크로 환경 점수 (단순화)
        macro_score = self._estimate_macro_environment(stock_data)
        signals['macro'] = {
            'score': macro_score,
            'environment': 'FAVORABLE' if macro_score > 60 else 'UNFAVORABLE' if macro_score < 40 else 'NEUTRAL'
        }
        
        return signals
    
    def _analyze_trend(self, stock_data: StockData, indicators: TechnicalIndicators) -> float:
        """추세 분석"""
        df = stock_data.price_data
        score = 0.0
        
        # 이동평균선 배열
        if indicators.sma_20 > indicators.sma_50:
            score += 25
        
        # 가격이 이동평균선 위에 있는지
        if indicators.close_price > indicators.sma_20:
            score += 20
        
        # MACD 추세
        if indicators.macd > indicators.macd_signal:
            score += 20
        
        # 장기 추세 (50일 기준)
        if len(df) >= 50:
            price_50d_ago = df['Close'].iloc[-50]
            if indicators.close_price > price_50d_ago:
                score += 25
        
        # ADX로 추세 강도 확인
        if indicators.adx > 25:
            score += 10
        
        return min(score, 100)
    
    def _analyze_volume(self, stock_data: StockData, indicators: TechnicalIndicators) -> Dict[str, Any]:
        """거래량 분석"""
        df = stock_data.price_data
        
        # 거래량 증가 확인
        volume_increasing = indicators.volume_ratio > self.parameters['volume_threshold']
        
        # 가격-거래량 관계
        recent_prices = df['Close'].tail(5)
        recent_volumes = df['Volume'].tail(5)
        
        price_volume_correlation = recent_prices.corr(recent_volumes)
        
        return {
            'increasing': volume_increasing,
            'ratio': indicators.volume_ratio,
            'price_volume_correlation': price_volume_correlation if not pd.isna(price_volume_correlation) else 0,
            'signal': 'STRONG' if volume_increasing and price_volume_correlation > 0.3 else 'WEAK'
        }
    
    def _calculate_risk_reward_ratio(self, stock_data: StockData, indicators: TechnicalIndicators) -> Dict[str, Any]:
        """리스크/보상 비율 계산"""
        df = stock_data.price_data
        
        # 최근 변동성
        returns = df['Close'].pct_change().tail(20)
        volatility = returns.std() * np.sqrt(252)  # 연환산
        
        # 지지/저항 레벨 (단순화)
        recent_high = df['High'].tail(20).max()
        recent_low = df['Low'].tail(20).min()
        current_price = indicators.close_price
        
        # 상승 여력 vs 하락 위험
        upside_potential = (recent_high - current_price) / current_price
        downside_risk = (current_price - recent_low) / current_price
        
        risk_reward_ratio = upside_potential / downside_risk if downside_risk > 0 else 0
        
        return {
            'ratio': risk_reward_ratio,
            'volatility': volatility,
            'upside_potential': upside_potential,
            'downside_risk': downside_risk,
            'quality': 'EXCELLENT' if risk_reward_ratio > 2 else 'GOOD' if risk_reward_ratio > 1.5 else 'POOR'
        }
    
    def _estimate_macro_environment(self, stock_data: StockData) -> float:
        """매크로 환경 추정 (단순화)"""
        df = stock_data.price_data
        
        # 시장 전반적 추세 (단순화)
        if len(df) >= 60:
            long_term_trend = (df['Close'].iloc[-1] - df['Close'].iloc[-60]) / df['Close'].iloc[-60]
            if long_term_trend > 0.1:
                return 75  # 호황
            elif long_term_trend < -0.1:
                return 25  # 불황
            else:
                return 50  # 중립
        
        return 50  # 기본값
    
    def _calculate_total_score(self, signals: Dict[str, Any], indicators: TechnicalIndicators) -> float:
        """종합 점수 계산"""
        # 드러켄밀러 전략의 가중치
        weights = {
            'trend': 0.35,      # 추세가 가장 중요
            'momentum': 0.25,   # 모멘텀
            'technical': 0.15,  # 기술적 지표
            'volume': 0.10,     # 거래량
            'risk_reward': 0.10, # 리스크/보상
            'macro': 0.05       # 매크로 환경
        }
        
        total_score = 0.0
        
        # 각 신호별 점수 계산
        total_score += signals['trend']['score'] * weights['trend']
        total_score += signals['momentum']['score'] * weights['momentum']
        total_score += signals['technical']['score'] * weights['technical']
        
        # 거래량 점수
        volume_score = 80 if signals['volume']['signal'] == 'STRONG' else 40
        total_score += volume_score * weights['volume']
        
        # 리스크/보상 점수
        rr_score = 90 if signals['risk_reward']['quality'] == 'EXCELLENT' else 60 if signals['risk_reward']['quality'] == 'GOOD' else 30
        total_score += rr_score * weights['risk_reward']
        
        # 매크로 점수
        total_score += signals['macro']['score'] * weights['macro']
        
        return min(total_score, 100.0)
    
    def _generate_investment_reasons(self, signals: Dict[str, Any], indicators: TechnicalIndicators) -> List[str]:
        """투자 사유 생성"""
        reasons = []
        
        # 추세 관련
        if signals['trend']['direction'] == 'UP':
            reasons.append("강한 상승 추세 확인")
        elif signals['trend']['direction'] == 'DOWN':
            reasons.append("하락 추세로 투자 부적합")
        
        # 모멘텀 관련
        if signals['momentum']['strength'] == 'STRONG':
            reasons.append("강한 상승 모멘텀 보유")
        elif signals['momentum']['strength'] == 'WEAK':
            reasons.append("모멘텀 부족으로 주의 필요")
        
        # 거래량 관련
        if signals['volume']['signal'] == 'STRONG':
            reasons.append("거래량 증가로 신뢰성 높음")
        
        # 기술적 지표
        if signals['technical']['quality'] == 'EXCELLENT':
            reasons.append("우수한 기술적 지표")
        elif signals['technical']['quality'] == 'POOR':
            reasons.append("기술적 지표 부정적")
        
        # 리스크/보상
        if signals['risk_reward']['quality'] == 'EXCELLENT':
            reasons.append("우수한 리스크/보상 비율")
        elif signals['risk_reward']['quality'] == 'POOR':
            reasons.append("리스크 대비 보상 부족")
        
        # RSI 과매수/과매도
        if indicators.rsi < 30:
            reasons.append("RSI 과매도 구간에서 반등 기대")
        elif indicators.rsi > 70:
            reasons.append("RSI 과매수 구간으로 조정 위험")
        
        return reasons
    
    def _calculate_confidence(self, signals: Dict[str, Any], indicators: TechnicalIndicators) -> float:
        """신뢰도 계산"""
        confidence_factors = []
        
        # 추세 신뢰도
        if signals['trend']['score'] > 70:
            confidence_factors.append(0.9)
        elif signals['trend']['score'] > 50:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.3)
        
        # 거래량 신뢰도
        if signals['volume']['signal'] == 'STRONG':
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.5)
        
        # 기술적 지표 일치도
        if signals['technical']['score'] > 70:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.4)
        
        # 리스크/보상 신뢰도
        if signals['risk_reward']['quality'] == 'EXCELLENT':
            confidence_factors.append(0.9)
        elif signals['risk_reward']['quality'] == 'GOOD':
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.4)
        
        return np.mean(confidence_factors) * 100
    
    def _calculate_volatility(self, stock_data: StockData) -> float:
        """변동성 계산"""
        df = stock_data.price_data
        returns = df['Close'].pct_change().dropna()
        
        if len(returns) < 20:
            return 0.3  # 기본값
        
        return returns.tail(20).std() * np.sqrt(252)  # 연환산 변동성
    
    def get_strategy_parameters(self) -> Dict[str, Any]:
        """전략 파라미터 반환"""
        return {
            'name': self.name,
            'description': self.description,
            'parameters': self.parameters,
            'focus_areas': [
                '추세 추종',
                '모멘텀 분석',
                '거래량 확인',
                '리스크 관리',
                '매크로 경제 고려'
            ],
            'key_indicators': [
                'SMA/EMA 이동평균선',
                'MACD',
                'RSI',
                '거래량',
                'ADX',
                '볼린저 밴드'
            ]
        } 