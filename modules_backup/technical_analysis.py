"""
기술적 분석 모듈 v2.0
ITechnicalAnalyzer 인터페이스 구현 및 표준화된 데이터 구조 사용
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import ta
from loguru import logger
from datetime import datetime
import os

from ..core.base_interfaces import (
    ITechnicalAnalyzer, StockData, TechnicalIndicators, 
    TechnicalSignals, TechnicalAnalysisResult, AnalysisError
)


class TechnicalAnalyzer(ITechnicalAnalyzer):
    """기술적 분석기 - ITechnicalAnalyzer 인터페이스 구현"""
    
    def __init__(self):
        """기술적 분석기 초기화"""
        self.indicators = {}
        logger.info("TechnicalAnalyzer 초기화 완료")
    
    async def analyze(self, data: Union[str, StockData]) -> TechnicalAnalysisResult:
        """기술적 분석 수행"""
        try:
            # 문자열인 경우 StockData 객체 생성 (Mock)
            if isinstance(data, str):
                symbol = data
                # Mock StockData 생성
                stock_data = StockData(
                    symbol=symbol,
                    name=f"Mock {symbol}",
                    current_price=100.0
                )
                logger.info(f"{symbol} Mock 기술적 분석 시작")
            else:
                stock_data = data
                symbol = stock_data.symbol
                logger.info(f"{symbol} 기술적 분석 시작")
            
            # Mock 모드에서는 가상 데이터로 분석
            if os.getenv('IS_MOCK', 'false').lower() == 'true' or isinstance(data, str):
                return self._generate_mock_analysis(symbol)
            
            # 실제 분석 로직
            if stock_data.historical_data is None or stock_data.historical_data.empty:
                logger.warning(f"{symbol}: 히스토리 데이터 없음, Mock 분석 사용")
                return self._generate_mock_analysis(symbol)
            
            # 기술적 지표 계산
            indicators = self.calculate_indicators(stock_data.historical_data)
            
            # 신호 생성
            signals = self.generate_signals(indicators, stock_data.current_price)
            
            # 신뢰도 계산
            confidence = self._calculate_confidence(indicators, signals)
            
            # 요약 생성
            summary = self._generate_summary(symbol, indicators, signals)
            
            result = TechnicalAnalysisResult(
                symbol=symbol,
                indicators=indicators,
                signals=signals,
                confidence=confidence,
                summary=summary
            )
            
            logger.info(f"{symbol} 기술적 분석 완료 (신뢰도: {confidence:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"기술적 분석 중 오류: {e}")
            return self._generate_mock_analysis(symbol if isinstance(data, str) else data.symbol)
    
    def calculate_indicators(self, price_data: pd.DataFrame) -> TechnicalIndicators:
        """
        기술적 지표 계산
        
        Args:
            price_data: OHLCV 데이터
            
        Returns:
            TechnicalIndicators: 계산된 기술적 지표들
        """
        try:
            if len(price_data) < 20:
                logger.warning("데이터가 부족합니다. 최소 20일 데이터가 필요합니다.")
                return TechnicalIndicators()
            
            # 컬럼명 정규화
            price_data = self._normalize_columns(price_data)
            
            # ta 라이브러리를 사용하여 모든 지표 계산
            enriched_data = ta.add_all_ta_features(
                price_data, 
                open="Open", high="High", low="Low", close="Close", volume="Volume",
                fillna=True
            )
            
            # 최신 값들 추출
            latest_idx = -1
            
            return TechnicalIndicators(
                rsi=self._safe_get_value(enriched_data, 'momentum_rsi', latest_idx),
                macd=self._safe_get_value(enriched_data, 'trend_macd', latest_idx),
                macd_signal=self._safe_get_value(enriched_data, 'trend_macd_signal', latest_idx),
                bb_upper=self._safe_get_value(enriched_data, 'volatility_bbh', latest_idx),
                bb_middle=self._safe_get_value(enriched_data, 'volatility_bbm', latest_idx),
                bb_lower=self._safe_get_value(enriched_data, 'volatility_bbl', latest_idx),
                sma_20=self._safe_get_value(enriched_data, 'trend_sma_fast', latest_idx),
                sma_50=self._safe_get_value(enriched_data, 'trend_sma_slow', latest_idx),
                sma_200=ta.trend.sma_indicator(close=price_data['Close'], window=200).iloc[latest_idx] if len(price_data) >= 200 else None,
                ema_20=self._safe_get_value(enriched_data, 'trend_ema_fast', latest_idx),
                volume_sma=ta.trend.sma_indicator(close=price_data['Volume'], window=20).iloc[latest_idx] if 'Volume' in price_data.columns else None,
                adx=self._safe_get_value(enriched_data, 'trend_adx', latest_idx),
                stoch_k=self._safe_get_value(enriched_data, 'momentum_stoch', latest_idx),
                stoch_d=self._safe_get_value(enriched_data, 'momentum_stoch_signal', latest_idx),
                obv=self._safe_get_value(enriched_data, 'volume_obv', latest_idx)
            )
            
        except Exception as e:
            logger.error(f"기술적 지표 계산 실패: {e}")
            return TechnicalIndicators()
    
    def generate_signals(self, indicators: TechnicalIndicators, 
                        current_price: float) -> TechnicalSignals:
        """
        기술적 신호 생성
        
        Args:
            indicators: 기술적 지표들
            current_price: 현재 가격
            
        Returns:
            TechnicalSignals: 생성된 기술적 신호들
        """
        try:
            signals = TechnicalSignals()
            
            # RSI 신호
            if indicators.rsi is not None:
                if indicators.rsi > 70:
                    signals.rsi_signal = "과매수"
                elif indicators.rsi < 30:
                    signals.rsi_signal = "과매도"
                elif indicators.rsi > 50:
                    signals.rsi_signal = "상승세"
                else:
                    signals.rsi_signal = "하락세"
            
            # MACD 신호
            if indicators.macd is not None and indicators.macd_signal is not None:
                if indicators.macd > indicators.macd_signal:
                    signals.macd_signal = "상승신호"
                else:
                    signals.macd_signal = "하락신호"
            
            # 볼린저밴드 신호
            if all(x is not None for x in [indicators.bb_upper, indicators.bb_middle, indicators.bb_lower]):
                if current_price >= indicators.bb_upper:
                    signals.bb_signal = "과매수"
                elif current_price <= indicators.bb_lower:
                    signals.bb_signal = "과매도"
                elif current_price > indicators.bb_middle:
                    signals.bb_signal = "상승세"
                else:
                    signals.bb_signal = "하락세"
            
            # 이동평균선 트렌드
            if all(x is not None for x in [indicators.sma_20, indicators.sma_50]):
                if indicators.sma_20 > indicators.sma_50:
                    if current_price > indicators.sma_20:
                        signals.ma_trend = "강한상승"
                    else:
                        signals.ma_trend = "약한상승"
                else:
                    if current_price < indicators.sma_20:
                        signals.ma_trend = "강한하락"
                    else:
                        signals.ma_trend = "약한하락"
            
            # 거래량 신호
            if indicators.volume_sma is not None:
                # 현재 거래량 정보가 없으므로 기본값 설정
                signals.volume_signal = "중립"
            
            # 전체 트렌드 종합 판단
            signals.overall_trend = self._determine_overall_trend(signals)
            
            return signals
            
        except Exception as e:
            logger.error(f"기술적 신호 생성 실패: {e}")
            return TechnicalSignals()
    
    # 기존 메서드들을 private로 유지 (하위 호환성)
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """RSI (Relative Strength Index) 계산 - 레거시 메서드"""
        try:
            rsi_indicator = ta.momentum.RSIIndicator(close=prices, window=period)
            return rsi_indicator.rsi()
        except Exception as e:
            logger.error(f"RSI 계산 실패: {e}")
            return pd.Series(dtype=float)
    
    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """MACD 계산 - 레거시 메서드"""
        try:
            macd_indicator = ta.trend.MACD(close=prices, window_slow=slow, window_fast=fast, window_sign=signal)
            
            return {
                'MACD': macd_indicator.macd(),
                'MACD_Signal': macd_indicator.macd_signal(),
                'MACD_Hist': macd_indicator.macd_diff()
            }
        except Exception as e:
            logger.error(f"MACD 계산 실패: {e}")
            return {}
    
    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std: float = 2) -> Dict[str, pd.Series]:
        """볼린저 밴드 계산 - 레거시 메서드"""
        try:
            bb_indicator = ta.volatility.BollingerBands(close=prices, window=period, window_dev=std)
            
            return {
                'BB_Upper': bb_indicator.bollinger_hband(),
                'BB_Middle': bb_indicator.bollinger_mavg(),
                'BB_Lower': bb_indicator.bollinger_lband()
            }
        except Exception as e:
            logger.error(f"볼린저 밴드 계산 실패: {e}")
            return {}
    
    def calculate_moving_averages(self, prices: pd.Series) -> Dict[str, pd.Series]:
        """이동평균선들 계산 - 레거시 메서드"""
        try:
            ma_periods = [5, 10, 20, 50, 100, 200]
            mas = {}
            
            for period in ma_periods:
                mas[f'SMA_{period}'] = ta.trend.sma_indicator(close=prices, window=period)
                mas[f'EMA_{period}'] = ta.trend.ema_indicator(close=prices, window=period)
            
            return mas
        except Exception as e:
            logger.error(f"이동평균 계산 실패: {e}")
            return {}
    
    def calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                           k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
        """스토캐스틱 오실레이터 계산 - 레거시 메서드"""
        try:
            stoch_indicator = ta.momentum.StochasticOscillator(high=high, low=low, close=close, window=k_period, smooth_window=d_period)
            
            return {
                'STOCH_K': stoch_indicator.stoch(),
                'STOCH_D': stoch_indicator.stoch_signal()
            }
        except Exception as e:
            logger.error(f"스토캐스틱 계산 실패: {e}")
            return {}
    
    def calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """ADX 계산 - 레거시 메서드"""
        try:
            adx_indicator = ta.trend.ADXIndicator(high=high, low=low, close=close, window=period)
            return adx_indicator.adx()
        except Exception as e:
            logger.error(f"ADX 계산 실패: {e}")
            return pd.Series(dtype=float)
    
    def calculate_volume_indicators(self, prices: pd.Series, volume: pd.Series) -> Dict[str, pd.Series]:
        """거래량 기반 지표들 계산 - 레거시 메서드"""
        try:
            indicators = {}
            
            # OBV (On Balance Volume)
            indicators['OBV'] = ta.volume.OnBalanceVolumeIndicator(close=prices, volume=volume).on_balance_volume()
            
            # Volume SMA
            indicators['Volume_SMA_20'] = ta.trend.sma_indicator(close=volume, window=20)
            
            return indicators
        except Exception as e:
            logger.error(f"거래량 지표 계산 실패: {e}")
            return {}
    
    def calculate_momentum_indicators(self, prices: pd.Series, high: pd.Series, low: pd.Series) -> Dict[str, pd.Series]:
        """모멘텀 지표들 계산 - 레거시 메서드"""
        try:
            indicators = {}
            
            # ROC (Rate of Change)
            indicators['ROC'] = ta.momentum.ROCIndicator(close=prices, window=10).roc()
            
            # Momentum
            indicators['Momentum'] = ta.momentum.ROCIndicator(close=prices, window=10).roc()
            
            # Williams %R
            indicators['Williams_R'] = ta.momentum.WilliamsRIndicator(high=high, low=low, close=prices, lbp=14).williams_r()
            
            return indicators
        except Exception as e:
            logger.error(f"모멘텀 지표 계산 실패: {e}")
            return {}
    
    def analyze_stock(self, stock_data: pd.DataFrame) -> TechnicalAnalysisResult:
        """레거시 호환성을 위한 메서드"""
        try:
            # StockData 객체로 변환
            symbol = "UNKNOWN"
            current_price = stock_data['Close'].iloc[-1] if not stock_data.empty else 0.0
            
            from ..core.base_interfaces import MarketType
            stock_obj = StockData(
                symbol=symbol,
                name=symbol,
                market=MarketType.KOSPI200,  # 기본값
                current_price=current_price,
                historical_data=stock_data
            )
            
            return self.analyze(stock_obj)
            
        except Exception as e:
            logger.error(f"레거시 분석 실패: {e}")
            return TechnicalAnalysisResult(
                symbol="UNKNOWN",
                indicators=TechnicalIndicators(),
                signals=TechnicalSignals()
            )
    
    # Private 헬퍼 메서드들
    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """DataFrame 컬럼명 정규화"""
        column_mapping = {
            'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume',
            'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume'
        }
        
        return df.rename(columns=column_mapping)
    
    def _safe_get_value(self, df: pd.DataFrame, column: str, index: int) -> Optional[float]:
        """안전한 값 추출"""
        try:
            if column not in df.columns:
                return None
            value = df[column].iloc[index]
            return float(value) if pd.notna(value) else None
        except:
            return None
    
    def _calculate_confidence(self, indicators: TechnicalIndicators, signals: TechnicalSignals) -> float:
        """신뢰도 계산"""
        try:
            confidence_score = 0.0
            total_indicators = 0
            
            # 각 지표의 유효성 검사 및 점수 계산
            if indicators.rsi is not None:
                total_indicators += 1
                if 30 < indicators.rsi < 70:  # 정상 범위
                    confidence_score += 0.15
            
            if indicators.macd is not None and indicators.macd_signal is not None:
                total_indicators += 1
                confidence_score += 0.15
            
            if all(x is not None for x in [indicators.bb_upper, indicators.bb_middle, indicators.bb_lower]):
                total_indicators += 1
                confidence_score += 0.2
            
            if all(x is not None for x in [indicators.sma_20, indicators.sma_50]):
                total_indicators += 1
                confidence_score += 0.25
            
            if indicators.adx is not None:
                total_indicators += 1
                if indicators.adx > 25:  # 강한 트렌드
                    confidence_score += 0.25
                else:
                    confidence_score += 0.1
            
            # 최종 신뢰도 계산
            if total_indicators > 0:
                return min(1.0, confidence_score)
            else:
                return 0.0
            
        except Exception as e:
            logger.error(f"신뢰도 계산 실패: {e}")
            return 0.5
    
    def _determine_overall_trend(self, signals: TechnicalSignals) -> str:
        """전체 트렌드 종합 판단"""
        try:
            bullish_signals = 0
            bearish_signals = 0
            
            # 각 신호별 점수 계산
            signal_weights = {
                'ma_trend': 0.3,
                'rsi_signal': 0.2,
                'macd_signal': 0.25,
                'bb_signal': 0.25
            }
            
            if signals.ma_trend in ['강한상승', '약한상승']:
                bullish_signals += signal_weights['ma_trend']
            elif signals.ma_trend in ['강한하락', '약한하락']:
                bearish_signals += signal_weights['ma_trend']
            
            if signals.rsi_signal in ['상승세']:
                bullish_signals += signal_weights['rsi_signal']
            elif signals.rsi_signal in ['하락세', '과매수']:
                bearish_signals += signal_weights['rsi_signal']
            elif signals.rsi_signal == '과매도':
                bullish_signals += signal_weights['rsi_signal'] * 0.7  # 반전 가능성
            
            if signals.macd_signal == '상승신호':
                bullish_signals += signal_weights['macd_signal']
            elif signals.macd_signal == '하락신호':
                bearish_signals += signal_weights['macd_signal']
            
            if signals.bb_signal in ['상승세']:
                bullish_signals += signal_weights['bb_signal']
            elif signals.bb_signal in ['하락세', '과매수']:
                bearish_signals += signal_weights['bb_signal']
            elif signals.bb_signal == '과매도':
                bullish_signals += signal_weights['bb_signal'] * 0.7
            
            # 최종 판단
            if bullish_signals > bearish_signals + 0.2:
                return "상승추세"
            elif bearish_signals > bullish_signals + 0.2:
                return "하락추세"
            else:
                return "횡보"
                
        except Exception as e:
            logger.error(f"전체 트렌드 판단 실패: {e}")
            return "중립"
    
    def _generate_summary(self, symbol: str, indicators: TechnicalIndicators, signals: TechnicalSignals) -> str:
        """분석 요약 생성"""
        try:
            summary_parts = []
            
            # RSI 요약
            if indicators.rsi is not None:
                summary_parts.append(f"RSI: {indicators.rsi:.1f} ({signals.rsi_signal})")
            
            # MACD 요약
            if signals.macd_signal != "중립":
                summary_parts.append(f"MACD: {signals.macd_signal}")
            
            # 볼린저밴드 요약
            if signals.bb_signal != "중립":
                summary_parts.append(f"볼린저밴드: {signals.bb_signal}")
            
            # 이동평균 요약
            if signals.ma_trend != "중립":
                summary_parts.append(f"이동평균: {signals.ma_trend}")
            
            # 전체 트렌드
            summary_parts.append(f"전체 추세: {signals.overall_trend}")
            
            return f"{symbol} - " + " | ".join(summary_parts) if summary_parts else f"{symbol} - 분석 데이터 부족"
            
        except Exception as e:
            logger.error(f"요약 생성 실패: {e}")
            return f"{symbol} - 분석 요약 생성 실패"
    
    def _generate_mock_analysis(self, symbol: str) -> TechnicalAnalysisResult:
        """Mock 기술적 분석 결과 생성"""
        import random
        
        # Mock 지표 생성
        indicators = TechnicalIndicators(
            rsi=random.uniform(30, 70),
            macd=random.uniform(-2, 2),
            macd_signal=random.uniform(-2, 2),
            bb_upper=random.uniform(105, 110),
            bb_middle=random.uniform(95, 105),
            bb_lower=random.uniform(85, 95),
            sma_20=random.uniform(90, 100),
            sma_50=random.uniform(85, 95),
            sma_200=random.uniform(80, 90)
        )
        
        # Mock 신호 생성
        signals = TechnicalSignals(
            rsi_signal="중립",
            macd_signal="매수",
            bb_signal="중립",
            ma_trend="상승",
            volume_signal="보통",
            overall_trend="중립"
        )
        
        return TechnicalAnalysisResult(
            symbol=symbol,
            indicators=indicators,
            signals=signals,
            confidence=0.75,
            summary=f"{symbol} Mock 기술적 분석 완료"
        )


# 전역 인스턴스 생성 (하위 호환성)
technical_analyzer = TechnicalAnalyzer()


# 레거시 호환성을 위한 클래스 (기존 코드와의 호환성 유지)
class TechnicalAnalysisResult_Legacy:
    """레거시 호환성을 위한 기술적 분석 결과 클래스"""
    
    def __init__(self, current_values: Dict[str, Any], signals: Dict[str, str], summary: Dict[str, Any]):
        self.current_values = current_values
        self.signals = signals
        self.summary = summary