#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: signal_generator.py
모듈: 매매 시그널 생성
목적: 전략별 매수/매도/청산 시그널 생성 및 타이밍 결정

Author: World-Class Python Engineer
Created: 2025-07-13
Version: 1.0.0
"""
from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class SignalType(Enum):
    """매매 시그널 타입"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

class SignalGenerator:
    """매매 시그널 생성"""
    
    def __init__(self):
        self.signal_history: List[Dict[str, Any]] = []
    
    def generate_signals(self, portfolio: Dict[str, Any], market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """포트폴리오 전체에 대한 매매 시그널 생성"""
        try:
            strategy = portfolio['strategy']
            stocks = portfolio['stocks']
            weights = portfolio['weights']
            
            all_signals = []
            
            for i, stock in enumerate(stocks):
                stock_code = stock['stock_code']
                weight = weights[i]
                
                if stock_code in market_data:
                    stock_data = market_data[stock_code]
                    signal = self._generate_stock_signal(stock_data, strategy, stock_code, weight)
                    all_signals.append(signal)
            
            # 포트폴리오 레벨 시그널 생성
            portfolio_signal = self._generate_portfolio_signal(all_signals, strategy)
            
            result = {
                'strategy': strategy,
                'individual_signals': all_signals,
                'portfolio_signal': portfolio_signal,
                'total_signals': len(all_signals)
            }
            
            self.signal_history.append(result)
            logger.info(f"시그널 생성 완료: {len(all_signals)}개 종목")
            return result
            
        except Exception as e:
            logger.error(f"시그널 생성 오류: {e}")
            raise
    
    def _generate_stock_signal(self, df: pd.DataFrame, strategy: str, stock_code: str, weight: float) -> Dict[str, Any]:
        """개별 종목 시그널 생성"""
        try:
            # 기술적 지표 계산
            ma_5 = df['종가'].rolling(5).mean()
            ma_20 = df['종가'].rolling(20).mean()
            ma_50 = df['종가'].rolling(50).mean()
            
            # RSI 계산
            delta = df['종가'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # 볼린저 밴드 계산
            bb_20 = df['종가'].rolling(20).mean()
            bb_std = df['종가'].rolling(20).std()
            bb_upper = bb_20 + (bb_std * 2)
            bb_lower = bb_20 - (bb_std * 2)
            
            # 현재 값들
            current_price = df['종가'].iloc[-1]
            current_rsi = rsi.iloc[-1]
            current_bb_upper = bb_upper.iloc[-1]
            current_bb_lower = bb_lower.iloc[-1]
            
            # 전략별 시그널 생성
            signal_type, confidence, reason = self._determine_signal(
                strategy, current_price, current_rsi, current_bb_upper, current_bb_lower,
                ma_5.iloc[-1], ma_20.iloc[-1], ma_50.iloc[-1]
            )
            
            return {
                'stock_code': stock_code,
                'signal_type': signal_type.value,
                'confidence': confidence,
                'reason': reason,
                'current_price': current_price,
                'weight': weight,
                'rsi': current_rsi,
                'bb_position': (current_price - current_bb_lower) / (current_bb_upper - current_bb_lower)
            }
            
        except Exception as e:
            logger.warning(f"종목 {stock_code} 시그널 생성 오류: {e}")
            return {
                'stock_code': stock_code,
                'signal_type': SignalType.HOLD.value,
                'confidence': 0.0,
                'reason': f"오류: {e}",
                'current_price': 0,
                'weight': weight,
                'rsi': 50,
                'bb_position': 0.5
            }
    
    def _determine_signal(self, strategy: str, price: float, rsi: float, bb_upper: float, 
                         bb_lower: float, ma_5: float, ma_20: float, ma_50: float) -> Tuple[SignalType, float, str]:
        """시그널 타입 결정"""
        try:
            if strategy == "DAY_TRADING":
                return self._day_trading_signal(price, rsi, bb_upper, bb_lower, ma_5)
            elif strategy == "SWING_TRADING":
                return self._swing_trading_signal(price, rsi, bb_upper, bb_lower, ma_20)
            else:  # MEDIUM_TERM
                return self._medium_term_signal(price, rsi, bb_upper, bb_lower, ma_50)
                
        except Exception as e:
            logger.warning(f"시그널 결정 오류: {e}")
            return SignalType.HOLD, 0.0, f"오류: {e}"
    
    def _day_trading_signal(self, price: float, rsi: float, bb_upper: float, bb_lower: float, ma_5: float) -> Tuple[SignalType, float, str]:
        """데이트레이딩 시그널"""
        # 단기 모멘텀 기반
        if price > ma_5 * 1.01 and rsi < 70:
            return SignalType.BUY, 0.8, "단기 상승 모멘텀"
        elif price < ma_5 * 0.99 or rsi > 80:
            return SignalType.SELL, 0.8, "단기 하락 모멘텀"
        else:
            return SignalType.HOLD, 0.5, "중립"
    
    def _swing_trading_signal(self, price: float, rsi: float, bb_upper: float, bb_lower: float, ma_20: float) -> Tuple[SignalType, float, str]:
        """스윙매매 시그널"""
        # 중기 추세 기반
        if price > ma_20 * 1.02 and 30 < rsi < 70:
            return SignalType.BUY, 0.7, "중기 상승 추세"
        elif price < ma_20 * 0.98 or rsi > 75:
            return SignalType.SELL, 0.7, "중기 하락 추세"
        else:
            return SignalType.HOLD, 0.6, "중립"
    
    def _medium_term_signal(self, price: float, rsi: float, bb_upper: float, bb_lower: float, ma_50: float) -> Tuple[SignalType, float, str]:
        """중기투자 시그널"""
        # 장기 추세 기반
        if price > ma_50 * 1.05 and 40 < rsi < 60:
            return SignalType.BUY, 0.6, "장기 상승 추세"
        elif price < ma_50 * 0.95 or rsi > 80:
            return SignalType.SELL, 0.6, "장기 하락 추세"
        else:
            return SignalType.HOLD, 0.7, "중립"
    
    def _generate_portfolio_signal(self, signals: List[Dict[str, Any]], strategy: str) -> Dict[str, Any]:
        """포트폴리오 레벨 시그널 생성"""
        try:
            # 가중 평균 신뢰도 계산
            total_weight = sum(signal['weight'] for signal in signals)
            weighted_confidence = sum(signal['confidence'] * signal['weight'] for signal in signals) / total_weight
            
            # 시그널 분포 분석
            buy_signals = [s for s in signals if s['signal_type'] == SignalType.BUY.value]
            sell_signals = [s for s in signals if s['signal_type'] == SignalType.SELL.value]
            
            buy_weight = sum(s['weight'] for s in buy_signals)
            sell_weight = sum(s['weight'] for s in sell_signals)
            
            # 포트폴리오 시그널 결정
            if buy_weight > sell_weight * 1.5:
                portfolio_signal = SignalType.BUY.value
                reason = f"매수 우세 ({len(buy_signals)}개 종목)"
            elif sell_weight > buy_weight * 1.5:
                portfolio_signal = SignalType.SELL.value
                reason = f"매도 우세 ({len(sell_signals)}개 종목)"
            else:
                portfolio_signal = SignalType.HOLD.value
                reason = f"중립 ({len(signals)}개 종목)"
            
            return {
                'signal_type': portfolio_signal,
                'confidence': weighted_confidence,
                'reason': reason,
                'buy_weight': buy_weight,
                'sell_weight': sell_weight,
                'buy_count': len(buy_signals),
                'sell_count': len(sell_signals)
            }
            
        except Exception as e:
            logger.error(f"포트폴리오 시그널 생성 오류: {e}")
            return {
                'signal_type': SignalType.HOLD.value,
                'confidence': 0.0,
                'reason': f"오류: {e}",
                'buy_weight': 0.0,
                'sell_weight': 0.0,
                'buy_count': 0,
                'sell_count': 0
            } 