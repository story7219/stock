#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: market_analyzer.py
모듈: 시장 상황 분석 및 자동 전략 선택
목적: 변동성, 추세, 거래량 분석으로 최적 전략 자동 선택

Author: World-Class Python Engineer
Created: 2025-07-13
Version: 1.0.0
"""
from __future__ import annotations
from typing import Dict, Any, List, Literal
import pandas as pd
import numpy as np
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class TradingStrategy(Enum):
    """거래 전략 열거형"""
    DAY_TRADING = "DAY_TRADING"
    SWING_TRADING = "SWING_TRADING" 
    MEDIUM_TERM = "MEDIUM_TERM"

class MarketAnalyzer:
    """시장 상황 분석 및 자동 전략 선택"""
    
    def __init__(self):
        self.strategy_history: List[TradingStrategy] = []
    
    def analyze_market_condition(self, df: pd.DataFrame) -> Dict[str, Any]:
        """시장 상황 분석: 변동성, 추세, 거래량 기반"""
        try:
            # 변동성 분석 (20일 기준)
            volatility = df['return'].rolling(20).std().iloc[-1] * np.sqrt(252)
            
            # 추세 분석 (50일 이동평균 기울기)
            ma_50 = df['종가'].rolling(50).mean()
            trend_strength = (ma_50.iloc[-1] - ma_50.iloc[-20]) / ma_50.iloc[-20]
            
            # 거래량 분석 (20일 평균 대비)
            volume_ratio = df['거래량'].iloc[-5:].mean() / df['거래량'].rolling(20).mean().iloc[-1]
            
            # 시장 상황 판단
            market_condition = self._determine_market_condition(volatility, trend_strength, volume_ratio)
            
            # 최적 전략 선택
            strategy = self._select_optimal_strategy(market_condition, volatility, trend_strength, volume_ratio)
            
            result = {
                'market_condition': market_condition,
                'strategy': strategy.value,
                'volatility': volatility,
                'trend_strength': trend_strength,
                'volume_ratio': volume_ratio,
                'analysis_date': df['날짜'].iloc[-1]
            }
            
            self.strategy_history.append(strategy)
            logger.info(f"시장 분석 완료: {result}")
            return result
            
        except Exception as e:
            logger.error(f"시장 분석 오류: {e}")
            raise
    
    def _determine_market_condition(self, volatility: float, trend_strength: float, volume_ratio: float) -> str:
        """시장 상황 판단"""
        if volatility > 0.3:  # 고변동성
            if volume_ratio > 1.5:  # 고거래량
                return "HIGH_VOLATILITY_HIGH_VOLUME"
            else:
                return "HIGH_VOLATILITY_LOW_VOLUME"
        elif trend_strength > 0.05:  # 강한 상승추세
            return "STRONG_UPTREND"
        elif trend_strength < -0.05:  # 강한 하락추세
            return "STRONG_DOWNTREND"
        elif abs(trend_strength) < 0.02:  # 횡보장
            return "SIDEWAYS_MARKET"
        else:
            return "NORMAL_MARKET"
    
    def _select_optimal_strategy(self, market_condition: str, volatility: float, trend_strength: float, volume_ratio: float) -> TradingStrategy:
        """최적 전략 선택"""
        if market_condition in ["HIGH_VOLATILITY_HIGH_VOLUME", "HIGH_VOLATILITY_LOW_VOLUME"]:
            return TradingStrategy.DAY_TRADING  # 고변동성 → 데이트레이딩
        elif market_condition in ["STRONG_UPTREND", "STRONG_DOWNTREND"]:
            return TradingStrategy.SWING_TRADING  # 강한 추세 → 스윙매매
        elif market_condition == "SIDEWAYS_MARKET":
            return TradingStrategy.MEDIUM_TERM  # 횡보장 → 중기투자
        else:
            # 기본값: 스윙매매
            return TradingStrategy.SWING_TRADING 