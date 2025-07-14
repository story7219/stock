#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: backtest_engine.py
모듈: 실제 백테스트 엔진
목적: 전체 자동매매 시스템 시뮬레이션 (시장분석→스크리닝→시그널→매매→성능)

Author: World-Class Python Engineer
Created: 2025-07-13
Version: 1.0.0
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

from .market_analyzer import MarketAnalyzer
from .stock_screener import StockScreener
from .signal_generator import SignalGenerator
from .performance import PerformanceAnalyzer
from .risk import RiskAnalyzer
from .checklist import ChecklistEvaluator

logger = logging.getLogger(__name__)

class BacktestEngine:
    """실제 백테스트 엔진: 전체 자동매매 시스템 시뮬레이션"""
    
    def __init__(self, initial_capital: float = 10000000):
        self.initial_capital = initial_capital
        self.market_analyzer = MarketAnalyzer()
        self.stock_screener = StockScreener()
        self.signal_generator = SignalGenerator()
        self.performance_analyzer = PerformanceAnalyzer()
        self.risk_analyzer = RiskAnalyzer()
        self.checklist_evaluator = ChecklistEvaluator()
        
        # 백테스트 결과 저장
        self.trading_history: List[Dict[str, Any]] = []
        self.portfolio_history: List[Dict[str, Any]] = []
        self.performance_history: List[Dict[str, Any]] = []
    
    def run_backtest(self, start_date: str = "2013-01-01", end_date: str = "2024-12-31", 
                    rebalance_frequency: str = "monthly") -> Dict[str, Any]:
        """전체 백테스트 실행"""
        try:
            logger.info(f"백테스트 시작: {start_date} ~ {end_date}")
            
            # 1. 시장 데이터 로드
            market_data = self._load_market_data()
            
            # 2. 백테스트 기간 설정
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            # 3. 백테스트 실행
            current_date = start_dt
            current_capital = self.initial_capital
            current_portfolio = {}
            
            while current_date <= end_dt:
                try:
                    # 시장 상황 분석
                    market_analysis = self._analyze_market_at_date(market_data, current_date)
                    
                    # 포트폴리오 재구성 (월별 또는 시장 변화 시)
                    if self._should_rebalance(current_date, market_analysis, rebalance_frequency):
                        portfolio = self._rebalance_portfolio(market_data, current_date, market_analysis)
                        current_portfolio = portfolio
                    
                    # 매매 시그널 생성
                    signals = self._generate_trading_signals(current_portfolio, market_data, current_date)
                    
                    # 매매 실행 및 성과 기록
                    trading_result = self._execute_trades(current_portfolio, signals, market_data, current_date)
                    
                    # 성과 업데이트
                    current_capital = trading_result['total_value']
                    
                    # 히스토리 저장
                    self._save_history(current_date, market_analysis, current_portfolio, signals, trading_result)
                    
                    # 다음 날짜로 이동
                    current_date += timedelta(days=1)
                    
                except Exception as e:
                    logger.warning(f"날짜 {current_date} 처리 오류: {e}")
                    current_date += timedelta(days=1)
                    continue
            
            # 4. 최종 성과 분석
            final_results = self._analyze_final_results()
            
            logger.info("백테스트 완료")
            return final_results
            
        except Exception as e:
            logger.error(f"백테스트 실행 오류: {e}")
            raise
    
    def _load_market_data(self) -> Dict[str, pd.DataFrame]:
        """시장 데이터 로드"""
        try:
            market_data = {}
            data_dir = "backup/krx_k200_kosdaq50/krx_backup_20250712_054858/"
            
            # 주요 종목들만 로드 (성능 고려)
            target_stocks = ['005930', '000660', '035420', '051910', '006400']  # 삼성전자, SK하이닉스, NAVER, LG화학, 삼성SDI
            
            for stock_code in target_stocks:
                try:
                    file_pattern = f"*_{stock_code}_backup_backup.parquet"
                    import glob
                    import os
                    files = glob.glob(os.path.join(data_dir, file_pattern))
                    
                    if files:
                        df = pd.read_parquet(files[0])
                        df = df.sort_values('날짜')
                        df['return'] = df['종가'].pct_change()
                        df = df.dropna()
                        market_data[stock_code] = df
                        logger.info(f"종목 {stock_code} 데이터 로드 완료")
                        
                except Exception as e:
                    logger.warning(f"종목 {stock_code} 데이터 로드 오류: {e}")
                    continue
            
            logger.info(f"총 {len(market_data)}개 종목 데이터 로드 완료")
            return market_data
            
        except Exception as e:
            logger.error(f"시장 데이터 로드 오류: {e}")
            raise
    
    def _analyze_market_at_date(self, market_data: Dict[str, pd.DataFrame], date: datetime) -> Dict[str, Any]:
        """특정 날짜의 시장 상황 분석"""
        try:
            # 가장 많은 데이터를 가진 종목으로 시장 분석
            reference_stock = max(market_data.keys(), key=lambda x: len(market_data[x]))
            df = market_data[reference_stock]
            
            # 해당 날짜까지의 데이터만 사용
            df_until_date = df[df['날짜'] <= date.strftime('%Y-%m-%d')]
            
            if len(df_until_date) < 50:
                return {
                    'market_condition': 'NORMAL_MARKET',
                    'strategy': 'SWING_TRADING',
                    'volatility': 0.2,
                    'trend_strength': 0.0,
                    'volume_ratio': 1.0
                }
            
            return self.market_analyzer.analyze_market_condition(df_until_date)
            
        except Exception as e:
            logger.warning(f"시장 분석 오류: {e}")
            return {
                'market_condition': 'NORMAL_MARKET',
                'strategy': 'SWING_TRADING',
                'volatility': 0.2,
                'trend_strength': 0.0,
                'volume_ratio': 1.0
            }
    
    def _should_rebalance(self, date: datetime, market_analysis: Dict[str, Any], frequency: str) -> bool:
        """포트폴리오 재구성 필요 여부 판단"""
        if frequency == "monthly":
            return date.day == 1  # 매월 1일
        elif frequency == "quarterly":
            return date.day == 1 and date.month in [1, 4, 7, 10]  # 분기별
        else:
            return True  # 매일
    
    def _rebalance_portfolio(self, market_data: Dict[str, pd.DataFrame], date: datetime, 
                           market_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """포트폴리오 재구성"""
        try:
            # 종목 스크리닝
            selected_stocks = self.stock_screener.screen_stocks(
                market_analysis['market_condition'],
                market_analysis['strategy'],
                max_stocks=5
            )
            
            # 포트폴리오 구성
            portfolio = self.stock_screener.create_portfolio(selected_stocks, market_analysis['strategy'])
            
            logger.info(f"포트폴리오 재구성 완료: {len(selected_stocks)}개 종목")
            return portfolio
            
        except Exception as e:
            logger.error(f"포트폴리오 재구성 오류: {e}")
            return {'strategy': 'SWING_TRADING', 'stocks': [], 'weights': [], 'total_stocks': 0}
    
    def _generate_trading_signals(self, portfolio: Dict[str, Any], market_data: Dict[str, pd.DataFrame], 
                                date: datetime) -> Dict[str, Any]:
        """매매 시그널 생성"""
        try:
            # 해당 날짜까지의 데이터만 사용
            filtered_data = {}
            for stock_code, df in market_data.items():
                df_until_date = df[df['날짜'] <= date.strftime('%Y-%m-%d')]
                if len(df_until_date) > 20:
                    filtered_data[stock_code] = df_until_date
            
            signals = self.signal_generator.generate_signals(portfolio, filtered_data)
            return signals
            
        except Exception as e:
            logger.error(f"시그널 생성 오류: {e}")
            return {'strategy': 'SWING_TRADING', 'individual_signals': [], 'portfolio_signal': {}, 'total_signals': 0}
    
    def _execute_trades(self, portfolio: Dict[str, Any], signals: Dict[str, Any], 
                       market_data: Dict[str, pd.DataFrame], date: datetime) -> Dict[str, Any]:
        """매매 실행 시뮬레이션"""
        try:
            total_value = 0
            trades = []
            
            for signal in signals.get('individual_signals', []):
                stock_code = signal['stock_code']
                signal_type = signal['signal_type']
                weight = signal['weight']
                
                if stock_code in market_data:
                    df = market_data[stock_code]
                    df_until_date = df[df['날짜'] <= date.strftime('%Y-%m-%d')]
                    
                    if len(df_until_date) > 0:
                        current_price = df_until_date['종가'].iloc[-1]
                        position_value = self.initial_capital * weight
                        
                        if signal_type == 'BUY':
                            shares = position_value / current_price
                            total_value += position_value
                        elif signal_type == 'SELL':
                            shares = 0
                            total_value += 0
                        else:  # HOLD
                            shares = position_value / current_price
                            total_value += position_value
                        
                        trades.append({
                            'stock_code': stock_code,
                            'signal_type': signal_type,
                            'price': current_price,
                            'shares': shares,
                            'value': position_value
                        })
            
            return {
                'date': date.strftime('%Y-%m-%d'),
                'total_value': total_value,
                'trades': trades,
                'portfolio_signal': signals.get('portfolio_signal', {})
            }
            
        except Exception as e:
            logger.error(f"매매 실행 오류: {e}")
            return {
                'date': date.strftime('%Y-%m-%d'),
                'total_value': self.initial_capital,
                'trades': [],
                'portfolio_signal': {}
            }
    
    def _save_history(self, date: datetime, market_analysis: Dict[str, Any], portfolio: Dict[str, Any], 
                     signals: Dict[str, Any], trading_result: Dict[str, Any]):
        """히스토리 저장"""
        self.trading_history.append({
            'date': date.strftime('%Y-%m-%d'),
            'market_analysis': market_analysis,
            'portfolio': portfolio,
            'signals': signals,
            'trading_result': trading_result
        })
    
    def _analyze_final_results(self) -> Dict[str, Any]:
        """최종 성과 분석"""
        try:
            # 성과 데이터 추출
            daily_returns = []
            portfolio_values = []
            
            for record in self.trading_history:
                total_value = record['trading_result']['total_value']
                portfolio_values.append(total_value)
                
                if len(portfolio_values) > 1:
                    daily_return = (total_value - portfolio_values[-2]) / portfolio_values[-2]
                    daily_returns.append(daily_return)
            
            # 성과 분석
            if len(daily_returns) > 0:
                perf_df = pd.DataFrame({'return': daily_returns})
                performance = self.performance_analyzer.analyze(perf_df)
                risk = self.risk_analyzer.analyze(perf_df)
                checklist = self.checklist_evaluator.evaluate(performance, risk)
            else:
                performance = {'total_return': 0, 'annualized_return': 0, 'sharpe': 0, 'max_drawdown': 0, 'volatility': 0}
                risk = {'p_value': 1.0, 'stress_loss': 0}
                checklist = {}
            
            return {
                'performance': performance,
                'risk': risk,
                'checklist': checklist,
                'trading_history': self.trading_history,
                'final_value': portfolio_values[-1] if portfolio_values else self.initial_capital,
                'total_trades': len(self.trading_history)
            }
            
        except Exception as e:
            logger.error(f"최종 성과 분석 오류: {e}")
            raise 