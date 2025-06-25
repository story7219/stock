#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📈 백테스팅 엔진 v2.0
투자 전략 백테스트 및 성과 분석
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import json
import sqlite3
from pathlib import Path
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class BacktestConfig:
    """백테스트 설정"""
    start_date: str
    end_date: str
    initial_capital: float = 10000000  # 초기 자본 1천만원
    commission: float = 0.0015  # 수수료 0.15%
    slippage: float = 0.001  # 슬리피지 0.1%
    max_positions: int = 10  # 최대 보유 종목 수
    rebalance_frequency: str = "weekly"  # daily, weekly, monthly
    benchmark: str = "^KS11"  # 코스피 지수

@dataclass
class Trade:
    """거래 기록"""
    symbol: str
    entry_date: datetime
    exit_date: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    quantity: int
    trade_type: str  # BUY, SELL
    strategy: str
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None

@dataclass
class PerformanceMetrics:
    """성과 지표"""
    total_return: float
    annual_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_return: float
    best_trade: float
    worst_trade: float
    calmar_ratio: float
    sortino_ratio: float
    beta: float
    alpha: float
    information_ratio: float

class BacktestingEngine:
    """백테스팅 엔진"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """백테스팅 엔진 초기화"""
        self.config = config or {
            'initial_capital': 1000000,
            'commission_rate': 0.001,
            'slippage_rate': 0.0005,
            'risk_free_rate': 0.02,
            'start_date': '2023-01-01',
            'end_date': '2024-01-01',
            'benchmark': 'SPY',
            'rebalance_frequency': 'monthly'
        }
        self.results = {}
        self.performance_metrics = {}
        logger.info("백테스팅 엔진 초기화 완료")
        self.trades: List[Trade] = []
        self.portfolio_value = []
        self.positions = {}  # symbol -> quantity
        self.cash = self.config['initial_capital']
        self.benchmark_data = None
        self.results_db = "data/backtest_results.db"
        self.initialize_database()
    
    def initialize_database(self):
        """데이터베이스 초기화"""
        Path("data").mkdir(exist_ok=True)
        conn = sqlite3.connect(self.results_db)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS backtest_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_name TEXT,
                start_date TEXT,
                end_date TEXT,
                total_return REAL,
                annual_return REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                win_rate REAL,
                total_trades INTEGER,
                config TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trade_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                backtest_id INTEGER,
                symbol TEXT,
                entry_date TEXT,
                exit_date TEXT,
                entry_price REAL,
                exit_price REAL,
                quantity INTEGER,
                pnl REAL,
                pnl_pct REAL,
                strategy TEXT,
                FOREIGN KEY (backtest_id) REFERENCES backtest_results (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    async def run_backtest(self, strategy_func, symbols: List[str], strategy_name: str) -> Dict[str, Any]:
        """백테스트 실행"""
        try:
            logger.info(f"백테스트 시작: {strategy_name}")
            
            # 데이터 수집
            price_data = await self._collect_price_data(symbols)
            if price_data.empty:
                raise ValueError("가격 데이터를 수집할 수 없습니다")
            
            # 벤치마크 데이터 수집
            self.benchmark_data = await self._collect_benchmark_data()
            
            # 백테스트 실행
            await self._execute_backtest(strategy_func, price_data, strategy_name)
            
            # 성과 분석
            performance = self._calculate_performance()
            
            # 결과 저장
            backtest_id = self._save_results(strategy_name, performance)
            
            # 리포트 생성
            report = self._generate_report(performance, strategy_name)
            
            logger.info(f"백테스트 완료: {strategy_name}")
            return {
                "backtest_id": backtest_id,
                "performance": performance,
                "report": report,
                "trades": [asdict(trade) for trade in self.trades]
            }
            
        except Exception as e:
            logger.error(f"백테스트 실행 오류: {e}")
            raise
    
    async def _collect_price_data(self, symbols: List[str]) -> pd.DataFrame:
        """가격 데이터 수집"""
        try:
            data_frames = []
            
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(
                        start=self.config['start_date'],
                        end=self.config['end_date'],
                        interval='1d'
                    )
                    
                    if not hist.empty:
                        hist['Symbol'] = symbol
                        data_frames.append(hist)
                        
                except Exception as e:
                    logger.warning(f"종목 {symbol} 데이터 수집 실패: {e}")
                    continue
            
            if data_frames:
                combined_data = pd.concat(data_frames)
                combined_data.reset_index(inplace=True)
                return combined_data
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"가격 데이터 수집 오류: {e}")
            return pd.DataFrame()
    
    async def _collect_benchmark_data(self) -> pd.DataFrame:
        """벤치마크 데이터 수집"""
        try:
            ticker = yf.Ticker(self.config['benchmark'])
            data = ticker.history(
                start=self.config['start_date'],
                end=self.config['end_date'],
                interval='1d'
            )
            return data
            
        except Exception as e:
            logger.error(f"벤치마크 데이터 수집 오류: {e}")
            return pd.DataFrame()
    
    async def _execute_backtest(self, strategy_func, price_data: pd.DataFrame, strategy_name: str):
        """백테스트 실행"""
        try:
            # 날짜별로 그룹화
            grouped_data = price_data.groupby('Date')
            
            for date, day_data in grouped_data:
                # 포트폴리오 가치 계산
                portfolio_value = self._calculate_portfolio_value(day_data)
                self.portfolio_value.append({
                    'Date': date,
                    'Value': portfolio_value
                })
                
                # 전략 실행
                signals = await strategy_func(day_data, self.positions)
                
                # 신호 처리
                for signal in signals:
                    await self._process_signal(signal, day_data, strategy_name)
                
                # 리밸런싱 체크
                if self._should_rebalance(date):
                    await self._rebalance_portfolio(day_data)
                    
        except Exception as e:
            logger.error(f"백테스트 실행 중 오류: {e}")
            raise
    
    def _calculate_portfolio_value(self, day_data: pd.DataFrame) -> float:
        """포트폴리오 가치 계산"""
        total_value = self.cash
        
        for symbol, quantity in self.positions.items():
            symbol_data = day_data[day_data['Symbol'] == symbol]
            if not symbol_data.empty:
                current_price = symbol_data['Close'].iloc[0]
                total_value += quantity * current_price
        
        return total_value
    
    async def _process_signal(self, signal: Dict[str, Any], day_data: pd.DataFrame, strategy_name: str):
        """매매 신호 처리"""
        try:
            symbol = signal['symbol']
            action = signal['action']  # BUY, SELL
            quantity = signal.get('quantity', 0)
            
            symbol_data = day_data[day_data['Symbol'] == symbol]
            if symbol_data.empty:
                return
            
            current_price = symbol_data['Close'].iloc[0]
            date = symbol_data['Date'].iloc[0]
            
            if action == 'BUY':
                await self._execute_buy(symbol, quantity, current_price, date, strategy_name)
            elif action == 'SELL':
                await self._execute_sell(symbol, quantity, current_price, date, strategy_name)
                
        except Exception as e:
            logger.error(f"신호 처리 오류: {e}")
    
    async def _execute_buy(self, symbol: str, quantity: int, price: float, date: datetime, strategy: str):
        """매수 실행"""
        try:
            # 수수료 및 슬리피지 적용
            adjusted_price = price * (1 + self.config['slippage_rate'])
            total_cost = quantity * adjusted_price * (1 + self.config['commission_rate'])
            
            if total_cost <= self.cash:
                # 거래 실행
                self.cash -= total_cost
                self.positions[symbol] = self.positions.get(symbol, 0) + quantity
                
                # 거래 기록
                trade = Trade(
                    symbol=symbol,
                    entry_date=date,
                    exit_date=None,
                    entry_price=adjusted_price,
                    exit_price=None,
                    quantity=quantity,
                    trade_type="BUY",
                    strategy=strategy
                )
                self.trades.append(trade)
                
                logger.debug(f"매수 실행: {symbol} {quantity}주 @ {adjusted_price:,.0f}")
                
        except Exception as e:
            logger.error(f"매수 실행 오류: {e}")
    
    async def _execute_sell(self, symbol: str, quantity: int, price: float, date: datetime, strategy: str):
        """매도 실행"""
        try:
            if symbol in self.positions and self.positions[symbol] >= quantity:
                # 수수료 및 슬리피지 적용
                adjusted_price = price * (1 - self.config['slippage_rate'])
                total_proceeds = quantity * adjusted_price * (1 - self.config['commission_rate'])
                
                # 거래 실행
                self.cash += total_proceeds
                self.positions[symbol] -= quantity
                
                if self.positions[symbol] == 0:
                    del self.positions[symbol]
                
                # 거래 기록 업데이트 (해당 종목의 가장 오래된 매수 거래 찾기)
                for trade in self.trades:
                    if (trade.symbol == symbol and 
                        trade.trade_type == "BUY" and 
                        trade.exit_date is None):
                        
                        trade.exit_date = date
                        trade.exit_price = adjusted_price
                        trade.pnl = (adjusted_price - trade.entry_price) * quantity
                        trade.pnl_pct = (adjusted_price - trade.entry_price) / trade.entry_price * 100
                        break
                
                logger.debug(f"매도 실행: {symbol} {quantity}주 @ {adjusted_price:,.0f}")
                
        except Exception as e:
            logger.error(f"매도 실행 오류: {e}")
    
    def _should_rebalance(self, date: datetime) -> bool:
        """리밸런싱 필요 여부 확인"""
        if self.config['rebalance_frequency'] == "daily":
            return True
        elif self.config['rebalance_frequency'] == "weekly":
            return date.weekday() == 0  # 월요일
        elif self.config['rebalance_frequency'] == "monthly":
            return date.day == 1  # 매월 1일
        return False
    
    async def _rebalance_portfolio(self, day_data: pd.DataFrame):
        """포트폴리오 리밸런싱"""
        # 현재는 단순 구현, 향후 정교한 리밸런싱 로직 추가
        pass
    
    def _calculate_performance(self) -> PerformanceMetrics:
        """성과 지표 계산"""
        try:
            if not self.portfolio_value:
                return PerformanceMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            
            # 포트폴리오 가치 시계열 생성
            df = pd.DataFrame(self.portfolio_value)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            df['Returns'] = df['Value'].pct_change()
            
            # 기본 성과 지표
            total_return = (df['Value'].iloc[-1] - df['Value'].iloc[0]) / df['Value'].iloc[0] * 100
            annual_return = self._calculate_annual_return(df)
            volatility = df['Returns'].std() * np.sqrt(252) * 100
            sharpe_ratio = self._calculate_sharpe_ratio(df)
            max_drawdown = self._calculate_max_drawdown(df)
            
            # 거래 관련 지표
            completed_trades = [t for t in self.trades if t.exit_date is not None]
            win_trades = [t for t in completed_trades if t.pnl > 0]
            
            win_rate = len(win_trades) / len(completed_trades) * 100 if completed_trades else 0
            profit_factor = self._calculate_profit_factor(completed_trades)
            avg_trade_return = np.mean([t.pnl_pct for t in completed_trades]) if completed_trades else 0
            best_trade = max([t.pnl_pct for t in completed_trades]) if completed_trades else 0
            worst_trade = min([t.pnl_pct for t in completed_trades]) if completed_trades else 0
            
            # 고급 지표
            calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
            sortino_ratio = self._calculate_sortino_ratio(df)
            beta, alpha = self._calculate_beta_alpha(df)
            information_ratio = self._calculate_information_ratio(df)
            
            return PerformanceMetrics(
                total_return=total_return,
                annual_return=annual_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                profit_factor=profit_factor,
                total_trades=len(completed_trades),
                avg_trade_return=avg_trade_return,
                best_trade=best_trade,
                worst_trade=worst_trade,
                calmar_ratio=calmar_ratio,
                sortino_ratio=sortino_ratio,
                beta=beta,
                alpha=alpha,
                information_ratio=information_ratio
            )
            
        except Exception as e:
            logger.error(f"성과 지표 계산 오류: {e}")
            return PerformanceMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    
    def _calculate_annual_return(self, df: pd.DataFrame) -> float:
        """연간 수익률 계산"""
        days = (df.index[-1] - df.index[0]).days
        years = days / 365.25
        total_return = (df['Value'].iloc[-1] - df['Value'].iloc[0]) / df['Value'].iloc[0]
        return (1 + total_return) ** (1/years) - 1 if years > 0 else 0
    
    def _calculate_sharpe_ratio(self, df: pd.DataFrame) -> float:
        """샤프 비율 계산"""
        excess_returns = df['Returns'] - self.config['risk_free_rate']/252  # 무위험 수익률 2.5% 가정
        return excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() != 0 else 0
    
    def _calculate_max_drawdown(self, df: pd.DataFrame) -> float:
        """최대 낙폭 계산"""
        peak = df['Value'].expanding().max()
        drawdown = (df['Value'] - peak) / peak * 100
        return drawdown.min()
    
    def _calculate_profit_factor(self, trades: List[Trade]) -> float:
        """수익 팩터 계산"""
        profits = sum([t.pnl for t in trades if t.pnl > 0])
        losses = abs(sum([t.pnl for t in trades if t.pnl < 0]))
        return profits / losses if losses != 0 else 0
    
    def _calculate_sortino_ratio(self, df: pd.DataFrame) -> float:
        """소르티노 비율 계산"""
        negative_returns = df['Returns'][df['Returns'] < 0]
        downside_deviation = negative_returns.std() * np.sqrt(252)
        excess_return = df['Returns'].mean() * 252 - self.config['risk_free_rate']  # 무위험 수익률 2.5%
        return excess_return / downside_deviation if downside_deviation != 0 else 0
    
    def _calculate_beta_alpha(self, df: pd.DataFrame) -> Tuple[float, float]:
        """베타와 알파 계산"""
        if self.benchmark_data is None or self.benchmark_data.empty:
            return 0, 0
        
        try:
            # 벤치마크 수익률 계산
            benchmark_returns = self.benchmark_data['Close'].pct_change().dropna()
            portfolio_returns = df['Returns'].dropna()
            
            # 날짜 맞추기
            common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
            if len(common_dates) < 2:
                return 0, 0
            
            portfolio_aligned = portfolio_returns.loc[common_dates]
            benchmark_aligned = benchmark_returns.loc[common_dates]
            
            # 베타 계산
            covariance = np.cov(portfolio_aligned, benchmark_aligned)[0, 1]
            benchmark_variance = np.var(benchmark_aligned)
            beta = covariance / benchmark_variance if benchmark_variance != 0 else 0
            
            # 알파 계산
            portfolio_mean = portfolio_aligned.mean() * 252
            benchmark_mean = benchmark_aligned.mean() * 252
            alpha = portfolio_mean - (self.config['risk_free_rate'] + beta * (benchmark_mean - self.config['risk_free_rate']))
            
            return beta, alpha
            
        except Exception as e:
            logger.error(f"베타/알파 계산 오류: {e}")
            return 0, 0
    
    def _calculate_information_ratio(self, df: pd.DataFrame) -> float:
        """정보 비율 계산"""
        if self.benchmark_data is None or self.benchmark_data.empty:
            return 0
        
        try:
            benchmark_returns = self.benchmark_data['Close'].pct_change().dropna()
            portfolio_returns = df['Returns'].dropna()
            
            common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
            if len(common_dates) < 2:
                return 0
            
            portfolio_aligned = portfolio_returns.loc[common_dates]
            benchmark_aligned = benchmark_returns.loc[common_dates]
            
            excess_returns = portfolio_aligned - benchmark_aligned
            tracking_error = excess_returns.std() * np.sqrt(252)
            
            return excess_returns.mean() * 252 / tracking_error if tracking_error != 0 else 0
            
        except Exception as e:
            logger.error(f"정보 비율 계산 오류: {e}")
            return 0
    
    def _save_results(self, strategy_name: str, performance: PerformanceMetrics) -> int:
        """결과 저장"""
        try:
            conn = sqlite3.connect(self.results_db)
            cursor = conn.cursor()
            
            # 백테스트 결과 저장
            cursor.execute('''
                INSERT INTO backtest_results 
                (strategy_name, start_date, end_date, total_return, annual_return, 
                 sharpe_ratio, max_drawdown, win_rate, total_trades, config)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                strategy_name,
                self.config['start_date'],
                self.config['end_date'],
                performance.total_return,
                performance.annual_return,
                performance.sharpe_ratio,
                performance.max_drawdown,
                performance.win_rate,
                performance.total_trades,
                json.dumps(self.config)
            ))
            
            backtest_id = cursor.lastrowid
            
            # 거래 기록 저장
            for trade in self.trades:
                if trade.exit_date is not None:
                    cursor.execute('''
                        INSERT INTO trade_history 
                        (backtest_id, symbol, entry_date, exit_date, entry_price, 
                         exit_price, quantity, pnl, pnl_pct, strategy)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        backtest_id,
                        trade.symbol,
                        trade.entry_date.isoformat(),
                        trade.exit_date.isoformat(),
                        trade.entry_price,
                        trade.exit_price,
                        trade.quantity,
                        trade.pnl,
                        trade.pnl_pct,
                        trade.strategy
                    ))
            
            conn.commit()
            conn.close()
            
            return backtest_id
            
        except Exception as e:
            logger.error(f"결과 저장 오류: {e}")
            return -1
    
    def _generate_report(self, performance: PerformanceMetrics, strategy_name: str) -> str:
        """백테스트 리포트 생성"""
        report = f"""
=== 백테스트 결과 리포트 ===
전략명: {strategy_name}
기간: {self.config['start_date']} ~ {self.config['end_date']}
초기 자본: {self.config['initial_capital']:,.0f}원

=== 수익률 지표 ===
총 수익률: {performance.total_return:.2f}%
연간 수익률: {performance.annual_return:.2f}%
변동성: {performance.volatility:.2f}%
샤프 비율: {performance.sharpe_ratio:.2f}
최대 낙폭: {performance.max_drawdown:.2f}%

=== 거래 지표 ===
총 거래 횟수: {performance.total_trades}
승률: {performance.win_rate:.2f}%
수익 팩터: {performance.profit_factor:.2f}
평균 거래 수익률: {performance.avg_trade_return:.2f}%
최고 거래: {performance.best_trade:.2f}%
최악 거래: {performance.worst_trade:.2f}%

=== 고급 지표 ===
칼마 비율: {performance.calmar_ratio:.2f}
소르티노 비율: {performance.sortino_ratio:.2f}
베타: {performance.beta:.2f}
알파: {performance.alpha:.2f}%
정보 비율: {performance.information_ratio:.2f}
"""
        return report
    
    def generate_charts(self, backtest_id: int, output_dir: str = "reports"):
        """차트 생성"""
        try:
            Path(output_dir).mkdir(exist_ok=True)
            
            # 포트폴리오 가치 차트
            df = pd.DataFrame(self.portfolio_value)
            df['Date'] = pd.to_datetime(df['Date'])
            
            plt.figure(figsize=(15, 10))
            
            # 포트폴리오 가치 추이
            plt.subplot(2, 2, 1)
            plt.plot(df['Date'], df['Value'])
            plt.title('포트폴리오 가치 추이')
            plt.xlabel('날짜')
            plt.ylabel('가치 (원)')
            plt.xticks(rotation=45)
            
            # 일별 수익률 분포
            plt.subplot(2, 2, 2)
            returns = df['Value'].pct_change().dropna() * 100
            plt.hist(returns, bins=50, alpha=0.7)
            plt.title('일별 수익률 분포')
            plt.xlabel('수익률 (%)')
            plt.ylabel('빈도')
            
            # 드로우다운 차트
            plt.subplot(2, 2, 3)
            peak = df['Value'].expanding().max()
            drawdown = (df['Value'] - peak) / peak * 100
            plt.fill_between(df['Date'], drawdown, 0, alpha=0.3, color='red')
            plt.title('드로우다운')
            plt.xlabel('날짜')
            plt.ylabel('드로우다운 (%)')
            plt.xticks(rotation=45)
            
            # 월별 수익률 히트맵
            plt.subplot(2, 2, 4)
            df['Month'] = df['Date'].dt.month
            df['Year'] = df['Date'].dt.year
            monthly_returns = df.groupby(['Year', 'Month'])['Value'].last().pct_change() * 100
            monthly_returns = monthly_returns.unstack(level=1)
            
            if not monthly_returns.empty:
                sns.heatmap(monthly_returns, annot=True, fmt='.1f', cmap='RdYlGn', center=0)
                plt.title('월별 수익률 히트맵')
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/backtest_{backtest_id}_charts.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"차트 생성 완료: {output_dir}/backtest_{backtest_id}_charts.png")
            
        except Exception as e:
            logger.error(f"차트 생성 오류: {e}")

# 사용 예제
async def example_strategy(day_data: pd.DataFrame, positions: Dict[str, int]) -> List[Dict[str, Any]]:
    """예제 전략 - 단순 이동평균 전략"""
    signals = []
    
    for _, row in day_data.iterrows():
        symbol = row['Symbol']
        close_price = row['Close']
        
        # 단순한 예제: 임의의 조건으로 매매 신호 생성
        if symbol not in positions:
            # 매수 신호
            signals.append({
                'symbol': symbol,
                'action': 'BUY',
                'quantity': 100
            })
    
    return signals 