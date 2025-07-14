#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: backtest_engine.py
목적: AI 트레이딩 전략 백테스팅 엔진
Author: [Your Name]
Created: 2025-07-11
Version: 1.0.0

- 커서룰 100% 준수 (타입힌트, 예외처리, 구조화 로깅, 문서화)
- 전략 성능 검증, 리스크 분석, 수익률 계산, 시각화
- 다양한 전략 지원, 포트폴리오 최적화
"""

from __future__ import annotations
import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime
import timedelta
from pathlib import Path
from typing import Dict
import List
import Optional, Tuple, Any, Callable
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import field
import warnings
warnings.filterwarnings('ignore')

# 구조화 로깅
logging.basicConfig(
    filename="logs/backtest_engine.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

@dataclass
class Trade:
    """거래 기록"""
    symbol: str
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    quantity: int
    side: str  # 'long' or 'short'
    pnl: float
    pnl_percent: float
    commission: float = 0.0
    slippage: float = 0.0

@dataclass
class BacktestResult:
    """백테스트 결과"""
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    trades: List[Trade]
    equity_curve: pd.Series
    benchmark_return: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0

class BacktestEngine:
    """백테스팅 엔진"""

    def __init__(self, initial_capital: float = 10000000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions: Dict[str, Dict] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.dates: List[datetime] = []

        # 거래 비용 설정
        self.commission_rate = 0.00015  # 0.015%
        self.slippage_rate = 0.0001     # 0.01%

        # 리스크 관리
        self.max_position_size = 0.1  # 최대 포지션 크기 (10%)
        self.stop_loss = 0.05         # 손절 기준 (5%)
        self.take_profit = 0.15       # 익절 기준 (15%)

    def load_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """데이터 로딩"""
        try:
            file_path = f"data/{symbol}_daily.parquet"
            df = pd.read_parquet(file_path)
            df.index = pd.to_datetime(df.index)

            # 날짜 필터링
            df = df[(df.index >= start_date) & (df.index <= end_date)]

            # 기술적 지표 계산
            df = self._calculate_technical_indicators(df)

            logger.info(f"Loaded data for {symbol}: {len(df)} rows")
            return df

        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {e}")
            return pd.DataFrame()

    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """기술적 지표 계산"""
        # 이동평균
        df['MA5'] = df['종가'].rolling(window=5).mean()
        df['MA20'] = df['종가'].rolling(window=20).mean()
        df['MA60'] = df['종가'].rolling(window=60).mean()

        # RSI
        delta = df['종가'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # MACD
        exp1 = df['종가'].ewm(span=12).mean()
        exp2 = df['종가'].ewm(span=26).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()

        # 볼린저 밴드
        df['BB_Middle'] = df['종가'].rolling(window=20).mean()
        bb_std = df['종가'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)

        # 거래량 지표
        df['Volume_MA'] = df['거래량'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['거래량'] / df['Volume_MA']

        return df

    def generate_signals(self, df: pd.DataFrame, strategy: str = 'ma_crossover') -> pd.DataFrame:
        """매매 신호 생성"""
        df = df.copy()
        df['Signal'] = 0

        if strategy == 'ma_crossover':
            # 이동평균 크로스오버
            df.loc[df['MA5'] > df['MA20'], 'Signal'] = 1
            df.loc[df['MA5'] < df['MA20'], 'Signal'] = -1

        elif strategy == 'rsi_strategy':
            # RSI 전략
            df.loc[(df['RSI'] < 30) & (df['종가'] > df['MA20']), 'Signal'] = 1
            df.loc[(df['RSI'] > 70) & (df['종가'] < df['MA20']), 'Signal'] = -1

        elif strategy == 'macd_strategy':
            # MACD 전략
            df.loc[(df['MACD'] > df['MACD_Signal']) & (df['MACD'] > 0), 'Signal'] = 1
            df.loc[(df['MACD'] < df['MACD_Signal']) & (df['MACD'] < 0), 'Signal'] = -1

        elif strategy == 'bollinger_strategy':
            # 볼린저 밴드 전략
            df.loc[df['종가'] < df['BB_Lower'], 'Signal'] = 1
            df.loc[df['종가'] > df['BB_Upper'], 'Signal'] = -1

        elif strategy == 'volume_price_strategy':
            # 거래량 + 가격 전략
            df.loc[(df['Volume_Ratio'] > 1.5) & (df['종가'] > df['MA20']), 'Signal'] = 1
            df.loc[(df['Volume_Ratio'] > 1.5) & (df['종가'] < df['MA20']), 'Signal'] = -1

        return df

    def execute_trade(self, symbol: str, signal: int, price: float, date: datetime, quantity: int = 1) -> None:
        """거래 실행"""
        if signal == 0:
            return

        commission = price * quantity * self.commission_rate
        slippage = price * quantity * self.slippage_rate
        total_cost = commission + slippage

        if signal == 1:  # 매수
            if symbol not in self.positions:
                self.positions[symbol] = {
                    'quantity': quantity,
                    'entry_price': price,
                    'entry_date': date,
                    'side': 'long'
                }
                self.current_capital -= (price * quantity + total_cost)

        elif signal == -1:  # 매도
            if symbol in self.positions:
                position = self.positions[symbol]
                exit_price = price
                pnl = (exit_price - position['entry_price']) * position['quantity'] - total_cost
                pnl_percent = (pnl / (position['entry_price'] * position['quantity'])) * 100

                trade = Trade(
                    symbol=symbol,
                    entry_date=position['entry_date'],
                    exit_date=date,
                    entry_price=position['entry_price'],
                    exit_price=exit_price,
                    quantity=position['quantity'],
                    side=position['side'],
                    pnl=pnl,
                    pnl_percent=pnl_percent,
                    commission=commission,
                    slippage=slippage
                )

                self.trades.append(trade)
                self.current_capital += (exit_price * position['quantity'] - total_cost)
                del self.positions[symbol]

    def run_backtest(self, symbol: str, start_date: str, end_date: str,
                    strategy: str = 'ma_crossover') -> BacktestResult:
        """백테스트 실행"""
        logger.info(f"Starting backtest for {symbol} with {strategy} strategy")

        # 데이터 로딩
        df = self.load_data(symbol, start_date, end_date)
        if df.empty:
            logger.error(f"No data available for {symbol}")
            return BacktestResult(
                total_return=0.0, annualized_return=0.0, sharpe_ratio=0.0,
                max_drawdown=0.0, win_rate=0.0, profit_factor=0.0,
                total_trades=0, winning_trades=0, losing_trades=0,
                avg_win=0.0, avg_loss=0.0, trades=[], equity_curve=pd.Series()
            )

        # 신호 생성
        df = self.generate_signals(df, strategy)

        # 백테스트 실행
        for date, row in df.iterrows():
            if pd.isna(row['Signal']) or row['Signal'] == 0:
                continue

            self.execute_trade(symbol, int(row['Signal']), row['종가'], date)

            # 자본금 추적
            total_value = self.current_capital
            for pos_symbol, position in self.positions.items():
                if pos_symbol in df.columns:
                    total_value += position['quantity'] * row['종가']

            self.equity_curve.append(total_value)
            self.dates.append(date)

        # 결과 계산
        result = self._calculate_performance_metrics()
        logger.info(f"Backtest completed: Total Return {result.total_return:.2f}%")

        return result

    def _calculate_performance_metrics(self) -> BacktestResult:
        """성과 지표 계산"""
        if not self.trades:
            return BacktestResult(
                total_return=0.0, annualized_return=0.0, sharpe_ratio=0.0,
                max_drawdown=0.0, win_rate=0.0, profit_factor=0.0,
                total_trades=0, winning_trades=0, losing_trades=0,
                avg_win=0.0, avg_loss=0.0, trades=[], equity_curve=pd.Series()
            )

        # 기본 지표
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t.pnl > 0])
        losing_trades = total_trades - winning_trades

        total_return = ((self.equity_curve[-1] - self.initial_capital) / self.initial_capital) * 100

        # 수익률 계산
        returns = pd.Series(self.equity_curve).pct_change().dropna()
        annualized_return = ((1 + total_return/100) ** (252/len(returns)) - 1) * 100

        # 샤프 비율
        sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0

        # 최대 낙폭
        equity_series = pd.Series(self.equity_curve, index=self.dates)
        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series - rolling_max) / rolling_max * 100
        max_drawdown = drawdown.min()

        # 승률 및 수익성
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0

        winning_pnls = [t.pnl for t in self.trades if t.pnl > 0]
        losing_pnls = [t.pnl for t in self.trades if t.pnl < 0]

        avg_win = np.mean(winning_pnls) if winning_pnls else 0
        avg_loss = np.mean(losing_pnls) if losing_pnls else 0

        profit_factor = abs(sum(winning_pnls) / sum(losing_pnls)) if sum(losing_pnls) != 0 else float('inf')

        return BacktestResult(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            avg_win=avg_win,
            avg_loss=avg_loss,
            trades=self.trades,
            equity_curve=pd.Series(self.equity_curve, index=self.dates)
        )

    def plot_results(self, result: BacktestResult, symbol: str, strategy: str) -> None:
        """결과 시각화"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 자본금 곡선
        axes[0, 0].plot(result.equity_curve.index, result.equity_curve.values)
        axes[0, 0].set_title(f'{symbol} - Equity Curve ({strategy})')
        axes[0, 0].set_ylabel('Portfolio Value')
        axes[0, 0].grid(True)

        # 수익률 분포
        returns = result.equity_curve.pct_change().dropna()
        axes[0, 1].hist(returns, bins=50, alpha=0.7)
        axes[0, 1].set_title('Returns Distribution')
        axes[0, 1].set_xlabel('Returns')
        axes[0, 1].grid(True)

        # 거래 PnL
        if result.trades:
            pnls = [t.pnl for t in result.trades]
            axes[1, 0].bar(range(len(pnls)), pnls, alpha=0.7)
            axes[1, 0].set_title('Trade PnL')
            axes[1, 0].set_xlabel('Trade Number')
            axes[1, 0].set_ylabel('PnL')
            axes[1, 0].grid(True)

        # 성과 지표
        metrics_text = f"""
        Total Return: {result.total_return:.2f}%
        Annualized Return: {result.annualized_return:.2f}%
        Sharpe Ratio: {result.sharpe_ratio:.2f}
        Max Drawdown: {result.max_drawdown:.2f}%
        Win Rate: {result.win_rate:.1f}%
        Total Trades: {result.total_trades}
        """
        axes[1, 1].text(0.1, 0.5, metrics_text, transform=axes[1, 1].transAxes,
                        fontsize=10, verticalalignment='center')
        axes[1, 1].set_title('Performance Metrics')
        axes[1, 1].axis('off')

        plt.tight_layout()
        plt.savefig(f'backtest_results_{symbol}_{strategy}.png', dpi=300, bbox_inches='tight')
        plt.show()

    def compare_strategies(self, symbol: str, start_date: str, end_date: str) -> Dict[str, BacktestResult]:
        """여러 전략 비교"""
        strategies = ['ma_crossover', 'rsi_strategy', 'macd_strategy', 'bollinger_strategy', 'volume_price_strategy']
        results = {}

        for strategy in strategies:
            logger.info(f"Testing {strategy} strategy")
            self.__init__(self.initial_capital)  # 초기화
            result = self.run_backtest(symbol, start_date, end_date, strategy)
            results[strategy] = result

        return results

def main():
    """메인 함수"""
    # 백테스트 엔진 초기화
    engine = BacktestEngine(initial_capital=10000000)

    # 삼성전자 백테스트
    symbol = "005930"
    start_date = "2020-01-01"
    end_date = "2024-12-31"

    # 단일 전략 테스트
    result = engine.run_backtest(symbol, start_date, end_date, 'ma_crossover')
    engine.plot_results(result, symbol, 'ma_crossover')

    # 전략 비교
    comparison_results = engine.compare_strategies(symbol, start_date, end_date)

    # 결과 요약
    print("\n=== 전략 비교 결과 ===")
    for strategy, result in comparison_results.items():
        print(f"{strategy}:")
        print(f"  총 수익률: {result.total_return:.2f}%")
        print(f"  샤프 비율: {result.sharpe_ratio:.2f}")
        print(f"  최대 낙폭: {result.max_drawdown:.2f}%")
        print(f"  승률: {result.win_rate:.1f}%")
        print(f"  총 거래 수: {result.total_trades}")
        print()

if __name__ == "__main__":
    main()
