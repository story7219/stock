#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
💼 포트폴리오 관리 시스템 v1.0
포트폴리오 최적화 및 리스크 관리
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
from scipy.optimize import minimize
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.covariance import LedoitWolf
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class PortfolioConfig:
    """포트폴리오 설정"""
    max_positions: int = 20
    min_weight: float = 0.01  # 최소 비중 1%
    max_weight: float = 0.15  # 최대 비중 15%
    rebalance_threshold: float = 0.05  # 리밸런싱 임계값 5%
    target_volatility: float = 0.15  # 목표 변동성 15%
    max_drawdown_limit: float = 0.20  # 최대 낙폭 한계 20%
    correlation_limit: float = 0.7  # 상관관계 한계 70%

@dataclass
class Position:
    """포지션 정보"""
    symbol: str
    quantity: int
    avg_price: float
    current_price: float
    market_value: float
    weight: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    sector: str
    country: str

@dataclass
class PortfolioMetrics:
    """포트폴리오 지표"""
    total_value: float
    total_return: float
    daily_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    var_95: float  # 95% VaR
    cvar_95: float  # 95% CVaR
    beta: float
    alpha: float
    correlation_matrix: Dict[str, Dict[str, float]]
    sector_allocation: Dict[str, float]
    country_allocation: Dict[str, float]

class PortfolioManager:
    """포트폴리오 관리자"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """포트폴리오 매니저 초기화"""
        self.config = config or {
            'max_portfolio_size': 10,
            'rebalance_threshold': 0.05,
            'risk_tolerance': 'medium',
            'default_position_size': 0.1
        }
        self.portfolios = {}
        self.performance_tracker = {}
        self.positions: Dict[str, Position] = {}
        self.cash = 0.0
        self.portfolio_history = []
        self.rebalance_history = []
        self.db_path = "data/portfolio.db"
        self.initialize_database()
        logger.info("포트폴리오 매니저 초기화 완료")
    
    async def create_portfolio(self, portfolio_name: str, initial_capital: float) -> Dict[str, Any]:
        """포트폴리오 생성"""
        try:
            logger.info(f"포트폴리오 '{portfolio_name}' 생성 중 - 초기 자본: {initial_capital:,.0f}원")
            
            portfolio = {
                'name': portfolio_name,
                'initial_capital': initial_capital,
                'current_value': initial_capital,
                'cash': initial_capital,
                'positions': {},
                'created_at': datetime.now(),
                'last_updated': datetime.now()
            }
            
            self.portfolios[portfolio_name] = portfolio
            self.cash = initial_capital
            
            logger.info(f"포트폴리오 '{portfolio_name}' 생성 완료")
            return portfolio
            
        except Exception as e:
            logger.error(f"포트폴리오 생성 중 오류: {e}")
            raise
    
    def initialize_database(self):
        """데이터베이스 초기화"""
        Path("data").mkdir(exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS portfolio_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                total_value REAL,
                cash REAL,
                positions TEXT,
                metrics TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS rebalance_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                reason TEXT,
                old_weights TEXT,
                new_weights TEXT,
                transaction_costs REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    async def update_portfolio(self, market_data: Dict[str, float]) -> PortfolioMetrics:
        """포트폴리오 업데이트"""
        try:
            # 현재 가격으로 포지션 업데이트
            for symbol, position in self.positions.items():
                if symbol in market_data:
                    position.current_price = market_data[symbol]
                    position.market_value = position.quantity * position.current_price
                    position.unrealized_pnl = (position.current_price - position.avg_price) * position.quantity
                    position.unrealized_pnl_pct = (position.current_price - position.avg_price) / position.avg_price * 100
            
            # 포트폴리오 지표 계산
            metrics = await self.calculate_portfolio_metrics()
            
            # 히스토리 저장
            await self.save_portfolio_snapshot(metrics)
            
            # 리밸런싱 필요 여부 확인
            if await self.should_rebalance(metrics):
                await self.rebalance_portfolio()
            
            return metrics
            
        except Exception as e:
            logger.error(f"포트폴리오 업데이트 오류: {e}")
            raise
    
    async def calculate_portfolio_metrics(self) -> PortfolioMetrics:
        """포트폴리오 지표 계산"""
        try:
            total_value = self.cash + sum(pos.market_value for pos in self.positions.values())
            
            if total_value == 0:
                return PortfolioMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, {}, {}, {})
            
            # 비중 계산
            for position in self.positions.values():
                position.weight = position.market_value / total_value
            
            # 과거 데이터 수집
            returns_data = await self._get_returns_data()
            
            if returns_data.empty:
                return PortfolioMetrics(total_value, 0, 0, 0, 0, 0, 0, 0, 0, 0, {}, {}, {})
            
            # 수익률 지표
            portfolio_returns = self._calculate_portfolio_returns(returns_data)
            total_return = (total_value - self._get_initial_value()) / self._get_initial_value() * 100
            daily_return = portfolio_returns.iloc[-1] if not portfolio_returns.empty else 0
            volatility = portfolio_returns.std() * np.sqrt(252) * 100
            
            # 리스크 지표
            sharpe_ratio = self._calculate_sharpe_ratio(portfolio_returns)
            max_drawdown = self._calculate_max_drawdown(portfolio_returns)
            var_95 = self._calculate_var(portfolio_returns, 0.95)
            cvar_95 = self._calculate_cvar(portfolio_returns, 0.95)
            
            # 베타, 알파 계산
            beta, alpha = await self._calculate_beta_alpha(portfolio_returns)
            
            # 상관관계 매트릭스
            correlation_matrix = self._calculate_correlation_matrix(returns_data)
            
            # 섹터/국가 배분
            sector_allocation = self._calculate_sector_allocation()
            country_allocation = self._calculate_country_allocation()
            
            return PortfolioMetrics(
                total_value=total_value,
                total_return=total_return,
                daily_return=daily_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                var_95=var_95,
                cvar_95=cvar_95,
                beta=beta,
                alpha=alpha,
                correlation_matrix=correlation_matrix,
                sector_allocation=sector_allocation,
                country_allocation=country_allocation
            )
            
        except Exception as e:
            logger.error(f"포트폴리오 지표 계산 오류: {e}")
            return PortfolioMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, {}, {}, {})
    
    async def _get_returns_data(self, period: str = "1y") -> pd.DataFrame:
        """수익률 데이터 수집"""
        try:
            if not self.positions:
                return pd.DataFrame()
            
            symbols = list(self.positions.keys())
            data_frames = []
            
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period=period)
                    if not hist.empty:
                        returns = hist['Close'].pct_change().dropna()
                        returns.name = symbol
                        data_frames.append(returns)
                except Exception as e:
                    logger.warning(f"종목 {symbol} 데이터 수집 실패: {e}")
                    continue
            
            if data_frames:
                return pd.concat(data_frames, axis=1).dropna()
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"수익률 데이터 수집 오류: {e}")
            return pd.DataFrame()
    
    def _calculate_portfolio_returns(self, returns_data: pd.DataFrame) -> pd.Series:
        """포트폴리오 수익률 계산"""
        if returns_data.empty:
            return pd.Series()
        
        weights = np.array([self.positions[symbol].weight for symbol in returns_data.columns])
        portfolio_returns = (returns_data * weights).sum(axis=1)
        return portfolio_returns
    
    def _get_initial_value(self) -> float:
        """초기 포트폴리오 가치"""
        if self.portfolio_history:
            return self.portfolio_history[0]['total_value']
        return sum(pos.quantity * pos.avg_price for pos in self.positions.values()) + self.cash
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.025) -> float:
        """샤프 비율 계산"""
        if returns.empty or returns.std() == 0:
            return 0
        
        excess_returns = returns.mean() * 252 - risk_free_rate
        volatility = returns.std() * np.sqrt(252)
        return excess_returns / volatility
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """최대 낙폭 계산"""
        if returns.empty:
            return 0
        
        cumulative = (1 + returns).cumprod()
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak
        return drawdown.min() * 100
    
    def _calculate_var(self, returns: pd.Series, confidence_level: float) -> float:
        """VaR 계산"""
        if returns.empty:
            return 0
        
        return np.percentile(returns, (1 - confidence_level) * 100) * 100
    
    def _calculate_cvar(self, returns: pd.Series, confidence_level: float) -> float:
        """CVaR 계산"""
        if returns.empty:
            return 0
        
        var = self._calculate_var(returns, confidence_level) / 100
        tail_returns = returns[returns <= var]
        return tail_returns.mean() * 100 if not tail_returns.empty else 0
    
    async def _calculate_beta_alpha(self, portfolio_returns: pd.Series) -> Tuple[float, float]:
        """베타, 알파 계산"""
        try:
            # 벤치마크 데이터 (코스피)
            benchmark = yf.Ticker("^KS11")
            benchmark_data = benchmark.history(period="1y")
            
            if benchmark_data.empty or portfolio_returns.empty:
                return 0, 0
            
            benchmark_returns = benchmark_data['Close'].pct_change().dropna()
            
            # 공통 날짜 찾기
            common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
            if len(common_dates) < 30:  # 최소 30일 데이터 필요
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
            alpha = portfolio_mean - (0.025 + beta * (benchmark_mean - 0.025))
            
            return beta, alpha * 100
            
        except Exception as e:
            logger.error(f"베타/알파 계산 오류: {e}")
            return 0, 0
    
    def _calculate_correlation_matrix(self, returns_data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """상관관계 매트릭스 계산"""
        if returns_data.empty:
            return {}
        
        corr_matrix = returns_data.corr()
        return corr_matrix.to_dict()
    
    def _calculate_sector_allocation(self) -> Dict[str, float]:
        """섹터 배분 계산"""
        sector_weights = {}
        total_value = sum(pos.market_value for pos in self.positions.values())
        
        if total_value == 0:
            return {}
        
        for position in self.positions.values():
            sector = position.sector
            if sector not in sector_weights:
                sector_weights[sector] = 0
            sector_weights[sector] += position.market_value / total_value * 100
        
        return sector_weights
    
    def _calculate_country_allocation(self) -> Dict[str, float]:
        """국가 배분 계산"""
        country_weights = {}
        total_value = sum(pos.market_value for pos in self.positions.values())
        
        if total_value == 0:
            return {}
        
        for position in self.positions.values():
            country = position.country
            if country not in country_weights:
                country_weights[country] = 0
            country_weights[country] += position.market_value / total_value * 100
        
        return country_weights
    
    async def should_rebalance(self, metrics: PortfolioMetrics) -> bool:
        """리밸런싱 필요 여부 확인"""
        try:
            # 비중 이탈 확인
            for position in self.positions.values():
                if (position.weight > self.config['max_weight'] or 
                    position.weight < self.config['min_weight']):
                    return True
            
            # 변동성 초과 확인
            if metrics.volatility > self.config['target_volatility'] * 100 * 1.2:
                return True
            
            # 최대 낙폭 초과 확인
            if abs(metrics.max_drawdown) > self.config['max_drawdown_limit'] * 100:
                return True
            
            # 상관관계 과도 확인
            for symbol1, correlations in metrics.correlation_matrix.items():
                for symbol2, corr in correlations.items():
                    if symbol1 != symbol2 and abs(corr) > self.config['correlation_limit']:
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"리밸런싱 확인 오류: {e}")
            return False
    
    async def rebalance_portfolio(self) -> Dict[str, Any]:
        """포트폴리오 리밸런싱"""
        try:
            logger.info("포트폴리오 리밸런싱 시작")
            
            # 현재 비중 저장
            old_weights = {symbol: pos.weight for symbol, pos in self.positions.items()}
            
            # 최적 비중 계산
            new_weights = await self.optimize_portfolio()
            
            # 리밸런싱 실행
            rebalance_trades = await self._execute_rebalancing(old_weights, new_weights)
            
            # 리밸런싱 기록 저장
            await self.save_rebalance_record("정기 리밸런싱", old_weights, new_weights, 0.0)
            
            logger.info("포트폴리오 리밸런싱 완료")
            
            return {
                "old_weights": old_weights,
                "new_weights": new_weights,
                "trades": rebalance_trades
            }
            
        except Exception as e:
            logger.error(f"리밸런싱 오류: {e}")
            return {}
    
    async def optimize_portfolio(self) -> Dict[str, float]:
        """포트폴리오 최적화"""
        try:
            # 수익률 데이터 수집
            returns_data = await self._get_returns_data()
            
            if returns_data.empty:
                return {}
            
            # 평균 수익률과 공분산 매트릭스 계산
            mean_returns = returns_data.mean() * 252
            
            # Ledoit-Wolf 추정을 사용한 공분산 매트릭스
            lw = LedoitWolf()
            cov_matrix = lw.fit(returns_data).covariance_ * 252
            
            # 제약 조건 설정
            n_assets = len(returns_data.columns)
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # 비중 합 = 1
            ]
            
            bounds = [(self.config['min_weight'], self.config['max_weight']) for _ in range(n_assets)]
            
            # 초기 추정값 (균등 비중)
            x0 = np.array([1/n_assets] * n_assets)
            
            # 목적 함수: 변동성 최소화 (리스크 패리티)
            def objective(weights):
                portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
                return portfolio_variance
            
            # 최적화 실행
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if result.success:
                optimal_weights = dict(zip(returns_data.columns, result.x))
                return optimal_weights
            else:
                logger.warning("포트폴리오 최적화 실패, 균등 비중 사용")
                return dict(zip(returns_data.columns, x0))
                
        except Exception as e:
            logger.error(f"포트폴리오 최적화 오류: {e}")
            return {}
    
    async def _execute_rebalancing(self, old_weights: Dict[str, float], 
                                 new_weights: Dict[str, float]) -> List[Dict[str, Any]]:
        """리밸런싱 실행"""
        trades = []
        total_value = sum(pos.market_value for pos in self.positions.values())
        
        for symbol, new_weight in new_weights.items():
            if symbol in old_weights:
                old_weight = old_weights[symbol]
                weight_diff = new_weight - old_weight
                
                if abs(weight_diff) > self.config['rebalance_threshold']:
                    target_value = new_weight * total_value
                    current_value = self.positions[symbol].market_value
                    trade_value = target_value - current_value
                    
                    if trade_value > 0:
                        action = "BUY"
                        quantity = int(trade_value / self.positions[symbol].current_price)
                    else:
                        action = "SELL"
                        quantity = int(abs(trade_value) / self.positions[symbol].current_price)
                    
                    trades.append({
                        "symbol": symbol,
                        "action": action,
                        "quantity": quantity,
                        "price": self.positions[symbol].current_price,
                        "value": trade_value
                    })
        
        return trades
    
    async def save_portfolio_snapshot(self, metrics: PortfolioMetrics):
        """포트폴리오 스냅샷 저장"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO portfolio_history (date, total_value, cash, positions, metrics)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                metrics.total_value,
                self.cash,
                json.dumps({symbol: asdict(pos) for symbol, pos in self.positions.items()}),
                json.dumps(asdict(metrics))
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"포트폴리오 스냅샷 저장 오류: {e}")
    
    async def save_rebalance_record(self, reason: str, old_weights: Dict[str, float], 
                                  new_weights: Dict[str, float], cost: float):
        """리밸런싱 기록 저장"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO rebalance_history (date, reason, old_weights, new_weights, transaction_costs)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                reason,
                json.dumps(old_weights),
                json.dumps(new_weights),
                cost
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"리밸런싱 기록 저장 오류: {e}")
    
    def generate_portfolio_report(self, metrics: PortfolioMetrics) -> str:
        """포트폴리오 리포트 생성"""
        report = f"""
=== 포트폴리오 현황 리포트 ===
생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

=== 포트폴리오 개요 ===
총 자산가치: {metrics.total_value:,.0f}원
현금: {self.cash:,.0f}원
총 수익률: {metrics.total_return:.2f}%
일일 수익률: {metrics.daily_return:.2f}%

=== 리스크 지표 ===
변동성: {metrics.volatility:.2f}%
샤프 비율: {metrics.sharpe_ratio:.2f}
최대 낙폭: {metrics.max_drawdown:.2f}%
VaR (95%): {metrics.var_95:.2f}%
CVaR (95%): {metrics.cvar_95:.2f}%

=== 시장 지표 ===
베타: {metrics.beta:.2f}
알파: {metrics.alpha:.2f}%

=== 보유 종목 ==="""
        
        for symbol, position in self.positions.items():
            report += f"""
{symbol}: {position.quantity:,}주 ({position.weight:.1f}%)
  현재가: {position.current_price:,.0f}원
  평가금액: {position.market_value:,.0f}원
  평가손익: {position.unrealized_pnl:,.0f}원 ({position.unrealized_pnl_pct:.2f}%)"""
        
        report += f"""

=== 섹터 배분 ==="""
        for sector, weight in metrics.sector_allocation.items():
            report += f"\n{sector}: {weight:.1f}%"
        
        report += f"""

=== 국가 배분 ==="""
        for country, weight in metrics.country_allocation.items():
            report += f"\n{country}: {weight:.1f}%"
        
        return report
    
    def generate_portfolio_charts(self, output_dir: str = "reports"):
        """포트폴리오 차트 생성"""
        try:
            Path(output_dir).mkdir(exist_ok=True)
            
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            
            # 포트폴리오 구성 파이 차트
            weights = [pos.weight * 100 for pos in self.positions.values()]
            labels = list(self.positions.keys())
            
            axes[0, 0].pie(weights, labels=labels, autopct='%1.1f%%')
            axes[0, 0].set_title('포트폴리오 구성')
            
            # 섹터 배분
            if hasattr(self, '_last_metrics') and self._last_metrics.sector_allocation:
                sectors = list(self._last_metrics.sector_allocation.keys())
                sector_weights = list(self._last_metrics.sector_allocation.values())
                
                axes[0, 1].pie(sector_weights, labels=sectors, autopct='%1.1f%%')
                axes[0, 1].set_title('섹터 배분')
            
            # 수익률 분포
            if self.portfolio_history:
                returns = []
                for i in range(1, len(self.portfolio_history)):
                    prev_value = self.portfolio_history[i-1]['total_value']
                    curr_value = self.portfolio_history[i]['total_value']
                    daily_return = (curr_value - prev_value) / prev_value * 100
                    returns.append(daily_return)
                
                if returns:
                    axes[0, 2].hist(returns, bins=30, alpha=0.7)
                    axes[0, 2].set_title('일일 수익률 분포')
                    axes[0, 2].set_xlabel('수익률 (%)')
                    axes[0, 2].set_ylabel('빈도')
            
            # 포트폴리오 가치 추이
            if self.portfolio_history:
                dates = [datetime.fromisoformat(h['date']) for h in self.portfolio_history]
                values = [h['total_value'] for h in self.portfolio_history]
                
                axes[1, 0].plot(dates, values)
                axes[1, 0].set_title('포트폴리오 가치 추이')
                axes[1, 0].set_xlabel('날짜')
                axes[1, 0].set_ylabel('가치 (원)')
                axes[1, 0].tick_params(axis='x', rotation=45)
            
            # 개별 종목 성과
            pnl_data = [(pos.symbol, pos.unrealized_pnl_pct) for pos in self.positions.values()]
            if pnl_data:
                symbols, pnls = zip(*pnl_data)
                colors = ['green' if pnl >= 0 else 'red' for pnl in pnls]
                
                axes[1, 1].bar(symbols, pnls, color=colors, alpha=0.7)
                axes[1, 1].set_title('개별 종목 수익률')
                axes[1, 1].set_xlabel('종목')
                axes[1, 1].set_ylabel('수익률 (%)')
                axes[1, 1].tick_params(axis='x', rotation=45)
            
            # 상관관계 히트맵
            if hasattr(self, '_last_metrics') and self._last_metrics.correlation_matrix:
                corr_df = pd.DataFrame(self._last_metrics.correlation_matrix)
                if not corr_df.empty:
                    sns.heatmap(corr_df, annot=True, cmap='coolwarm', center=0, 
                              ax=axes[1, 2], fmt='.2f')
                    axes[1, 2].set_title('상관관계 매트릭스')
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/portfolio_dashboard.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"포트폴리오 차트 생성 완료: {output_dir}/portfolio_dashboard.png")
            
        except Exception as e:
            logger.error(f"포트폴리오 차트 생성 오류: {e}")

# 사용 예제
async def main():
    """포트폴리오 관리 예제"""
    config = PortfolioConfig(
        max_positions=15,
        max_weight=0.12,
        target_volatility=0.18
    )
    
    portfolio_manager = PortfolioManager(config)
    
    # 예제 포지션 추가
    portfolio_manager.positions["AAPL"] = Position(
        symbol="AAPL",
        quantity=100,
        avg_price=150.0,
        current_price=155.0,
        market_value=15500.0,
        weight=0.0,
        unrealized_pnl=500.0,
        unrealized_pnl_pct=3.33,
        sector="Technology",
        country="US"
    )
    
    # 포트폴리오 업데이트
    market_data = {"AAPL": 158.0}
    metrics = await portfolio_manager.update_portfolio(market_data)
    
    # 리포트 생성
    report = portfolio_manager.generate_portfolio_report(metrics)
    print(report)

if __name__ == "__main__":
    asyncio.run(main()) 