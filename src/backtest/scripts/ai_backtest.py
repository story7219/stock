from datetime import datetime
import timedelta
from typing import Dict
import List, Optional, Tuple, Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
import yfinance as yf
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI 주식 자동매매 백테스트 시스템
ML + DL + AI + 알고리즘 판단으로 과거 데이터 백테스트

Author: AI Trading System
Created: 2025-07-08
Version: 1.0.0

Features:
- 실시간 AI 판단 로직을 과거 데이터에 적용
- ML/DL 모델 시뮬레이션
- 주요 성과 지표 계산 (누적 수익률, 최대 낙폭, 승률 등)
- 삼성전자(005930) 데이터 사용
"""

warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class AITradingStrategy:
    """AI 트레이딩 전략 클래스"""

    def __init__(self, initial_capital: float = 10000000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}  # 보유 종목
        self.trades = []     # 거래 내역
        self.equity_curve = []  # 자본 곡선

        # AI 모델 파라미터 (실제로는 학습된 모델 사용)
        self.ml_confidence_threshold = 0.7
        self.risk_threshold = 0.5
        self.position_size_ratio = 0.1  # 자본의 10%

        # 기술적 지표 파라미터
        self.rsi_period = 14
        self.ma_short = 5
        self.ma_long = 20
        self.volume_ma_period = 20

        # 성과 지표
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.max_drawdown = 0
        self.peak_capital = initial_capital

    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """기술적 지표 계산"""
        df = data.copy()

        # 이동평균
        df['MA_short'] = df['Close'].rolling(window=self.ma_short).mean()
        df['MA_long'] = df['Close'].rolling(window=self.ma_long).mean()

        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # 거래량 이동평균
        df['Volume_MA'] = df['Volume'].rolling(window=self.volume_ma_period).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']

        # 볼린저 밴드
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)

        # MACD
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_signal'] = df['MACD'].ewm(span=9).mean()

        # 가격 변화율
        df['Price_Change'] = df['Close'].pct_change()
        df['Price_Change_5'] = df['Close'].pct_change(periods=5)

        return df

    def run_ai_analysis(self, row: pd.Series) -> Dict[str, Any]:
        """AI 분석 (ML/DL 모델 시뮬레이션)"""
        # 실제로는 학습된 ML/DL 모델을 사용
        # 여기서는 기술적 지표를 바탕으로 AI 판단을 시뮬레이션

        # 1. 기술적 지표 기반 점수 계산
        technical_score = 0

        # RSI 기반 점수
        if row['RSI'] < 30:
            technical_score += 0.3  # 과매도
        elif row['RSI'] > 70:
            technical_score -= 0.3  # 과매수

        # 이동평균 크로스오버
        if row['MA_short'] > row['MA_long']:
            technical_score += 0.2
        else:
            technical_score -= 0.2

        # 볼린저 밴드 위치
        bb_position = (row['Close'] - row['BB_lower']) / (row['BB_upper'] - row['BB_lower'])
        if bb_position < 0.2:
            technical_score += 0.2  # 하단 근처
        elif bb_position > 0.8:
            technical_score -= 0.2  # 상단 근처

        # MACD 신호
        if row['MACD'] > row['MACD_signal']:
            technical_score += 0.1
        else:
            technical_score -= 0.1

        # 거래량 급증
        if row['Volume_Ratio'] > 2.0:
            technical_score += 0.2

        # 2. 감정 점수 (뉴스/시장 심리 시뮬레이션)
        # 실제로는 뉴스 감정 분석, 소셜 미디어 분석 등
        sentiment_score = np.random.normal(0, 0.1)  # 랜덤 시뮬레이션

        # 3. 리스크 점수
        price_change = row['Price_Change']
        volatility = abs(price_change) if pd.notna(price_change) else 0
        risk_score = min(volatility * 10, 1.0)  # 변동성 기반 리스크

        # 4. 종합 AI 점수
        ai_score = technical_score + sentiment_score
        confidence = min(abs(ai_score) + 0.5, 1.0)  # 신뢰도

        return {
            'ai_score': ai_score,
            'technical_score': technical_score,
            'sentiment_score': sentiment_score,
            'risk_score': risk_score,
            'confidence': confidence,
            'trend_prediction': 'UP' if ai_score > 0 else 'DOWN' if ai_score < 0 else 'NEUTRAL'
        }

    def make_trading_decision(self, ai_analysis: Dict[str, Any], current_price: float) -> Dict[str, Any]:
        """AI 분석 결과를 바탕으로 매매 판단"""

        confidence = ai_analysis['confidence']
        ai_score = ai_analysis['ai_score']
        risk_score = ai_analysis['risk_score']

        # 매매 조건
        should_trade = (confidence > self.ml_confidence_threshold and
                       risk_score < self.risk_threshold)

        # 매매 방향 결정
        if ai_score > 0.1:
            action = 'BUY'
        elif ai_score < -0.1:
            action = 'SELL'
        else:
            action = 'HOLD'

        # 포지션 크기 계산
        position_size = self.current_capital * self.position_size_ratio * confidence

        return {
            'should_trade': should_trade,
            'action': action,
            'confidence': confidence,
            'position_size': position_size,
            'reasoning': f"AI점수: {ai_score:.3f}, 신뢰도: {confidence:.3f}, 리스크: {risk_score:.3f}"
        }

    def execute_trade(self, decision: Dict[str, Any], price: float, date: datetime, symbol: str):
        """거래 실행"""
        if not decision['should_trade']:
            return

        action = decision['action']
        position_size = decision['position_size']
        quantity = int(position_size / price)

        if quantity <= 0:
            return

        # 수수료 및 슬리피지 (실제 거래 시뮬레이션)
        commission_rate = 0.00015  # 0.015%
        slippage_rate = 0.0001     # 0.01%

        if action == 'BUY':
            # 매수
            total_cost = quantity * price * (1 + commission_rate + slippage_rate)

            if total_cost <= self.current_capital:
                self.current_capital -= total_cost

                if symbol not in self.positions:
                    self.positions[symbol] = {'quantity': 0, 'avg_price': 0}

                pos = self.positions[symbol]
                new_quantity = pos['quantity'] + quantity
                new_avg_price = ((pos['quantity'] * pos['avg_price']) + (quantity * price)) / new_quantity

                pos['quantity'] = new_quantity
                pos['avg_price'] = new_avg_price

                # 거래 기록
                trade = {
                    'date': date,
                    'symbol': symbol,
                    'action': action,
                    'quantity': quantity,
                    'price': price,
                    'total_cost': total_cost,
                    'commission': quantity * price * commission_rate,
                    'slippage': quantity * price * slippage_rate,
                    'reasoning': decision['reasoning']
                }
                self.trades.append(trade)
                self.total_trades += 1

        elif action == 'SELL':
            # 매도
            if symbol in self.positions and self.positions[symbol]['quantity'] > 0:
                pos = self.positions[symbol]
                sell_quantity = min(quantity, pos['quantity'])

                if sell_quantity > 0:
                    gross_amount = sell_quantity * price
                    commission = gross_amount * commission_rate
                    slippage = gross_amount * slippage_rate
                    net_amount = gross_amount - commission - slippage

                    self.current_capital += net_amount
                    pos['quantity'] -= sell_quantity

                    if pos['quantity'] == 0:
                        del self.positions[symbol]

                    # 거래 기록
                    trade = {
                        'date': date,
                        'symbol': symbol,
                        'action': action,
                        'quantity': sell_quantity,
                        'price': price,
                        'gross_amount': gross_amount,
                        'net_amount': net_amount,
                        'commission': commission,
                        'slippage': slippage,
                        'reasoning': decision['reasoning']
                    }
                    self.trades.append(trade)
                    self.total_trades += 1

    def calculate_portfolio_value(self, current_price: float, symbol: str) -> float:
        """포트폴리오 가치 계산"""
        portfolio_value = self.current_capital

        if symbol in self.positions:
            position_value = self.positions[symbol]['quantity'] * current_price
            portfolio_value += position_value

        return portfolio_value

    def update_performance_metrics(self, portfolio_value: float):
        """성과 지표 업데이트"""
        # 최대 낙폭 계산
        if portfolio_value > self.peak_capital:
            self.peak_capital = portfolio_value

        current_drawdown = (self.peak_capital - portfolio_value) / self.peak_capital
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown

        # 자본 곡선 업데이트
        self.equity_curve.append(portfolio_value)

    def calculate_final_metrics(self) -> Dict[str, float]:
        """최종 성과 지표 계산"""
        if not self.trades:
            return {
                'total_return': 0.0,
                'annual_return': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'sharpe_ratio': 0.0,
                'total_trades': 0
            }

        # 수익/손실 거래 분리
        buy_trades = [t for t in self.trades if t['action'] == 'BUY']
        sell_trades = [t for t in self.trades if t['action'] == 'SELL']

        # 매수/매도 쌍으로 수익 계산
        profits = []
        losses = []

        for i in range(min(len(buy_trades), len(sell_trades))):
            buy_trade = buy_trades[i]
            sell_trade = sell_trades[i]

            if sell_trade['date'] > buy_trade['date']:
                profit = sell_trade['net_amount'] - buy_trade['total_cost']
                if profit > 0:
                    profits.append(profit)
                else:
                    losses.append(abs(profit))

        # 성과 지표 계산
        total_profit = sum(profits)
        total_loss = sum(losses)
        net_profit = total_profit - total_loss

        total_return = (self.equity_curve[-1] - self.initial_capital) / self.initial_capital

        # 승률
        total_trades = len(profits) + len(losses)
        win_rate = len(profits) / total_trades if total_trades > 0 else 0

        # 수익 팩터
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')

        # 샤프 비율 (간단한 계산)
        returns = pd.Series(self.equity_curve).pct_change().dropna()
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

        return {
            'total_return': total_return,
            'annual_return': total_return * (252 / len(self.equity_curve)) if len(self.equity_curve) > 0 else 0,
            'max_drawdown': self.max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'total_trades': total_trades,
            'net_profit': net_profit,
            'total_profit': total_profit,
            'total_loss': total_loss
        }

class AIBacktestEngine:
    """AI 백테스트 엔진"""

    def __init__(self, initial_capital: float = 10000000):
        self.strategy = AITradingStrategy(initial_capital)
        self.data = None

    def load_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """데이터 로드 (yfinance 사용)"""
        print(f"데이터 로딩 중: {symbol} ({start_date} ~ {end_date})")

        try:
            # yfinance로 데이터 다운로드
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)

            if data.empty:
                raise ValueError(f"데이터를 찾을 수 없습니다: {symbol}")

            # 컬럼명 정규화
            data.columns = [col.title() for col in data.columns]

            print(f"데이터 로딩 완료: {len(data)}개 데이터")
            return data

        except Exception as e:
            print(f"데이터 로딩 실패: {e}")
            # 예시 데이터 생성
            return self.generate_sample_data(start_date, end_date)

    def generate_sample_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """예시 데이터 생성 (삼성전자 시뮬레이션)"""
        print("예시 데이터 생성 중...")

        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        dates = pd.date_range(start=start, end=end, freq='D')

        # 삼성전자와 유사한 가격 패턴 생성
        np.random.seed(42)
        initial_price = 70000
        returns = np.random.normal(0.0005, 0.02, len(dates))  # 일간 수익률
        prices = [initial_price]

        for ret in returns[1:]:
            new_price = prices[-1] * (1 + ret)
            prices.append(max(new_price, 1000))  # 최소 가격 보장

        # 거래량 생성
        volumes = np.random.lognormal(15, 0.5, len(dates))

        data = pd.DataFrame({
            'Open': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
            'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'Close': prices,
            'Volume': volumes
        }, index=dates)

        # OHLC 정렬
        for i in range(len(data)):
            high = max(data.iloc[i]['Open'], data.iloc[i]['Close'])
            low = min(data.iloc[i]['Open'], data.iloc[i]['Close'])
            data.iloc[i, data.columns.get_loc('High')] = high
            data.iloc[i, data.columns.get_loc('Low')] = low

        print(f"예시 데이터 생성 완료: {len(data)}개 데이터")
        return data

    def run_backtest(self, data: pd.DataFrame) -> Dict[str, Any]:
        """백테스트 실행"""
        print("AI 백테스트 시작...")

        # 기술적 지표 계산
        data_with_indicators = self.strategy.calculate_technical_indicators(data)

        # 백테스트 루프
        for i in range(len(data_with_indicators)):
            row = data_with_indicators.iloc[i]
            date = data_with_indicators.index[i]

            # NaN 체크
            if pd.isna(row['RSI']) or pd.isna(row['MA_short']) or pd.isna(row['MA_long']):
                continue

            # 1. AI 분석
            ai_analysis = self.strategy.run_ai_analysis(row)

            # 2. 매매 판단
            trading_decision = self.strategy.make_trading_decision(ai_analysis, row['Close'])

            # 3. 거래 실행
            self.strategy.execute_trade(trading_decision, row['Close'], date, '005930')

            # 4. 포트폴리오 가치 계산
            portfolio_value = self.strategy.calculate_portfolio_value(row['Close'], '005930')
            self.strategy.update_performance_metrics(portfolio_value)

            # 진행상황 출력 (10%마다)
            if i % (len(data_with_indicators) // 10) == 0:
                progress = (i / len(data_with_indicators)) * 100
                print(f"진행률: {progress:.1f}%")

        # 최종 성과 지표 계산
        final_metrics = self.strategy.calculate_final_metrics()

        print("AI 백테스트 완료!")
        return {
            'metrics': final_metrics,
            'trades': self.strategy.trades,
            'equity_curve': self.strategy.equity_curve,
            'data': data_with_indicators
        }

    def plot_results(self, results: Dict[str, Any]):
        """결과 시각화"""
        metrics = results['metrics']
        equity_curve = results['equity_curve']
        data = results['data']

        # 그래프 설정
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('AI 주식 자동매매 백테스트 결과 (삼성전자 005930)', fontsize=16, fontweight='bold')

        # 1. 자본 곡선
        axes[0, 0].plot(equity_curve, label='포트폴리오 가치', color='blue', linewidth=2)
        axes[0, 0].axhline(y=self.strategy.initial_capital, color='red', linestyle='--',
                          label=f'초기 자본: {self.strategy.initial_capital:,.0f}원')
        axes[0, 0].set_title('자본 곡선')
        axes[0, 0].set_ylabel('포트폴리오 가치 (원)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. 가격 차트와 거래 포인트
        axes[0, 1].plot(data.index, data['Close'], label='삼성전자 주가', color='black', linewidth=1)

        # 매수/매도 포인트 표시
        buy_trades = [t for t in results['trades'] if t['action'] == 'BUY']
        sell_trades = [t for t in results['trades'] if t['action'] == 'SELL']

        if buy_trades:
            buy_dates = [t['date'] for t in buy_trades]
            buy_prices = [t['price'] for t in buy_trades]
            axes[0, 1].scatter(buy_dates, buy_prices, color='green', marker='^', s=100,
                              label=f'매수 ({len(buy_trades)}회)', alpha=0.7)

        if sell_trades:
            sell_dates = [t['date'] for t in sell_trades]
            sell_prices = [t['price'] for t in sell_trades]
            axes[0, 1].scatter(sell_dates, sell_prices, color='red', marker='v', s=100,
                              label=f'매도 ({len(sell_trades)}회)', alpha=0.7)

        axes[0, 1].set_title('주가 차트와 거래 포인트')
        axes[0, 1].set_ylabel('주가 (원)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 성과 지표 테이블
        metrics_text = f"""
        📊 AI 백테스트 성과 지표

        💰 총 수익률: {metrics['total_return']:.2%}
        📈 연간 수익률: {metrics['annual_return']:.2%}
        📉 최대 낙폭: {metrics['max_drawdown']:.2%}
        🎯 승률: {metrics['win_rate']:.2%}
        📊 수익 팩터: {metrics['profit_factor']:.2f}
        📈 샤프 비율: {metrics['sharpe_ratio']:.2f}
        🔄 총 거래 횟수: {metrics['total_trades']}회
        💵 순손익: {metrics['net_profit']:.0f}원
        """

        axes[1, 0].text(0.1, 0.5, metrics_text, transform=axes[1, 0].transAxes,
                       fontsize=12, verticalalignment='center', fontfamily='monospace')
        axes[1, 0].set_title('성과 지표')
        axes[1, 0].axis('off')

        # 4. 거래 분포
        if results['trades']:
            trade_returns = []
            for i in range(0, len(results['trades'])-1, 2):
                if i+1 < len(results['trades']):
                    buy_trade = results['trades'][i]
                    sell_trade = results['trades'][i+1]
                    if buy_trade['action'] == 'BUY' and sell_trade['action'] == 'SELL':
                        return_pct = (sell_trade['net_amount'] - buy_trade['total_cost']) / buy_trade['total_cost']
                        trade_returns.append(return_pct)

            if trade_returns:
                axes[1, 1].hist(trade_returns, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                axes[1, 1].axvline(x=0, color='red', linestyle='--', alpha=0.7)
                axes[1, 1].set_title('거래별 수익률 분포')
                axes[1, 1].set_xlabel('수익률')
                axes[1, 1].set_ylabel('거래 횟수')
                axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def print_detailed_results(self, results: Dict[str, Any]):
        """상세 결과 출력"""
        metrics = results['metrics']
        trades = results['trades']

        print("\n" + "="*60)
        print("🤖 AI 주식 자동매매 백테스트 결과")
        print("="*60)
        print(f"📅 백테스트 기간: {results['data'].index[0].strftime('%Y-%m-%d')} ~ {results['data'].index[-1].strftime('%Y-%m-%d')}")
        print(f"💰 초기 자본: {self.strategy.initial_capital:,.0f}원")
        print(f"💵 최종 자본: {results['equity_curve'][-1]:,.0f}원")
        print()

        print("📊 주요 성과 지표")
        print("-" * 40)
        print(f"총 수익률:     {metrics['total_return']:.2%}")
        print(f"연간 수익률:   {metrics['annual_return']:.2%}")
        print(f"최대 낙폭:     {metrics['max_drawdown']:.2%}")
        print(f"승률:          {metrics['win_rate']:.2%}")
        print(f"수익 팩터:     {metrics['profit_factor']:.2f}")
        print(f"샤프 비율:     {metrics['sharpe_ratio']:.2f}")
        print(f"총 거래 횟수:  {metrics['total_trades']}회")
        print()

        print("💰 손익 분석")
        print("-" * 40)
        print(f"총 수익:       {metrics['total_profit']:.0f}원")
        print(f"총 손실:       {metrics['total_loss']:.0f}원")
        print(f"순손익:        {metrics['net_profit']:.0f}원")
        print()

        if trades:
            print("🔍 최근 거래 내역 (상위 5개)")
            print("-" * 40)
            for i, trade in enumerate(trades[-5:], 1):
                print(f"{i}. {trade['date'].strftime('%Y-%m-%d')} | "
                      f"{trade['action']} | {trade['quantity']}주 | "
                      f"{trade['price']:.0f}원 | {trade['reasoning'][:30]}...")

        print("="*60)

def main():
    """메인 함수"""
    print("🚀 AI 주식 자동매매 백테스트 시스템 시작")
    print("="*50)

    # 백테스트 엔진 초기화
    engine = AIBacktestEngine(initial_capital=10000000)  # 1천만원

    # 데이터 로드 (삼성전자)
    data = engine.load_data(
        symbol="005930.KS",  # 삼성전자
        start_date="2023-01-01",
        end_date="2024-12-31"
    )

    # 백테스트 실행
    results = engine.run_backtest(data)

    # 결과 출력
    engine.print_detailed_results(results)

    # 결과 시각화
    engine.plot_results(results)

    print("\n✅ AI 백테스트 완료!")

if __name__ == "__main__":
    main()

