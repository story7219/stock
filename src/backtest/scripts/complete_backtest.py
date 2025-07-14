from datetime import datetime
import timedelta
from typing import Dict
import List
import Optional, Tuple, Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
import yfinance as yf
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
완전한 백테스트 시스템
- 예측 모델 시뮬레이션
- 성과 지표 계산 (CAGR, MDD, 승률, 샤프지수)
- 시각화 (자산곡선, 매매시점, 드로우다운)
"""

warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class PredictionModel:
    """예측 모델 시뮬레이션"""

    def __init__(self):
        self.is_trained = False

    def predict_signal(self, df: pd.DataFrame) -> pd.Series:
        """예측 신호 생성 (0~1 사이 값)"""
        # 실제로는 학습된 ML/DL 모델을 사용
        # 여기서는 기술적 지표 기반으로 예측값 시뮬레이션

        predictions = []

        for i in range(len(df)):
            if i < 20:  # 초기 데이터는 예측 불가
                predictions.append(0.5)
                continue

            # 기술적 지표 기반 예측
            current_price = df.iloc[i]['Close']
            ma_20 = df.iloc[i]['MA_20']
            rsi = df.iloc[i]['RSI']

            # 예측 점수 계산
            score = 0.5  # 기본값

            # 이동평균 기반
            if current_price > ma_20:
                score += 0.2
            else:
                score -= 0.2

            # RSI 기반
            if rsi < 30:
                score += 0.2  # 과매도
            elif rsi > 70:
                score -= 0.2  # 과매수

            # 랜덤 노이즈 추가 (실제 예측의 불확실성 시뮬레이션)
            noise = np.random.normal(0, 0.1)
            score += noise

            # 0~1 범위로 제한
            score = max(0, min(1, score))
            predictions.append(score)

        return pd.Series(predictions, index=df.index)

class CompleteBacktestEngine:
    """완전한 백테스트 엔진"""

    def __init__(self, initial_capital: float = 10000000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []

        # 전략 파라미터
        self.buy_threshold = 0.7
        self.sell_threshold = 0.3
        self.position_size_ratio = 0.1

        # 성과 지표
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.max_drawdown = 0
        self.peak_capital = initial_capital

        # 예측 모델
        self.model = PredictionModel()

    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """기술적 지표 계산"""
        df = data.copy()

        # 이동평균
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['MA_60'] = df['Close'].rolling(window=60).mean()

        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # MACD
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_signal'] = df['MACD'].ewm(span=9).mean()

        # 볼린저 밴드
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)

        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """매매 신호 생성"""
        # 예측값 생성
        df['prediction'] = self.model.predict_signal(df)

        # 신호 생성
        df['signal'] = 0
        df.loc[df['prediction'] > self.buy_threshold, 'signal'] = 1   # 매수
        df.loc[df['prediction'] < self.sell_threshold, 'signal'] = -1 # 매도

        # 포지션 계산 (신호가 바뀔 때만 진입/청산)
        df['position'] = df['signal'].replace(to_replace=0, method='ffill').fillna(0)

        return df

    def execute_trades(self, df: pd.DataFrame) -> pd.DataFrame:
        """거래 실행 및 수익률 계산"""
        # 수익률 계산
        df['return'] = df['Close'].pct_change().fillna(0)
        df['strategy_return'] = df['return'] * df['position'].shift(1).fillna(0)
        df['cum_return'] = (1 + df['strategy_return']).cumprod()

        # 드로우다운 계산
        df['drawdown'] = df['cum_return'] / df['cum_return'].cummax() - 1

        # 거래 기록
        for i in range(1, len(df)):
            if df.iloc[i]['signal'] != 0:
                trade = {
                    'date': df.index[i],
                    'price': df.iloc[i]['Close'],
                    'signal': df.iloc[i]['signal'],
                    'prediction': df.iloc[i]['prediction'],
                    'return': df.iloc[i]['strategy_return']
                }
                self.trades.append(trade)

        return df

    def calculate_performance_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """성과 지표 계산"""
        # 기본 지표
        total_return = df['cum_return'].iloc[-1] - 1
        buy_hold_return = (1 + df['return']).cumprod().iloc[-1] - 1

        # CAGR (연복리 수익률)
        days = len(df)
        years = days / 252
        cagr = (df['cum_return'].iloc[-1] ** (1/years)) - 1

        # MDD (최대 낙폭)
        mdd = df['drawdown'].min()

        # 승률
        trades_df = pd.DataFrame(self.trades)
        if len(trades_df) > 0:
            win_trades = len(trades_df[trades_df['return'] > 0])
            win_rate = win_trades / len(trades_df)
        else:
            win_rate = 0

        # 샤프 비율
        returns = df['strategy_return'].dropna()
        if returns.std() > 0:
            sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()
        else:
            sharpe_ratio = 0

        # 수익 팩터
        if len(trades_df) > 0:
            profits = trades_df[trades_df['return'] > 0]['return'].sum()
            losses = abs(trades_df[trades_df['return'] < 0]['return'].sum())
            profit_factor = profits / losses if losses > 0 else float('inf')
        else:
            profit_factor = 0

        return {
            'total_return': total_return,
            'buy_hold_return': buy_hold_return,
            'cagr': cagr,
            'mdd': mdd,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe_ratio,
            'profit_factor': profit_factor,
            'total_trades': len(self.trades)
        }

    def plot_results(self, df: pd.DataFrame, metrics: Dict[str, float]):
        """결과 시각화"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('백테스트 결과 분석', fontsize=16, fontweight='bold')

        # 1. 자산곡선과 매매시점
        axes[0, 0].plot(df.index, df['cum_return'], label='전략 수익률', color='blue', linewidth=2)
        axes[0, 0].plot(df.index, (1 + df['return']).cumprod(), label='Buy & Hold',
                       color='gray', linestyle='--', alpha=0.7)

        # 매수/매도 포인트
        buy_signals = df[df['signal'] == 1]
        sell_signals = df[df['signal'] == -1]

        if len(buy_signals) > 0:
            axes[0, 0].scatter(buy_signals.index, buy_signals['cum_return'],
                             color='green', marker='^', s=100, label='매수', alpha=0.7)
        if len(sell_signals) > 0:
            axes[0, 0].scatter(sell_signals.index, sell_signals['cum_return'],
                             color='red', marker='v', s=100, label='매도', alpha=0.7)

        axes[0, 0].set_title('자산곡선 및 매매시점')
        axes[0, 0].set_ylabel('누적 수익률')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. 드로우다운
        axes[0, 1].fill_between(df.index, df['drawdown'], 0, alpha=0.3, color='red')
        axes[0, 1].plot(df.index, df['drawdown'], color='red', linewidth=1)
        axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[0, 1].set_title('드로우다운')
        axes[0, 1].set_ylabel('드로우다운 (%)')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 예측값과 신호
        axes[1, 0].plot(df.index, df['prediction'], label='예측값', color='purple', alpha=0.7)
        axes[1, 0].axhline(y=self.buy_threshold, color='green', linestyle='--', alpha=0.7, label='매수 임계값')
        axes[1, 0].axhline(y=self.sell_threshold, color='red', linestyle='--', alpha=0.7, label='매도 임계값')
        axes[1, 0].set_title('예측값과 매매 신호')
        axes[1, 0].set_ylabel('예측값')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. 성과 지표 테이블
        metrics_text = f"""
        📊 백테스트 성과 지표

        💰 총 수익률: {metrics['total_return']:.2%}
        📈 연복리 수익률: {metrics['cagr']:.2%}
        📉 최대 낙폭: {metrics['mdd']:.2%}
        🎯 승률: {metrics['win_rate']:.2%}
        📊 샤프 비율: {metrics['sharpe_ratio']:.2f}
        💵 수익 팩터: {metrics['profit_factor']:.2f}
        🔄 총 거래 횟수: {metrics['total_trades']}회

        📈 Buy & Hold: {metrics['buy_hold_return']:.2%}
        """

        axes[1, 1].text(0.1, 0.5, metrics_text, transform=axes[1, 1].transAxes,
                       fontsize=11, verticalalignment='center', fontfamily='monospace')
        axes[1, 1].set_title('성과 지표')
        axes[1, 1].axis('off')

        plt.tight_layout()
        plt.show()

    def print_results(self, df: pd.DataFrame, metrics: Dict[str, float]):
        """결과 출력"""
        print("\n" + "="*60)
        print("📊 백테스트 결과")
        print("="*60)
        print(f"📅 백테스트 기간: {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}")
        print(f"💰 초기 자본: {self.initial_capital:,.0f}원")
        print()

        print("📈 주요 성과 지표")
        print("-" * 40)
        print(f"총 수익률:     {metrics['total_return']:>10.2%}")
        print(f"연복리 수익률: {metrics['cagr']:>10.2%}")
        print(f"최대 낙폭:     {metrics['mdd']:>10.2%}")
        print(f"승률:          {metrics['win_rate']:>10.2%}")
        print(f"샤프 비율:     {metrics['sharpe_ratio']:>10.2f}")
        print(f"수익 팩터:     {metrics['profit_factor']:>10.2f}")
        print(f"총 거래 횟수:  {metrics['total_trades']:>10d}회")
        print()

        print("📊 비교 분석")
        print("-" * 40)
        print(f"전략 수익률:   {metrics['total_return']:>10.2%}")
        print(f"Buy & Hold:    {metrics['buy_hold_return']:>10.2%}")
        print(f"초과 수익률:   {metrics['total_return'] - metrics['buy_hold_return']:>10.2%}")
        print("="*60)

def load_data(symbol: str = "005930.KS", start_date: str = "1900-01-01", end_date: str = "2025-12-31") -> pd.DataFrame:
    """데이터 로드"""
    print(f"데이터 로딩 중: {symbol}")

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
        print("예시 데이터 생성 중...")
        return generate_sample_data(start_date, end_date)

def generate_sample_data(start_date: str, end_date: str) -> pd.DataFrame:
    """예시 데이터 생성"""
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    dates = pd.date_range(start=start, end=end, freq='D')

    # 삼성전자와 유사한 가격 패턴 생성
    np.random.seed(42)
    initial_price = 50000
    prices = [initial_price]

    for i in range(1, len(dates)):
        # 연도별 다른 변동성
        year = dates[i].year
        if year < 2005:
            volatility = 0.03
        elif year < 2010:
            volatility = 0.025
        elif year < 2015:
            volatility = 0.02
        elif year < 2020:
            volatility = 0.015
        else:
            volatility = 0.02

        # 장기 상승 트렌드
        trend = 0.0001
        daily_return = np.random.normal(trend, volatility)

        # 특별한 이벤트 시뮬레이션
        if year == 2008:  # 금융위기
            daily_return -= 0.01
        elif year == 2020:  # 코로나
            daily_return -= 0.005
        elif year == 2021:  # 반도체 호황
            daily_return += 0.002

        new_price = prices[-1] * (1 + daily_return)
        new_price = max(new_price, 1000)
        prices.append(new_price)

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

def run_complete_backtest():
    """완전한 백테스트 실행"""
    print("🚀 완전한 백테스트 시스템 시작")
    print("="*50)

    # 백테스트 엔진 초기화
    engine = CompleteBacktestEngine(initial_capital=10000000)  # 1천만원

    # 데이터 로드
    data = load_data("005930.KS", "1900-01-01", "2025-12-31")

    # 백테스트 실행
    print("백테스트 실행 중...")

    # 1. 기술적 지표 계산
    data_with_indicators = engine.calculate_technical_indicators(data)

    # 2. 매매 신호 생성
    data_with_signals = engine.generate_signals(data_with_indicators)

    # 3. 거래 실행 및 수익률 계산
    final_data = engine.execute_trades(data_with_signals)

    # 4. 성과 지표 계산
    metrics = engine.calculate_performance_metrics(final_data)

    # 5. 결과 출력
    engine.print_results(final_data, metrics)

    # 6. 결과 시각화
    engine.plot_results(final_data, metrics)

    # 7. 예측 결과 CSV 저장
    final_data.to_csv('backtest/backtest_results.csv')
    print("\n✅ 백테스트 완료! 결과가 'backtest/backtest_results.csv'에 저장되었습니다.")

    return final_data, metrics

if __name__ == "__main__":
    run_complete_backtest()

