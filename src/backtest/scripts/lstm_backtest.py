from datetime import datetime
import timedelta
from sklearn.metrics import mean_squared_error
import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM
import Dense
import Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from typing import Dict
import List
import Optional, Tuple, Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import warnings
import yfinance as yf
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM 기반 주가 예측 백테스트 시스템
삼성전자(005930) 과거 최대치부터 2025년 현재까지

Author: AI Trading System
Created: 2025-07-08
Version: 1.0.0

Features:
- LSTM 딥러닝 모델로 주가 예측
- 기술적 지표 + LSTM 예측 결합
- 과거 최대치부터 현재까지 백테스트
- 주요 성과 지표 계산 (누적 수익률, 최대 낙폭, 승률 등)
"""

warnings.filterwarnings('ignore')

# 딥러닝 라이브러리

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class LSTMPredictor:
    """LSTM 기반 주가 예측 모델"""

    def __init__(self, sequence_length: int = 60, prediction_days: int = 1):
        self.sequence_length = sequence_length
        self.prediction_days = prediction_days
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_trained = False

    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """LSTM 모델용 데이터 준비"""
        # 종가와 거래량을 사용
        features = data[['Close', 'Volume']].values

        # 정규화
        scaled_features = self.scaler.fit_transform(features)

        X, y = [], []
        for i in range(self.sequence_length, len(scaled_features) - self.prediction_days + 1):
            X.append(scaled_features[i-self.sequence_length:i])
            y.append(scaled_features[i:i+self.prediction_days, 0])  # 종가만 예측

        return np.array(X), np.array(y)

    def build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """LSTM 모델 구축"""
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(units=50, return_sequences=True),
            Dropout(0.2),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(units=25),
            Dense(units=self.prediction_days)
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mean_squared_error',
            metrics=['mae']
        )

        return model

    def train(self, data: pd.DataFrame, epochs: int = 50, batch_size: int = 32):
        """LSTM 모델 학습"""
        print("LSTM 모델 학습 시작...")

        # 데이터 준비
        X, y = self.prepare_data(data)

        # 학습/검증 데이터 분할
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # 모델 구축
        self.model = self.build_model((X.shape[1], X.shape[2]))

        # 모델 학습
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=1
        )

        self.is_trained = True
        print("LSTM 모델 학습 완료!")

        return history

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """주가 예측"""
        if not self.is_trained:
            raise ValueError("모델이 학습되지 않았습니다.")

        # 예측용 데이터 준비
        features = data[['Close', 'Volume']].values
        scaled_features = self.scaler.transform(features)

        predictions = []
        for i in range(self.sequence_length, len(scaled_features)):
            X = scaled_features[i-self.sequence_length:i].reshape(1, self.sequence_length, 2)
            pred = self.model.predict(X, verbose=0)
            predictions.append(pred[0, 0])

        # 역정규화
        predictions = np.array(predictions).reshape(-1, 1)
        dummy_array = np.zeros((len(predictions), 2))
        dummy_array[:, 0] = predictions.flatten()
        predictions = self.scaler.inverse_transform(dummy_array)[:, 0]

        return predictions

    def calculate_prediction_metrics(self, actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        """예측 성능 지표 계산"""
        mse = mean_squared_error(actual, predicted)
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mse)

        # 방향성 정확도
        actual_direction = np.diff(actual) > 0
        predicted_direction = np.diff(predicted) > 0
        direction_accuracy = np.mean(actual_direction == predicted_direction)

        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'direction_accuracy': direction_accuracy
        }

class LSTMBacktestStrategy:
    """LSTM 기반 백테스트 전략"""

    def __init__(self, initial_capital: float = 10000000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []

        # LSTM 모델
        self.lstm_predictor = LSTMPredictor(sequence_length=60, prediction_days=1)

        # 전략 파라미터
        self.confidence_threshold = 0.6
        self.position_size_ratio = 0.1
        self.stop_loss_ratio = 0.05
        self.take_profit_ratio = 0.10

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
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['MA_60'] = df['Close'].rolling(window=60).mean()

        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

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

        # 거래량 지표
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']

        return df

    def analyze_with_lstm(self, data: pd.DataFrame, current_index: int) -> Dict[str, Any]:
        """LSTM + 기술적 지표 결합 분석"""
        if current_index < 60:  # 충분한 데이터가 없으면
            return {
                'lstm_prediction': None,
                'technical_score': 0,
                'combined_score': 0,
                'confidence': 0,
                'action': 'HOLD'
            }

        # LSTM 예측
        historical_data = data.iloc[:current_index+1]
        try:
            lstm_prediction = self.lstm_predictor.predict(historical_data)
            current_price = data.iloc[current_index]['Close']
            predicted_price = lstm_prediction[-1]

            # 예측 방향과 크기
            price_change_ratio = (predicted_price - current_price) / current_price
            lstm_score = np.tanh(price_change_ratio * 10)  # -1 ~ 1 범위로 정규화

        except Exception as e:
            print(f"LSTM 예측 오류: {e}")
            lstm_score = 0
            price_change_ratio = 0

        # 기술적 지표 점수
        row = data.iloc[current_index]
        technical_score = 0

        # RSI
        if row['RSI'] < 30:
            technical_score += 0.2
        elif row['RSI'] > 70:
            technical_score -= 0.2

        # 이동평균 크로스오버
        if row['MA_5'] > row['MA_20']:
            technical_score += 0.15
        else:
            technical_score -= 0.15

        # 볼린저 밴드
        bb_position = (row['Close'] - row['BB_lower']) / (row['BB_upper'] - row['BB_lower'])
        if bb_position < 0.2:
            technical_score += 0.15
        elif bb_position > 0.8:
            technical_score -= 0.15

        # MACD
        if row['MACD'] > row['MACD_signal']:
            technical_score += 0.1
        else:
            technical_score -= 0.1

        # 거래량
        if row['Volume_Ratio'] > 1.5:
            technical_score += 0.1

        # 종합 점수 (LSTM 60%, 기술적 지표 40%)
        combined_score = lstm_score * 0.6 + technical_score * 0.4
        confidence = min(abs(combined_score) + 0.3, 1.0)

        # 매매 방향 결정
        if combined_score > 0.2:
            action = 'BUY'
        elif combined_score < -0.2:
            action = 'SELL'
        else:
            action = 'HOLD'

        return {
            'lstm_prediction': predicted_price if 'predicted_price' in locals() else None,
            'price_change_ratio': price_change_ratio if 'price_change_ratio' in locals() else 0,
            'lstm_score': lstm_score,
            'technical_score': technical_score,
            'combined_score': combined_score,
            'confidence': confidence,
            'action': action
        }

    def execute_trade(self, analysis: Dict[str, Any], current_price: float,
                     date: datetime, symbol: str) -> bool:
        """거래 실행"""
        if analysis['action'] == 'HOLD' or analysis['confidence'] < self.confidence_threshold:
            return False

        action = analysis['action']
        position_size = self.current_capital * self.position_size_ratio * analysis['confidence']
        quantity = int(position_size / current_price)

        if quantity <= 0:
            return False

        # 수수료 및 슬리피지
        commission_rate = 0.00015
        slippage_rate = 0.0001

        if action == 'BUY':
            # 매수
            total_cost = quantity * current_price * (1 + commission_rate + slippage_rate)

            if total_cost <= self.current_capital:
                self.current_capital -= total_cost

                if symbol not in self.positions:
                    self.positions[symbol] = {'quantity': 0, 'avg_price': 0}

                pos = self.positions[symbol]
                new_quantity = pos['quantity'] + quantity
                new_avg_price = ((pos['quantity'] * pos['avg_price']) + (quantity * current_price)) / new_quantity

                pos['quantity'] = new_quantity
                pos['avg_price'] = new_avg_price

                # 거래 기록
                trade = {
                    'date': date,
                    'symbol': symbol,
                    'action': action,
                    'quantity': quantity,
                    'price': current_price,
                    'total_cost': total_cost,
                    'lstm_prediction': analysis['lstm_prediction'],
                    'confidence': analysis['confidence'],
                    'reasoning': f"LSTM점수: {analysis['lstm_score']:.3f}, 기술점수: {analysis['technical_score']:.3f}"
                }
                self.trades.append(trade)
                self.total_trades += 1
                return True

        elif action == 'SELL':
            # 매도
            if symbol in self.positions and self.positions[symbol]['quantity'] > 0:
                pos = self.positions[symbol]
                sell_quantity = min(quantity, pos['quantity'])

                if sell_quantity > 0:
                    gross_amount = sell_quantity * current_price
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
                        'price': current_price,
                        'gross_amount': gross_amount,
                        'net_amount': net_amount,
                        'lstm_prediction': analysis['lstm_prediction'],
                        'confidence': analysis['confidence'],
                        'reasoning': f"LSTM점수: {analysis['lstm_score']:.3f}, 기술점수: {analysis['technical_score']:.3f}"
                    }
                    self.trades.append(trade)
                    self.total_trades += 1
                    return True

        return False

    def calculate_portfolio_value(self, current_price: float, symbol: str) -> float:
        """포트폴리오 가치 계산"""
        portfolio_value = self.current_capital

        if symbol in self.positions:
            position_value = self.positions[symbol]['quantity'] * current_price
            portfolio_value += position_value

        return portfolio_value

    def update_performance_metrics(self, portfolio_value: float):
        """성과 지표 업데이트"""
        if portfolio_value > self.peak_capital:
            self.peak_capital = portfolio_value

        current_drawdown = (self.peak_capital - portfolio_value) / self.peak_capital
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown

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

        # 매수/매도 쌍으로 수익 계산
        buy_trades = [t for t in self.trades if t['action'] == 'BUY']
        sell_trades = [t for t in self.trades if t['action'] == 'SELL']

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

        # 샤프 비율
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

class LSTMBacktestEngine:
    """LSTM 백테스트 엔진"""

    def __init__(self, initial_capital: float = 10000000):
        self.strategy = LSTMBacktestStrategy(initial_capital)
        self.data = None

    def load_samsung_data(self) -> pd.DataFrame:
        """삼성전자 데이터 로드 (과거 최대치부터 현재까지)"""
        print("삼성전자 데이터 로딩 중...")

        try:
            # 삼성전자 데이터 다운로드 (2000년부터 현재까지)
            ticker = yf.Ticker("005930.KS")
            data = ticker.history(start="2000-01-01", end="2025-12-31")

            if data.empty:
                raise ValueError("삼성전자 데이터를 찾을 수 없습니다.")

            # 컬럼명 정규화
            data.columns = [col.title() for col in data.columns]

            print(f"삼성전자 데이터 로딩 완료: {len(data)}개 데이터")
            print(f"데이터 기간: {data.index[0].strftime('%Y-%m-%d')} ~ {data.index[-1].strftime('%Y-%m-%d')}")

            return data

        except Exception as e:
            print(f"데이터 로딩 실패: {e}")
            print("예시 데이터 생성 중...")
            return self.generate_samsung_sample_data()

    def generate_samsung_sample_data(self) -> pd.DataFrame:
        """삼성전자 예시 데이터 생성 (과거 최대치부터 현재까지)"""
        print("삼성전자 예시 데이터 생성 중...")

        # 2000년부터 2025년까지
        start = pd.to_datetime("2000-01-01")
        end = pd.to_datetime("2025-07-08")
        dates = pd.date_range(start=start, end=end, freq='D')

        # 삼성전자와 유사한 가격 패턴 생성 (과거 최대치 포함)
        np.random.seed(42)

        # 2000년 초기 가격
        initial_price = 50000

        # 주요 가격 변동 이벤트 시뮬레이션
        prices = [initial_price]
        current_price = initial_price

        for i, date in enumerate(dates[1:], 1):
            # 연도별 다른 변동성
            year = date.year

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

            # 랜덤 변동
            daily_return = np.random.normal(trend, volatility)

            # 특별한 이벤트 시뮬레이션
            if year == 2008:  # 금융위기
                daily_return -= 0.01
            elif year == 2020:  # 코로나
                daily_return -= 0.005
            elif year == 2021:  # 반도체 호황
                daily_return += 0.002
            elif year == 2022:  # 반도체 침체
                daily_return -= 0.003

            new_price = current_price * (1 + daily_return)
            new_price = max(new_price, 1000)  # 최소 가격 보장

            prices.append(new_price)
            current_price = new_price

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

        print(f"삼성전자 예시 데이터 생성 완료: {len(data)}개 데이터")
        return data

    def run_backtest(self, data: pd.DataFrame) -> Dict[str, Any]:
        """LSTM 백테스트 실행"""
        print("LSTM 백테스트 시작...")

        # 기술적 지표 계산
        data_with_indicators = self.strategy.calculate_technical_indicators(data)

        # LSTM 모델 학습 (처음 80% 데이터로 학습)
        train_size = int(len(data_with_indicators) * 0.8)
        train_data = data_with_indicators.iloc[:train_size]

        print("LSTM 모델 학습 중...")
        history = self.strategy.lstm_predictor.train(train_data, epochs=30, batch_size=32)

        # 백테스트 루프 (학습 데이터 이후부터)
        print("백테스트 실행 중...")
        for i in range(train_size, len(data_with_indicators)):
            row = data_with_indicators.iloc[i]
            date = data_with_indicators.index[i]

            # NaN 체크
            if pd.isna(row['RSI']) or pd.isna(row['MA_5']) or pd.isna(row['MA_20']):
                continue

            # LSTM + 기술적 지표 분석
            analysis = self.strategy.analyze_with_lstm(data_with_indicators, i)

            # 거래 실행
            trade_executed = self.strategy.execute_trade(analysis, row['Close'], date, '005930')

            # 포트폴리오 가치 계산
            portfolio_value = self.strategy.calculate_portfolio_value(row['Close'], '005930')
            self.strategy.update_performance_metrics(portfolio_value)

            # 진행상황 출력
            if (i - train_size) % ((len(data_with_indicators) - train_size) // 10) == 0:
                progress = ((i - train_size) / (len(data_with_indicators) - train_size)) * 100
                print(f"백테스트 진행률: {progress:.1f}%")

        # 최종 성과 지표 계산
        final_metrics = self.strategy.calculate_final_metrics()

        print("LSTM 백테스트 완료!")
        return {
            'metrics': final_metrics,
            'trades': self.strategy.trades,
            'equity_curve': self.strategy.equity_curve,
            'data': data_with_indicators,
            'train_size': train_size
        }

    def plot_results(self, results: Dict[str, Any]):
        """결과 시각화"""
        metrics = results['metrics']
        equity_curve = results['equity_curve']
        data = results['data']
        train_size = results['train_size']

        # 그래프 설정
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('LSTM 기반 주가 예측 백테스트 결과 (삼성전자 005930)', fontsize=16, fontweight='bold')

        # 1. 자본 곡선
        axes[0, 0].plot(equity_curve, label='포트폴리오 가치', color='blue', linewidth=2)
        axes[0, 0].axhline(y=self.strategy.initial_capital, color='red', linestyle='--',
                          label=f'초기 자본: {self.strategy.initial_capital:,.0f}원')
        axes[0, 0].axvline(x=train_size, color='green', linestyle='--', alpha=0.7,
                          label='LSTM 학습 완료')
        axes[0, 0].set_title('자본 곡선')
        axes[0, 0].set_ylabel('포트폴리오 가치 (원)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. 주가 차트와 거래 포인트
        axes[0, 1].plot(data.index, data['Close'], label='삼성전자 주가', color='black', linewidth=1)
        axes[0, 1].axvline(x=data.index[train_size], color='green', linestyle='--', alpha=0.7)

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
        📊 LSTM 백테스트 성과 지표

        💰 총 수익률: {metrics['total_return']:.2%}
        📈 연간 수익률: {metrics['annual_return']:.2%}
        📉 최대 낙폭: {metrics['max_drawdown']:.2%}
        🎯 승률: {metrics['win_rate']:.2%}
        📊 수익 팩터: {metrics['profit_factor']:.2f}
        📈 샤프 비율: {metrics['sharpe_ratio']:.2f}
        🔄 총 거래 횟수: {metrics['total_trades']}회
        💵 순손익: {metrics['net_profit']:,.0f}원

        🤖 LSTM 모델 정보
        📚 학습 데이터: {train_size}일
        🔮 예측 기간: {len(data) - train_size}일
        """

        axes[1, 0].text(0.1, 0.5, metrics_text, transform=axes[1, 0].transAxes,
                       fontsize=11, verticalalignment='center', fontfamily='monospace')
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
        data = results['data']
        train_size = results['train_size']

        print("\n" + "="*70)
        print("🤖 LSTM 기반 주가 예측 백테스트 결과")
        print("="*70)
        print(f"📅 백테스트 기간: {data.index[0].strftime('%Y-%m-%d')} ~ {data.index[-1].strftime('%Y-%m-%d')}")
        print(f"💰 초기 자본: {self.strategy.initial_capital:,.0f}원")
        print(f"💵 최종 자본: {results['equity_curve'][-1]:,.0f}원")
        print(f"🤖 LSTM 학습 기간: {data.index[0].strftime('%Y-%m-%d')} ~ {data.index[train_size-1].strftime('%Y-%m-%d')}")
        print(f"🔮 예측 기간: {data.index[train_size].strftime('%Y-%m-%d')} ~ {data.index[-1].strftime('%Y-%m-%d')}")
        print()

        print("📊 주요 성과 지표")
        print("-" * 50)
        print(f"총 수익률:     {metrics['total_return']:>12.2%}")
        print(f"연간 수익률:   {metrics['annual_return']:.2%}")
        print(f"최대 낙폭:     {metrics['max_drawdown']:.2%}")
        print(f"승률:          {metrics['win_rate']:.2%}")
        print(f"수익 팩터:     {metrics['profit_factor']:.2f}")
        print(f"샤프 비율:     {metrics['sharpe_ratio']:.2f}")
        print(f"총 거래 횟수:  {metrics['total_trades']}회")
        print()

        print("💰 손익 분석")
        print("-" * 50)
        print(f"총 수익:       {metrics['total_profit']:.0f}원")
        print(f"총 손실:       {metrics['total_loss']:.0f}원")
        print(f"순손익:        {metrics['net_profit']:.0f}원")
        print()

        if trades:
            print("🔍 최근 거래 내역 (상위 5개)")
            print("-" * 50)
            for i, trade in enumerate(trades[-5:], 1):
                lstm_pred = trade.get('lstm_prediction', 'N/A')
                if lstm_pred is not None:
                    lstm_pred_str = f"{lstm_pred:,.0f}원"
                else:
                    lstm_pred_str = "N/A"

                print(f"{i}. {trade['date'].strftime('%Y-%m-%d')} | "
                      f"{trade['action']} | {trade['quantity']}주 | "
                      f"{trade['price']:,.0f}원 | LSTM예측: {lstm_pred_str} | "
                      f"신뢰도: {trade['confidence']:.2f}")

        print("="*70)

def main():
    """메인 함수"""
    print("🚀 LSTM 기반 주가 예측 백테스트 시스템 시작")
    print("="*60)

    # 백테스트 엔진 초기화
    engine = LSTMBacktestEngine(initial_capital=10000000)  # 1천만원

    # 삼성전자 데이터 로드
    data = engine.load_samsung_data()

    # LSTM 백테스트 실행
    results = engine.run_backtest(data)

    # 결과 출력
    engine.print_detailed_results(results)

    # 결과 시각화
    engine.plot_results(results)

    print("\n✅ LSTM 백테스트 완료!")

if __name__ == "__main__":
    main()

