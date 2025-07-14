from sklearn.preprocessing import StandardScaler
from collections import deque
from pathlib import Path
from tensorflow.keras.layers import Dense
import InputLayer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from typing import Tuple
import List
import Deque, Any
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import tensorflow as tf
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: rl_dqn_trader.py
목적: 주식 데이트레이딩/스윙매매 자동화를 위한 DQN 강화학습 트레이더
작성일: 2025-07-08
Author: AI Assistant
"""


# 경로 설정
DATA_PATH = r"C:\data\intraday_stock.csv"  # 또는 daily_stock.csv
MODEL_PATH = r"C:\models\rl_trader.h5"
RESULT_PATH = r"C:\results\rl_trader_report.csv"
PLOT_PATH = r"C:\results\rl_trader_equity.png"

# 경로 자동 생성
for p in [Path(MODEL_PATH).parent, Path(RESULT_PATH).parent]:
    p.mkdir(parents=True, exist_ok=True)

STATE_WINDOW = 30  # 최근 30개 캔들
BATCH_SIZE = 64
MEMORY_SIZE = 5000
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
EPISODES = 1000
LEARNING_RATE = 0.001

ACTIONS = [0, 1, 2]  # 0: 관망, 1: 매수, 2: 매도
ACTION_NAMES = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}

FEATURES = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'MA5', 'MA20', 'MA60', 'RSI', 'MACD'
]


class ReplayBuffer:
    """경험 리플레이 버퍼"""
    def __init__(self, maxlen: int = 5000):
        self.buffer: Deque = deque(maxlen=maxlen)

    def add(self, experience: Tuple):
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> List:
        return random.sample(self.buffer, min(len(self.buffer), batch_size))

    def __len__(self):
        return len(self.buffer)


def load_data(path: str) -> pd.DataFrame:
    """CSV 데이터 로드 및 전처리"""
    df = pd.read_csv(path)
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime')
        df = df.reset_index(drop=True)
    elif 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        df = df.reset_index(drop=True)
    return df


def scale_features(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """특성 정규화"""
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[features] = scaler.fit_transform(df[features])
    return df_scaled


class TradingEnv:
    """주식 트레이딩 환경 (OpenAI Gym 스타일)"""
    def __init__(self, df: pd.DataFrame, window: int = 30):
        self.df = df.reset_index(drop=True)
        self.window = window
        self.current_step = window
        self.position = 0  # 0: 없음, 1: 매수
        self.entry_price = 0.0
        self.done = False
        self.total_profit = 0.0
        self.equity_curve = []
        self.actions = []
        self.rewards = []

    def reset(self) -> np.ndarray:
        self.current_step = self.window
        self.position = 0
        self.entry_price = 0.0
        self.done = False
        self.total_profit = 0.0
        self.equity_curve = [1.0]
        self.actions = []
        self.rewards = []
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        state = self.df[FEATURES].iloc[self.current_step - self.window : self.current_step].values
        state = state.flatten()
        state = np.append(state, [self.position])  # 포지션 정보 포함
        return state.astype(np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        reward = 0.0
        info = {}
        price = self.df['Close'].iloc[self.current_step]
        self.actions.append(action)

        # 행동: 0=관망, 1=매수, 2=매도
        if action == 1 and self.position == 0:  # 매수
            self.position = 1
            self.entry_price = price
        elif action == 2 and self.position == 1:  # 매도
            profit = price - self.entry_price
            reward = profit
            self.total_profit += profit
            self.position = 0
            self.entry_price = 0.0
        # 관망 또는 불가능한 행동은 reward=0

        # 에피소드 종료 조건: 마지막 step 또는 강제 청산
        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            self.done = True
            # 포지션 보유 중이면 청산
            if self.position == 1:
                profit = price - self.entry_price
                reward += profit
                self.total_profit += profit
                self.position = 0
                self.entry_price = 0.0
        self.equity_curve.append(self.equity_curve[-1] + reward)
        self.rewards.append(reward)
        return self._get_state(), reward, self.done, info


class DQNAgent:
    """DQN 에이전트"""
    def __init__(self, state_size: int, action_size: int):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayBuffer(MEMORY_SIZE)
        self.gamma = GAMMA
        self.epsilon = EPSILON_START
        self.epsilon_min = EPSILON_END
        self.epsilon_decay = EPSILON_DECAY
        self.model = self._build_model()

    def _build_model(self) -> tf.keras.Model:
        model = Sequential([
            InputLayer(input_shape=(self.state_size,)),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mse')
        return model

    def act(self, state: np.ndarray) -> int:
        if np.random.rand() <= self.epsilon:
            return random.choice(ACTIONS)
        q_values = self.model.predict(state[np.newaxis, :], verbose=0)[0]
        return int(np.argmax(q_values))

    def remember(self, state, action, reward, next_state, done):
        self.memory.add((state, action, reward, next_state, done))

    def replay(self, batch_size: int):
        minibatch = self.memory.sample(batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(next_state[np.newaxis, :], verbose=0)[0])
            target_f = self.model.predict(state[np.newaxis, :], verbose=0)[0]
            target_f[action] = target
            self.model.fit(state[np.newaxis, :], target_f[np.newaxis, :], epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path: str):
        self.model.save(path)

    def load(self, path: str):
        self.model = tf.keras.models.load_model(path)


def train_dqn():
    """DQN 훈련 및 평가 전체 파이프라인"""
    df = load_data(DATA_PATH)
    df = scale_features(df, FEATURES)
    env = TradingEnv(df, window=STATE_WINDOW)
    state_size = len(FEATURES) * STATE_WINDOW + 1  # 포지션 포함
    action_size = len(ACTIONS)
    agent = DQNAgent(state_size, action_size)
    equity_history = []
    episode_rewards = []

    for e in range(1, EPISODES + 1):
        state = env.reset()
        total_reward = 0.0
        while True:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                break
        agent.replay(BATCH_SIZE)
        equity_history.append(env.equity_curve[-1])
        episode_rewards.append(total_reward)
        if e % 50 == 0:
            print(f"에피소드 {e}/{EPISODES} | 누적수익: {env.equity_curve[-1]:.2f} | 에이전트 탐험률: {agent.epsilon:.3f}")
    agent.save(MODEL_PATH)
    print(f"모델 저장: {MODEL_PATH}")

    # 결과 리포트 저장
    report_df = pd.DataFrame({
        'episode': np.arange(1, EPISODES + 1),
        'total_reward': episode_rewards,
        'final_equity': equity_history
    })
    report_df.to_csv(RESULT_PATH, index=False, encoding='utf-8-sig')
    print(f"리포트 저장: {RESULT_PATH}")

    # 수익률 그래프 저장
    plt.figure(figsize=(10, 5))
    plt.plot(equity_history, label='Equity Curve')
    plt.xlabel('Episode')
    plt.ylabel('Equity')
    plt.title('DQN Trader Equity Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=200)
    print(f"수익률 그래프 저장: {PLOT_PATH}")
    plt.close()

if __name__ == "__main__":
    train_dqn()

