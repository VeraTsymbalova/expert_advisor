import gym
from gym import spaces
import numpy as np
import pandas as pd
import pickle

class ForexTradingEnv(gym.Env):
    def __init__(self, data=None, initial_balance=10000, leverage=30, tp=0.002, sl=0.001, account_type='cent'):
        super(ForexTradingEnv, self).__init__()

        # Если data не передано напрямую — загружаем из файла
        if data is None:
            with open('train_data.pkl', 'rb') as f:
                data = pickle.load(f)

        self.data = data.reset_index(drop=True)
        self.account_type = account_type
        if self.account_type == 'cent':
            self.initial_balance = initial_balance * 0.01  # перевод из центов в доллары
        else:
            self.initial_balance = initial_balance
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.position = 0  # 0: no position, 1: long, -1: short
        self.leverage = leverage
        self.tp = tp  # take profit threshold (relative)
        self.sl = sl  # stop loss threshold (relative)
        self.current_step = 0

        # Actions: 0 - hold, 1 - buy, 2 - sell
        self.action_space = spaces.Discrete(3)

        # Observations: все доступные признаки
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.data.shape[1],), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.position = 0

    # случайно начинаем внутри первых N% данных
        self.current_step = np.random.randint(0, int(len(self.data) * 0.8))
        return self._next_observation()

    def _next_observation(self):
        obs = self.data.drop(columns=['DATE']).iloc[self.current_step].values
        return obs.astype(np.float32)

    def step(self, action):
        reward = 0
        terminated = False
        truncated = False

        current_price = self.data.iloc[self.current_step]['Unscaled_Close']
        spread = self.data.iloc[self.current_step]['Unscaled_Spread'] / 10000  # в цену

        if action == 1:  # BUY
            if self.position == 0:
                self.position = 1
                self.entry_price = current_price + (spread / 2)
            elif self.position == -1:
                close_price = current_price - (spread / 2)
                price_diff = self.entry_price - close_price
                reward = price_diff * self.leverage * 10000
                self.balance += reward
                self.position = 0

        elif action == 2:  # SELL
            if self.position == 0:
                self.position = -1
                self.entry_price = current_price - (spread / 2)
            elif self.position == 1:
                close_price = current_price + (spread / 2)
                price_diff = close_price - self.entry_price
                reward = price_diff * self.leverage * 10000
                self.balance += reward
                self.position = 0

    # Проверка Take Profit / Stop Loss
        if self.position != 0:
            if self.position == 1:  # Long
            # Выйдем по BID цене
                current_bid = current_price - (spread / 2)
                price_diff = current_bid - self.entry_price
            else:  # Short
            # Выйдем по ASK цене
                current_ask = current_price + (spread / 2)
                price_diff = self.entry_price - current_ask

            relative_change = price_diff / self.entry_price

            if relative_change >= self.tp:
                reward = price_diff * self.leverage * 10000
                self.balance += reward
                self.position = 0
            elif relative_change <= -self.sl:
                reward = price_diff * self.leverage * 10000
                self.balance += reward
                self.position = 0

        self.net_worth = self.balance

        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            terminated = True

        obs = self._next_observation()

        return obs, reward, terminated, truncated, {}

    def render(self, mode='human'):
        profit = self.net_worth - self.initial_balance
        print(f'Step: {self.current_step}, Balance: {self.balance:.2f}, Net Worth: {self.net_worth:.2f}, Profit: {profit:.2f}')

    def close(self):
        pass
