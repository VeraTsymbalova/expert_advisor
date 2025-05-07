import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import gym
from gym import spaces
import numpy as np
import pandas as pd
import pickle

# Класс среды для торговли на рынке Forex
class ForexTradingEnv(gym.Env):
    def __init__(self, data=None, initial_balance=10000, leverage=100, tp=0.002, sl=0.004, account_type='cent'):
        super(ForexTradingEnv, self).__init__() # Инициализация базового класса среды

        # Загрузка данных
        if data is None:
            with open('train_data.pkl', 'rb') as f:
                data = pickle.load(f)

        self.data = data.reset_index(drop=True) # Сброс индексов
        self.account_type = account_type # Тип счёта (cent/dollar)
        if self.account_type == 'cent':
            self.initial_balance = initial_balance * 0.01  # перевод из центов в доллары
        else:
            self.initial_balance = initial_balance
        self.balance = self.initial_balance # Баланс счёта
        self.net_worth = self.initial_balance # Общая стоимость счёта
        self.position = 0  # 0: no position, 1: long, -1: short
        self.leverage = leverage # Кредитное плечо
        self.tp = tp  # Порог take-profit (относительный)
        self.sl = sl  # Порог stop-loss (относительный)
        self.current_step = 0 # Текущий шаг по времени
        self.action_counts = {0: 0, 1: 0, 2: 0} # Счетчики действий

        # Дискретное пространство: 0 - hold, 1 - buy, 2 - sell
        self.action_space = spaces.Discrete(3)

        # Observations
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.data.shape[1],), dtype=np.float32
        ) # Все признаки наблюдения в одном векторе


    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed) # Установка зерна генерации
        self.balance = self.initial_balance # Сброс баланса
        self.net_worth = self.initial_balance # Сброс стоимости счета
        self.position = 0 # Обнуляем позицию

        self.current_step = np.random.randint(0, int(len(self.data) * 0.8)) # Начинаем случайно внутри первых 80% данных
        return self._next_observation() # Возвращаем начальное наблюдение

    def _next_observation(self):
        obs = self.data.drop(columns=['DATE']).iloc[self.current_step].values # Убираем столбец DATE и берём текущую строку
        return obs.astype(np.float32) # Приводим к float32 для совместимости с моделью

    def step(self, action):
        self.action_counts[action] += 1
        reward = 0 # Награда
        terminated = False
        truncated = False

        current_price = self.data.iloc[self.current_step]['Unscaled_Close'] # Текущая цена закрытия
        spread = self.data.iloc[self.current_step]['Unscaled_Spread'] / 10000  # # Спред в ценовых пунктах


        if action == 1:  # BUY
            if self.position == 0:
                self.position = 1 # Открытие длинной позиции
                self.entry_price = current_price + (spread / 2) # Цена входа с учетом спреда
                reward += 0.1  # # Награда за открытие позиции
            elif self.position == -1:
                close_price = current_price - (spread / 2)
                price_diff = self.entry_price - close_price
                reward = price_diff * self.leverage * 10000 # Расчет прибыли/убытка
                self.balance += reward
                self.position = 0 # Закрытие позиции


        elif action == 2:  # SELL
            if self.position == 0:
                self.position = -1 # Открытие короткой позиции
                self.entry_price = current_price - (spread / 2)
                reward += 0.1
            elif self.position == 1:
                close_price = current_price + (spread / 2)
                price_diff = close_price - self.entry_price
                reward = price_diff * self.leverage * 10000
                self.balance += reward
                self.position = 0

        if self.current_step % 100 == 0:
            print(f"[STEP {self.current_step}] Action counts: {self.action_counts}")
            profit = self.net_worth - self.initial_balance
            print(f'Step: {self.current_step}, Balance: {self.balance:.2f}, Net Worth: {self.net_worth:.2f}, Profit: {profit:.2f}')

    # Проверка Take Profit / Stop Loss
        if self.position != 0:
            if self.position == 1:  # Long
                current_bid = current_price - (spread / 2) # Цена продажи (BID)
                price_diff = current_bid - self.entry_price
            else:  # Short
                current_ask = current_price + (spread / 2) # Цена покупки (ASK)
                price_diff = self.entry_price - current_ask

            relative_change = price_diff / self.entry_price # Относительное изменение цены

            if relative_change >= self.tp: # Достигли TP
                reward = price_diff * self.leverage * 10000
                self.balance += reward
                self.position = 0
            elif relative_change <= -self.sl: # Достигли SL
                reward = price_diff * self.leverage * 10000
                self.balance += reward
                self.position = 0

        self.net_worth = self.balance # Обновление стоимости счета

        self.current_step += 1 # Переход к следующему шагу
        if self.current_step >= len(self.data) - 1:
            terminated = True # Завершение эпизода

        obs = self._next_observation() # Получение нового наблюдения
        
        # === Наказания за неэффективные стратегии ===
        if self.current_step % 1000 == 0 and self.current_step > 0:
            total_actions = sum(self.action_counts.values())
            if total_actions > 0:
                freq = {k: v / total_actions for k, v in self.action_counts.items()}
        
            if freq.get(1, 0) < 0.05: # Редкий BUY
                reward -= 1.0

            if freq.get(2, 0) < 0.05: # Редкий SELL
                reward -= 1.0
        
            if freq.get(0, 0) > 0.80: # Частый HOLD
                reward -= 1.0

        if action in [1, 2]:
            if self.position == 0:
                reward += 0.1  # За открытие позиции
            else:
                reward += 0.02  # За активность

        return obs, reward, terminated, truncated, {} # Возврат следующего состояния и награды

    def render(self, mode='human'):
        profit = self.net_worth - self.initial_balance

    def close(self): # Закрытие среды
        pass
