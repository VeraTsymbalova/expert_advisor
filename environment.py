import gymnasium as gym
from gymnasium import spaces
import numpy as np

class ForexTradingEnv(gym.Env):
    def __init__(self, df, episode_length=None):
        super(ForexTradingEnv, self).__init__()

        self.long_positions_history = []
        self.short_positions_history = []

        self.df = df.reset_index(drop=True)
        self.total_steps = len(self.df)
        self.episode_length = episode_length

        self.initial_balance = 1000
        self.balance = self.initial_balance
        self.equity = self.initial_balance

        self.leverage = 200
        self.margin_call_level = 100
        self.stop_out_level = 50

        self.used_margin = 0
        self.free_margin = self.balance - self.used_margin
        self.margin_level = np.inf

        self.open_positions = []

        self.balance_history = []
        self.equity_history = []
        self.position_history = []
        self.used_margin_history = []
        self.free_margin_history = []
        self.margin_level_history = []

        price_columns = ['Unscaled_Open', 'Unscaled_High', 'Unscaled_Low', 'Unscaled_Close', 'Spread', 'Unscaled_Spread']
        self.numeric_columns = [col for col in self.df.select_dtypes(include=[np.number]).columns.tolist() if col not in price_columns]

        # Количество признаков = числовые признаки + 3 маржи + 1 кол-во открытых позиций
        num_features = len(self.numeric_columns) + 4

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(num_features,), dtype=np.float32)

        self.trade_log = []

    def reset(self, *, seed=None, options=None):
        """
        Сброс состояния среды в начальное
        """
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.open_positions = []

        # Сброс переменных маржи
        self.used_margin = 0
        self.free_margin = self.balance - self.used_margin
        self.margin_level = np.inf

        # Сброс истории
        self.balance_history = [self.balance]
        self.equity_history = [self.equity]
        self.position_history = [len(self.open_positions)]
        self.used_margin_history = [self.used_margin]
        self.free_margin_history = [self.free_margin]
        self.margin_level_history = [self.margin_level]

        # Инициализация истории длинных и коротких позиций
        num_long_positions = 0
        num_short_positions = 0

        self.long_positions_history = [num_long_positions]
        self.short_positions_history = [num_short_positions]

        # Сброс журнала сделок
        self.trade_log = []

        return self._next_observation(), {}

    def _next_observation(self):
        """
        Получение наблюдения для текущего шага
        """
        # Извлечение текущей строки
        obs = self.df.loc[self.current_step, self.numeric_columns].values.astype(np.float32)

        # Заменить NaN на нули для стабильности
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

        # Добавить переменные маржи к наблюдению
        # Обработка бесконечного значения margin_level заменой на большое конечное
        capped_margin_level = self.margin_level if np.isfinite(self.margin_level) else 1e6
        additional_obs = np.array([
            self.used_margin / self.initial_balance,
            self.free_margin / self.initial_balance,
            capped_margin_level / 1000,  # To normalize
            len(self.open_positions) / 200
        ], dtype=np.float32)

        # Проверка NaN или бесконечностей в дополнительных наблюдениях
        additional_obs = np.nan_to_num(additional_obs, nan=0.0, posinf=1e6 / 1000, neginf=0.0)

        # Объединение маржинальной информации с наблюдением
        obs = np.concatenate((obs, additional_obs))

        # Финальная проверка на NaN в наблюдении
        if np.any(np.isnan(obs)):
            print(f"NaN detected in observation at step {self.current_step}")
            obs = np.nan_to_num(obs, nan=0.0, posinf=1e6 / 1000, neginf=0.0)

        return obs

    def step(self, action):
        """
        Выполнение одиного шага в среде
        """
        terminated = False
        truncated = False

        # Сохранить предыдущую стоимость капитала для расчёта вознаграждения
        previous_equity = self.equity

        # Использование немасштабированной цены закрытия для расчета прибыли
        current_price = self.df.loc[self.current_step, 'Unscaled_Close']
        current_spread = self.df.loc[self.current_step, 'Unscaled_Spread'] * 0.00001  # Конвертация спреда в ценовые единицы

        total_realized_profit = 0

        # Инициализация штрафа за неверные действия
        penalty = 0

        # Process action
        # Тип действия - целое число от 0 до 3
        action_type = action

        # Использование фиксированного лота
        lot_size = 0.1  # Fixed lot size
        position_size = lot_size * 100000  # Convert lot size to units

        if action_type == 0:
            # Hold, do nothing
            reward_penalty = -0.00001  # Small penalty for holding
        elif action_type == 1 or action_type == 2:
            # Buy or Sell
            if len(self.open_positions) < 200:
                required_margin = position_size / self.leverage

                if self.free_margin >= required_margin and position_size > 0:
                    # Корректировка цены входа с учётом спреда
                    if action_type == 1:  # Buy
                        adjusted_entry_price = current_price + current_spread / 2
                        position_type = 'long'
                    else:  # Sell
                        adjusted_entry_price = current_price - current_spread / 2
                        position_type = 'short'

                    position = {
                        'type': position_type,
                        'entry_price': adjusted_entry_price,
                        'size': position_size,
                        'step_opened': self.current_step
                    }
                    print(f"[STEP {self.current_step}] OPEN {position_type.upper()} @ {adjusted_entry_price:.5f} | Size: {position_size}")
                    # Открытие позиции
                    self.open_positions.append(position)
                    # Обновление маржи
                    self.used_margin += required_margin
                    self.free_margin -= required_margin
                    # Запись сделки
                    trade = {
                        'step': self.current_step,
                        'date': self.df.loc[self.current_step, 'Date'],
                        'action': 'Buy' if action_type == 1 else 'Sell',
                        'position_size_lots': lot_size,
                        'entry_price': adjusted_entry_price
                    }
                    self.trade_log.append(trade)
                else:
                    # Недостаточно маржи или неправильный размер позиции
                    penalty -= 0.1  # Наказание за неправильное действие
            else:
                # Слишком много открытых позиций
                penalty -= 0.1  # Наказание за неправильное действие
            reward_penalty = 0  # нет штрафов за совершение действий
        elif action_type == 3:
            # Попытка закрытия всех позиций
            positions_to_close = []
            for pos in self.open_positions:
                # Корректировка цены выхода с учетом спреда
                if pos['type'] == 'long':
                    exit_price = current_price - current_spread / 2
                    pnl = (exit_price - pos['entry_price']) * pos['size']
                else:  # 'short'
                    exit_price = current_price + current_spread / 2
                    pnl = (pos['entry_price'] - exit_price) * pos['size']
                total_realized_profit += pnl
                # Обновление баланса
                self.balance += pnl
                # Освобождение маржи
                required_margin = pos['size'] / self.leverage
                self.used_margin -= required_margin
                self.free_margin += required_margin
                positions_to_close.append(pos)
                # Запись сделки
                trade = {
                    'step': self.current_step,
                    'date': self.df.loc[self.current_step, 'Date'],
                    'action': 'Close',
                    'position_type': 'Buy' if pos['type'] == 'long' else 'Sell',
                    'position_size_lots': pos['size'] / 100000,
                    'entry_price': pos['entry_price'],
                    'exit_price': exit_price,
                    'profit_loss': pnl
                }
                self.trade_log.append(trade)
            # Удаление закрытых позиций
            for pos in positions_to_close:
                self.open_positions.remove(pos)
            reward_penalty = 0  # нет штрафов за совершение действий

        # Расчёт нереализованной прибыли/убытка
        unrealized_pnl = 0
        for pos in self.open_positions:
            # Корректировка потенциальной цены выхода с учетом спреда
            if pos['type'] == 'long':
                potential_exit_price = current_price - current_spread / 2
                pnl = (potential_exit_price - pos['entry_price']) * pos['size']
            else:  # 'short'
                potential_exit_price = current_price + current_spread / 2
                pnl = (pos['entry_price'] - potential_exit_price) * pos['size']
            unrealized_pnl += pnl

        # Обновление equity
        self.equity = self.balance + unrealized_pnl

        # Обновление свободной маржи
        self.free_margin = self.equity - self.used_margin

        # Обновление уровня маржи с защитой от деления на ноль
        if self.used_margin > 0:
            self.margin_level = (self.equity / self.used_margin) * 100
        else:
            self.margin_level = 1e6  #

        # Проверка на маржин-колл или стоп-аут
        if self.margin_level <= self.stop_out_level:
            # Закрытие позиций до восстановления уровня маржи
            positions_to_close = []
            # Сортировка позиций по прибыли/убытку
            positions_with_pnl = []
            for pos in self.open_positions:
                # Корректировка потенциальной цены выхода с учетом спреда
                if pos['type'] == 'long':
                    exit_price = current_price - current_spread / 2
                    pnl = (exit_price - pos['entry_price']) * pos['size']
                else:  # 'short'
                    exit_price = current_price + current_spread / 2
                    pnl = (pos['entry_price'] - exit_price) * pos['size']
                positions_with_pnl.append({'position': pos, 'pnl': pnl})

            positions_with_pnl.sort(key=lambda x: x['pnl'])

            for p in positions_with_pnl:
                # Закрытие позиции
                pnl = p['pnl']
                self.balance += pnl
                total_realized_profit += pnl

                # Маржа
                required_margin = p['position']['size'] / self.leverage
                self.used_margin -= required_margin
                self.free_margin += required_margin

                positions_to_close.append(p['position'])
                # Запись сделки
                trade = {
                    'step': self.current_step,
                    'date': self.df.loc[self.current_step, 'Date'],
                    'action': 'Margin Call Close',
                    'position_type': 'Buy' if p['position']['type'] == 'long' else 'Sell',
                    'position_size_lots': p['position']['size'] / 100000,
                    'entry_price': p['position']['entry_price'],
                    'exit_price': exit_price,
                    'profit_loss': pnl
                }
                self.trade_log.append(trade)

                # Пересчет equity и маржи
                unrealized_pnl -= pnl  # вычитание PnL закрытой позиции
                self.equity = self.balance + unrealized_pnl

                if self.used_margin > 0:
                    self.margin_level = (self.equity / self.used_margin) * 100
                else:
                    self.margin_level = 1e6  #

                if self.margin_level > self.stop_out_level:
                    break  # Margin level is acceptable

            # Удаление закрытых позиций
            for pos in positions_to_close:
                if pos in self.open_positions:
                    self.open_positions.remove(pos)

        # Добавление в историю
        self.balance_history.append(self.balance)
        self.equity_history.append(self.equity)
        self.position_history.append(len(self.open_positions))
        self.used_margin_history.append(self.used_margin)
        self.free_margin_history.append(self.free_margin)
        self.margin_level_history.append(self.margin_level)

        num_long_positions = sum(1 for pos in self.open_positions if pos['type'] == 'long')
        num_short_positions = sum(1 for pos in self.open_positions if pos['type'] == 'short')

        self.long_positions_history.append(num_long_positions)
        self.short_positions_history.append(num_short_positions)

        # переход к следующему шагу
        self.current_step += 1

        # Проверка завершения эпизода
        if self.episode_length is not None:
            if self.current_step >= self.episode_length:
                terminated = True
        else:
            if self.current_step >= self.total_steps - 1:
                terminated = True

        observation = self._next_observation()

        # Расчёт вознаграждения на основе изменения equity и штрафа за просадку
        equity_change = (self.equity - previous_equity) / self.initial_balance

        # Calculate drawdown
        peak_equity = max(self.equity_history)
        drawdown = max(0, peak_equity - self.equity)
        drawdown_penalty_factor = 0.1  # Adjust as needed
        drawdown_penalty = (drawdown * drawdown_penalty_factor) / self.initial_balance

        # Расчёт просадки
        reward = equity_change - drawdown_penalty + penalty + reward_penalty + (total_realized_profit / self.initial_balance)

        # Проверка на NaN и бесконечности
        if not np.isfinite(self.equity):
            print(f"Non-finite equity detected at step {self.current_step}")
            self.equity = self.balance  #

        if not np.isfinite(reward):
            print(f"Non-finite reward detected at step {self.current_step}")
            reward = -1  #

        return observation, reward, terminated, truncated, {}