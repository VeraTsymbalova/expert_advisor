import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
import os
import pickle
import pandas as pd
import logging
from datetime import datetime

import MetaTrader5 as mt5
from import_data import initialize_mt5, shutdown_mt5, get_candles, calculate_spreads, merge_and_clean_data
from add_trend_indicators import add_trend_indicators
from preprocess_data import preprocess_data
import register_env
from train import WrappedVAC
from ding.entry import serial_pipeline
from ppo_forex_config import ppo_forex_config
from ppo_forex_create_config import ppo_forex_create_config
import torch
import torch.nn as nn
from export_to_onnx import ExportableActor
from ding.envs import DingEnvWrapper
from ding.utils import ENV_REGISTRY
from easydict import EasyDict

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Класс для модели, совместимый с ONNX
class ExportableActor(nn.Module):
    def __init__(self, vac_model):
        super().__init__() # Инициализация базового класса
        self.vac_model = vac_model # Сохранение модели

    def forward(self, x):
        return self.vac_model(x, mode='compute_actor')['logit'] # Возврат логитов из модели
    
def prepare_data():
    symbol = "EURUSD"
    start_date = datetime(2023, 4, 1)
    end_date = datetime(2025, 4, 30)

    logging.info(f"Загрузка данных с {start_date.strftime('%Y-%m-%d')} по {end_date.strftime('%Y-%m-%d')} для символа {symbol}")

    initialize_mt5()
    try:
        # Проверка доступных символов
        symbols = mt5.symbols_get()
        available = [s.name for s in symbols if 'EURUSD' in s.name]
        print("Доступные символы:", available)

        # Получение данных
        df_candles = get_candles(symbol, timeframe=mt5.TIMEFRAME_H1, start_date=start_date, end_date=end_date)
        if df_candles.empty:
            logging.error("Полученные свечи пусты. Завершаем выполнение.")
            shutdown_mt5()
            return

        df_spread = calculate_spreads(symbol, start_date, end_date)
        df_final = merge_and_clean_data(df_candles, df_spread)

        # Добавление технических индикаторов
        df_final = add_trend_indicators(df_final)
        logging.info("Технические индикаторы добавлены как признаки")

        # Сохранение расширенного набора в CSV (опционально)
        df_final.to_csv("EURUSD-H1-indicators.csv", index=False, sep='\t')

        # Препроцессинг данных: масштабирование и разбиение
        df_train, df_test, df_val = preprocess_data("EURUSD-H1-indicators.csv")
        logging.info(f"Данные разделены: обучение — {len(df_train)}, тест — {len(df_test)}, валидация — {len(df_val)}")
        
        # Сохранение данных для среды
        with open('train_data.pkl', 'wb') as f:
            pickle.dump(df_train, f)
        with open('val_data.pkl', 'wb') as f:
            pickle.dump(df_val, f)
        with open('test_data.pkl', 'wb') as f:
            pickle.dump(df_test, f)

        logging.info(f"Данные подготовлены")
    finally:
        shutdown_mt5()

# Функция обучения модели
def train_model():
    trained_policy = serial_pipeline(
        [ppo_forex_config, ppo_forex_create_config],
        seed=0,
    )

    from torch.utils.tensorboard import SummaryWriter # Импорт TensorBoard для логов
    writer = SummaryWriter(log_dir="./log/custom_metrics") # Инициализация логирования

    print("Обучение завершено")

    return trained_policy

# Функция экспорта обученной модели
def export_model(trained_policy):
    vac_model = trained_policy._model # Получение обученной модели
    exportable_model = ExportableActor(vac_model) # Обертка для экспорта
    exportable_model.eval() # Перевод модели в режим инференса

    obs_shape = ppo_forex_config.policy.model.obs_shape # Получение формы входных данных
    dummy_input = torch.randn(1, obs_shape) # Входной шаблон для ONNX

    # Проверка, что модель действительно возвращает 3 логита
    with torch.no_grad():
        logits = exportable_model(dummy_input)
        print("Logits shape:", logits.shape)
        print("Logits:", logits)

    # Экспорт в ONNX
    onnx_path = 'ppo_forex_model.onnx'
    torch.onnx.export(
        exportable_model,
        dummy_input,
        onnx_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        opset_version=11
    )
    print(f"Модель экспортирована в {onnx_path}")

    checkpoint_path = './exp/ppo_forex_trading/ckpt/ckpt_best.pth.tar' # Путь к чекпоинту
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True) # Создание папки при необходимости
    torch.save(trained_policy._model.state_dict(), checkpoint_path) # Сохранение состояния модели
    print(f"Состояние модели сохранено в {checkpoint_path}")

    from environment import ForexTradingEnv # Импорт среды

    # Загрузка обучающего набора для теста
    with open('train_data.pkl', 'rb') as f:
        test_data = pickle.load(f)

    env = ForexTradingEnv(data=test_data) # Инициализация среды
    obs = env.reset() # Сброс среды

    trained_model = trained_policy._model # Получение обученной модели
    trained_model.eval() # Перевод в режим инференса

    action_counts = {0: 0, 1: 0, 2: 0} # Счетчик действий
    total_steps = 0
    step = 0

    for _ in range(1000):
        obs_tensor = torch.tensor(obs).unsqueeze(0).float() # Подготовка наблюдения
        with torch.no_grad():
            logits = trained_model(obs_tensor, mode='compute_actor')['logit'] # Получение логитов
            probs = torch.softmax(logits, dim=-1) # Вычисление вероятностей действий
            action = torch.multinomial(probs, num_samples=1).item() # Выбор действия

        obs, reward, done, _, _ = env.step(action) # Шаг в среде
        action_counts[action] += 1
        print(f"[TEST STEP {step}] Action: {action}, Reward: {reward:.4f}") # Лог теста
        step += 1

        if env.net_worth < env.initial_balance * 0.1: # Проверка минимального капитала
            print(f"[ALERT] Net worth below threshold: {env.net_worth:.2f}")
            break
        
        if done:
            break

    print(f"\n[TEST] Завершено {total_steps} шагов.")
    print(f"[TEST] Распределение действий: {action_counts}")

# Функция оценки модели
def evaluate_model(policy, data, label="Validation"):
    from environment import ForexTradingEnv
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(log_dir="./log/custom_metrics")
    env = ForexTradingEnv(data=data)
    obs = env.reset()
    unique_states = set()
    total_reward = 0
    actions = {0: 0, 1: 0, 2: 0}
    step = 0

    model = policy._model
    model.eval()

    while True:
        obs_tensor = torch.tensor(obs).unsqueeze(0).float()
        with torch.no_grad():
            logits = model(obs_tensor, mode='compute_actor')['logit']
            probs = torch.softmax(logits, dim=-1)
            action = torch.multinomial(probs, num_samples=1).item()

        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        actions[action] += 1
        unique_states.add(tuple(obs))
        step += 1
        if done or step >= 1000:
            break

    # Подсчет метрик
    avg_reward = total_reward / step
    total_actions = sum(actions.values())
    exploit_vs_explore = max(actions.values()) / (total_actions + 1e-6)

    # Логирование
    writer.add_scalar(f"{label}/TotalReward", total_reward, step)
    writer.add_scalar(f"{label}/AverageReward", avg_reward, step)
    writer.add_scalar(f"{label}/EpisodeLength", step, step)
    writer.add_scalar(f"{label}/UniqueStates", len(unique_states), step)
    writer.add_scalar(f"{label}/ExploitVsExplore", exploit_vs_explore, step)

    print(f"\n[{label}] Total reward: {total_reward:.2f}, Steps: {step}, Unique states: {len(unique_states)}")
    print(f"[{label}] Action distribution: {actions}") 
    print(f"[{label}] Exploit vs Explore Ratio: {exploit_vs_explore:2f}")

    writer.close()

# Блок запуска
if __name__ == '__main__':
    prepare_data() # Подготовка данных
    
    with open('val_data.pkl', 'rb') as f:
        df_val = pickle.load(f) # Загрузка валидационного датасета
    with open('test_data.pkl', 'rb') as f:
        df_test = pickle.load(f) # Загрузка тестового датасета

    trained_policy = train_model() # Обучение модели
    export_model(trained_policy) # Экспорт модели
    
    evaluate_model(trained_policy, df_val, label="Validation") # Оценка на валидации
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir="./log/custom_metrics")

    evaluate_model(trained_policy, df_test, label="Test") # Оценка на тесте
    writer = SummaryWriter(log_dir="./log/custom_metrics")
