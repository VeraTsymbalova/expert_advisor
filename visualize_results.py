# visualize_results.py

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import matplotlib.pyplot as plt
from add_trend_indicators import add_trend_indicators
from preprocess_data import preprocess_data
from environment import ForexTradingEnv
from stable_baselines3 import PPO
from visualization import plot_trade_results, display_trade_log

def main():
    print("Загружаем и подготавливаем данные...")
    # Применяем preprocess_data (как в обучении)
    df_train, df_test, _ = preprocess_data('EURUSD-H1.csv', train_size_ratio=0.7)

    # Для визуализации берём тестовую часть
    df = df_test.copy()

    print("Добавляем технические индикаторы...")
    df = add_trend_indicators(df)
    df = df.dropna()

    print("Создаём окружение...")
    env = ForexTradingEnv(df)

    model_path = 'training/models/best_model.zip'
    print(f"Загружаем модель из {model_path}...")
    try:
        model = PPO.load(model_path)
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        return

    obs_space_shape = env.observation_space.shape[0]
    model_input_shape = model.policy.observation_space.shape[0]

    if obs_space_shape != model_input_shape:
        print(f"Ошибка: несовпадение формы признаков. Модель ожидает {model_input_shape}, окружение даёт {obs_space_shape}")
        return

    print("Запуск модели...")
    state, _ = env.reset()
    done = False

    while not done:
        action, _ = model.predict(state, deterministic=True)
        action_int = int(action)
        state, reward, done, truncated, _ = env.step(action_int)  # Добавлено 'truncated' для совместимости с Gymnasium

    if hasattr(env, 'trade_log') and env.trade_log:
        print("Отображение результатов торговли...")
        display_trade_log(env.trade_log)
        plot_trade_results(env)
        plt.show()
    else:
        print("Нет сделок для отображения.")

if __name__ == "__main__":
    main()