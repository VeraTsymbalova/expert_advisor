import sys
import os
# Добавление корневой директории проекта в путь Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from preprocess_data import preprocess_data
from add_trend_indicators import add_trend_indicators
from environment import ForexTradingEnv
import json

def objective(trial):
    # Определение пространства поиска гиперпараметров
    n_steps = trial.suggest_categorical('n_steps', [128, 256, 512])
    gamma = trial.suggest_float('gamma', 0.9, 0.9999)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    ent_coef = trial.suggest_float('ent_coef', 1e-8, 1e-2, log=True)
    clip_range = trial.suggest_float('clip_range', 0.1, 0.4)
    gae_lambda = trial.suggest_float('gae_lambda', 0.8, 1.0)
    max_grad_norm = trial.suggest_float('max_grad_norm', 0.5, 1.0)
    vf_coef = trial.suggest_float('vf_coef', 0.5, 1.0)

    # Предобработка данных
    df_train, df_test, _ = preprocess_data('EURUSD-H1.csv')
    df_train = add_trend_indicators(df_train)
    df_test = add_trend_indicators(df_test)

    # Используем rolling window для среды
    episode_length = 1024  # фиксированная длина эпизода

    # Инициализация среды
    env = ForexTradingEnv(df_train, episode_length=episode_length)

    model = PPO(
        'MlpPolicy',
        env,
        verbose=0,
        tensorboard_log='logs/',
        n_steps=n_steps,
        gamma=gamma,
        learning_rate=learning_rate,
        ent_coef=ent_coef,
        clip_range=clip_range,
        gae_lambda=gae_lambda,
        max_grad_norm=max_grad_norm,
        vf_coef=vf_coef,
    )

    # Оценка модели на тестовых данных с тем же episode_length
    eval_env = Monitor(ForexTradingEnv(df_test, episode_length=episode_length))
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='training/models',
        log_path='logs/',
        eval_freq=10000,
        deterministic=True,
        render=False,
    )

    model.learn(total_timesteps=500_000, callback=eval_callback, progress_bar=True)

    # Получение результатов оценки
    mean_reward = eval_callback.best_mean_reward

    return -mean_reward  # Optuna минимизирует objective

if __name__ == '__main__':
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    print("Best hyperparameters:", study.best_params)

    with open('best_hyperparameters.json', 'w') as f:
        json.dump(study.best_params, f)

    print("Лучшая модель сохранена в 'training/models/best_model.zip'")