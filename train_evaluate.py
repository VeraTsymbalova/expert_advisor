from stable_baselines3 import PPO
from environment import ForexTradingEnv
import numpy as np
import json

def train_model(env_train, total_timesteps=101326, model_name="best_model"):
    # Загрузка лучших гиперпараметров, если они существуют
    try:
        with open('best_hyperparameters.json', 'r') as f:
            best_params = json.load(f)
        print("Loaded best hyperparameters:", best_params)
    except FileNotFoundError:
        print("best_hyperparameters.json not found. Using default hyperparameters.")
        best_params = {}

    # Определение модели PPO с учебной средой
    model = PPO("MlpPolicy", env_train, verbose=1, **best_params)
    
    # Тренировка модели
    model.learn(total_timesteps=total_timesteps)
    
    # Путь сохранения модели
    model_path = "training/models/" + model_name

    # Сохранение модели
    model.save(model_path)
    
    return model

def evaluate_model(model, env_test):
    # Сбрасывание настроек среды для начала нового оценочного эпизода
    state, _ = env_test.reset()
    done = False
    truncated = False  # 
    trade_log = []  # 

    while not done and not truncated:
        # Получение действия от обученной модели
        action, _ = model.predict(state)
        
        # 
        state, reward, done, truncated, info = env_test.step(int(action))
        
        # 
        if action_type in [1, 2]:  # 
            #
            trade_log.append({
                "action": "Buy" if action_type == 1 else "Sell",
                "lot_size": lot_size,
                "reward": reward,
                "equity": env_test.equity,
                "balance": env_test.balance,
                "free_margin": env_test.free_margin,
                "margin_level": env_test.margin_level,
                "num_open_positions": len(env_test.open_positions)
            })

    return trade_log