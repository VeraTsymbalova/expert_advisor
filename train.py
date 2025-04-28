from ding.entry import serial_pipeline
from ppo_forex_config import ppo_forex_config
from ppo_forex_create_config import ppo_forex_create_config
import register_env  # здесь ты регистрируешь ForexTradingEnv
import pandas as pd
from add_trend_indicators import add_trend_indicators
from preprocess_data import preprocess_data
import pickle
import torch
import torch.nn as nn
from ding.envs import DingEnvWrapper
from ding.utils import ENV_REGISTRY

ENV_REGISTRY.register('gym', DingEnvWrapper)

class WrappedVAC(nn.Module):
    def __init__(self, vac_model):
        super(WrappedVAC, self).__init__()
        self.vac_model = vac_model

    def forward(self, x):
        output = self.vac_model(x, mode='compute_actor')['logit']
        return output

if __name__ == '__main__':
    # 1. Загружаем и обрабатываем данные
    df = pd.read_csv('EURUSD-H1.csv', delimiter='\t')
    df = add_trend_indicators(df)
    df.to_csv('EURUSD-H1-with-indicators.csv', index=False, sep='\t')
    df_train, df_test, df_val = preprocess_data('EURUSD-H1-with-indicators.csv')

    # 2. Сохраняем данные для среды
    with open('train_data.pkl', 'wb') as f:
        pickle.dump(df_train, f)

    # 3. Запускаем пайплайн и сохраняем обученную модель
    trained_policy = serial_pipeline(
        [ppo_forex_config, ppo_forex_create_config],
        seed=0,
    )
