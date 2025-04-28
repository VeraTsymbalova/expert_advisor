import os
import pickle
import pandas as pd

from import_data import df_final as raw_data
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

def prepare_data():
    # 1. Сохраняем сырые данные
    raw_data.to_csv('EURUSD-H1.csv', index=False, sep='\t')

    # 2. Добавляем индикаторы
    df = pd.read_csv('EURUSD-H1.csv', delimiter='\t')
    df = add_trend_indicators(df)
    df.to_csv('EURUSD-H1-with-indicators.csv', index=False, sep='\t')

    # 3. Препроцессинг и разделение данных
    df_train, df_test, df_val = preprocess_data('EURUSD-H1-with-indicators.csv')

    # 4. Сохраняем тренировочные данные для среды
    with open('train_data.pkl', 'wb') as f:
        pickle.dump(df_train, f)

    print("Данные подготовлены")

def train_model():
    # Обучение модели
    trained_policy = serial_pipeline(
        [ppo_forex_config, ppo_forex_create_config],
        seed=0,
    )
    print("Обучение завершено")

    return trained_policy

def export_model(trained_policy):
    # Экспорт модели в ONNX
    vac_model = trained_policy._model
    exportable_model = ExportableActor(vac_model)
    exportable_model.eval()

    obs_shape = ppo_forex_config.policy.model.obs_shape
    dummy_input = torch.randn(1, obs_shape)

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

if __name__ == '__main__':
    prepare_data()
    trained_policy = train_model()
    export_model(trained_policy)