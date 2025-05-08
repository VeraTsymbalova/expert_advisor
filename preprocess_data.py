import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def preprocess_data(filepath, train_size_ratio=0.7, test_size_ratio=0.2, val_size_ratio=0.1, random_state=42):
    # Загрузка данных
    df = pd.read_csv(filepath, delimiter='\t')

    # Преобразование дат и времени
    df['DATE'] = pd.to_datetime(df['DATE'])
    df['TIME'] = pd.to_datetime(df['TIME'], format='%H:%M:%S').dt.time
    df['DateTime'] = pd.to_datetime(df['DATE'].astype(str) + ' ' + df['TIME'].astype(str))

    # Извлечение признаков времени
    df['Hour'] = df['TIME'].apply(lambda x: x.hour)
    df['Minute'] = df['TIME'].apply(lambda x: x.minute)

    # Циклическое кодирование времени
    df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
    df['Minute_sin'] = np.sin(2 * np.pi * df['Minute'] / 60)
    df['Minute_cos'] = np.cos(2 * np.pi * df['Minute'] / 60)

    # Удаляем лишние колонки
    df.drop(['Hour', 'Minute', 'TIME', 'Minute_sin', 'Minute_cos', 'DateTime'], axis=1, errors='ignore', inplace=True)

    # Заполнение пропусков
    df = df.ffill().dropna().reset_index(drop=True)

    # Сохраняем немасштабированные значения цен и индикаторов
    df['Unscaled_Open'] = df['OPEN']
    df['Unscaled_High'] = df['HIGH']
    df['Unscaled_Low'] = df['LOW']
    df['Unscaled_Close'] = df['CLOSE']
    df['Unscaled_Spread'] = df['SPREAD']

    technical_indicators = [
        'EMA_12', 'EMA_26', 'MACD', 'MACD_Signal', 'MACD_Diff',
        'RSI', 'ATR', 'OBV', 'VWAP', 'Ichimoku_A'
    ]

    for col in technical_indicators:
        if col in df.columns:
            df[f'Unscaled_{col}'] = df[col]

    # Масштабирование всех числовых признаков, включая OHLC и SPREAD
    exclude_from_scaling = ['DATE'] + [col for col in df.columns if col.startswith('Unscaled_')]

    numeric_columns = [col for col in df.select_dtypes(include=[np.number]).columns if col not in exclude_from_scaling]

    scaler = MinMaxScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    # Финальный порядок колонок
    preserved = ['DATE', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'TICKVOL', 'SPREAD', 'Hour_sin', 'Hour_cos'] + technical_indicators
    unscaled = [col for col in df.columns if col.startswith('Unscaled_')]
    rest = [col for col in df.columns if col not in preserved + unscaled]
    df = df[preserved + rest + unscaled]

    # Разделение на выборки
    df_temp, df_val = train_test_split(df, test_size=val_size_ratio, random_state=random_state, shuffle=True)
    test_ratio_adjusted = test_size_ratio / (train_size_ratio + test_size_ratio)
    df_train, df_test = train_test_split(df_temp, test_size=test_ratio_adjusted, random_state=random_state, shuffle=True)

    # Логирование размеров
    logging.info(f"Размер выборок: train={len(df_train)}, test={len(df_test)}, val={len(df_val)}")

    # Сохранение выборок в файлы
    df_train.to_csv("train.csv", index=False)
    df_test.to_csv("test.csv", index=False)
    df_val.to_csv("val.csv", index=False)

    return df_train.reset_index(drop=True), df_test.reset_index(drop=True), df_val.reset_index(drop=True)
