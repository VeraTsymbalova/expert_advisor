import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(filepath, train_size_ratio=0.7, test_size_ratio=0.2, val_size_ratio=0.1):
    # Загрузка данных
    df = pd.read_csv(filepath, delimiter='\t')
    df.columns = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'TickVol', 'Spread']

    # Преобразование дат и времени
    df['Date'] = pd.to_datetime(df['Date'])
    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.time
    df['DateTime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))

    # Извлечение признаков времени
    df['Hour'] = df['Time'].apply(lambda x: x.hour)
    df['Minute'] = df['Time'].apply(lambda x: x.minute)

    # Циклическое кодирование
    df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
    df['Minute_sin'] = np.sin(2 * np.pi * df['Minute'] / 60)
    df['Minute_cos'] = np.cos(2 * np.pi * df['Minute'] / 60)

    # Удаляем лишнее
    df.drop(['Hour', 'Minute', 'Time', 'Minute_sin', 'Minute_cos', 'DateTime'], axis=1, inplace=True)

    # Заполнение пропусков
    df = df.ffill().dropna().reset_index(drop=True)

    # Сохраняем немасштабированные значения
    df['Unscaled_Open'] = df['Open']
    df['Unscaled_High'] = df['High']
    df['Unscaled_Low'] = df['Low']
    df['Unscaled_Close'] = df['Close']
    df['Unscaled_Spread'] = df['Spread']

    # Выбираем числовые колонки, которые будем масштабировать
    exclude_from_scaling = ['Open', 'High', 'Low', 'Close', 'Spread',
                            'Unscaled_Open', 'Unscaled_High', 'Unscaled_Low', 'Unscaled_Close', 'Unscaled_Spread']
    numeric_columns = [col for col in df.select_dtypes(include=[np.number]).columns if col not in exclude_from_scaling]

    # Масштабируем
    scaler = MinMaxScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    # Формируем итоговый порядок колонок
    final_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'TickVol', 'Spread',
                     'Hour_sin', 'Hour_cos', 'Unscaled_Open', 'Unscaled_High',
                     'Unscaled_Low', 'Unscaled_Close', 'Unscaled_Spread']
    df = df[final_columns]

    # Разделение на train/test/val
    total_len = len(df)
    train_end = int(total_len * train_size_ratio)
    test_end = train_end + int(total_len * test_size_ratio)

    df_train = df.iloc[:train_end].reset_index(drop=True)
    df_test = df.iloc[train_end:test_end].reset_index(drop=True)
    df_val = df.iloc[test_end:].reset_index(drop=True)

    return df_train, df_test, df_val