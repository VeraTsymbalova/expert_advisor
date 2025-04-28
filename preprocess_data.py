import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

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

    # Циклическое кодирование
    df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
    df['Minute_sin'] = np.sin(2 * np.pi * df['Minute'] / 60)
    df['Minute_cos'] = np.cos(2 * np.pi * df['Minute'] / 60)

    # Удаляем лишнее
    df.drop(['Hour', 'Minute', 'TIME', 'Minute_sin', 'Minute_cos', 'DateTime'], axis=1, errors='ignore', inplace=True)

    # Заполнение пропусков
    df = df.ffill().dropna().reset_index(drop=True)

    # Сохраняем немасштабированные значения
    df['Unscaled_Open'] = df['OPEN']
    df['Unscaled_High'] = df['HIGH']
    df['Unscaled_Low'] = df['LOW']
    df['Unscaled_Close'] = df['CLOSE']
    df['Unscaled_Spread'] = df['SPREAD']

    # Выбираем числовые колонки, которые будем масштабировать
    exclude_from_scaling = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'SPREAD',
                            'Unscaled_Open', 'Unscaled_High', 'Unscaled_Low', 'Unscaled_Close', 'Unscaled_Spread']
    numeric_columns = [col for col in df.select_dtypes(include=[np.number]).columns if col not in exclude_from_scaling]

    # Масштабируем
    scaler = MinMaxScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    # Формируем итоговый порядок колонок
    final_columns = ['DATE', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'TICKVOL', 'SPREAD',
                     'Hour_sin', 'Hour_cos', 'Unscaled_Open', 'Unscaled_High',
                     'Unscaled_Low', 'Unscaled_Close', 'Unscaled_Spread']
    df = df[final_columns]

    # Разделение на train/test/val через train_test_split
    df_temp, df_val = train_test_split(df, test_size=val_size_ratio, random_state=random_state, shuffle=True)
    test_ratio_adjusted = test_size_ratio / (train_size_ratio + test_size_ratio)
    df_train, df_test = train_test_split(df_temp, test_size=test_ratio_adjusted, random_state=random_state, shuffle=True)

    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)

    return df_train, df_test, df_val