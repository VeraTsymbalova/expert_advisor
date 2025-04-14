import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(filepath, train_size_ratio=0.7, test_size_ratio=0.2, val_size_ratio=0.1):
    # Загрузка всех данных
    df = pd.read_csv(filepath, delimiter='\t')
    df.columns = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'TickVol']

    # Преобразование даты
    df['Date'] = pd.to_datetime(df['Date'])
    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.time

    # Объединение даты и времени
    df['DateTime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))

    # Извлечение часа и минуты
    df['Hour'] = df['Time'].apply(lambda x: x.hour)
    df['Minute'] = df['Time'].apply(lambda x: x.minute)

    # Циклическое кодирование времени
    df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
    df['Minute_sin'] = np.sin(2 * np.pi * df['Minute'] / 60)
    df['Minute_cos'] = np.cos(2 * np.pi * df['Minute'] / 60)

    # Удаление ненужных колонок
    df.drop(['Hour', 'Minute', 'Time', 'Minute_sin', 'Minute_cos', 'DateTime'], axis=1, inplace=True)

    # Заполнение пропущенных значений
    df = df.ffill()
    df = df.dropna().reset_index(drop=True)

    # Сохраняем немасштабированные цены
    df['Unscaled_Open'] = df['Open']
    df['Unscaled_High'] = df['High']
    df['Unscaled_Low'] = df['Low']
    df['Unscaled_Close'] = df['Close']

    # Определяем, какие числовые колонки масштабировать
    price_columns = ['Open', 'High', 'Low', 'Close', 'Unscaled_Open', 'Unscaled_High', 'Unscaled_Low', 'Unscaled_Close']
    numeric_columns = [col for col in df.select_dtypes(include=[np.number]).columns if col not in price_columns]

    # Масштабирование
    scaler = MinMaxScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    # Разделение на train/test/val
    total_len = len(df)
    train_end = int(total_len * train_size_ratio)
    test_end = train_end + int(total_len * test_size_ratio)

    df_train = df.iloc[:train_end].reset_index(drop=True)
    df_test = df.iloc[train_end:test_end].reset_index(drop=True)
    df_val = df.iloc[test_end:].reset_index(drop=True)

    return df_train, df_test, df_val

if __name__ == '__main__':
    # Имя файла
    filepath = 'EURUSD-H1-2020-2025.csv'

    # Запуск обработки
    df_train, df_test, df_val = preprocess_data(filepath)

    # Вывод размеров датасетов
    print(f"Размер Train: {df_train.shape[0]} строк, {df_train.shape[1]} столбцов")
    print(f"Размер Test: {df_test.shape[0]} строк, {df_test.shape[1]} столбцов")
    print(f"Размер Validation: {df_val.shape[0]} строк, {df_val.shape[1]} столбцов")
    print()

    # Вывод первых строк
    print("Train (первые 5 строк):")
    print(df_train.head())
    print()

    print("Test (первые 5 строк):")
    print(df_test.head())
    print()

    print("Validation (первые 5 строк):")
    print(df_val.head())