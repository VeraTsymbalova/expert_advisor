import pandas as pd

# Путь к исходному файлу
input_file = 'EURUSD60.csv'

# Путь для сохранения результата
output_file = 'EURUSD-H1-2020-2025.csv'

# Чтение CSV-файла
df = pd.read_csv(
    input_file,
    header=None,
    names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'],
    delimiter=','
)

# Объединяем дату и время в один datetime-объект
df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M')

# Оставляем только диапазон с 2020-01-01 по 2025-01-01
df = df[(df['datetime'] >= '2020-01-01') & (df['datetime'] < '2025-01-01')]

# Формируем таблицу
result = pd.DataFrame({
    'DATE': df['datetime'].dt.strftime('%Y.%m.%d'),
    'TIME': df['datetime'].dt.strftime('%H:%M:%S'),
    'OPEN': df['Open'],
    'HIGH': df['High'],
    'LOW': df['Low'],
    'CLOSE': df['Close'],
    'TICKVOL': df['Volume']
})

# Сохраняем
result.to_csv(output_file, sep='\t', index=False)

print(f'Файл успешно сохранен: {output_file}')