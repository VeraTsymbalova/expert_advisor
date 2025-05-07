import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta

# Инициализация подключения к Metatrader5
if not mt5.initialize():
    print("Ошибка инициализации MT5:", mt5.last_error())
    quit()

symbol = "EURUSD"
start_date = datetime(2023, 4, 1)
end_date = datetime(2025, 4, 30)

# Получение свечей с timeframe H1
rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_H1, start_date, end_date)
df_candles = pd.DataFrame(rates)
df_candles['time'] = pd.to_datetime(df_candles['time'], unit='s')
df_candles.rename(columns={
    'time': 'DATETIME)',
    'open': 'OPEN',
    'high': 'HIGH',
    'low': 'LOW',
    'close': 'CLOSE',
    'tick_volume': 'TICKVOL'
}, inplace=True)

# Загрузка тиков по дням и расчет спреда
all_spreads = []
curr_date = start_date
while curr_date < end_date:
    next_date = curr_date + timedelta(days=1)
    ticks = mt5.copy_ticks_range(symbol, curr_date, next_date, mt5.COPY_TICKS_ALL)
    if ticks is not None and len(ticks) > 0:
        df_ticks = pd.DataFrame(ticks)
        df_ticks['time'] = pd.to_datetime(df_ticks['time'], unit='s')
        df_ticks['spread'] = (df_ticks['ask'] - df_ticks['bid']) * 10000  # в пунктах
        hourly_spread = df_ticks.resample('1h', on='time')['spread'].mean().reset_index()
        hourly_spread.rename(columns={'time': 'DATETIME', 'spread': 'SPREAD'}, inplace=True)
        all_spreads.append(hourly_spread)
    curr_date = next_date

# Объединение спредов в dataframe
if all_spreads:
    df_spread = pd.concat(all_spreads, ignore_index=True)
else:
    df_spread = pd.DataFrame(columns=['DATETIME', 'SPREAD'])

# Объединение свечей с расчетным спредом
df_final = pd.merge(df_candles, df_spread, on='DATETIME', how='left')

# Разбивка DATETIME на DATE и TIME
df_final['DATE'] = df_final['DATETIME'].dt.strftime('%Y.%m.%d')
df_final['TIME'] = df_final['DATETIME'].dt.strftime('%H:%M:%S')

# Итоговая структура
df_final = df_final[['DATE', 'TIME', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'TICKVOL', 'SPREAD']]

# Сохранение в CSV-файл
df_final.to_csv("EURUSD-H1.csv", index=False, sep='\t')
print("Файл сохранён как EURUSD-H1.csv")

# Проверка на пропущенные значения
missing_info = df_final[['OPEN', 'HIGH', 'LOW', 'CLOSE', 'TICKVOL', 'SPREAD']].isna().sum()
total_missing = missing_info.sum()

if total_missing > 0:
    print("Обнаружены пропущенные значения:")
    for column, missing_count in missing_info.items():
        if missing_count > 0:
            print(f"- {column}: {missing_count} пропущенных из {len(df_final)}")

# Отключение от терминала Metatrader5
mt5.shutdown()
