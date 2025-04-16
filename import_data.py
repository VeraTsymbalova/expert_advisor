import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta

# Инициализация подключения
if not mt5.initialize():
    print("Ошибка инициализации MT5:", mt5.last_error())
    quit()

symbol = "EURUSD"
start_date = datetime(2023, 4, 1)
end_date = datetime.now()

# Получаем H1 свечи
rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_H1, start_date, end_date)
df_candles = pd.DataFrame(rates)
df_candles['time'] = pd.to_datetime(df_candles['time'], unit='s')
df_candles.rename(columns={
    'time': 'DATETIME',
    'open': 'OPEN',
    'high': 'HIGH',
    'low': 'LOW',
    'close': 'CLOSE',
    'tick_volume': 'TICKVOL'
}, inplace=True)

# Загружаем тики по дням и рассчитываем спред
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

# Объединяем все спреды
if all_spreads:
    df_spread = pd.concat(all_spreads, ignore_index=True)
else:
    df_spread = pd.DataFrame(columns=['DATETIME', 'SPREAD'])

# Объединяем свечи с расчетным спредом
df_final = pd.merge(df_candles, df_spread, on='DATETIME', how='left')

# Разбиваем DATETIME на DATE и TIME
df_final['DATE'] = df_final['DATETIME'].dt.strftime('%Y.%m.%d')
df_final['TIME'] = df_final['DATETIME'].dt.strftime('%H:%M:%S')

# Итоговая структура
df_final = df_final[['DATE', 'TIME', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'TICKVOL', 'SPREAD']]

# Сохраняем в CSV с табуляцией
df_final.to_csv("EURUSD-H1.csv", index=False, sep='\t')
print("Файл сохранён как EURUSD-H1.csv")

# Инфо по пропущенным значениям спреда
missing_count = df_final['SPREAD'].isna().sum()
print(f"Пропущено значений спреда: {missing_count} из {len(df_final)}")