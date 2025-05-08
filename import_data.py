import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Инициализация MT5

def initialize_mt5():
    if not mt5.initialize():
        logging.error(f"Ошибка инициализации MT5: {mt5.last_error()}")
        raise SystemExit(1)
    logging.info("MetaTrader5 успешно инициализирован")

# Завершение сессии MT5

def shutdown_mt5():
    mt5.shutdown()
    logging.info("MetaTrader5 отключен")

# Получение часовых свечей (H1)

def get_candles(symbol, timeframe, start_date, end_date):
    rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
    if rates is None or len(rates) == 0:
        logging.error("Не удалось загрузить свечные данные или они пусты.")
        return pd.DataFrame()  # Возвращаем пустой DataFrame

    df = pd.DataFrame(rates)
    if 'time' not in df.columns:
        logging.error("Нет столбца 'time' в полученных данных.")
        return pd.DataFrame()

    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.rename(columns={
        'time': 'DATETIME',
        'open': 'OPEN',
        'high': 'HIGH',
        'low': 'LOW',
        'close': 'CLOSE',
        'tick_volume': 'TICKVOL'
    }, inplace=True)
    return df

# Расчёт почасового спреда на основе тиковых данных

def calculate_spreads(symbol, start_date, end_date):
    all_spreads = []
    curr_date = start_date
    while curr_date < end_date:
        next_date = curr_date + timedelta(days=1)
        ticks = mt5.copy_ticks_range(symbol, curr_date, next_date, mt5.COPY_TICKS_ALL)
        if ticks is not None and len(ticks) > 0:
            df_ticks = pd.DataFrame(ticks)
            df_ticks['time'] = pd.to_datetime(df_ticks['time'], unit='s')
            df_ticks['spread'] = (df_ticks['ask'] - df_ticks['bid']) * 10000
            hourly_spread = df_ticks.resample('1h', on='time')['spread'].mean().reset_index()
            hourly_spread.rename(columns={'time': 'DATETIME', 'spread': 'SPREAD'}, inplace=True)
            all_spreads.append(hourly_spread)
        curr_date = next_date

    return pd.concat(all_spreads, ignore_index=True) if all_spreads else pd.DataFrame(columns=['DATETIME', 'SPREAD'])

# Объединение свечей и спредов, заполнение пропусков

def merge_and_clean_data(df_candles, df_spread):
    df = pd.merge(df_candles, df_spread, on='DATETIME', how='left')
    df['DATE'] = df['DATETIME'].dt.strftime('%Y.%m.%d')
    df['TIME'] = df['DATETIME'].dt.strftime('%H:%M:%S')

    missing_info = df[['OPEN', 'HIGH', 'LOW', 'CLOSE', 'TICKVOL', 'SPREAD']].isna().sum()
    total_missing = missing_info.sum()
    if total_missing > 0:
        logging.warning("Обнаружены пропущенные значения:")
        for col, cnt in missing_info.items():
            if cnt > 0:
                logging.warning(f"- {col}: {cnt} пропущенных из {len(df)}")
        df[['OPEN', 'HIGH', 'LOW', 'CLOSE', 'TICKVOL', 'SPREAD']] = df[['OPEN', 'HIGH', 'LOW', 'CLOSE', 'TICKVOL', 'SPREAD']].fillna(
            df[['OPEN', 'HIGH', 'LOW', 'CLOSE', 'TICKVOL', 'SPREAD']].mean())
        logging.info("Пропущенные значения были заполнены средними значениями.")

    return df[['DATETIME', 'DATE', 'TIME', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'TICKVOL', 'SPREAD']]
