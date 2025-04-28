import ta
import pandas as pd

def add_trend_indicators(df):
    # Проверка, что есть нужные для расчетов столбцы
    required_cols = {'OPEN', 'HIGH', 'LOW', 'CLOSE', 'TICKVOL'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"DataFrame должен содержать колонки: {required_cols}")

    # EMA (тренд)
    df['EMA_12'] = ta.trend.EMAIndicator(close=df['CLOSE'], window=12).ema_indicator()
    df['EMA_26'] = ta.trend.EMAIndicator(close=df['CLOSE'], window=26).ema_indicator()

    # MACD
    macd = ta.trend.MACD(close=df['CLOSE'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Diff'] = macd.macd_diff()

    # RSI
    df['RSI'] = ta.momentum.RSIIndicator(close=df['CLOSE'], window=14).rsi()

    # ATR (волатильность)
    df['ATR'] = ta.volatility.AverageTrueRange(high=df['HIGH'], low=df['LOW'], close=df['CLOSE'], window=14).average_true_range()

    # OBV (объем)
    df['OBV'] = ta.volume.OnBalanceVolumeIndicator(close=df['CLOSE'], volume=df['TICKVOL']).on_balance_volume()

    # VWAP
    df['VWAP'] = ta.volume.VolumeWeightedAveragePrice(high=df['HIGH'], low=df['LOW'], close=df['CLOSE'], volume=df['TICKVOL']).volume_weighted_average_price()

    # Ichimoku (одна линия)
    ichimoku = ta.trend.IchimokuIndicator(high=df['HIGH'], low=df['LOW'], window1=9, window2=26, window3=52)
    df['Ichimoku_A'] = ichimoku.ichimoku_a()

    # Заполнение пропусков
    df = df.ffill().bfill()

    return df