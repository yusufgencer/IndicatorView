import pandas as pd

# ------------------------------
# RSI Al/Sat Sinyalleri
# ------------------------------
def generate_rsi_signals(df: pd.DataFrame, buy_threshold: float = 30, sell_threshold: float = 70) -> pd.DataFrame:
    df = df.copy()
    df['Signal'] = None
    position = False

    for i in range(1, len(df)):
        rsi = df['RSI'].iloc[i]
        if not position and rsi < buy_threshold:
            df.at[df.index[i], 'Signal'] = 'BUY'
            position = True
        elif position and rsi > sell_threshold:
            df.at[df.index[i], 'Signal'] = 'SELL'
            position = False

    return df


# ------------------------------
# MACD Al/Sat Sinyalleri
# ------------------------------
def generate_macd_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Signal'] = None
    position = False

    for i in range(1, len(df)):
        macd = df['MACD'].iloc[i]
        signal = df['MACD_signal'].iloc[i]
        prev_macd = df['MACD'].iloc[i - 1]
        prev_signal = df['MACD_signal'].iloc[i - 1]

        # MACD'nin sinyal çizgisini aşağıdan yukarıya kesmesi
        if not position and prev_macd < prev_signal and macd > signal:
            df.at[df.index[i], 'Signal'] = 'BUY'
            position = True
        # Yukarıdan aşağıya kesmesi
        elif position and prev_macd > prev_signal and macd < signal:
            df.at[df.index[i], 'Signal'] = 'SELL'
            position = False

    return df


# ------------------------------
# Bollinger Bands Al/Sat Sinyalleri
# ------------------------------
def generate_bollinger_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Signal'] = None
    position = False

    for i in range(1, len(df)):
        price = df['Close'].iloc[i]
        lower = df['BB_lower'].iloc[i]
        upper = df['BB_upper'].iloc[i]

        if not position and price < lower:
            df.at[df.index[i], 'Signal'] = 'BUY'
            position = True
        elif position and price > upper:
            df.at[df.index[i], 'Signal'] = 'SELL'
            position = False

    return df


# ------------------------------
# MFI Al/Sat Sinyalleri
# ------------------------------
def generate_mfi_signals(df: pd.DataFrame, buy_threshold: float = 20, sell_threshold: float = 80) -> pd.DataFrame:
    df = df.copy()
    df['Signal'] = None
    position = False

    for i in range(1, len(df)):
        mfi = df['MFI'].iloc[i]
        if not position and mfi < buy_threshold:
            df.at[df.index[i], 'Signal'] = 'BUY'
            position = True
        elif position and mfi > sell_threshold:
            df.at[df.index[i], 'Signal'] = 'SELL'
            position = False

    return df

def generate_obv_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Signal'] = None
    df['OBV'] = (df['Volume'] * (df['Close'].diff() > 0).apply(lambda x: 1 if x else -1)).cumsum()
    df['OBV_SMA'] = df['OBV'].rolling(window=10).mean()

    position = False
    for i in range(1, len(df)):
        obv = df['OBV'].iloc[i]
        obv_sma = df['OBV_SMA'].iloc[i]
        if not position and obv > obv_sma:
            df.at[df.index[i], 'Signal'] = 'BUY'
            position = True
        elif position and obv < obv_sma:
            df.at[df.index[i], 'Signal'] = 'SELL'
            position = False

    return df

def generate_atr_signals(df: pd.DataFrame, atr_multiplier: float = 1.5) -> pd.DataFrame:
    df = df.copy()
    df['Signal'] = None
    position = False  # Pozisyon yokken sadece AL yapılabilir

    for i in range(1, len(df)):
        if pd.isna(df['ATR'].iloc[i]):
            continue  # ATR değeri eksikse atla

        close = df['Close'].iloc[i]
        prev_close = df['Close'].iloc[i - 1]
        atr = df['ATR'].iloc[i]

        upper_band = prev_close + atr * atr_multiplier
        lower_band = prev_close - atr * atr_multiplier

        if not position and close > upper_band:
            df.at[df.index[i], 'Signal'] = 'BUY'
            position = True

        elif position and close < lower_band:
            df.at[df.index[i], 'Signal'] = 'SELL'
            position = False

    return df

def generate_stochrsi_signals(df: pd.DataFrame, lower: float = 0.2, upper: float = 0.8) -> pd.DataFrame:
    df = df.copy()
    df['Signal'] = None
    position = False

    for i in range(1, len(df)):
        stochrsi = df['StochRSI'].iloc[i]
        if not position and stochrsi < lower:
            df.at[df.index[i], 'Signal'] = 'BUY'
            position = True
        elif position and stochrsi > upper:
            df.at[df.index[i], 'Signal'] = 'SELL'
            position = False

    return df

def generate_cci_signals(df: pd.DataFrame, buy_threshold: float = -100, sell_threshold: float = 100) -> pd.DataFrame:
    df = df.copy()
    df['Signal'] = None
    position = False

    for i in range(1, len(df)):
        cci = df["CCI"].iloc[i]
        if not position and cci < buy_threshold:
            df.at[df.index[i], "Signal"] = "BUY"
            position = True
        elif position and cci > sell_threshold:
            df.at[df.index[i], "Signal"] = "SELL"
            position = False

    return df

def generate_supertrend_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Signal"] = None
    position = False

    for i in range(1, len(df)):
        if not position and df["SuperTrend"].iloc[i]:
            df.at[df.index[i], "Signal"] = "BUY"
            position = True
        elif position and not df["SuperTrend"].iloc[i]:
            df.at[df.index[i], "Signal"] = "SELL"
            position = False

    return df

def generate_williamsr_signals(df: pd.DataFrame, overbought: float = -20, oversold: float = -80) -> pd.DataFrame:
    df = df.copy()
    df["Signal"] = None
    position = False

    for i in range(1, len(df)):
        wr = df["WilliamsR"].iloc[i]
        if not position and wr < oversold:
            df.at[df.index[i], "Signal"] = "BUY"
            position = True
        elif position and wr > overbought:
            df.at[df.index[i], "Signal"] = "SELL"
            position = False

    return df

def generate_adx_signals(df: pd.DataFrame, threshold: float = 20) -> pd.DataFrame:
    df = df.copy()
    df["Signal"] = None
    position = False

    for i in range(1, len(df)):
        adx = df["ADX"].iloc[i]
        plus_di = df["+DI"].iloc[i]
        minus_di = df["-DI"].iloc[i]

        if not position and adx > threshold and plus_di > minus_di:
            df.at[df.index[i], "Signal"] = "BUY"
            position = True

        elif position and adx > threshold and minus_di > plus_di:
            df.at[df.index[i], "Signal"] = "SELL"
            position = False

    return df

def generate_trix_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Signal"] = None
    position = False

    for i in range(1, len(df)):
        prev_trix = df["TRIX"].iloc[i - 1]
        trix = df["TRIX"].iloc[i]

        if not position and prev_trix < 0 and trix > 0:
            df.at[df.index[i], "Signal"] = "BUY"
            position = True
        elif position and prev_trix > 0 and trix < 0:
            df.at[df.index[i], "Signal"] = "SELL"
            position = False

    return df

def generate_roc_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Signal"] = None
    position = False

    for i in range(1, len(df)):
        prev_roc = df["ROC"].iloc[i - 1]
        roc = df["ROC"].iloc[i]

        if not position and prev_roc < 0 and roc > 0:
            df.at[df.index[i], "Signal"] = "BUY"
            position = True
        elif position and prev_roc > 0 and roc < 0:
            df.at[df.index[i], "Signal"] = "SELL"
            position = False

    return df

def generate_vwap_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Signal'] = None
    position = False

    for i in range(1, len(df)):
        price = df['Close'].iloc[i]
        vwap = df['VWAP'].iloc[i]

        if not position and price > vwap:
            df.at[df.index[i], 'Signal'] = 'BUY'
            position = True
        elif position and price < vwap:
            df.at[df.index[i], 'Signal'] = 'SELL'
            position = False

    return df

def generate_keltner_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Signal"] = None
    position = False

    for i in range(1, len(df)):
        close = df["Close"].iloc[i]
        upper = df["KC_Upper"].iloc[i]
        lower = df["KC_Lower"].iloc[i]

        if not position and close < lower:
            df.at[df.index[i], "Signal"] = "BUY"
            position = True
        elif position and close > upper:
            df.at[df.index[i], "Signal"] = "SELL"
            position = False

    return df

def generate_donchian_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Signal"] = None
    position = False

    for i in range(1, len(df)):
        close = df["Close"].iloc[i]
        upper = df["DC_Upper"].iloc[i]
        lower = df["DC_Lower"].iloc[i]

        if not position and close > upper:
            df.at[df.index[i], "Signal"] = "BUY"
            position = True
        elif position and close < lower:
            df.at[df.index[i], "Signal"] = "SELL"
            position = False

    return df

def generate_parabolic_sar_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Signal"] = None
    position = False

    for i in range(1, len(df)):
        trend = df["PSAR_dir"].iloc[i]

        if not position and trend:
            df.at[df.index[i], "Signal"] = "BUY"
            position = True
        elif position and not trend:
            df.at[df.index[i], "Signal"] = "SELL"
            position = False

    return df


def generate_zscore_signals(df: pd.DataFrame, lower: float = -1.5, upper: float = 1.5) -> pd.DataFrame:
    df = df.copy()
    df["Signal"] = None
    position = False

    for i in range(1, len(df)):
        z = df["ZScore"].iloc[i]

        if not position and z < lower:
            df.at[df.index[i], "Signal"] = "BUY"
            position = True
        elif position and z > upper:
            df.at[df.index[i], "Signal"] = "SELL"
            position = False

    return df

def generate_ema_crossover_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Signal"] = None
    position = False

    for i in range(1, len(df)):
        prev_short = df["EMA_Short"].iloc[i - 1]
        prev_long = df["EMA_Long"].iloc[i - 1]
        curr_short = df["EMA_Short"].iloc[i]
        curr_long = df["EMA_Long"].iloc[i]

        if not position and prev_short < prev_long and curr_short > curr_long:
            df.at[df.index[i], "Signal"] = "BUY"
            position = True

        elif position and prev_short > prev_long and curr_short < curr_long:
            df.at[df.index[i], "Signal"] = "SELL"
            position = False

    return df
