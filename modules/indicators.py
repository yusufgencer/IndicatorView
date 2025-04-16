import pandas as pd
import numpy as np

# ------------------------------
# RSI (Relative Strength Index)
# ------------------------------
def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df


# ------------------------------
# MACD (Moving Average Convergence Divergence)
# ------------------------------
def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    df['EMA_fast'] = df['Close'].ewm(span=fast, adjust=False).mean()
    df['EMA_slow'] = df['Close'].ewm(span=slow, adjust=False).mean()

    df['MACD'] = df['EMA_fast'] - df['EMA_slow']
    df['MACD_signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    return df


# ------------------------------
# Bollinger Bands
# ------------------------------
def add_bollinger_bands(df: pd.DataFrame, window: int = 20, std_multiplier: float = 2.0) -> pd.DataFrame:
    rolling_mean = df['Close'].rolling(window).mean()
    rolling_std = df['Close'].rolling(window).std()

    df['BB_upper'] = rolling_mean + (rolling_std * std_multiplier)
    df['BB_lower'] = rolling_mean - (rolling_std * std_multiplier)
    df['BB_middle'] = rolling_mean
    return df


# ------------------------------
# MFI (Money Flow Index)
# ------------------------------
def add_mfi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    money_flow = typical_price * df['Volume']

    delta = typical_price.diff()
    positive_flow = money_flow.where(delta > 0, 0.0)
    negative_flow = money_flow.where(delta < 0, 0.0)

    pos_mf = positive_flow.rolling(window).sum()
    neg_mf = negative_flow.rolling(window).sum()

    mfi = 100 - (100 / (1 + (pos_mf / neg_mf)))
    df['MFI'] = mfi
    return df


# ------------------------------
# ATR (Average True Range)
# ------------------------------
def add_atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=window).mean()
    return df


def add_obv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['OBV'] = (df['Volume'] * (df['Close'].diff() > 0).apply(lambda x: 1 if x else -1)).cumsum()
    return df


def add_stochrsi(df: pd.DataFrame, rsi_window: int = 14, stoch_window: int = 14) -> pd.DataFrame:
    df = df.copy()
    
    # RSI hesapla
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=rsi_window).mean()
    avg_loss = loss.rolling(window=rsi_window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    df['RSI'] = rsi

    # Stochastic RSI hesapla
    min_rsi = rsi.rolling(window=stoch_window).min()
    max_rsi = rsi.rolling(window=stoch_window).max()
    df['StochRSI'] = (rsi - min_rsi) / (max_rsi - min_rsi)

    return df

def add_cci(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    df = df.copy()
    tp = (df["High"] + df["Low"] + df["Close"]) / 3  # Typical Price
    ma = tp.rolling(window=window).mean()
    md = tp.rolling(window=window).apply(lambda x: (abs(x - x.mean())).mean())
    df["CCI"] = (tp - ma) / (0.015 * md)
    return df

def add_supertrend(df: pd.DataFrame, atr_period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    df = df.copy()

    # ATR hesapla
    high_low = df["High"] - df["Low"]
    high_close = abs(df["High"] - df["Close"].shift())
    low_close = abs(df["Low"] - df["Close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(atr_period).mean()

    # HL2 + bandlar
    hl2 = (df["High"] + df["Low"]) / 2
    upperband = hl2 + (multiplier * atr)
    lowerband = hl2 - (multiplier * atr)

    supertrend = [True] * len(df)

    for i in range(1, len(df)):
        close = df["Close"].iloc[i]

        if close > upperband.iloc[i - 1]:
            supertrend[i] = True
        elif close < lowerband.iloc[i - 1]:
            supertrend[i] = False
        else:
            supertrend[i] = supertrend[i - 1]
            if supertrend[i] and lowerband.iloc[i] < lowerband.iloc[i - 1]:
                lowerband.iloc[i] = lowerband.iloc[i - 1]
            if not supertrend[i] and upperband.iloc[i] > upperband.iloc[i - 1]:
                upperband.iloc[i] = upperband.iloc[i - 1]

    df["SuperTrend"] = supertrend
    return df


def add_williams_r(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    df = df.copy()
    highest_high = df["High"].rolling(window=window).max()
    lowest_low = df["Low"].rolling(window=window).min()

    df["WilliamsR"] = -100 * ((highest_high - df["Close"]) / (highest_high - lowest_low))
    return df

def add_adx(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    df = df.copy()

    high = df['High']
    low = df['Low']
    close = df['Close']

    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(window).mean()
    plus_di = 100 * (plus_dm.rolling(window).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window).mean() / atr)

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(window).mean()

    df['ADX'] = adx
    df['+DI'] = plus_di
    df['-DI'] = minus_di
    return df

def add_trix(df: pd.DataFrame, trix_length: int = 15) -> pd.DataFrame:
    df = df.copy()
    ema1 = df["Close"].ewm(span=trix_length, adjust=False).mean()
    ema2 = ema1.ewm(span=trix_length, adjust=False).mean()
    ema3 = ema2.ewm(span=trix_length, adjust=False).mean()

    df["TRIX"] = ema3.pct_change() * 100  # Yüzde değişim
    return df

def add_roc(df: pd.DataFrame, period: int = 12) -> pd.DataFrame:
    df = df.copy()
    df["ROC"] = ((df["Close"] - df["Close"].shift(period)) / df["Close"].shift(period)) * 100
    return df

def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    cumulative_tp_vol = (typical_price * df['Volume']).cumsum()
    cumulative_vol = df['Volume'].cumsum()
    df['VWAP'] = cumulative_tp_vol / cumulative_vol
    return df

def add_keltner_channels(df: pd.DataFrame, ema_period: int = 20, atr_period: int = 10, atr_multiplier: float = 2.0) -> pd.DataFrame:
    df = df.copy()
    df["KC_Middle"] = df["Close"].ewm(span=ema_period, adjust=False).mean()

    high_low = df["High"] - df["Low"]
    high_close = abs(df["High"] - df["Close"].shift())
    low_close = abs(df["Low"] - df["Close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(atr_period).mean()

    df["KC_Upper"] = df["KC_Middle"] + (atr_multiplier * atr)
    df["KC_Lower"] = df["KC_Middle"] - (atr_multiplier * atr)

    return df

def add_donchian_channels(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    df = df.copy()
    df["DC_Upper"] = df["High"].rolling(window=period).max()
    df["DC_Lower"] = df["Low"].rolling(window=period).min()
    df["DC_Middle"] = (df["DC_Upper"] + df["DC_Lower"]) / 2
    return df

def add_parabolic_sar(df: pd.DataFrame, af: float = 0.02, af_max: float = 0.2) -> pd.DataFrame:
    df = df.copy()
    df["PSAR"] = df["Close"].copy()
    df["PSAR_dir"] = True  # True: uptrend, False: downtrend

    ep = df["Low"].iloc[0]
    sar = df["High"].iloc[0]
    trend = True  # Başlangıçta yukarı trend
    acc = af

    for i in range(2, len(df)):
        prev_sar = sar

        if trend:
            sar = sar + acc * (ep - sar)
            sar = min(sar, df["Low"].iloc[i - 1], df["Low"].iloc[i - 2])

            if df["Low"].iloc[i] < sar:
                trend = False
                sar = ep
                ep = df["Low"].iloc[i]
                acc = af
        else:
            sar = sar + acc * (ep - sar)
            sar = max(sar, df["High"].iloc[i - 1], df["High"].iloc[i - 2])

            if df["High"].iloc[i] > sar:
                trend = True
                sar = ep
                ep = df["High"].iloc[i]
                acc = af

        if trend:
            if df["High"].iloc[i] > ep:
                ep = df["High"].iloc[i]
                acc = min(acc + af, af_max)
        else:
            if df["Low"].iloc[i] < ep:
                ep = df["Low"].iloc[i]
                acc = min(acc + af, af_max)

        df.at[df.index[i], "PSAR"] = sar
        df.at[df.index[i], "PSAR_dir"] = trend

    return df

def add_zscore(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    df = df.copy()
    sma = df["Close"].rolling(window=window).mean()
    std = df["Close"].rolling(window=window).std()

    df["ZScore"] = (df["Close"] - sma) / std
    return df

def add_ema_crossover(df: pd.DataFrame, short_period: int = 12, long_period: int = 26) -> pd.DataFrame:
    df = df.copy()
    df["EMA_Short"] = df["Close"].ewm(span=short_period, adjust=False).mean()
    df["EMA_Long"] = df["Close"].ewm(span=long_period, adjust=False).mean()
    return df

def add_sma(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    window = 20  # sabit pencere uzunluğu
    df['SMA'] = df['Close'].rolling(window=window).mean()
    return df


def add_cmf(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    window = 20  # sabit pencere
    mf_multiplier = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
    mf_multiplier = mf_multiplier.replace([np.inf, -np.inf], 0).fillna(0)

    mf_volume = mf_multiplier * df['Volume']
    df['CMF'] = mf_volume.rolling(window=window).sum() / df['Volume'].rolling(window=window).sum()
    return df


def add_ichimoku_cloud(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    high_9 = df['High'].rolling(window=9).max()
    low_9 = df['Low'].rolling(window=9).min()
    df['Tenkan_sen'] = (high_9 + low_9) / 2

    high_26 = df['High'].rolling(window=26).max()
    low_26 = df['Low'].rolling(window=26).min()
    df['Kijun_sen'] = (high_26 + low_26) / 2

    df['Senkou_span_a'] = ((df['Tenkan_sen'] + df['Kijun_sen']) / 2).shift(26)

    high_52 = df['High'].rolling(window=52).max()
    low_52 = df['Low'].rolling(window=52).min()
    df['Senkou_span_b'] = ((high_52 + low_52) / 2).shift(26)

    df['Chikou_span'] = df['Close'].shift(-26)

    return df

def add_dpo(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    period = 20  # sabit pencere
    shift = int(period / 2) + 1

    df['SMA_dpo'] = df['Close'].rolling(window=period).mean()
    df['DPO'] = df['Close'].shift(shift) - df['SMA_dpo']
    return df

def add_vortex(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    period = 14

    tr = pd.Series(index=df.index)
    tr = df[['High', 'Low', 'Close']].copy()
    tr['HL'] = df['High'] - df['Low']
    tr['HC'] = abs(df['High'] - df['Close'].shift(1))
    tr['LC'] = abs(df['Low'] - df['Close'].shift(1))
    tr['TR'] = tr[['HL', 'HC', 'LC']].max(axis=1)

    vm_plus = abs(df['High'] - df['Low'].shift(1))
    vm_minus = abs(df['Low'] - df['High'].shift(1))

    tr_sum = tr['TR'].rolling(window=period).sum()
    vi_plus = vm_plus.rolling(window=period).sum() / tr_sum
    vi_minus = vm_minus.rolling(window=period).sum() / tr_sum

    df['VI_plus'] = vi_plus
    df['VI_minus'] = vi_minus
    return df

def add_ultimate_oscillator(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    high = df['High']
    low = df['Low']
    close = df['Close']

    bp = close - pd.concat([low, close.shift(1)], axis=1).min(axis=1)
    tr = pd.concat([high, close.shift(1)], axis=1).max(axis=1) - pd.concat([low, close.shift(1)], axis=1).min(axis=1)

    avg7 = bp.rolling(window=7).sum() / tr.rolling(window=7).sum()
    avg14 = bp.rolling(window=14).sum() / tr.rolling(window=14).sum()
    avg28 = bp.rolling(window=28).sum() / tr.rolling(window=28).sum()

    df['UO'] = 100 * (4 * avg7 + 2 * avg14 + avg28) / 7
    return df

def add_elder_ray(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ema_period = 13
    df['EMA'] = df['Close'].ewm(span=ema_period, adjust=False).mean()
    df['BullPower'] = df['High'] - df['EMA']
    df['BearPower'] = df['Low'] - df['EMA']
    return df

def add_price_oscillator(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    fast = 12
    slow = 26
    signal = 9

    df['EMA_fast'] = df['Close'].ewm(span=fast, adjust=False).mean()
    df['EMA_slow'] = df['Close'].ewm(span=slow, adjust=False).mean()
    df['PPO'] = ((df['EMA_fast'] - df['EMA_slow']) / df['EMA_slow']) * 100
    df['PPO_signal'] = df['PPO'].ewm(span=signal, adjust=False).mean()
    return df

def add_bop(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    bop = (df['Close'] - df['Open']) / (df['High'] - df['Low'])
    df['BOP'] = bop.replace([np.inf, -np.inf], 0).fillna(0)
    return df

def add_ac_oscillator(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    median_price = (df['High'] + df['Low']) / 2

    ao_short = median_price.rolling(window=5).mean()
    ao_long = median_price.rolling(window=34).mean()
    df['AO'] = ao_short - ao_long

    df['AC'] = df['AO'] - df['AO'].rolling(window=5).mean()
    return df

def add_chaikin_oscillator(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    mfm = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
    mfm = mfm.replace([np.inf, -np.inf], 0).fillna(0)
    mfv = mfm * df['Volume']

    df['ADL'] = mfv.cumsum()
    df['CO'] = df['ADL'].ewm(span=3, adjust=False).mean() - df['ADL'].ewm(span=10, adjust=False).mean()
    return df

def add_tema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    span = 20  # sabit periyot

    ema1 = df['Close'].ewm(span=span, adjust=False).mean()
    ema2 = ema1.ewm(span=span, adjust=False).mean()
    ema3 = ema2.ewm(span=span, adjust=False).mean()

    df['TEMA'] = 3 * ema1 - 3 * ema2 + ema3
    return df

def add_fractal_indicator(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    highs = df['High']
    lows = df['Low']

    df['Fractal_High'] = (
        (highs.shift(2) < highs.shift(0)) &
        (highs.shift(1) < highs.shift(0)) &
        (highs.shift(-1) < highs.shift(0)) &
        (highs.shift(-2) < highs.shift(0))
    )

    df['Fractal_Low'] = (
        (lows.shift(2) > lows.shift(0)) &
        (lows.shift(1) > lows.shift(0)) &
        (lows.shift(-1) > lows.shift(0)) &
        (lows.shift(-2) > lows.shift(0))
    )

    return df

def add_rvi(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    numerator = df['Close'] - df['Open']
    denominator = df['High'] - df['Low']
    rvi_raw = numerator / denominator.replace(0, np.nan)
    rvi_raw = rvi_raw.replace([np.inf, -np.inf], 0).fillna(0)

    df['RVI'] = rvi_raw.rolling(window=4).mean()
    df['RVI_signal'] = df['RVI'].rolling(window=4).mean()
    return df
