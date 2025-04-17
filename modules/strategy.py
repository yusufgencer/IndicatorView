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

def generate_sma_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Signal'] = None
    position = False

    for i in range(1, len(df)):
        close = df['Close'].iloc[i]
        sma = df['SMA'].iloc[i]
        prev_close = df['Close'].iloc[i - 1]
        prev_sma = df['SMA'].iloc[i - 1]

        # Aşağıdan yukarı keserse BUY
        if not position and prev_close < prev_sma and close > sma:
            df.at[df.index[i], 'Signal'] = 'BUY'
            position = True
        # Yukarıdan aşağı keserse SELL
        elif position and prev_close > prev_sma and close < sma:
            df.at[df.index[i], 'Signal'] = 'SELL'
            position = False

    return df


def generate_cmf_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Signal'] = None
    position = False

    buy_threshold = 0.05
    sell_threshold = -0.05

    for i in range(1, len(df)):
        cmf = df['CMF'].iloc[i]

        if not position and cmf > buy_threshold:
            df.at[df.index[i], 'Signal'] = 'BUY'
            position = True
        elif position and cmf < sell_threshold:
            df.at[df.index[i], 'Signal'] = 'SELL'
            position = False

    return df


def generate_ichimoku_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Signal'] = None
    position = False

    for i in range(52, len(df)):  # İlk 52 gün boşluk oluşabilir
        price = df['Close'].iloc[i]
        kijun = df['Kijun_sen'].iloc[i]
        tenkan = df['Tenkan_sen'].iloc[i]
        span_a = df['Senkou_span_a'].iloc[i]
        span_b = df['Senkou_span_b'].iloc[i]

        # Fiyat bulutun üstünde ve Tenkan > Kijun ise al sinyali
        if not position and price > max(span_a, span_b) and tenkan > kijun:
            df.at[df.index[i], 'Signal'] = 'BUY'
            position = True

        # Fiyat bulutun altında ve Tenkan < Kijun ise sat sinyali
        elif position and price < min(span_a, span_b) and tenkan < kijun:
            df.at[df.index[i], 'Signal'] = 'SELL'
            position = False

    return df

def generate_dpo_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Signal'] = None
    position = False

    for i in range(1, len(df)):
        dpo = df['DPO'].iloc[i]
        prev_dpo = df['DPO'].iloc[i - 1]

        # DPO sıfır çizgisini aşağıdan yukarı keserse BUY
        if not position and prev_dpo < 0 and dpo > 0:
            df.at[df.index[i], 'Signal'] = 'BUY'
            position = True

        # Yukarıdan aşağı keserse SELL
        elif position and prev_dpo > 0 and dpo < 0:
            df.at[df.index[i], 'Signal'] = 'SELL'
            position = False

    return df

def generate_vortex_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Signal'] = None
    position = False

    for i in range(1, len(df)):
        vi_plus = df['VI_plus'].iloc[i]
        vi_minus = df['VI_minus'].iloc[i]
        prev_vi_plus = df['VI_plus'].iloc[i - 1]
        prev_vi_minus = df['VI_minus'].iloc[i - 1]

        # BUY sinyali: VI+ aşağıdan yukarı VI-’yı keserse
        if not position and prev_vi_plus < prev_vi_minus and vi_plus > vi_minus:
            df.at[df.index[i], 'Signal'] = 'BUY'
            position = True

        # SELL sinyali: VI+ yukarıdan aşağı VI-’nın altına inerse
        elif position and prev_vi_plus > prev_vi_minus and vi_plus < vi_minus:
            df.at[df.index[i], 'Signal'] = 'SELL'
            position = False

    return df

def generate_ultimate_oscillator_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Signal'] = None
    position = False

    for i in range(1, len(df)):
        uo = df['UO'].iloc[i]

        if not position and uo < 30:
            df.at[df.index[i], 'Signal'] = 'BUY'
            position = True
        elif position and uo > 70:
            df.at[df.index[i], 'Signal'] = 'SELL'
            position = False

    return df

def generate_elder_ray_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Signal'] = None
    position = False

    for i in range(2, len(df)):
        bull_now = df['BullPower'].iloc[i]
        bull_prev = df['BullPower'].iloc[i - 1]
        bear_now = df['BearPower'].iloc[i]
        bear_prev = df['BearPower'].iloc[i - 1]

        # BUY: BullPower pozitif ve artıyorsa
        if not position and bull_now > 0 and bull_now > bull_prev:
            df.at[df.index[i], 'Signal'] = 'BUY'
            position = True

        # SELL: BearPower negatif ve düşüyorsa
        elif position and bear_now < 0 and bear_now < bear_prev:
            df.at[df.index[i], 'Signal'] = 'SELL'
            position = False

    return df

def generate_price_oscillator_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Signal'] = None
    position = False

    for i in range(1, len(df)):
        ppo = df['PPO'].iloc[i]
        signal = df['PPO_signal'].iloc[i]
        prev_ppo = df['PPO'].iloc[i - 1]
        prev_signal = df['PPO_signal'].iloc[i - 1]

        if not position and prev_ppo < prev_signal and ppo > signal:
            df.at[df.index[i], 'Signal'] = 'BUY'
            position = True
        elif position and prev_ppo > prev_signal and ppo < signal:
            df.at[df.index[i], 'Signal'] = 'SELL'
            position = False

    return df

def generate_bop_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Signal'] = None
    position = False

    for i in range(1, len(df)):
        bop = df['BOP'].iloc[i]

        if not position and bop > 0:
            df.at[df.index[i], 'Signal'] = 'BUY'
            position = True
        elif position and bop < 0:
            df.at[df.index[i], 'Signal'] = 'SELL'
            position = False

    return df

def generate_ac_oscillator_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Signal'] = None
    position = False

    for i in range(2, len(df)):
        ac_now = df['AC'].iloc[i]
        ac_prev = df['AC'].iloc[i - 1]

        # AC pozitif ve artıyorsa BUY
        if not position and ac_now > 0 and ac_now > ac_prev:
            df.at[df.index[i], 'Signal'] = 'BUY'
            position = True

        # AC negatif ve azalıyorsa SELL
        elif position and ac_now < 0 and ac_now < ac_prev:
            df.at[df.index[i], 'Signal'] = 'SELL'
            position = False

    return df

def generate_chaikin_oscillator_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Signal'] = None
    position = False

    for i in range(1, len(df)):
        co = df['CO'].iloc[i]
        prev_co = df['CO'].iloc[i - 1]

        # BUY: CO sıfır çizgisini aşağıdan yukarı keserse
        if not position and prev_co < 0 and co > 0:
            df.at[df.index[i], 'Signal'] = 'BUY'
            position = True

        # SELL: CO sıfır çizgisini yukarıdan aşağı keserse
        elif position and prev_co > 0 and co < 0:
            df.at[df.index[i], 'Signal'] = 'SELL'
            position = False

    return df

def generate_tema_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Signal'] = None
    position = False

    for i in range(1, len(df)):
        close = df['Close'].iloc[i]
        tema = df['TEMA'].iloc[i]
        prev_close = df['Close'].iloc[i - 1]
        prev_tema = df['TEMA'].iloc[i - 1]

        # BUY: fiyat TEMA'yı aşağıdan yukarı keserse
        if not position and prev_close < prev_tema and close > tema:
            df.at[df.index[i], 'Signal'] = 'BUY'
            position = True

        # SELL: fiyat TEMA'yı yukarıdan aşağı keserse
        elif position and prev_close > prev_tema and close < tema:
            df.at[df.index[i], 'Signal'] = 'SELL'
            position = False

    return df

def generate_fractal_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Signal'] = None
    position = False

    for i in range(2, len(df) - 2):
        if not position and df['Fractal_Low'].iloc[i]:
            df.at[df.index[i], 'Signal'] = 'BUY'
            position = True
        elif position and df['Fractal_High'].iloc[i]:
            df.at[df.index[i], 'Signal'] = 'SELL'
            position = False

    return df

def generate_rvi_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Signal'] = None
    position = False

    for i in range(1, len(df)):
        rvi = df['RVI'].iloc[i]
        signal = df['RVI_signal'].iloc[i]
        prev_rvi = df['RVI'].iloc[i - 1]
        prev_signal = df['RVI_signal'].iloc[i - 1]

        if not position and prev_rvi < prev_signal and rvi > signal:
            df.at[df.index[i], 'Signal'] = 'BUY'
            position = True
        elif position and prev_rvi > prev_signal and rvi < signal:
            df.at[df.index[i], 'Signal'] = 'SELL'
            position = False

    return df

