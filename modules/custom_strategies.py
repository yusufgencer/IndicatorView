from modules.indicators import add_rsi, add_macd, add_bollinger_bands, add_atr
import pandas as pd

# 1. Triple Confirmation: RSI + MACD + Bollinger Bands

def strategy_triple_confirmation(df: pd.DataFrame):
    # RSI
    window_rsi = 14
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window_rsi).mean()
    avg_loss = loss.rolling(window=window_rsi).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    ema_fast = df['Close'].ewm(span=12, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_fast - ema_slow
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    ma = df['Close'].rolling(window=20).mean()
    std = df['Close'].rolling(window=20).std()
    df['BB_upper'] = ma + 2 * std
    df['BB_lower'] = ma - 2 * std

    df['Signal'] = None
    in_position = False

    for i in range(len(df)):
        if i < 26:  # MACD ve RSI için yeterli veri yoksa atla
            continue

        row = df.iloc[i]
        if pd.isna(row[['RSI', 'MACD', 'MACD_signal', 'BB_lower', 'BB_upper']]).any():
            continue

        # Alım sinyali
        if (not in_position and
            row['RSI'] < 35 and
            row['MACD'] > row['MACD_signal'] and
            row['Close'] <= row['BB_lower']):
            df.at[i, 'Signal'] = 'BUY'
            in_position = True

        # Satım sinyali
        elif (in_position and
              row['RSI'] > 65 and
              row['MACD'] < row['MACD_signal'] and
              row['Close'] >= row['BB_upper']):
            df.at[i, 'Signal'] = 'SELL'
            in_position = False

    return df

# 2. Volatility Breakout + Volume Confirmation (manual calculation)

def strategy_volatility_breakout_with_volume(df: pd.DataFrame):
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['STD20'] = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['MA20'] + 2 * df['STD20']
    df['BB_lower'] = df['MA20'] - 2 * df['STD20']
    df['BB_width'] = df['BB_upper'] - df['BB_lower']
    df['BB_width_avg'] = df['BB_width'].rolling(window=20).mean()
    df['VolumeAvg'] = df['Volume'].rolling(window=20).mean()

    df['Signal'] = None
    in_position = False

    for i in range(len(df)):
        if i < 20:
            continue  # İlk 20 gün sinyal üretme

        row = df.iloc[i]
        prev_row = df.iloc[i - 1]

        # Tüm değerler geçerli mi kontrol et
        if pd.isna(row['Close']) or pd.isna(row['BB_upper']) or pd.isna(row['BB_lower']) or pd.isna(row['BB_width']) or pd.isna(row['BB_width_avg']) or pd.isna(row['Volume']) or pd.isna(row['VolumeAvg']):
            continue

        # Alım sinyali
        if (not in_position and
            row['BB_width'] < row['BB_width_avg'] and
            row['Close'] > row['BB_upper'] and
            row['Volume'] > 1.5 * row['VolumeAvg']):
            df.at[i, 'Signal'] = 'BUY'
            in_position = True

        # Satım sinyali
        elif (in_position and
              row['BB_width'] < row['BB_width_avg'] and
              row['Close'] < row['BB_lower'] and
              row['Volume'] > 1.5 * row['VolumeAvg']):
            df.at[i, 'Signal'] = 'SELL'
            in_position = False

    return df

def strategy_rsi_bollinger_with_position(df: pd.DataFrame) -> pd.DataFrame:
    # RSI hesapla
    window_rsi = 14
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window_rsi).mean()
    avg_loss = loss.rolling(window=window_rsi).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Bollinger Bands hesapla
    window_bb = 20
    ma = df['Close'].rolling(window_bb).mean()
    std = df['Close'].rolling(window_bb).std()
    df['BB_upper'] = ma + 2 * std
    df['BB_lower'] = ma - 2 * std

    # Sinyal sütunu ve pozisyon durumu
    df['Signal'] = None
    in_position = False

    # Sinyal üretimi
    for i in range(len(df)):
        if i < max(window_rsi, window_bb):
            continue

        row = df.iloc[i]
        if pd.isna(row[['RSI', 'BB_lower', 'BB_upper']]).any():
            continue

        # Alım koşulu
        if (not in_position and
            row['RSI'] < 30 and
            row['Close'] < row['BB_lower']):
            df.at[i, 'Signal'] = 'BUY'
            in_position = True

        # Satım koşulu
        elif (in_position and
              row['RSI'] > 70 and
              row['Close'] > row['BB_upper']):
            df.at[i, 'Signal'] = 'SELL'
            in_position = False

    return df

import pandas as pd
import numpy as np

def strategy_triexdev_superbuyselltrend(df: pd.DataFrame,
                                                 atr_period: int = 10,
                                                 multiplier1: float = 0.8,
                                                 multiplier2: float = 1.6,
                                                 changeATR: bool = True) -> pd.DataFrame:
    """
    Geliştirilmiş dinamik al-sat sinyalleri veren TriexDev - SuperBuySellTrend stratejisinin Python uyarlaması.
    Bu sürümde, sinyallerin daha sık tetiklenmesi için trend geçişi kontrolü,
    klasik Pine kodundaki koşullara benzer şekilde ama daha hassas olacak şekilde yeniden düzenlenmiştir.
    
    Parametreler:
      - df: 'High', 'Low', 'Close' sütunlarını içeren DataFrame.
      - atr_period: ATR hesaplamasında kullanılacak periyot (default: 10).
      - multiplier1: Birincil ATR çarpanı (default: 0.8).
      - multiplier2: Onay (confirmation) ATR çarpanı (default: 1.6).
      - changeATR: Eğer True ise Wilder tipi ATR (üstel hareketli ortalama) kullanılır, False ise TR’nin SMA’sı alınır.
    """
    df = df.copy()
    # hl2 hesapla
    df['hl2'] = (df['High'] + df['Low']) / 2

    # True Range (TR) hesaplaması: ilk bar için high-low, sonraki barlarda
    # current high - current low, |current high - prev close|, |current low - prev close| değerlerinin maksimumu
    tr = [df['High'].iloc[0] - df['Low'].iloc[0]]
    for i in range(1, len(df)):
        current_high = df['High'].iloc[i]
        current_low = df['Low'].iloc[i]
        prev_close = df['Close'].iloc[i-1]
        tr_val = max(current_high - current_low,
                     abs(current_high - prev_close),
                     abs(current_low - prev_close))
        tr.append(tr_val)
    df['TR'] = tr

    # ATR hesaplama: changeATR True ise Wilder tarzı (EMA benzeri) hesaplama, yoksa SMA
    if changeATR:
        df['ATR'] = df['TR'].ewm(alpha=1/atr_period, adjust=False).mean()
    else:
        df['ATR'] = df['TR'].rolling(window=atr_period).mean()

    src = df['hl2']
    atr = df['ATR']
    n = len(df)

    # Birincil ve onay (confirmation) serileri için listeler
    up = [np.nan] * n
    dn = [np.nan] * n
    trend = [np.nan] * n

    upx = [np.nan] * n
    dnx = [np.nan] * n
    trendx = [np.nan] * n

    # İlk bar için başlangıç değerleri
    up[0] = src.iloc[0] - multiplier1 * atr.iloc[0]
    dn[0] = src.iloc[0] + multiplier1 * atr.iloc[0]
    trend[0] = 1  # Varsayılan trend yukarı

    upx[0] = src.iloc[0] - multiplier2 * atr.iloc[0]
    dnx[0] = src.iloc[0] + multiplier2 * atr.iloc[0]
    trendx[0] = 1

    # Trend serilerinin dinamik güncellenmesi:
    # Daha dinamik sinyal için, önceki trend durumunu sorgulamak yerine, 
    # fiyatın önceki dinamik eşik değerlerle karşılaştırılmasına dayalı geçişler kullanılacak.
    for i in range(1, n):
        # Primary seriler hesaplama:
        computed_up = src.iloc[i] - multiplier1 * atr.iloc[i]
        computed_dn = src.iloc[i] + multiplier1 * atr.iloc[i]
        # Önceki barın kapanışına göre eşik güncelleme:
        up[i] = computed_up if df['Close'].iloc[i-1] <= up[i-1] else max(computed_up, up[i-1])
        dn[i] = computed_dn if df['Close'].iloc[i-1] >= dn[i-1] else min(computed_dn, dn[i-1])
        
        # Trend geçişini daha dinamik yapmak:
        if df['Close'].iloc[i] > dn[i-1]:
            trend[i] = 1
        elif df['Close'].iloc[i] < up[i-1]:
            trend[i] = -1
        else:
            trend[i] = trend[i-1]

        # Confirmation seriler hesaplama:
        computed_upx = src.iloc[i] - multiplier2 * atr.iloc[i]
        computed_dnx = src.iloc[i] + multiplier2 * atr.iloc[i]
        upx[i] = computed_upx if df['Close'].iloc[i-1] <= upx[i-1] else max(computed_upx, upx[i-1])
        dnx[i] = computed_dnx if df['Close'].iloc[i-1] >= dnx[i-1] else min(computed_dnx, dnx[i-1])
        
        if df['Close'].iloc[i] > dnx[i-1]:
            trendx[i] = 1
        elif df['Close'].iloc[i] < upx[i-1]:
            trendx[i] = -1
        else:
            trendx[i] = trendx[i-1]

    # Tetiklenen sinyalleri hesaplamak: 
    # Burada, önceki barın trend durumuyla farklı bir durum oluştuğunda (yukarı veya aşağıya geçiş)
    # alım/satım sinyali üretilecek.
    buy_signal_primary = [False] * n
    sell_signal_primary = [False] * n
    buy_signal_confirm = [False] * n
    sell_signal_confirm = [False] * n

    for i in range(1, n):
        buy_signal_primary[i] = (trend[i] == 1 and trend[i-1] != 1)
        sell_signal_primary[i] = (trend[i] == -1 and trend[i-1] != -1)
        buy_signal_confirm[i] = (trendx[i] == 1 and trendx[i-1] != 1)
        sell_signal_confirm[i] = (trendx[i] == -1 and trendx[i-1] != -1)

    # Pozisyon mantığı ile sinyal entegrasyonu: 
    # Hem primary hem onaylanmış trend geçişleri olduğunda sinyal üretilir.
    df['Signal'] = None
    in_position = False
    for i in range(1, n):
        if not in_position and buy_signal_primary[i] and buy_signal_confirm[i]:
            df.at[df.index[i], 'Signal'] = 'BUY'
            in_position = True
        elif in_position and sell_signal_primary[i] and sell_signal_confirm[i]:
            df.at[df.index[i], 'Signal'] = 'SELL'
            in_position = False

    # Opsiyonel: Ek bilgi sütunları ekleyebilirsiniz.
    df['Trend'] = trend
    df['TrendConfirm'] = trendx
    df['Up'] = up
    df['Dn'] = dn
    df['Up_Confirm'] = upx
    df['Dn_Confirm'] = dnx

    return df

from pykalman import KalmanFilter

def strategy_kalman_rsi(df: pd.DataFrame,
                         rsi_period: int = 14,
                         rsi_overbought: float = 70,
                         rsi_oversold: float = 30) -> pd.DataFrame:
    """
    Kalman Filtreli Trend ve RSI Stratejisi:
      - Kalman filtresi ile fiyatın dinamik trendi belirlenir.
      - RSI ile momentum teyidi yapılarak aşırı alım/aşırı satım durumları tespit edilir.
      - Fiyatın KalmanTrend seviyesini kesmesi ve RSI sinyali sinyal üretir.
    
    Parametreler:
      df: 'Close' sütununu içeren ve en az 'Close' fiyat verisini barındıran DataFrame.
      rsi_period: RSI hesaplaması için periyot (varsayılan 14)
      rsi_overbought: RSI aşırı alım eşik değeri (varsayılan 70)
      rsi_oversold: RSI aşırı satım eşik değeri (varsayılan 30)
      
    Çıktı:
      Sinyallerin üretilmiş olduğu ve KalmanTrend ile RSI sütunlarını içeren DataFrame.
    """
    
    df = df.copy()
    
    # RSI hesaplaması:
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=rsi_period).mean()
    avg_loss = loss.rolling(window=rsi_period).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Kalman filtresi kullanarak trend tahmini
    # Kalman filtresinin parametrelerinde modelin sade tutulması amaçlanmıştır.
    kf = KalmanFilter(
        transition_matrices = [1],
        observation_matrices = [1],
        initial_state_mean = df['Close'].iloc[0],
        initial_state_covariance = 1,
        observation_covariance = 1,
        transition_covariance = 0.01
    )
    state_means, _ = kf.smooth(df['Close'].values)
    df['KalmanTrend'] = state_means.flatten()
    
    # Sinyal üretimi: 
    # - Alım sinyali: Fiyat, önceki barda KalmanTrend'in altında olup, şu barda üzerinde kesişiyor ve RSI aşırı satım bölgesinde (RSI < 30)
    # - Satım sinyali: Fiyat, önceki barda KalmanTrend'in üstünde olup, şu barda altında kesişiyor ve RSI aşırı alım bölgesinde (RSI > 70)
    df['Signal'] = None
    in_position = False
    for i in range(1, len(df)):
        # Alım sinyali kontrolü
        if (not in_position and 
            df['Close'].iloc[i-1] < df['KalmanTrend'].iloc[i-1] and
            df['Close'].iloc[i] >= df['KalmanTrend'].iloc[i] and
            df['RSI'].iloc[i] < rsi_oversold):
            df.at[df.index[i], 'Signal'] = 'BUY'
            in_position = True
        # Satım sinyali kontrolü
        elif (in_position and
              df['Close'].iloc[i-1] > df['KalmanTrend'].iloc[i-1] and
              df['Close'].iloc[i] <= df['KalmanTrend'].iloc[i] and
              df['RSI'].iloc[i] > rsi_overbought):
            df.at[df.index[i], 'Signal'] = 'SELL'
            in_position = False

    return df



def strategy_fractal_alligator(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Alligator bileşenleri
    df['Jaw'] = df['Close'].rolling(window=13).mean().shift(8)
    df['Teeth'] = df['Close'].rolling(window=8).mean().shift(5)
    df['Lips'] = df['Close'].rolling(window=5).mean().shift(3)

    # Fractal tespiti (sadece geçmişe bakar)
    df['Fractal_High'] = False
    df['Fractal_Low'] = False
    for i in range(2, len(df)):
        if df['High'].iloc[i - 2] < df['High'].iloc[i] and df['High'].iloc[i - 1] < df['High'].iloc[i]:
            df.at[df.index[i], 'Fractal_High'] = True
        if df['Low'].iloc[i - 2] > df['Low'].iloc[i] and df['Low'].iloc[i - 1] > df['Low'].iloc[i]:
            df.at[df.index[i], 'Fractal_Low'] = True

    # Sinyal üretimi
    df['Signal'] = None
    in_position = False

    for i in range(2, len(df)):
        close = df['Close'].iloc[i]
        jaw = df['Jaw'].iloc[i]
        if pd.isna(jaw):
            continue

        if not in_position and df['Fractal_High'].iloc[i - 1] and close > jaw:
            df.at[df.index[i], 'Signal'] = 'BUY'
            in_position = True
        elif in_position and df['Fractal_Low'].iloc[i - 1] and close < jaw:
            df.at[df.index[i], 'Signal'] = 'SELL'
            in_position = False

    return df
