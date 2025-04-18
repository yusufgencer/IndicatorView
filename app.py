import streamlit as st
import pandas as pd
from datetime import date, timedelta
from modules.data_loader import PriceDataPipeline
from modules.indicators import (
    add_atr,
    add_bollinger_bands,
    add_mfi,
    add_macd,
    add_obv,
    add_rsi,
    add_stochrsi,
    add_cci,
    add_supertrend,
    add_williams_r,
    add_adx,
    add_trix,
    add_roc,
    add_vwap,
    add_keltner_channels,
    add_donchian_channels,
    add_parabolic_sar,
    add_zscore,
    add_ema_crossover,
    add_sma,
    add_cmf,
    add_ichimoku_cloud,
    add_dpo,
    add_vortex,
    add_ultimate_oscillator,
    add_elder_ray,
    add_price_oscillator,
    add_bop,
    add_ac_oscillator,
    add_chaikin_oscillator,
    add_tema,
    add_fractal_indicator,
    add_rvi,

)

from modules.strategy import (
    generate_atr_signals,
    generate_bollinger_signals,
    generate_macd_signals,
    generate_mfi_signals,
    generate_obv_signals,
    generate_rsi_signals,
    generate_stochrsi_signals,
    generate_cci_signals,
    generate_supertrend_signals,
    generate_williamsr_signals,
    generate_adx_signals,
    generate_trix_signals,
    generate_roc_signals,
    generate_vwap_signals,
    generate_keltner_signals,
    generate_donchian_signals,
    generate_parabolic_sar_signals,
    generate_zscore_signals,
    generate_ema_crossover_signals,
    generate_sma_signals,
    generate_cmf_signals,
    generate_ichimoku_signals,
    generate_dpo_signals,
    generate_vortex_signals,
    generate_ultimate_oscillator_signals,
    generate_elder_ray_signals,
    generate_price_oscillator_signals,
    generate_bop_signals,
    generate_ac_oscillator_signals,
    generate_chaikin_oscillator_signals,   
    generate_tema_signals,
    generate_fractal_signals,
    generate_rvi_signals,
)


from modules.backtester import run_backtest
from modules.performance import evaluate_performance
from modules.plotter import plot_price_with_signals, plot_candlestick
from modules import custom_strategies

# ------------------------ PARAMETRE MAPPINGLER ------------------------
parametric_indicator_mapping = {
    "RSI": {
        "add_func": add_rsi,
        "signal_func": generate_rsi_signals,
        "add_params": ["window"],
        "signal_params": {"buy": "buy_threshold", "sell": "sell_threshold"}
    },
    "MACD": {
        "add_func": add_macd,
        "signal_func": generate_macd_signals,
        "add_params": ["fast", "slow", "signal"],
        "signal_params": {}
    },
    "Bollinger Bands": {
        "add_func": add_bollinger_bands,
        "signal_func": generate_bollinger_signals,
        "add_params": ["window", "std_multiplier"],
        "signal_params": {}
    },
    "MFI": {
        "add_func": add_mfi,
        "signal_func": generate_mfi_signals,
        "add_params": ["window"],
        "signal_params": {"buy": "buy_threshold", "sell": "sell_threshold"}
    }
}

nonparametric_indicator_mapping = {
    "ATR": {
        "add_func": add_atr,
        "signal_func": generate_atr_signals,
        "add_params": ["window"],
        "signal_params": {}
    },
    "OBV": {
        "add_func": add_obv,
        "signal_func": generate_obv_signals,
        "add_params": [],
        "signal_params": {}
    },
    "Stochastic RSI": {
    "add_func": add_stochrsi,
    "signal_func": generate_stochrsi_signals,
    "add_params": [],
    "signal_params": {}
    },
    "CCI": {
        "add_func": add_cci,
        "signal_func": generate_cci_signals,
        "add_params": [],
        "signal_params": {}
    },
    "SuperTrend": {
    "add_func": add_supertrend,
    "signal_func": generate_supertrend_signals,
    "add_params": [],
    "signal_params": {}
    },
    "Williams %R": {
    "add_func": add_williams_r,
    "signal_func": generate_williamsr_signals,
    "add_params": [],
    "signal_params": {}
    },
    "ADX/DMI": {
    "add_func": add_adx,
    "signal_func": generate_adx_signals,
    "add_params": [],
    "signal_params": {}
    },
    "TRIX": {
    "add_func": add_trix,
    "signal_func": generate_trix_signals,
    "add_params": [],
    "signal_params": {}
    },
    "ROC": {
    "add_func": add_roc,
    "signal_func": generate_roc_signals,
    "add_params": [],
    "signal_params": {}
    },
    "VWAP": {
    "add_func": add_vwap,
    "signal_func": generate_vwap_signals,
    "add_params": [],
    "signal_params": {}
    },
    "Keltner Channels": {
    "add_func": add_keltner_channels,
    "signal_func": generate_keltner_signals,
    "add_params": [],
    "signal_params": {}
    },
    "Donchian Channels": {
    "add_func": add_donchian_channels,
    "signal_func": generate_donchian_signals,
    "add_params": [],
    "signal_params": {}
    },
    "Parabolic SAR": {
    "add_func": add_parabolic_sar,
    "signal_func": generate_parabolic_sar_signals,
    "add_params": [],
    "signal_params": {}
    },
    "Z-Score": {
    "add_func": add_zscore,
    "signal_func": generate_zscore_signals,
    "add_params": [],
    "signal_params": {}
    },
    "EMA Crossover": {
    "add_func": add_ema_crossover,
    "signal_func": generate_ema_crossover_signals,
    "add_params": [],
    "signal_params": {}
    },
    "SMA": {
    "add_func": add_sma,
    "signal_func": generate_sma_signals,
    "add_params": [],
    "signal_params": {}
    },
    "CMF": {
    "add_func": add_cmf,
    "signal_func": generate_cmf_signals,
    "add_params": [],
    "signal_params": {}
    },
    "Ichimoku Cloud": {
        "add_func": add_ichimoku_cloud,
        "signal_func": generate_ichimoku_signals,
        "add_params": [],
        "signal_params": {}
    },
    "DPO": {
    "add_func": add_dpo,
    "signal_func": generate_dpo_signals,
    "add_params": [],
    "signal_params": {}
    },
    "Vortex Indicator": {
    "add_func": add_vortex,
    "signal_func": generate_vortex_signals,
    "add_params": [],
    "signal_params": {}
    },
    "Ultimate Oscillator": {
        "add_func": add_ultimate_oscillator,
        "signal_func": generate_ultimate_oscillator_signals,
        "add_params": [],
        "signal_params": {}
    },
    "Elder Ray Index": {
        "add_func": add_elder_ray,
        "signal_func": generate_elder_ray_signals,
        "add_params": [],
        "signal_params": {}
    },
    "Price Oscillator": {
        "add_func": add_price_oscillator,
        "signal_func": generate_price_oscillator_signals,
        "add_params": [],
        "signal_params": {}
    },
    "Balance of Power": {
    "add_func": add_bop,
    "signal_func": generate_bop_signals,
    "add_params": [],
    "signal_params": {}
    },
    "Acceleration/Deceleration Oscillator": {
    "add_func": add_ac_oscillator,
    "signal_func": generate_ac_oscillator_signals,
    "add_params": [],
    "signal_params": {}
    },
    "Chaikin Oscillator": {
    "add_func": add_chaikin_oscillator,
    "signal_func": generate_chaikin_oscillator_signals,
    "add_params": [],
    "signal_params": {}
    },
    "TEMA": {
    "add_func": add_tema,
    "signal_func": generate_tema_signals,
    "add_params": [],
    "signal_params": {}
    },
    "Fractal Indicator": {
    "add_func": add_fractal_indicator,
    "signal_func": generate_fractal_signals,
    "add_params": [],
    "signal_params": {}
    },
    "Relative Vigor Index": {
        "add_func": add_rvi,
        "signal_func": generate_rvi_signals,
        "add_params": [],
        "signal_params": {}
    },



}

import pandas as pd

def generate_composite_strategy_signals(
    df: pd.DataFrame,
    selected_indicators: list,
    parametric_indicator_mapping: dict,
    nonparametric_indicator_mapping: dict,
    indicator_params: dict,
    default_indicator_params: dict,
    threshold: float = 0.6
) -> pd.DataFrame:
    """
    Seçilen indikatörlerden gelen sinyallere göre çoğunluğa dayalı BUY/SELL sinyali üretir.
    Position mantığına göre sadece BUY → SELL → BUY döngüsüne izin verir.
    """
    df = df.copy()
    signal_matrix = pd.DataFrame(index=df.index)

    # 1. Tüm göstergelerden sinyal matrisi oluştur
    for ind in selected_indicators:
        mapping = parametric_indicator_mapping.get(ind, nonparametric_indicator_mapping.get(ind))
        if mapping is None:
            continue

        all_params = indicator_params.get(ind, default_indicator_params.get(ind, {}))

        valid_add_keys = mapping.get("add_params", [])
        add_args = {k: all_params[k] for k in valid_add_keys if k in all_params}
        df = mapping["add_func"](df, **add_args)

        signal_func = mapping["signal_func"]
        signal_param_map = mapping.get("signal_params", {})
        signal_args = {
            target_key: all_params[source_key]
            for source_key, target_key in signal_param_map.items()
            if source_key in all_params
        }

        df_with_signal = signal_func(df.copy(), **signal_args)
        signal_matrix[ind] = df_with_signal["Signal"]

    # 2. Threshold bazlı çoğunluk sinyali + pozisyon kontrolü
    df["Signal"] = None
    position = False  # Başlangıçta pozisyon yok

    for i in range(len(df)):
        signals_today = signal_matrix.iloc[i]
        buy_count = (signals_today == "BUY").sum()
        sell_count = (signals_today == "SELL").sum()
        total_valid = signals_today.notna().sum()

        if total_valid == 0:
            continue

        buy_ratio = buy_count / total_valid
        sell_ratio = sell_count / total_valid

        if not position and buy_ratio >= threshold:
            df.at[df.index[i], "Signal"] = "BUY"
            position = True
        elif position and sell_ratio >= threshold:
            df.at[df.index[i], "Signal"] = "SELL"
            position = False

    return df


import pandas as pd

def generate_composite_strategy_signals_with_sl_tp(
    df: pd.DataFrame,
    selected_indicators: list,
    parametric_indicator_mapping: dict,
    nonparametric_indicator_mapping: dict,
    indicator_params: dict,
    default_indicator_params: dict,
    threshold: float = 0.6,
    stop_loss: float = 0.02,
    take_profit: float = 0.05
) -> pd.DataFrame:
    """
    - Majority‑voting BUY/SELL sinyali
    - Stop‑Loss / Take‑Profit
    - Pozisyon durumuna göre yalnızca BUY→SELL→BUY döngüsüne izin verir
    """
    df = df.copy()

    # 1) Her gösterge için sinyal matrisi oluştur
    signal_matrix = pd.DataFrame(index=df.index)
    for ind in selected_indicators:
        mapping = parametric_indicator_mapping.get(ind) or nonparametric_indicator_mapping.get(ind)
        if mapping is None:
            continue

        # Göstergeleri ekle
        params   = indicator_params.get(ind, default_indicator_params.get(ind, {}))
        add_args = {k: params[k] for k in mapping.get("add_params", []) if k in params}
        df       = mapping["add_func"](df, **add_args)

        # Sinyal fonksiyonunu çalıştır
        sig_args = {
            tgt: params[src]
            for src, tgt in mapping.get("signal_params", {}).items()
            if src in params
        }
        tmp = mapping["signal_func"](df.copy(), **sig_args)
        signal_matrix[ind] = tmp["Signal"]

    # 2) Composite + SL/TP + pozisyon döngüsü
    df["Signal"]    = None
    position        = False
    entry_price     = None

    for i, idx in enumerate(df.index):
        price = df.at[idx, "Close"]

        # 2a) Pozisyonda isek önce SL/TP kontrolü
        if position and entry_price is not None:
            # Stop‑Loss
            if price <= entry_price * (1 - stop_loss):
                df.at[idx, "Signal"] = "SELL"
                position    = False
                entry_price = None
                continue
            # Take‑Profit
            if price >= entry_price * (1 + take_profit):
                df.at[idx, "Signal"] = "SELL"
                position    = False
                entry_price = None
                continue

        # 2b) Majority‑voting
        today       = signal_matrix.iloc[i]
        valid_count = today.notna().sum()
        if valid_count == 0:
            continue

        buy_ratio  = (today == "BUY").sum()  / valid_count
        sell_ratio = (today == "SELL").sum() / valid_count

        # 2c) BUY: yalnızca pozisyon kapalıyken
        if not position and buy_ratio >= threshold:
            df.at[idx, "Signal"] = "BUY"
            position    = True
            entry_price = price
            continue

        # 2d) SELL: yalnızca pozisyondayken
        if position and sell_ratio >= threshold:
            df.at[idx, "Signal"] = "SELL"
            position    = False
            entry_price = None
            continue

    return df

def generate_composite_strategy_signals_with_trailing_sl_tp(
    df: pd.DataFrame,
    selected_indicators: list,
    parametric_indicator_mapping: dict,
    nonparametric_indicator_mapping: dict,
    indicator_params: dict,
    default_indicator_params: dict,
    threshold: float = 0.6,
    trailing_stop: float = 0.02,
    take_profit: float = 0.05
) -> pd.DataFrame:
    """
    - Majority‑voting BUY/SELL sinyali
    - Sabit Take‑Profit + Dinamik (Trailing) Stop‑Loss
    - Pozisyon döngüsü: sadece BUY→SELL→BUY
    """
    df = df.copy()

    # 1) Sinyal matrisi oluşturma (diğer fonksiyonla aynı)
    signal_matrix = pd.DataFrame(index=df.index)
    for ind in selected_indicators:
        mapping = parametric_indicator_mapping.get(ind) or nonparametric_indicator_mapping.get(ind)
        if not mapping:
            continue
        params   = indicator_params.get(ind, default_indicator_params.get(ind, {}))
        add_args = {k: params[k] for k in mapping.get("add_params", []) if k in params}
        df       = mapping["add_func"](df, **add_args)
        sig_args = {tgt: params[src] for src, tgt in mapping.get("signal_params", {}).items() if src in params}
        tmp = mapping["signal_func"](df.copy(), **sig_args)
        signal_matrix[ind] = tmp["Signal"]

    # 2) Composite + Dinamik SL + Sabit TP
    df["Signal"]     = None
    position          = False
    entry_price       = None
    highest_price     = None

    for i, idx in enumerate(df.index):
        price = df.at[idx, "Close"]

        # Pozisyondayken önce TP/Trailing SL kontrolü
        if position:
            # En yüksek fiyatı güncelle
            highest_price = max(highest_price, price)
            # Sabit kâr hedefi ve dinamik stop seviyesi
            tp_price = entry_price * (1 + take_profit)
            ts_price = highest_price * (1 - trailing_stop)

            # Sabit Take‑Profit
            if price >= tp_price:
                df.at[idx, "Signal"]    = "SELL"
                position                   = False
                entry_price, highest_price = None, None
                continue
            # Dinamik Trailing Stop‑Loss
            if price <= ts_price:
                df.at[idx, "Signal"]    = "SELL"
                position                   = False
                entry_price, highest_price = None, None
                continue

        # Composite SELL sinyali (pozisyondayken)
        if position:
            today       = signal_matrix.iloc[i]
            valid_count = today.notna().sum()
            if valid_count:
                sell_ratio = (today == "SELL").sum() / valid_count
                if sell_ratio >= threshold:
                    df.at[idx, "Signal"]    = "SELL"
                    position                  = False
                    entry_price, highest_price = None, None
                    continue

        # Composite BUY sinyali (pozisyon kapalıyken)
        if not position:
            today       = signal_matrix.iloc[i]
            valid_count = today.notna().sum()
            if valid_count:
                buy_ratio = (today == "BUY").sum() / valid_count
                if buy_ratio >= threshold:
                    df.at[idx, "Signal"] = "BUY"
                    position               = True
                    entry_price            = price
                    highest_price          = price
                    continue

    return df

def generate_buy_only_composite_strategy_with_trailing_sl_tp(
    df: pd.DataFrame,
    selected_indicators: list,
    parametric_indicator_mapping: dict,
    nonparametric_indicator_mapping: dict,
    indicator_params: dict,
    default_indicator_params: dict,
    threshold: float = 0.6,
    trailing_stop: float = 0.02,
    take_profit: float = 0.05
) -> pd.DataFrame:
    """
    - BUY kararları çoğunluk sinyale göre verilir.
    - SELL kararları sadece TP veya Trailing SL ile olur.
    - SELL sinyali göstergelerden asla gelmez.
    """
    df = df.copy()
    signal_matrix = pd.DataFrame(index=df.index)

    for ind in selected_indicators:
        mapping = parametric_indicator_mapping.get(ind) or nonparametric_indicator_mapping.get(ind)
        if not mapping:
            continue
        params   = indicator_params.get(ind, default_indicator_params.get(ind, {}))
        add_args = {k: params[k] for k in mapping.get("add_params", []) if k in params}
        df       = mapping["add_func"](df, **add_args)
        sig_args = {tgt: params[src] for src, tgt in mapping.get("signal_params", {}).items() if src in params}
        tmp = mapping["signal_func"](df.copy(), **sig_args)
        signal_matrix[ind] = tmp["Signal"]

    df["Signal"]      = None
    position          = False
    entry_price       = None
    highest_price     = None

    for i, idx in enumerate(df.index):
        price = df.at[idx, "Close"]

        # Pozisyondayken sadece TP veya SL kontrol edilir
        if position:
            highest_price = max(highest_price, price)
            tp_price = entry_price * (1 + take_profit)
            ts_price = highest_price * (1 - trailing_stop)

            if price >= tp_price or price <= ts_price:
                df.at[idx, "Signal"] = "SELL"
                position = False
                entry_price, highest_price = None, None
                continue

        # Pozisyon yoksa BUY sinyali değerlendir
        if not position:
            today = signal_matrix.iloc[i]
            valid_count = today.notna().sum()
            if valid_count:
                buy_ratio = (today == "BUY").sum() / valid_count
                if buy_ratio >= threshold:
                    df.at[idx, "Signal"] = "BUY"
                    position = True
                    entry_price = price
                    highest_price = price

    return df

# ------------------------ STREAMLIT BAŞLANGIÇ ------------------------
st.set_page_config(page_title="📊 İndikatör Bazlı Strateji Simülatörü", layout="wide")
st.title("🤖 İndikatör Tabanlı Backtest Uygulaması")

# ------------------------ HİSSE SEÇİMİ ------------------------
st.sidebar.header("📈 Hisse Seçimi")
selected_market = st.sidebar.selectbox("Borsa", ["NASDAQ", "BIST"])

nasdaq_symbols = {
    "Apple (AAPL)": "AAPL", "Tesla (TSLA)": "TSLA", "Amazon (AMZN)": "AMZN",
    "NVIDIA (NVDA)": "NVDA", "Microsoft (MSFT)": "MSFT", "Meta (META)": "META",
    "Netflix (NFLX)": "NFLX", "Google (GOOGL)": "GOOGL", "Intel (INTC)": "INTC", "AMD (AMD)": "AMD"
}
bist_symbols = {
    "ASELSAN (ASELS.IS)": "ASELS.IS", "THYAO (THYAO.IS)": "THYAO.IS", "EREGLİ (EREGL.IS)": "EREGL.IS",
    "SASA (SASA.IS)": "SASA.IS", "BIMAS (BIMAS.IS)": "BIMAS.IS", "SISE (SISE.IS)": "SISE.IS",
    "KORDS (KORDS.IS)": "KORDS.IS", "TUPRS (TUPRS.IS)": "TUPRS.IS", "PETKM (PETKM.IS)": "PETKM.IS", "ISCTR (ISCTR.IS)": "ISCTR.IS"
}
symbol_dict = nasdaq_symbols if selected_market == "NASDAQ" else bist_symbols
symbol_label = st.sidebar.selectbox("Hisse Senedi Seçin", list(symbol_dict.keys()))
symbol = symbol_dict[symbol_label]

# ------------------------ TARİH ARALIĞI ------------------------
user_start_date = st.sidebar.date_input("Başlangıç Tarihi", date.today() - timedelta(days=365))
start_date = user_start_date - timedelta(days=30)
end_date = st.sidebar.date_input("Bitiş Tarihi", date.today())

# ------------------------ GÖSTERGE SEÇİMİ ------------------------
st.sidebar.subheader("🔧 Parametreli İndikatörler")
selected_indicators = st.sidebar.multiselect("Parametresi girilebilen göstergeler:", list(parametric_indicator_mapping.keys()))

st.sidebar.subheader("📌 Sabit İndikatörler")

# 1) "Tümünü Seç" butonu
if st.sidebar.button("▶️ Tümünü Seç"):
    # session_state içindeki anahtar, multiselect'in key'iyle aynı olmalı
    st.session_state['selected_basic_indicators'] = list(nonparametric_indicator_mapping.keys())

# 2) Multiselect'i session_state üzerinden yönet
selected_basic_indicators = st.sidebar.multiselect(
    "Parametresiz göstergeler:",
    options=list(nonparametric_indicator_mapping.keys()),
    key='selected_basic_indicators'
)
# ------------------------ GÖSTERGE PARAMETRELERİ ------------------------
user_indicator_params = {}
default_indicator_params = {
    "RSI": {"window": 14, "buy": 30, "sell": 70},
    "MACD": {"fast": 12, "slow": 26, "signal": 9},
    "Bollinger Bands": {"window": 20, "std_multiplier": 2.0},
    "MFI": {"window": 14, "buy": 20, "sell": 80},
    "ATR": {"window": 14}
}

for ind in selected_indicators:
    user_indicator_params[ind] = {}
    for param_key, default_val in default_indicator_params.get(ind, {}).items():
        label = f"{ind} - {param_key}"
        if isinstance(default_val, int):
            user_indicator_params[ind][param_key] = st.sidebar.slider(label, 1, 100, default_val)
        elif isinstance(default_val, float):
            user_indicator_params[ind][param_key] = st.sidebar.slider(label, 0.1, 5.0, default_val, step=0.1)

# ------------------------ GELİŞMİŞ STRATEJİLER ------------------------
strategy_funcs = {
    "🔥 Triple Confirmation": custom_strategies.strategy_triple_confirmation,
    "📊 Volatility Breakout + Volume": custom_strategies.strategy_volatility_breakout_with_volume,
    "📉 RSI + Bollinger Bands": custom_strategies.strategy_rsi_bollinger_with_position,
    "🧠 Kalman Trend + RSI": custom_strategies.strategy_kalman_rsi,
    "🐊 Fractal + Alligator": custom_strategies.strategy_fractal_alligator
}

selected_strategies = st.sidebar.multiselect("📘 Gelişmiş Stratejiler", list(strategy_funcs.keys()))

st.sidebar.subheader("🧠 Ortak Strateji Ayarları")

composite_threshold = st.sidebar.slider(
    "Majority (çoğunluk) için eşik oranı (%)",
    min_value=0,
    max_value=100,
    value=60,
    step=1,
    help="Örn: 60% seçersen, seçili göstergelerin en az %60'ı aynı sinyali vermelidir."
) / 100  # yüzdelikten orana çeviriyoruz

st.sidebar.subheader("⚙️ Stop‑Loss & Take‑Profit Ayarları")
stop_loss_pct = st.sidebar.slider(
    "Stop‑Loss (%)",
    min_value=0.0,
    max_value=20.0,
    value=2.0,
    step=0.1,
    help="Pozisyon açıldıktan sonra en fazla kaç yüzde zarar tolere edilsin?"
) / 100

take_profit_pct = st.sidebar.slider(
    "Take‑Profit (%)",
    min_value=0.0,
    max_value=50.0,
    value=5.0,
    step=0.1,
    help="Pozisyon açıldıktan sonra kaç yüzde karla çıkılsın?"
) / 100

# Yeni: Dinamik (Trailing) Stop‑Loss
st.sidebar.subheader("⚙️ Dinamik (Trailing) Stop‑Loss Ayarı")
trailing_stop_pct = st.sidebar.slider(
    "Trailing Stop (%)",
    min_value=0.0,
    max_value=20.0,
    value=2.0,
    step=0.1,
    help="Pozisyon açıldıktan sonra zararı takip eden stop seviyesi yüzde kaç olsun?"
) / 100

# ------------------------ VERİYİ ÇEK ------------------------
pipeline = PriceDataPipeline(symbol, str(start_date), str(end_date))
full_df = pipeline.fetch_daily_stock_data()
user_start_date_aware = pd.to_datetime(user_start_date).tz_localize(full_df['Date'].dt.tz)
base_df = full_df[full_df['Date'] >= user_start_date_aware].reset_index(drop=True)

st.subheader("📉 Seçilen Hissenin Fiyat Hareketi")
candle_fig = plot_candlestick(base_df, title=f"{symbol} - Mum Grafiği")
st.plotly_chart(candle_fig, use_container_width=True)

# ------------------------ STRATEJİ TESTİ ------------------------
if st.sidebar.button("🚀 Strateji Testini Başlat"):
    st.success("Strateji testi başladı...")
    strategy_results = {}
    performances = []

    # BUY & HOLD
    bh_df = base_df.copy()
    bh_df['Signal'] = ["BUY"] + [None] * (len(bh_df) - 1)
    bh_df = run_backtest(bh_df, initial_cash=10000)
    perf = evaluate_performance(bh_df, initial_cash=10000)
    performances.append({"Strateji": "BUY & HOLD", **perf})
    st.subheader("📊 BUY & HOLD Strateji Grafiği")
    st.plotly_chart(plot_price_with_signals(bh_df, title="BUY & HOLD - Fiyat ve Portföy"), use_container_width=True)

    # Parametreli indikatörler
    for ind in selected_indicators:
        df = full_df.copy()
        mapping = parametric_indicator_mapping[ind]
        add_args = {k: user_indicator_params[ind][k] for k in mapping["add_params"]}
        df = mapping["add_func"](df, **add_args)
        signal_args = {target_key: user_indicator_params[ind][source_key] for source_key, target_key in mapping["signal_params"].items()}
        df = mapping["signal_func"](df, **signal_args)
        df = df[df['Date'] >= user_start_date_aware].reset_index(drop=True)
        df = run_backtest(df, initial_cash=10000)
        strategy_results[ind] = df
        perf = evaluate_performance(df, initial_cash=10000)
        performances.append({"Strateji": ind, **perf})
        st.subheader(f"📊 {ind} Strateji Grafiği")
        st.plotly_chart(plot_price_with_signals(df, title=f"{ind} - Fiyat, Portföy ve Al/Sat Sinyalleri"), use_container_width=True)

    # Sabit indikatörler
    for ind in selected_basic_indicators:
        df = full_df.copy()
        mapping = nonparametric_indicator_mapping[ind]
        default_args = {k: default_indicator_params[ind][k] for k in mapping["add_params"]}
        df = mapping["add_func"](df, **default_args)
        signal_args = {target_key: default_indicator_params[ind][source_key] for source_key, target_key in mapping["signal_params"].items()}
        df = mapping["signal_func"](df, **signal_args)
        df = df[df['Date'] >= user_start_date_aware].reset_index(drop=True)
        df = run_backtest(df, initial_cash=10000)
        strategy_results[ind] = df
        perf = evaluate_performance(df, initial_cash=10000)
        performances.append({"Strateji": ind, **perf})
        st.subheader(f"📊 {ind} Strateji Grafiği")
        st.plotly_chart(plot_price_with_signals(df, title=f"{ind} - Fiyat, Portföy ve Al/Sat Sinyalleri"), use_container_width=True)

    # ------------------------ ORTAK STRATEJİ ------------------------
    if selected_indicators or selected_basic_indicators:
        st.subheader("🤝 Ortak Strateji Karşılaştırması")

        all_selected = selected_indicators + selected_basic_indicators

        # ——————————————————————————————————————————————
        # 1) Normal Composite (SL/TP yok)
        # ——————————————————————————————————————————————
        df_normal = full_df.copy()
        df_normal = generate_composite_strategy_signals(
            df=df_normal,
            selected_indicators=all_selected,
            parametric_indicator_mapping=parametric_indicator_mapping,
            nonparametric_indicator_mapping=nonparametric_indicator_mapping,
            indicator_params=user_indicator_params,
            default_indicator_params=default_indicator_params,
            threshold=composite_threshold
        )
        df_normal = df_normal[df_normal['Date'] >= user_start_date_aware].reset_index(drop=True)
        df_normal = run_backtest(df_normal, initial_cash=10000)
        perf_normal = evaluate_performance(df_normal, initial_cash=10000)

        st.markdown("### 📊 Composite (Normal)")
        st.plotly_chart(
            plot_price_with_signals(df_normal, title="Ortak Strateji (Normal)"),
            use_container_width=True
        )

        # Performans kaydı
        performances.append({"Strateji": "🧠 Composite (Normal)", **perf_normal})
        strategy_results["🧠 Composite (Normal)"] = df_normal


        # ——————————————————————————————————————————————
        # 2) Composite with SL/TP
        # ——————————————————————————————————————————————
        df_sltp = full_df.copy()
        df_sltp = generate_composite_strategy_signals_with_sl_tp(
            df=df_sltp,
            selected_indicators=all_selected,
            parametric_indicator_mapping=parametric_indicator_mapping,
            nonparametric_indicator_mapping=nonparametric_indicator_mapping,
            indicator_params=user_indicator_params,
            default_indicator_params=default_indicator_params,
            threshold=composite_threshold,
            stop_loss=stop_loss_pct,
            take_profit=take_profit_pct
        )
        df_sltp = df_sltp[df_sltp['Date'] >= user_start_date_aware].reset_index(drop=True)
        df_sltp = run_backtest(df_sltp, initial_cash=10000)
        perf_sltp = evaluate_performance(df_sltp, initial_cash=10000)

        st.markdown("### 📊 Composite (Stop‑Loss & Take‑Profit)")
        st.plotly_chart(
            plot_price_with_signals(df_sltp, title="Ortak Strateji (SL/TP)"),
            use_container_width=True
        )
        # Performans kaydı
        performances.append({"Strateji": "🧠 Composite (SL/TP)", **perf_sltp})
        strategy_results["🧠 Composite (SL/TP)"] = df_sltp

        # ——————————————————————————————————————————————
        # 3) Composite (Buy Only + Trailing SL & TP)
        # ——————————————————————————————————————————————
        df_trailing = full_df.copy()
        df_trailing = generate_buy_only_composite_strategy_with_trailing_sl_tp(
            df=df_trailing,
            selected_indicators=all_selected,
            parametric_indicator_mapping=parametric_indicator_mapping,
            nonparametric_indicator_mapping=nonparametric_indicator_mapping,
            indicator_params=user_indicator_params,
            default_indicator_params=default_indicator_params,
            threshold=composite_threshold,
            trailing_stop=stop_loss_pct,
            take_profit=take_profit_pct
        )
        df_trailing = df_trailing[df_trailing['Date'] >= user_start_date_aware].reset_index(drop=True)
        df_trailing = run_backtest(df_trailing, initial_cash=10000)
        perf_trailing = evaluate_performance(df_trailing, initial_cash=10000)

        st.markdown("### 📊 Composite (Buy Only + Trailing SL & TP)")
        st.plotly_chart(
            plot_price_with_signals(df_trailing, title="Ortak Strateji (Buy Only + Trailing SL/TP)"),
            use_container_width=True
        )

        # Performans kaydı
        performances.append({"Strateji": "🧠 Composite (Buy Only + Trailing SL/TP)", **perf_trailing})
        strategy_results["🧠 Composite (Buy Only + Trailing SL/TP)"] = df_trailing


        # Dinamik SL + Sabit TP testi
        df_dynamic = full_df.copy()
        df_dynamic = generate_composite_strategy_signals_with_trailing_sl_tp(
            df=df_dynamic,
            selected_indicators=all_selected,
            parametric_indicator_mapping=parametric_indicator_mapping,
            nonparametric_indicator_mapping=nonparametric_indicator_mapping,
            indicator_params=user_indicator_params,
            default_indicator_params=default_indicator_params,
            threshold=composite_threshold,
            trailing_stop=trailing_stop_pct,
            take_profit=take_profit_pct
        )
        df_dynamic = df_dynamic[df_dynamic['Date'] >= user_start_date_aware].reset_index(drop=True)
        df_dynamic = run_backtest(df_dynamic, initial_cash=10000)
        perf_dynamic = evaluate_performance(df_dynamic, initial_cash=10000)

        st.markdown("### 📊 Composite (Trailing SL & TP)")
        st.plotly_chart(
            plot_price_with_signals(df_dynamic, title="Ortak Strateji (Trailing SL & TP)"),
            use_container_width=True
        )
        performances.append({"Strateji": "🧠 Composite (Trailing SL & TP)", **perf_dynamic})
        strategy_results["🧠 Composite (Trailing SL & TP)"] = df_dynamic





    # Gelişmiş stratejiler
    for strat_name in selected_strategies:
        df = full_df.copy()
        df = strategy_funcs[strat_name](df)
        df = df[df['Date'] >= user_start_date_aware].reset_index(drop=True)
        df = run_backtest(df, initial_cash=10000)
        perf = evaluate_performance(df, initial_cash=10000)
        performances.append({"Strateji": strat_name, **perf})
        st.subheader(f"📊 {strat_name} Strateji Grafiği")
        st.plotly_chart(plot_price_with_signals(df, title=f"{strat_name} - Fiyat, Portföy ve Al/Sat Sinyalleri"), use_container_width=True)

    # Performans Tablosu
    st.subheader("📋 Karşılaştırma Tablosu")
    st.dataframe(pd.DataFrame(performances))
