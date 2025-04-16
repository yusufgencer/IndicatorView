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
}
}

import pandas as pd

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
selected_basic_indicators = st.sidebar.multiselect("Parametresiz göstergeler:", list(nonparametric_indicator_mapping.keys()))

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
    "🧠 Kalman Trend + RSI": custom_strategies.strategy_kalman_rsi
}
selected_strategies = st.sidebar.multiselect("📘 Gelişmiş Stratejiler", list(strategy_funcs.keys()))

st.sidebar.subheader("🧠 Ortak Strateji Ayarları")

composite_threshold = st.sidebar.slider(
    "Majority (çoğunluk) için eşik oranı (%)",
    min_value=0,
    max_value=100,
    value=60,
    step=5,
    help="Örn: 60% seçersen, seçili göstergelerin en az %60'ı aynı sinyali vermelidir."
) / 100  # yüzdelikten orana çeviriyoruz


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

    # Ortak Strateji (Majority Voting)
    if selected_indicators or selected_basic_indicators:
        st.subheader("🤝 Ortak Strateji (Çoğunluğa Dayalı Sinyal)")

        df = full_df.copy()
        all_selected_indicators = selected_indicators + selected_basic_indicators

        df = generate_composite_strategy_signals(
            df=full_df,
            selected_indicators=selected_indicators + selected_basic_indicators,
            parametric_indicator_mapping=parametric_indicator_mapping,
            nonparametric_indicator_mapping=nonparametric_indicator_mapping,
            indicator_params=user_indicator_params,
            default_indicator_params=default_indicator_params,
            threshold=composite_threshold
        )


        df = df[df['Date'] >= user_start_date_aware].reset_index(drop=True)
        df = run_backtest(df, initial_cash=10000)

        # Performans kaydı
        perf = evaluate_performance(df, initial_cash=10000)
        performances.append({"Strateji": "🧠 Ortak Strateji", **perf})
        strategy_results["🧠 Ortak Strateji"] = df

        # Grafik
        st.subheader("📊 Ortak Strateji Grafiği")
        st.plotly_chart(
            plot_price_with_signals(df, title="🧠 Ortak Strateji - Fiyat, Portföy ve Al/Sat Sinyalleri"),
            use_container_width=True
        )


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
