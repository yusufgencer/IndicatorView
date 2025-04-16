import numpy as np
import pandas as pd

def calculate_arr(df: pd.DataFrame, initial_cash: float, final_value: float) -> float:
    num_days = (df['Date'].iloc[-1] - df['Date'].iloc[0]).days
    num_years = num_days / 365.0
    if num_years == 0:
        return 0.0
    arr = ((final_value / initial_cash) ** (1 / num_years)) - 1
    return arr * 100


def calculate_sharpe_ratio(df: pd.DataFrame, risk_free_rate: float = 0.01) -> float:
    df = df.copy()
    df['DailyReturn'] = df['Portfolio'].pct_change()
    excess_return = df['DailyReturn'] - (risk_free_rate / 252)
    std_dev = np.std(excess_return)

    if std_dev == 0:
        return 0.0

    sharpe = np.mean(excess_return) / std_dev
    return sharpe * np.sqrt(252)


def calculate_max_drawdown(df: pd.DataFrame) -> float:
    cumulative_max = df['Portfolio'].cummax()
    drawdown = (df['Portfolio'] - cumulative_max) / cumulative_max
    max_drawdown = drawdown.min()
    return max_drawdown * 100


def evaluate_performance(df: pd.DataFrame, initial_cash: float) -> dict:
    start_value = df['Portfolio'].iloc[0]
    final_value = df['Portfolio'].iloc[-1]
    
    arr = calculate_arr(df, initial_cash, final_value)
    sharpe = calculate_sharpe_ratio(df)
    mdd = calculate_max_drawdown(df)

    total_return = ((final_value - start_value) / start_value) * 100

    return {
        "Başlangıç Portföy ($)": round(start_value, 2),
        "Bitiş Portföy ($)": round(final_value, 2),
        "Toplam Getiri (%)": round(total_return, 2),
        "Yıllık Ortalama Getiri (%)": round(arr, 2),
        "Sharpe Oranı": round(sharpe, 2),
        "Maksimum Düşüş (%)": round(mdd, 2)
    }
