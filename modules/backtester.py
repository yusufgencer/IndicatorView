import pandas as pd

def run_backtest(df: pd.DataFrame, initial_cash: float = 10000.0) -> pd.DataFrame:
    df = df.copy()
    cash = initial_cash
    position = 0.0
    portfolio_values = []

    for i in range(len(df)):
        price = df['Close'].iloc[i]
        signal = df['Signal'].iloc[i]

        if signal == 'BUY' and cash > 0:
            position = cash / price
            cash = 0

        elif signal == 'SELL' and position > 0:
            cash = position * price
            position = 0

        total_value = cash + (position * price)
        portfolio_values.append(total_value)

    df['Portfolio'] = portfolio_values
    return df