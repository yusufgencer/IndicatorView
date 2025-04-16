import pandas as pd
import yfinance as yf
from typing import Any

class PriceDataPipeline:
    def __init__(self, symbol: str, start_date: str, end_date: str):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.data: pd.DataFrame = pd.DataFrame()

    def fetch_daily_stock_data(self) -> pd.DataFrame:
        ticker: Any = yf.Ticker(self.symbol)
        history: pd.DataFrame = ticker.history(start=self.start_date, end=self.end_date)
        self.data = history[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        self.data.reset_index(inplace=True)
        return self.data