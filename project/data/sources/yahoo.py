import datetime
import typing
import pandas as pd
import yfinance as yf
from yahoo_fin import stock_info as si


class Yahoo:
    def __init__(self) -> None:
        self._tickers = [
            si.tickers_sp500(),
            si.tickers_nasdaq(),
            si.tickers_dow(),
            si.tickers_other(),
        ]

    @property
    def tickers(self) -> list:
        return sorted(set(sum(self._tickers, [])))

    def history(self, ticket: str, period: typing.Optional[str] = None) -> pd.DataFrame:
        period = period if period else "max"
        data = yf.Ticker(ticket).history(period=period)
        data.index = pd.to_datetime(data.index, format="%Y-%m-%d %H:%M:%S%z")
        return data
