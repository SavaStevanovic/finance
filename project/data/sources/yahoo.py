import typing
import pandas as pd
import yfinance as yf
from yahoo_fin import stock_info as si

from data.sources.data_source import Provider


class Yahoo(Provider):
    def __init__(self) -> None:
        self._tickers = [si.tickers_sp500(),
            si.tickers_nasdaq(),
            si.tickers_dow()]

    @property
    def tickers(self) -> list:
        return sorted(set(sum(self._tickers, [])))

    def history(self, ticket: str, period: typing.Optional[str] = None) -> pd.DataFrame:
        return yf.Ticker(ticket).history(period=period).reset_index()
