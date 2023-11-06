from abc import abstractmethod
import typing
import pandas as pd


class Provider:
    @property
    @abstractmethod
    def tickers(self) -> list:
        pass

    @abstractmethod
    def history(self, ticket: str, period: typing.Optional[str] = None) -> pd.DataFrame:
        pass


class Source:
    def __init__(self, provider: Provider) -> None:
        self._provider = provider
        self._columns = [
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Adj Close",
            "Volume",
            "Capital Gains",
            "Dividends",
            "Splits",
            "Stock Splits",
        ]

    def history(self, ticket: str, period: typing.Optional[str] = None) -> pd.DataFrame:
        period = period if period else "max"
        data = self._provider.history(ticket, period=period)
        data["Date"] = pd.to_datetime(data["Date"], format="%Y-%m-%d %H:%M:%S%z")
        for c in self._columns:
            if c not in data:
                data[c] = None
        data = data.assign(Symbol=ticket)
        data.set_index(["Symbol", "Date"], inplace=True)
        return data
