import pandas as pd
import torch


class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, data: pd.DataFrame, sequence_length: int):
        self._sequence_length = sequence_length
        cols = list(data.columns)
        cols.remove("Symbol")
        self._groups = {
            key: subdata[cols] for key, subdata in data.groupby("Symbol")[cols]
        }
        self._ids = [
            (k, kid)
            for k, v in self._groups.items()
            for kid in range(len(v) - self._sequence_length)
        ]

    def __len__(self):
        return len(self._ids)

    def __getitem__(self, idx):
        k, idv = self._ids[idx]
        sequence = self._groups[k].iloc[idv : idv + self._sequence_length]
        return torch.tensor(
            sequence.drop(columns=["Date"]).fillna(0)[["Open"]].astype("float32").values
        )
