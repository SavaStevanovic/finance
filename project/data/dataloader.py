import typing
import pandas as pd
import torch


class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, groups: typing.Dict[str, pd.DataFrame], sequence_length: int):
        self._sequence_length = sequence_length
        self._groups = groups
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
        sequence = self.preprocess(sequence)
        return (
            sequence[:-1],
            sequence[1:],
        )

    @staticmethod
    def preprocess(sequence):
        sequence = sequence.drop(columns=["Date"]).astype("float32")
        # sequence = sequence.subtract(sequence.mean()).divide(sequence.std() + 1e-7)
        return torch.tensor(sequence[["Open"]].values)
