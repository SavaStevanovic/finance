import typing
import numpy as np
import pandas as pd
import torch


class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        groups: typing.Dict[str, pd.DataFrame],
        sequence_length: int,
        features: list,
        single_sample: bool = False,
    ):
        self._sequence_length = sequence_length + 1
        self._features = features
        self._groups = groups
        select = slice(0, -1)
        if single_sample:
            select = slice(-2, -1)
        self._ids = [
            (k, kid)
            for k, v in self._groups.items()
            for kid in range(len(v) - self._sequence_length)[select]
        ]

    def __len__(self):
        return len(self._ids)

    def __getitem__(self, idx):
        k, idv = self._ids[idx]
        sequence = self._groups[k].iloc[idv : idv + self._sequence_length]
        sequence = self.preprocess(sequence, self._features)
        channels = [sequence[:, i].numpy() for i in range(0, sequence.shape[1])]
        seq_len = 13
        seq = (
            np.concatenate((np.ones(seq_len), np.zeros(seq_len - 1)), axis=0) / seq_len
        )

        mean_stack = [
            np.convolve(
                np.pad(c, (len(seq) // 2, len(seq) // 2), mode="edge"),
                seq,
                mode="valid",
            )
            for c in channels
        ]
        sequence_mean = np.stack(
            mean_stack,
            axis=1,
        ).astype("float32")
        sequence = torch.tensor(
            np.concatenate((sequence_mean, sequence.numpy()), axis=1)
        )
        sequence = sequence[1:] - sequence[:-1]
        return (
            sequence[:-1],
            sequence[1:],
        )

    @staticmethod
    def preprocess(sequence, features):
        sequence = sequence.drop(columns=["Date"]).astype("float32")
        # sequence = sequence.subtract(sequence.mean()).divide(sequence.std() + 1e-7)
        return torch.tensor(sequence[features].values)
