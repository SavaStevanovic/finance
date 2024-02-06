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
    ):
        self._sequence_length = sequence_length + 1
        self._features = features
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
        sequence = self.preprocess(sequence, self._features)
        # sequence_mean = self.get_running_mean(sequence)
        # inp = torch.tensor(list(range(50)) + list(range(50, 0, -1))).unsqueeze(1)
        # print([round(x, 2) for x in self.get_running_mean(inp).squeeze().tolist()])
        # print([round(x, 2) for x in self.get_rsi(inp).squeeze().tolist()])
        # sequence_rsi = self.get_rsi(sequence)
        orig_sequence = torch.tensor(sequence.numpy())
        sequence = orig_sequence
        sequence = (sequence[1:] - sequence[:-1]) / (sequence[1:] + 1e-7)
        # sequence = sequence.nan_to_num(0)
        return (sequence[:-1], sequence[1:], orig_sequence[1:-1])

    def get_running_mean(self, sequence):
        channels = [sequence[:, i].numpy() for i in range(0, sequence.shape[1])]
        seq_len = 14
        seq = (
            np.concatenate((np.zeros(seq_len - 1), np.ones(seq_len)), axis=0) / seq_len
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

        return sequence_mean

    def get_rsi(self, sequence):
        channels = [
            sequence[1:, i].numpy() - sequence[:-1, i].numpy()
            for i in range(0, sequence.shape[1])
        ]
        seq_len = 14
        seq = (
            np.concatenate((np.zeros(seq_len - 1), np.ones(seq_len)), axis=0) / seq_len
        )

        channels = [np.stack([c * (c > 0), -c * (c < 0)]) for c in channels]
        channels = [
            np.stack(
                [
                    np.convolve(
                        np.pad(
                            c,
                            (len(seq) // 2, len(seq) // 2),
                            mode="edge",
                        ),
                        seq,
                        mode="valid",
                    )
                    for c in subc
                ]
            )
            for subc in channels
        ]
        channels = [c[0] / (c[1] + 1e-7) for c in channels]
        channels = [100 - (100 / (1 + c)) for c in channels]

        channels = np.stack(
            channels,
            axis=1,
        ).astype("float32")
        channels = np.pad(channels, [(0, 1), (0, 0)], mode="edge")

        return channels

    @staticmethod
    def preprocess(sequence, features):
        sequence = sequence.drop(columns=["Date"]).astype("float32")
        # sequence = sequence.subtract(sequence.mean()).divide(sequence.std() + 1e-7)
        return torch.tensor(sequence[features].values)


class TimeSeriesDatasetInference(torch.utils.data.Dataset):
    def __init__(
        self,
        groups: typing.Dict[str, pd.DataFrame],
        sequence_length: int,
        features: list,
    ):
        self._dataset = TimeSeriesDataset(
            {k: v.iloc[-sequence_length - 2 :] for k, v in groups.items()},
            sequence_length,
            features,
        )

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        ticker, _ = self._dataset._ids[idx]
        return ticker, self._dataset[idx][1], self._dataset[idx][2]
