from abc import abstractmethod
import typing

import torch


class Metric:
    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def __call__(self, output, sample):
        pass


class Loss(Metric):
    def __init__(self, loss: typing.Callable) -> None:
        super().__init__()
        self._loss = loss

    @property
    def name(self):
        return f"LossSeqence"

    def __call__(self, output, sample):
        return self._loss(torch.tensor(output), torch.tensor(sample))


class WholeSeqence(Metric):
    @property
    def name(self):
        return f"WholeSeqence"

    def __call__(self, output, sample):
        assert output.shape==sample.shape
        if not isinstance(output, torch.Tensor):
            output = torch.tensor(output)
        sample = torch.tensor(sample)
        return (
            torch.clamp(torch.abs(sample - output), max=1)
        ).mean()


class Seqence(Metric):
    def __init__(self, distance: int) -> None:
        super().__init__()
        self._distance = distance

    @property
    def name(self):
        return f"Seqence_{self._distance}"

    def __call__(self, output, sample):
        assert len(sample) == len(output)
        return abs(sample[self._distance] - output[self._distance]) / (
            abs(sample[self._distance]) + abs(output[self._distance])
        )
