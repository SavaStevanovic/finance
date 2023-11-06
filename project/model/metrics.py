from abc import abstractmethod


class Metric:
    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def __call__(self, sample, output):
        pass


class Seqence(Metric):
    def __init__(self, distance: int) -> None:
        super().__init__()
        self._distance = distance

    @property
    def name(self):
        return f"Seqence_{self._distance}"

    def __call__(self, sample, output):
        return abs(sample[self._distance] - output[self._distance])
