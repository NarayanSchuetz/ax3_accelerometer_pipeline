from scipy import signal
import numpy as np
import pandas as pd

from .base import *


class BaseButterworthFilter(DataProcessor):

    def __init__(self, order, sampling_frequency, cutoff_frequency, btype):
        super().__init__()
        self.order = order
        self.sampling_frequency = sampling_frequency
        self.cutoff_frequency = cutoff_frequency

        nyquist_frequency = 0.5 * sampling_frequency
        self._fc = cutoff_frequency / nyquist_frequency

        self._b, self._a = signal.butter(self.order, self._fc, btype=btype, analog=False)

    def process(self, *data: np.array):
        r = [pd.Series(signal.filtfilt(self._b, self._a, d), dtype=np.float32) for d in data if len(d) > 0]
        assert len(r) == len(data), "number of input does not match number of output arrays, maybe one array was empty"
        return tuple(r)


class LowpassButterworthFilter(BaseButterworthFilter):

    def __init__(self, order=4, sampling_frequency=100, cutoff_frequency=20):
        super().__init__(order, sampling_frequency, cutoff_frequency, "low")


class HighpassButterworthFilter(BaseButterworthFilter):

    def __init__(self, order=2, sampling_frequency=100, cutoff_frequency=0.1):
        super().__init__(order, sampling_frequency, cutoff_frequency, "high")
