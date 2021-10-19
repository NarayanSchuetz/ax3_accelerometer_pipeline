import warnings

from scipy.fftpack import fft
from scipy.integrate import cumtrapz, trapz
import numpy as np
import pandas as pd
from typing import Sequence

from .base import *


class TrapezoidalIntegrator(DataProcessor):

    def __init__(self, first_value=0, time_delta_seconds=0.01, cumulative=True):
        super().__init__()
        self._first_value = first_value
        self._time_delta_s = time_delta_seconds
        self._cumulative = cumulative

    def process(self, *data):
        if self._cumulative:
            return tuple([cumtrapz(d, initial=self._first_value, dx=self._time_delta_s) for d in data])
        else:
            return tuple([trapz(d, dx=self._time_delta_s) for d in data])


class EpochGenerator(DataProcessor):

    def __init__(self, timestamps_ms=None, n_samples_per_epoch=500, name=None):
        """
        If a timestamp sequence is provided it will check whether it is continuous (no gaps inside an epoch) and remove
        and epochs with gaps.
        :param timestamps_ms: POSIX timestamps with millisecond precision
        :param n_samples_per_epoch: the number of samples in each epoch (defaults to 500=5s)
        :param name:
        """
        super().__init__(name)
        if timestamps_ms is None:
            self._timestamps = None
        else:
            if isinstance(timestamps_ms, pd.Series):
                timestamps_ms = timestamps_ms.to_numpy()
            if not isinstance(timestamps_ms, np.ndarray):
                raise TypeError(
                    "timestamp_ms must be either a numpy array or pandas Series, got: %s" % type(timestamps_ms))

            self._delta_t = int(np.median(np.diff(timestamps_ms)))
            timestamps_ms = timestamps_ms.reshape(-1, n_samples_per_epoch)
            self._bad_epoch_indices = np.where(np.diff(timestamps_ms) != self._delta_t)[0]

            if len(self._bad_epoch_indices > 0):
                warnings.warn(
                    "Found %d bad epochs where delta_t != estimated delta_t of %d ms. Bad epochs will be removed!" %
                    (len(self._bad_epoch_indices), self._delta_t))
                timestamps_ms = np.delete(timestamps_ms, self._bad_epoch_indices, axis=0)

            self._timestamps = timestamps_ms

        self._n_samples_per_epoch = n_samples_per_epoch

    @property
    def timestamps(self):
        """
        :return: epoch start timestamps if a timestamp sequence was provided
        """
        if self._timestamps is None:
            raise ValueError("No timestamp sequence provided")
        return self._timestamps[:, 0]

    def process(self, *data: Sequence):
        """
        :param data: sequences of floating point values. Should be either np.array or pd.Series
        :return: reshaped sequences into 2d arrays where rows correspond to epochs and columns to the respective values
        """
        r = []
        for d in data:
            if isinstance(d, pd.Series):
                d = d.to_numpy()
            d = d.reshape(-1, self._n_samples_per_epoch)
            if self._timestamps is not None:
                d = np.delete(d, self._bad_epoch_indices, axis=0)

            r.append(d.reshape(-1, self._n_samples_per_epoch))
        return tuple(r)


class Dft1d(DataProcessor):

    def __init__(self, name=None):
        super().__init__(name)

    def process(self, *data):
        return tuple([fft(d) for d in data])


class VectorMagnitude(DataProcessor):

    def process(self, *data):
        if len(data) != 3:
            warnings.warn("Expected three arrays corresponding to x, y, z axis. Result might not be as expected!")
            # the generic calculation is at least an order of magnitude slower
            return self._calc_generic_vector_norm(data)
        else:
            squared = self._square(data)
            return (np.sqrt(squared[0] + squared[1] + squared[2]),)

    def _calc_generic_vector_norm(self, data):
        return (np.sqrt(np.sum(self._square(data), axis=0)),)

    def _square(self, data):
        return np.array([np.power(d, 2) for d in data])


class EuclideanNormMinusOne(VectorMagnitude):

    def process(self, *data):
        norm = super().process(*data)[0]
        norm_minus_one = norm - 1
        norm_minus_one[norm_minus_one < 0] = 0
        return (norm_minus_one,)
