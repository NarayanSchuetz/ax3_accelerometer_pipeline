from .base import *
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Sequence

from ..util import merge_intervals_with_short_interruptions


class FeatureConcat(DataProcessor):

    def __init__(self, epoch_timestamps_ms, name=None):
        super().__init__(name)
        self._timestamps = epoch_timestamps_ms

    def process(self, *data: pd.Series):
        df = pd.concat([*data], axis=1)
        df.index = pd.to_datetime(self._timestamps, unit="ms")
        return (df,)


class NonWeartimeCalculator(DataProcessor):

    def __init__(
            self,
            duration_s=3600,
            non_wear_threshold_mG=13/1000,
            min_non_wear_interruption_s=3600,
            remove=True,
            name=None):
        """
        :param duration_s: minimum duration (in seconds) of non-wear interval
        :param non_wear_threshold_mG: milli-G threshold
        :param min_non_wear_interruption_s: the minimum wear time (in seconds) between non-wear segments.
        :param remove: whether to remove non-wear segments or just set them to NaN
        :param name:
        """
        super().__init__(name)
        self._duration = duration_s
        self._non_wear_threshold_mG = non_wear_threshold_mG
        self._min_non_wear_interruption = min_non_wear_interruption_s
        self._remove = remove

    def process(self, *data: pd.DataFrame) -> Tuple[pd.DataFrame]:
        """
        Calculate epochs as wear or non-wear based on a predefined mG threshold. If the SD in none of the three axes
        exceeds the threshold for the duration it is labelled as non-wear.
        :param data:
        :return:
        """
        assert len(data) == 1, "Expected exactly one epoch DataFrame"
        assert isinstance(data[0], pd.DataFrame), "Expected type pandas DataFrame"
        assert isinstance(data[0].index, pd.DatetimeIndex), "Expected datetime indexed DataFrame"

        data = data[0].copy()
        self.lable_non_weartime(data, self._duration, self._non_wear_threshold_mG, self._min_non_wear_interruption)
        return (data.dropna().drop(columns=["nw"]),) if self._remove else (data,)

    @staticmethod
    def lable_non_weartime(e, min_duration=3600, max_std=13/1000, min_between_time_s=3600):
        e['nw'] = np.where((e['x_std'] < max_std) & (e['y_std'] < max_std) &
                           (e['z_std'] < max_std), True, False)
        print(e.nw.head())
        starts = e.index[(e['nw']) & (~e['nw'].shift(1).fillna(False))]
        ends = e.index[(e['nw']) & (~e['nw'].shift(-1).fillna(False))]

        non_wear_intervals = [(start, end) for start, end in zip(starts, ends)
                              if end > start + np.timedelta64(min_duration, 's')]

        if not non_wear_intervals:
            return

        valid_intervals = NonWeartimeCalculator._remove_short_interruptions(non_wear_intervals, min_between_time_s)

        for episode in valid_intervals:
            e.nw[episode[0]:episode[1]] = np.nan

    @staticmethod
    def _remove_short_interruptions(non_wear_intervals: Sequence[Tuple[pd.Timestamp, pd.Timestamp]],
                                    min_weartime_time: int) -> Dict[Tuple[pd.Timestamp, pd.Timestamp], None]:
        """
        Basically merges non-wear intervals interleaved with short wear-time segments to remove too short wear-time
        intervals (NOTE: after investigating on multiple persons it seems like these are often caused by external
        influences e.g. if someone puts something on a desk where the accelerometer is placed.
        :param non_wear_intervals:
        :param min_weartime_time: minimal weartime (in seconds) for an interleaved non-wear semgent to be left unchanged.
        :return:
        """
        merge_intervals_with_short_interruptions(non_wear_intervals, min_weartime_time)
