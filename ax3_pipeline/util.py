from typing import Dict, Tuple, Sequence

import pandas as pd
from scipy.stats import linregress


def plot_timeseries_trend(timeseries: pd.Series, color="black", suffix="_trend", **kwargs):
    """
    Plots the trend of a DateTimeIndexed timeseries based on a OLS fitted linear regression.
    :param timeseries: pandas Series with DateTimeIndex
    :param color: color of the trend line (default: black)
    :param suffix: the suffix to be appended to the input series name (useful for legends)
    :param kwargs: keyword arguments to be provided to the pandas series plot method
    :return: matplotlib axes object
    """
    timeseries = timeseries.copy().sort_index().dropna()
    timestamps = [int(i.timestamp()) for i in timeseries.index]

    slope, intercept, _, _, _ = linregress(timestamps, timeseries)

    y_start = intercept + slope*timestamps[0]
    y_stop = intercept + slope*timestamps[-1]

    s_new = pd.Series([y_start, y_stop], index=[timeseries.index[0], timeseries.index[-1]])
    s_new.name = timeseries.name + suffix
    
    
    return s_new.plot(color=color, **kwargs)


def sortidx_rmna_rmdup_df(df: pd.DataFrame, inplace=True):
    """
    Sorts DataFrame by index, removes NaNs and duplicates.
    :param df: input DataFrame
    :param inplace: whether to do the operations in place or on a copy of the DataFrame
    :return:
    """
    if inplace:
        df.dropna(inplace=True)
        df.sort_index(inplace=True)
        df.drop_duplicates(inplace=True)
        return df

    return df.dropna().sort_index().drop_duplicates()


def merge_intervals_with_short_interruptions(timestamp_intervals: Sequence[Tuple[pd.Timestamp, pd.Timestamp]],
                                             max_interruption_duration: int) -> Dict[Tuple[pd.Timestamp, pd.Timestamp],
                                                                                     None]:
    """
    Based on a sequence of intervals (defined by timestamp start and stop tuples) merges time-intervals with short
    interruptions in-between. What is considered as a short interruption is defined by the max_interruption_duration.
    intervals (NOTE: after investigating on multiple persons it seems like these are often caused by external
    influences e.g. if someone puts something on a desk where the accelerometer is placed.
    :param timestamp_intervals:
    :param max_interruption_duration: maximum time duration for interruptions to be removed (seconds).
    :return:
    """
    valid_intervals = {}
    nw_0 = timestamp_intervals[0]

    for nw_1 in timestamp_intervals[1:]:
        t0 = nw_0[1]
        t1 = nw_1[0]

        delta = t1.timestamp() - t0.timestamp()
        if delta < max_interruption_duration:
            valid_intervals[(nw_0[0], nw_1[1])] = None
        else:
            valid_intervals[nw_0] = None
            valid_intervals[nw_1] = None

        nw_0 = nw_1
    return valid_intervals


def get_participant_subset(df: pd.DataFrame) -> Dict[object, pd.DataFrame]:
    """
    In most of our datasets the 'participant_id' column identifies the respective anonymized participants, this utility
    function returns a dict with individual DataFrame subsets.
    :param df: DataFrame with 'participant_id' column
    :return: dictionary where participant_id is the key and respective DataFrame subset the value
    """
    return {participant: df[df.participant_id == participant]
            for participant in set(df.participant_id)}
