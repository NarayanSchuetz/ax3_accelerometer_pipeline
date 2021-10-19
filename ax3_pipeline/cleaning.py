import warnings

import pandas as pd


def filter_noncompliant_weartime_days(acc_epoch_df, min_weartime_percentage=0.5, max_weartime_percentage=0.8,
                                      epoch_length_s=5):
    assert isinstance(acc_epoch_df.index, pd.DatetimeIndex), "Epoch DataFrame must have a DateTimeIndex"

    days = []
    for idx, day in acc_epoch_df.resample("d"):
        p_weartime = day.x_mean.count() * epoch_length_s / 86400
        if p_weartime >= min_weartime_percentage and p_weartime <= max_weartime_percentage:
            days.append(day)
    if not days:
        warnings.warn("No weartime compliant days found for participant df: %s" % str(acc_epoch_df.head(1)))
        return pd.DataFrame()

    return pd.concat(days)