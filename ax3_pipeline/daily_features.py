"""
Extract daily aggregations from axivity epochs DataFrame.
"""


def mean_daily_enmo(acc_epoch_df):
    s = acc_epoch_df.mean_enmo.resample("d").mean()
    s.name = "mean_daily_enmo"
    return s


def sum_daily_enmo(acc_epoch_df):
    s = acc_epoch_df.mean_enmo.resample("d").sum()
    s.name = "sum_daily_enmo"
    return s[s > 0]


def mean_daily_std_sum(acc_epoch_df):
    sum_std = acc_epoch_df.x_std + acc_epoch_df.y_std + acc_epoch_df.z_std
    sum_std.name = "mean_daily_std_sum"
    return sum_std.resample("d").mean()


def mean_daily_signal_energy(acc_epoch_df):
    s = acc_epoch_df.total_energy.resample("d").mean()
    s.name = "mean_daily_signal_energy"
    return s


def sum_daily_signal_energy(acc_epoch_df):
    s = acc_epoch_df.total_energy.resample("d").sum()
    s.name = "sum_daily_signal_energy"
    return s[s > 0]


def mean_daily_velocity(acc_epoch_df):
    s = acc_epoch_df.mean_velocity.resample("d").mean()
    s.name = "mean_daily_velocity"
    return s


def sum_daily_distance(acc_epoch_df):
    s = acc_epoch_df.approximate_distance.resample("d").sum()
    s.name = "total_daily_distance"
    return s[s > 0]


def relative_daily_sedentary(acc_epoch_df):
    n_sed = acc_epoch_df.mean_enmo[acc_epoch_df.mean_enmo < 0.03].resample("d").count()
    n_tot = acc_epoch_df.mean_enmo.resample("d").count()
    s = n_sed / n_tot
    s.name = "relative_daily_sedentary"
    return s.dropna()


def relative_daily_lpa(acc_epoch_df):
    n_light = acc_epoch_df.mean_enmo[(acc_epoch_df.mean_enmo >= 0.03) &
                                   (acc_epoch_df.mean_enmo < 0.1)].resample("d").count()
    n_tot = acc_epoch_df.mean_enmo.resample("d").count()
    s = n_light / n_tot
    s.name = "relative_daily_lpa"
    return s.dropna()


def relative_daily_mvpa(acc_epoch_df):
    n_mvpa = acc_epoch_df.mean_enmo[acc_epoch_df.mean_enmo >= 0.1].resample("d").count()
    n_tot = acc_epoch_df.mean_enmo.resample("d").count()
    s = n_mvpa / n_tot
    s.name = "relative_daily_mvpa"
    return s.dropna()


def mean_daily_spectral_entropy(acc_epoch_df):
    s = acc_epoch_df.spectral_entropy.resample("d").mean()
    s.name = "mean_daily_spectral_entropy"
    return s


def enmo_25_quantile(acc_epoch_df):
    s = acc_epoch_df.mean_enmo.resample("d").quantile(0.25)
    s.name = "enmo_25_quantile"
    return s

def enmo_50_quantile(acc_epoch_df):
    s = acc_epoch_df.mean_enmo.resample("d").quantile(0.5)
    s.name = "enmo_50_quantile"
    return s


def enmo_75_quantile(acc_epoch_df):
    s = acc_epoch_df.mean_enmo.resample("d").quantile(0.75)
    s.name = "enmo_75_quantile"
    return s