from .base import *
import pandas as pd
import numpy as np

"""
If viewed as a graph features represent the sinks (although there are post-processors that may invalidate
this view)
"""


class FeatureExtractor(DataProcessor):

    def __init__(self, feature_names, name=None):
        super().__init__(name)
        self._feature_names = feature_names
        self._axes = ["x_", "y_", "z_"]

    @abc.abstractmethod
    def process(self, *data):
        raise NotImplementedError


class ApproximateDistance(FeatureExtractor):

    def __init__(self, name=None):
        super().__init__(feature_names="approximate_distance", name=name)
        self.si_unit = "m"

    def process(self, *data):
        axis = [""] if len(data) == 1 else self._axes
        return tuple([pd.Series(d, name=axis[i] + self._feature_names) for i, d in enumerate(data)])


class TimeDomainSummaryStatistics(FeatureExtractor):

    def __init__(self, feature_names=None, name=None):
        super().__init__(feature_names=(
            "mean",
            "median",
            "std",
            "min",
            "max",
            "q_5",
            "q_95"
        ) if feature_names is None else feature_names, name=name)

    def process(self, *data):
        r = []
        data_size = len(data)
        for i, d in enumerate(data):
            assert len(
                d.shape) == 2, "data is supposed to be a 2d array where rows correspond to epochs and columns to the " \
                               "values of a given epoch"
            axis = "" if data_size == 1 else self._axes[i]
            features = [
                pd.Series(np.mean(d, axis=1), name=axis + self._feature_names[0]),
                pd.Series(np.median(d, axis=1), name=axis + self._feature_names[1]),
                pd.Series(np.std(d, axis=1), name=axis + self._feature_names[2]),
                pd.Series(np.min(d, axis=1), name=axis + self._feature_names[3]),
                pd.Series(np.max(d, axis=1), name=axis + self._feature_names[4]),
                pd.Series(np.quantile(d, 0.05, axis=1).astype(np.float32), name=axis + self._feature_names[5]),
                pd.Series(np.quantile(d, 0.95, axis=1).astype(np.float32), name=axis + self._feature_names[6])
            ]
            r.extend(features)
        return tuple(r)


class EnmoSummaryStatistics(TimeDomainSummaryStatistics):

    def __init__(self, name=None):
        super().__init__(feature_names=(
            "mean_enmo",
            "median_enmo",
            "std_enmo",
            "min_enmo",
            "max_enmo",
            "q_5_enmo",
            "q_95_enmo"
        ), name=name)


class ApproximateVelocity(TimeDomainSummaryStatistics):

    def __init__(self, name=None):
        super().__init__(feature_names=(
            "mean_velocity",
            "median_velocity",
            "std_velocity",
            "min_velocity",
            "max_velocity",
            "q_5_velocity",
            "q_95_velocity"
        ), name=name)

        self.si_unit = "m/s"


class TotalEnergy(FeatureExtractor):

    def __init__(self, name=None):
        super().__init__(feature_names="total_energy", name=name)

    def process(self, *data):
        axis = [""] if len(data) == 1 else self._axes
        return tuple([pd.Series(np.sum(np.power(np.abs(d), 2), axis=1)/d.shape[1], name=axis[i] + self._feature_names)
                      for i, d in enumerate(data)])


class SpectralEntropy(FeatureExtractor):

    def __init__(self, name=None):
        super().__init__(feature_names="spectral_entropy", name=name)

    def process(self, *data):
        r = []
        axis = [""] if len(data) == 1 else self._axes
        for i, d in enumerate(data):
            assert len(d.shape) == 2, "Operations are only defined on an Epoch 2d array"
            amp = (np.power(np.abs(d), 2) / d.shape[1]) + 0.000000000000000000000001
            scaled_amp = amp / np.sum(amp, axis=1).reshape(-1, 1)
            entropy = -1*np.sum(scaled_amp*np.log(scaled_amp), axis=1)
            r.append(pd.Series(entropy, name=axis[i] + self._feature_names))
        return tuple(r)


class ActivityClasses(FeatureExtractor):

    def __init__(self, moderate_to_vigorous_pa_threshold_g=68.7/1000, vigorous_pa_threshold_g=266.8/1000, name=None):
        """
        Default thresholds were set according to 'Age Group Comparability of Raw Accelerometer Output from Wrist- and
        Hip-Worn Monitors' by Hildebrand et al. 2014

        :param moderate_to_vigorous_pa_threshold_g: (in g) values below will be classified as sedentary
        :param vigorous_pa_threshold_g: (in g) values below will be classified as moderate-to-vigorous
        """
        super().__init__(feature_names=["sedentary", "moderate", "vigorous"], name=name)
        self._moderate_to_vigorous_pa_threshold_g = moderate_to_vigorous_pa_threshold_g
        self._vigorous_pa_threshold_g = vigorous_pa_threshold_g

    def process(self, *data):
        r = []
        axis = [""] if len(data) == 1 else self._axes
        for i, d in enumerate(data):
            means = np.mean(d, axis=1)
            sedentary = pd.Series((means < self._moderate_to_vigorous_pa_threshold_g) * 1,
                                  name=axis[i]+self._feature_names[0])
            moderate = pd.Series(((means >= self._moderate_to_vigorous_pa_threshold_g) &
                                  (means < self._vigorous_pa_threshold_g)) * 1,
                                 name=axis[i]+self._feature_names[1])
            vigorous = pd.Series((means >= self._vigorous_pa_threshold_g) * 1,
                                 name=axis[i]+self._feature_names[2])
            r.extend([sedentary.astype(np.int32), moderate.astype(np.int32), vigorous.astype(np.int32)])
        return tuple(r)
