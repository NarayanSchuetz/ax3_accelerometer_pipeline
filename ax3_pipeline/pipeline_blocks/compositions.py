from collections import OrderedDict

from .base import *


class DataProcessorComposition(DataProcessor):
    """
    If viewed as a graph, compositions represent the directed edges between executing parts (nodes).
    """

    def __init__(self, *data_processors, name=None):
        super().__init__(name)
        self._processors = [*data_processors]

    @property
    def processors(self):
        return OrderedDict(("%d_%s" % (i, p.name), p) for i, p in enumerate(self._processors))

    @abc.abstractmethod
    def process(self, *data):
        raise NotImplementedError()


class ParallelComposition(DataProcessorComposition):

    def process(self, *data):
        r = []
        for dp in self._processors:
            r.extend([*dp.process(*data)])
        return tuple(r)


class SequentialComposition(DataProcessorComposition):

    def process(self, *data):
        n_processors = len(self._processors)
        assert n_processors > 0, "at least one processor must be provided"

        if n_processors == 1:
            return self._processors[0].process(*data)
        else:
            tmp_data = self._processors[0].process(*data)
            for p in self._processors[1:]:
                tmp_data = p.process(*tmp_data)
            return tmp_data
