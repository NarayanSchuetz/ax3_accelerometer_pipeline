import abc


class DataProcessor(abc.ABC):

    def __init__(self, name=None):
        self._callbacks = []
        self._callback_arguments = []
        self._name = name

    def __str__(self):
        return self.name

    @property
    def name(self):
        return self.__class__.__name__ if self._name is None else self._name

    def add_callback(self, callback_function, arguments=()):
        if not isinstance(arguments, tuple):
            raise ValueError(
                "Provided callback argument/s must be a tuple \n(NOTE: a one element tuple is declare as ('element',) "
                "- notice the comma)")
        self._callbacks.append(callback_function)
        self._callback_arguments.append(arguments)
        return self

    @abc.abstractmethod
    def process(self, *data):
        raise NotImplementedError


def dataprocessor_hook(f):
        def wrapper(self, *args):
            assert isinstance(self, DataProcessor), "Behavior is not defined for instances that are not of type " \
                                                    "DataProcessor"
            r = f(self, *args)
            for cb_func, cb_args in zip(self._callbacks, self._callback_arguments):
                if len(cb_args) == 0:
                    cb_func(r)
                elif len(cb_args) == 1:
                    cb_func(r, cb_args[0])
                else:
                    cb_func(r, *cb_args)
        return wrapper
