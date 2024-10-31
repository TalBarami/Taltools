from abc import ABC, abstractmethod
from typing import Any, Dict

from matplotlib.figure import Figure
from pandas import DataFrame


class BaseLogger(ABC):

    def __init__(self):
        super(BaseLogger, self).__init__()

    @abstractmethod
    def debug(self, msg: str):
        pass

    @abstractmethod
    def info(self, msg: str):
        pass

    @abstractmethod
    def warning(self, msg: str):
        pass

    @abstractmethod
    def error(self, msg: str):
        pass

    @abstractmethod
    def log(self, name: str, data: Any, step=None):
        pass

    @abstractmethod
    def log_dict(self, name: str, data: Dict[str, Any], step=None):
        pass

    @abstractmethod
    def log_table(self, name: str, data: DataFrame, step=None):
        pass

    @abstractmethod
    def log_file(self, name: str, file_path: str):
        pass

    @abstractmethod
    def plot(self, name: str, data: Figure, step=None):
        pass


class CompositeLogger(BaseLogger):
    def __init__(self, loggers):
        super(CompositeLogger, self).__init__()
        self.loggers = loggers

    def debug(self, msg: str):
        for logger in self.loggers:
            logger.debug(msg)

    def info(self, msg: str):
        for logger in self.loggers:
            logger.info(msg)

    def warning(self, msg: str):
        for logger in self.loggers:
            logger.warning(msg)

    def error(self, msg: str):
        for logger in self.loggers:
            logger.error(msg)

    def log(self, name: str, data: Any, step=None):
        for logger in self.loggers:
            logger.log(name, data, step)

    def log_dict(self, name: str, data: Dict[str, Any], step=None):
        for logger in self.loggers:
            logger.log_dict(name, data, step)

    def log_table(self, name: str, data: DataFrame, step=None):
        for logger in self.loggers:
            logger.log_table(name, data, step)

    def log_file(self, name: str, file_path: str):
        for logger in self.loggers:
            logger.log_file(name, file_path)

    def plot(self, name: str, data: Figure, step=None):
        for logger in self.loggers:
            logger.plot(name, data, step)