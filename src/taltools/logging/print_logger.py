import logging
from os import path as osp
from typing import Any, Dict

from matplotlib.figure import Figure
from pandas import DataFrame
from tabulate import tabulate

from taltools.io.files import init_directories
from taltools.logging.base_logger import BaseLogger


class PrintLogger(BaseLogger):
    def __init__(self, name: str, show=False):
        super().__init__()
        self.name = name
        self.show = show
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        sh = logging.StreamHandler()
        sh.setLevel(logging.DEBUG)
        sh.setFormatter(self.formatter)
        self.logger.addHandler(sh)

    def debug(self, msg: str):
        self.logger.debug(msg)

    def info(self, msg: str):
        self.logger.info(msg)

    def warning(self, msg: str):
        self.logger.warning(msg)

    def error(self, msg: str):
        self.logger.error(msg)

    def log(self, name: str, data: Any, step=None):
        self.logger.info(f'{name}: {data}')

    def log_dict(self, name: str, data: Dict[str, Any], step=None):
        self.log(name, data, step)

    def log_table(self, name: str, data: DataFrame, step=None):
        self.logger.info(f'{name}:\n{tabulate(data, headers="keys", tablefmt="psql")}')

    def log_file(self, name: str, file_path: str):
        with open(file_path, 'r') as f:
            self.logger.info(f'{name}: {f.read()}')

    def plot(self, name: str, fig: Figure, step=None):
        if self.show:
            fig.show()


class FileLogger(PrintLogger):
    def __init__(self, name, log_path, show=False):
        super().__init__(name, show)
        self.path = log_path
        init_directories(self.path)
        fh = logging.FileHandler(osp.join(self.path, f'{self.name}.log'))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(self.formatter)
        self.logger.addHandler(fh)

    def plot(self, name: str, fig: Figure, step=None):
        super().plot(name, fig)
        _name = name.replace('/', '_').replace('\\', '_')
        fig.savefig(osp.join(self.path, f'{step}_{_name}.png'), dpi=300)