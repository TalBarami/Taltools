from os import path as osp
from pathlib import Path
from typing import Any, Dict

import neptune
from matplotlib.figure import Figure
from neptune.types import File
from pandas import DataFrame

from taltools.logging.base_logger import BaseLogger


class NeptuneLogger(BaseLogger):
    def __init__(self, project, tags=None, capture_stdout=False, capture_stderr=False):
        super(NeptuneLogger, self).__init__()
        api_token = Path.home().joinpath('.neptune', 'token.txt')
        api_token = api_token.read_text().strip()
        if isinstance(tags, str):
            tags = [tags]
        self.run = neptune.init_run(
            project=project,
            api_token=api_token,
            tags=list(tags) if tags is not None else None,
            capture_stdout=capture_stdout,
            capture_stderr=capture_stderr
        )

    def debug(self, msg: str, *args, **kwargs):
        pass

    def info(self, msg: str, *args, **kwargs):
        pass

    def warning(self, msg: str, *args, **kwargs):
        pass

    def error(self, msg: str, *args, **kwargs):
        pass

    def log(self, name: str, data, step=None):
        self.run[name].log(data, step=step)

    def log_dict(self, name: str, data: Dict[str, Any], step=None):
        for k, v in data.items():
            if isinstance(v, (dict, list)):
                v = str(v)
            self.log(f'{name}/{k}', v, step)

    def log_table(self, name: str, data: DataFrame, step=None):
        self.run[f'{name}/{step}'].upload(File.as_html(data))

    def log_file(self, name: str, file_path: str):
        self.run[name].upload(osp.join(file_path))

    def plot(self, name: str, fig: Figure, step=None):
        self.run[name].log(fig)