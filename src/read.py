import os
import sys
import yaml

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)

from config import *


class GoConfig:
    def __init__(self) -> None:
        for k, v in pre_prosessing_config.items():
            setattr(self, k, v)
        for k, v in training_config.items():
            setattr(self, k, v)
        for k, v in mel_specrogram_config.items():
            setattr(self, k, v)


class GoModelInfo:
    def __init__(self, model_path) -> None:
        self.model_path = model_path
        self._read_info()

    @property
    def info_path(self):
        return os.path.join(os.path.dirname(self.model_path), "model_info.yaml")

    def _read_info(self):
        with open(self.info_path, "r") as f:
            info = yaml.load(f, Loader=yaml.Loader)
        for k, v in info.items():
            setattr(self, k, v)
