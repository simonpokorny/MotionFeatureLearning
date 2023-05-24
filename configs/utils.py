import os
from pathlib import Path
from typing import Union

import yaml

def load_config(path: str):
    with open(path, "r") as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return cfg


if __name__ == "__main__":

    # cfg_dataset = load_config("datasets/rawkitti.yaml")

    a = None
