import yaml
from pathlib import Path
from typing import Union

def load_config_from_file(config_path: Union[str, Path]) -> dict:
    config = yaml.load(open(config_path), Loader=yaml.FullLoader)
    if not isinstance(config, dict):
        raise ValueError(f"Config file {config_path} should contain a dictionary")
    return config