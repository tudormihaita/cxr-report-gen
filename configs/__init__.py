import yaml

def load_config_from_file(config_path: str):
    config = yaml.load(open(config_path), Loader=yaml.FullLoader)
    if not isinstance(config, dict):
        raise ValueError(f"Config file {config_path} should contain a dictionary")
    return config