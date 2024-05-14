from pathlib import Path
from typing import Dict, Any
import yaml
import logging
from typing import Union


def update_dict_deep(original_dict = Dict[str, Any], update_dict = Dict[str, Any]) -> Dict[str, Any]:
    for key, value in update_dict.items():
        if isinstance(value, dict):
            original_dict[key] = update_dict_deep(original_dict.get(key, {}), value)
        else:
            original_dict[key] = value
    return original_dict


def update_config(config: Dict[str, Any] = None, config_new: Union[dict, Path] = None) -> Dict[str, Any]:
    # update config w/ new config
    if isinstance(config_new, Path):
        if config_new.exists():
            with open(config_new) as f:
                config = update_dict_deep(config, yaml.safe_load(f))
    elif isinstance(config_new, dict):
        config = update_dict_deep(config, config_new)
    else:
        raise TypeError("config_new must be a dict or a Path.")

    # data type repairs
    for key, value in config.get('paths', {}).items():
        if isinstance(value, str):
            config['paths'][key] = Path(value)
    for key, value in config.get('logging_level', {}).items():
        if isinstance(value, str):
            config['logging_level'][key] = getattr(logging, value)

    # return
    return config