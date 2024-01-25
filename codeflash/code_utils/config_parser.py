import os
import sys
from typing import List

from codeflash.code_utils.config_consts import MIN_IMPROVEMENT_THRESHOLD

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


def supported_config_keys() -> List[str]:
    return ["test-framework", "tests-root", "root"]


def find_pyproject_toml(config_file=None):
    # Find the pyproject.toml file on the root of the project

    if config_file is not None:
        if not config_file.lower().endswith(".toml"):
            raise ValueError(
                f"Config file {config_file} is not a valid toml file. Please recheck the path to pyproject.toml"
            )
        if not os.path.exists(config_file):
            raise ValueError(
                f"Config file {config_file} does not exist. Please recheck the path to pyproject.toml"
            )
        return config_file

    else:
        dir_path = os.getcwd()

        while not os.path.dirname(dir_path) == dir_path:
            config_file = os.path.join(dir_path, "pyproject.toml")
            if os.path.exists(config_file):
                return config_file
            # Search for pyproject.toml in the parent directories
            dir_path = os.path.dirname(dir_path)
        raise ValueError(
            f"Could not find pyproject.toml in the current directory {os.getcwd()} or any of the parent directories. Please pass the path to pyproject.toml with --config-file argument"
        )


def parse_config_file(config_file_path=None):
    config_file = find_pyproject_toml(config_file_path)
    try:
        with open(config_file, "rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ValueError(
            f"Error while parsing the config file {config_file}. Please recheck the file for syntax errors. Error: {e}"
        )
    tool = data["tool"]
    assert isinstance(tool, dict)
    config = tool["codeflash"]
    # todo nice error message whe ncodeflash block is missing
    assert isinstance(config, dict)
    path_keys = ["root", "test-root"]
    path_list_keys = ["ignore-paths"]
    # TODO: minimum-peformance-gain should become a more dynamic auto-detection in the future
    float_keys = {
        "minimum-performance-gain": MIN_IMPROVEMENT_THRESHOLD,
    }  # the value is the default value
    str_keys = {
        "pytest-cmd": "pytest",
    }

    for key in float_keys:
        if key in config:
            config[key] = float(config[key])
        else:
            config[key] = float_keys[key]
    for key in str_keys:
        if key in config:
            config[key] = str(config[key])
        else:
            config[key] = str_keys[key]
    for key in path_keys:
        if key in config:
            config[key] = os.path.join(os.path.dirname(config_file), config[key])

    for key in path_list_keys:
        if key in config:
            config[key] = [os.path.join(os.path.dirname(config_file), path) for path in config[key]]
        else:  # Default to empty list
            config[key] = []
    assert config["test-framework"] in [
        "pytest",
        "unittest",
    ], "CodeFlash only supports pytest and unittest in pyproject.toml"
    for key in list(config.keys()):
        if "-" in key:
            config[key.replace("-", "_")] = config[key]
            del config[key]

    return config
