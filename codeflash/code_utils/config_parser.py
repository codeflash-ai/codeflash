from __future__ import annotations

from pathlib import Path
from typing import Any


def find_pyproject_toml(config_file: Path | None = None) -> Path:
    # Find the pyproject.toml file on the root of the project

    if config_file is not None:
        config_file = Path(config_file)
        if config_file.suffix.lower() != ".toml":
            msg = f"Config file {config_file} is not a valid toml file. Please recheck the path to pyproject.toml"
            raise ValueError(msg)
        if not config_file.exists():
            msg = f"Config file {config_file} does not exist. Please recheck the path to pyproject.toml"
            raise ValueError(msg)
        return config_file
    dir_path = Path.cwd()

    while dir_path != dir_path.parent:
        config_file = dir_path / "pyproject.toml"
        if config_file.exists():
            return config_file
        # Search for pyproject.toml in the parent directories
        dir_path = dir_path.parent
    msg = f"Could not find pyproject.toml in the current directory {Path.cwd()} or any of the parent directories. Please create it by running `poetry init`, or pass the path to pyproject.toml with the --config-file argument."

    raise ValueError(msg)


def parse_config_file(
    config_file_path: Path | None = None, override_formatter_check: bool = False
) -> tuple[dict[str, Any], Path]:
    import tomlkit

    from codeflash.code_utils.config_parser import find_pyproject_toml

    config_file_path = find_pyproject_toml(config_file_path)
    try:
        with config_file_path.open("rb") as f:
            data = tomlkit.parse(f.read())
    except tomlkit.exceptions.ParseError as e:
        raise ValueError(
            f"Error while parsing the config file {config_file_path}. Please recheck the file for syntax errors. Error: {e}"
        ) from e

    try:
        config = data["tool"]["codeflash"]
    except KeyError as e:
        raise ValueError(
            f"Could not find the 'codeflash' block in the config file {config_file_path}. Please run 'codeflash init' to create the config file."
        ) from e

    # default values:
    path_keys = ["module-root", "tests-root"]
    path_list_keys = ["ignore-paths"]
    str_keys = {"pytest-cmd": "pytest", "git-remote": "origin"}
    bool_keys = {"disable-telemetry": False, "disable-imports-sorting": False}
    list_str_keys = {"formatter-cmds": ["black $file"]}

    config = {**{k: str_keys[k] for k in str_keys if k not in config}, **config}
    config = {**{k: bool_keys[k] for k in bool_keys if k not in config}, **config}
    config = {**config, **{k: list_str_keys[k] for k in list_str_keys if k not in config}}

    for key in path_keys:
        if key in config:
            config[key] = str((Path(config_file_path).parent / Path(config[key])).resolve())
    for key in path_list_keys:
        if key in config:
            config[key] = [str((Path(config_file_path).parent / path).resolve()) for path in config[key]]

    assert config["test-framework"] in ["pytest", "unittest"], (
        "In pyproject.toml, Codeflash only supports the 'test-framework' as pytest and unittest."
    )
    if config["formatter-cmds"] and not override_formatter_check:
        assert config["formatter-cmds"][0] != "your-formatter $file", (
            "The formatter command is not set correctly in pyproject.toml. Please set the "
            "formatter command in the 'formatter-cmds' key. More info - https://docs.codeflash.ai/configuration"
        )
    config = {k.replace("-", "_"): v for k, v in config.items()}

    return config, config_file_path
