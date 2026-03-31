from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomlkit

from codeflash.code_utils.config_js import find_package_json, parse_package_json_config
from codeflash.languages.language_enum import Language
from codeflash.lsp.helpers import is_LSP_enabled

logger = logging.getLogger("codeflash")

PYPROJECT_TOML_CACHE: dict[Path, Path] = {}
ALL_CONFIG_FILES: dict[Path, dict[str, Path]] = {}


@dataclass
class LanguageConfig:
    config: dict[str, Any]
    config_path: Path
    language: Language


def _try_parse_java_build_config() -> tuple[dict[str, Any], Path] | None:
    """Detect Java project from build files and parse config from pom.xml/gradle.properties.

    Returns (config_dict, project_root) if a Java project is found, None otherwise.
    """
    dir_path = Path.cwd()
    while dir_path != dir_path.parent:
        if (
            (dir_path / "pom.xml").exists()
            or (dir_path / "build.gradle").exists()
            or (dir_path / "build.gradle.kts").exists()
        ):
            from codeflash.languages.java.build_config_strategy import parse_java_project_config

            config = parse_java_project_config(dir_path)
            if config is not None:
                return config, dir_path
        dir_path = dir_path.parent
    return None


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
    cur_path = dir_path
    # see if it was encountered before in search
    if cur_path in PYPROJECT_TOML_CACHE:
        return PYPROJECT_TOML_CACHE[cur_path]
    while dir_path != dir_path.parent:
        config_file = dir_path / "pyproject.toml"
        if config_file.exists():
            PYPROJECT_TOML_CACHE[cur_path] = config_file
            return config_file
        dir_path = dir_path.parent
    msg = f"Could not find pyproject.toml in the current directory {Path.cwd()} or any of the parent directories. Please create it by running `codeflash init`, or pass the path to the config file with the --config-file argument."

    raise ValueError(msg) from None


def get_all_closest_config_files() -> list[Path]:
    all_closest_config_files = []
    for file_type in ["pyproject.toml", "pytest.ini", ".pytest.ini", "tox.ini", "setup.cfg"]:
        closest_config_file = find_closest_config_file(file_type)
        if closest_config_file:
            all_closest_config_files.append(closest_config_file)
    return all_closest_config_files


def find_closest_config_file(file_type: str) -> Path | None:
    # Find the closest pyproject.toml, pytest.ini, tox.ini, or setup.cfg file on the root of the project
    dir_path = Path.cwd()
    cur_path = dir_path
    if cur_path in ALL_CONFIG_FILES and file_type in ALL_CONFIG_FILES[cur_path]:
        return ALL_CONFIG_FILES[cur_path][file_type]
    while dir_path != dir_path.parent:
        config_file = dir_path / file_type
        if config_file.exists():
            if cur_path not in ALL_CONFIG_FILES:
                ALL_CONFIG_FILES[cur_path] = {}
            ALL_CONFIG_FILES[cur_path][file_type] = config_file
            return config_file
        # Search for pyproject.toml in the parent directories
        dir_path = dir_path.parent
    return None


def find_conftest_files(test_paths: list[Path]) -> list[Path]:
    list_of_conftest_files = set()
    for test_path in test_paths:
        # Find the conftest file on the root of the project
        dir_path = Path.cwd()
        cur_path = test_path
        while cur_path != dir_path:
            config_file = cur_path / "conftest.py"
            if config_file.exists():
                list_of_conftest_files.add(config_file)
            # Search for conftest.py in the parent directories
            cur_path = cur_path.parent
    return list(list_of_conftest_files)


def normalize_toml_config(config: dict[str, Any], config_file_path: Path) -> dict[str, Any]:
    path_keys = ["module-root", "tests-root", "benchmarks-root"]
    path_list_keys = ["ignore-paths"]
    str_keys = {"pytest-cmd": "pytest", "git-remote": "origin"}
    bool_keys = {
        "override-fixtures": False,
        "disable-telemetry": False,
        "disable-imports-sorting": False,
        "benchmark": False,
    }
    list_str_keys = {"formatter-cmds": []}

    for key, default_value in str_keys.items():
        if key in config:
            config[key] = str(config[key])
        else:
            config[key] = default_value
    for key, default_value in bool_keys.items():
        if key in config:
            config[key] = bool(config[key])
        else:
            config[key] = default_value
    for key in path_keys:
        if key in config:
            config[key] = str((config_file_path.parent / Path(config[key])).resolve())
    for key, default_value in list_str_keys.items():
        if key in config:
            config[key] = [str(cmd) for cmd in config[key]]
        else:
            config[key] = default_value
    for key in path_list_keys:
        if key in config:
            config[key] = [str((config_file_path.parent / path).resolve()) for path in config[key]]
        else:
            config[key] = []

    # Convert hyphenated keys to underscored keys
    for key in list(config.keys()):
        if "-" in key:
            config[key.replace("-", "_")] = config[key]
            del config[key]

    return config


def _parse_java_config_for_dir(dir_path: Path) -> dict[str, Any] | None:
    from codeflash.languages.java.build_config_strategy import parse_java_project_config

    return parse_java_project_config(dir_path)


_SUBDIR_SKIP = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        ".venv",
        "venv",
        "__pycache__",
        "target",
        "build",
        "dist",
        ".tox",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
    }
)


def _check_dir_for_configs(dir_path: Path, configs: list[LanguageConfig], seen_languages: set[Language]) -> None:
    if Language.PYTHON not in seen_languages:
        pyproject = dir_path / "pyproject.toml"
        if pyproject.exists():
            try:
                with pyproject.open("rb") as f:
                    data = tomlkit.parse(f.read())
                tool = data.get("tool", {})
                if isinstance(tool, dict) and "codeflash" in tool:
                    raw_config = dict(tool["codeflash"])
                    normalized = normalize_toml_config(raw_config, pyproject)
                    seen_languages.add(Language.PYTHON)
                    configs.append(LanguageConfig(config=normalized, config_path=pyproject, language=Language.PYTHON))
            except Exception:
                logger.debug("Failed to parse Python config in %s", dir_path, exc_info=True)

    if Language.JAVASCRIPT not in seen_languages and Language.TYPESCRIPT not in seen_languages:
        package_json = dir_path / "package.json"
        if package_json.exists():
            try:
                result = parse_package_json_config(package_json)
                if result is not None:
                    config, path = result
                    lang = Language(config.get("language", "javascript"))
                    seen_languages.add(lang)
                    configs.append(LanguageConfig(config=config, config_path=path, language=lang))
            except Exception:
                logger.debug("Failed to parse JS/TS config in %s", dir_path, exc_info=True)

    if Language.JAVA not in seen_languages:
        if (
            (dir_path / "pom.xml").exists()
            or (dir_path / "build.gradle").exists()
            or (dir_path / "build.gradle.kts").exists()
        ):
            try:
                java_config = _parse_java_config_for_dir(dir_path)
                if java_config is not None:
                    seen_languages.add(Language.JAVA)
                    configs.append(LanguageConfig(config=java_config, config_path=dir_path, language=Language.JAVA))
            except Exception:
                logger.debug("Failed to parse Java config in %s", dir_path, exc_info=True)


def find_all_config_files(start_dir: Path | None = None) -> list[LanguageConfig]:
    if start_dir is None:
        start_dir = Path.cwd()

    configs: list[LanguageConfig] = []
    seen_languages: set[Language] = set()

    # Walk upward from start_dir to filesystem root (closest config wins per language)
    dir_path = start_dir.resolve()
    while True:
        _check_dir_for_configs(dir_path, configs, seen_languages)

        parent = dir_path.parent
        if parent == dir_path:
            break
        dir_path = parent

    # Scan immediate subdirectories for monorepo language subprojects
    resolved_start = start_dir.resolve()
    try:
        subdirs = sorted(p for p in resolved_start.iterdir() if p.is_dir() and p.name not in _SUBDIR_SKIP)
    except OSError:
        subdirs = []
    for subdir in subdirs:
        _check_dir_for_configs(subdir, configs, seen_languages)

    return configs


def parse_config_file(
    config_file_path: Path | None = None, override_formatter_check: bool = False
) -> tuple[dict[str, Any], Path]:
    # Detect all config sources — Java build files, package.json, pyproject.toml
    java_result = _try_parse_java_build_config() if config_file_path is None else None
    package_json_path = find_package_json(config_file_path)
    pyproject_toml_path = find_closest_config_file("pyproject.toml") if config_file_path is None else None

    # Use Java config only if no closer JS/Python config exists (monorepo support).
    # In a monorepo with a parent pom.xml and a child package.json, the closer config wins.
    if java_result is not None:
        java_depth = len(java_result[1].parts)
        has_closer = (package_json_path is not None and len(package_json_path.parent.parts) >= java_depth) or (
            pyproject_toml_path is not None and len(pyproject_toml_path.parent.parts) >= java_depth
        )
        if not has_closer:
            return java_result

    # When both config files exist, prefer the one closer to CWD.
    # This prevents a parent-directory package.json (e.g., monorepo root)
    # from overriding a closer pyproject.toml.
    use_package_json = False
    if package_json_path:
        if pyproject_toml_path is None:
            use_package_json = True
        else:
            package_json_depth = len(package_json_path.parent.parts)
            toml_depth = len(pyproject_toml_path.parent.parts)
            use_package_json = package_json_depth >= toml_depth

    if use_package_json:
        assert package_json_path is not None
        result = parse_package_json_config(package_json_path)
        if result is not None:
            config, path = result
            # Validate formatter if needed
            if not override_formatter_check and config.get("formatter_cmds"):
                formatter_cmds = config.get("formatter_cmds", [])
                if formatter_cmds and formatter_cmds[0] == "your-formatter $file":
                    raise ValueError(
                        "The formatter command is not set correctly in package.json. Please set the "
                        "formatter command in the 'formatterCmds' key."
                    )
            return config, path

    # Fall back to pyproject.toml
    config_file_path = find_pyproject_toml(config_file_path)
    try:
        with config_file_path.open("rb") as f:
            data = tomlkit.parse(f.read())
    except tomlkit.exceptions.ParseError as e:
        msg = f"Error while parsing the config file {config_file_path}. Please recheck the file for syntax errors. Error: {e}"
        raise ValueError(msg) from None

    lsp_mode = is_LSP_enabled()

    try:
        tool = data["tool"]
        assert isinstance(tool, dict)
        config = tool["codeflash"]
    except tomlkit.exceptions.NonExistentKey as e:
        if lsp_mode:
            # don't fail in lsp mode if codeflash config is not found.
            return {}, config_file_path
        msg = f"Could not find the 'codeflash' block in the config file {config_file_path}. Please run 'codeflash init' to add Codeflash config."
        raise ValueError(msg) from e
    assert isinstance(config, dict)

    if config == {} and lsp_mode:
        return {}, config_file_path

    config = normalize_toml_config(config, config_file_path)

    # see if this is happening during GitHub actions setup
    if config.get("formatter_cmds") and len(config.get("formatter_cmds")) > 0 and not override_formatter_check:
        assert config.get("formatter_cmds")[0] != "your-formatter $file", (
            "The formatter command is not set correctly in pyproject.toml. Please set the "
            "formatter command in the 'formatter-cmds' key. More info - https://docs.codeflash.ai/configuration"
        )

    return config, config_file_path
