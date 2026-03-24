from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import tomlkit


class EffortLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class EffortKeys(str, Enum):
    N_OPTIMIZER_CANDIDATES = "N_OPTIMIZER_CANDIDATES"
    N_OPTIMIZER_LP_CANDIDATES = "N_OPTIMIZER_LP_CANDIDATES"
    N_GENERATED_TESTS = "N_GENERATED_TESTS"
    MAX_CODE_REPAIRS_PER_TRACE = "MAX_CODE_REPAIRS_PER_TRACE"
    REPAIR_UNMATCHED_PERCENTAGE_LIMIT = "REPAIR_UNMATCHED_PERCENTAGE_LIMIT"
    TOP_VALID_CANDIDATES_FOR_REFINEMENT = "TOP_VALID_CANDIDATES_FOR_REFINEMENT"
    ADAPTIVE_OPTIMIZATION_THRESHOLD = "ADAPTIVE_OPTIMIZATION_THRESHOLD"
    MAX_ADAPTIVE_OPTIMIZATIONS_PER_TRACE = "MAX_ADAPTIVE_OPTIMIZATIONS_PER_TRACE"


HIGH_EFFORT_TOP_N = 15
MAX_TEST_REPAIR_CYCLES = 2

EFFORT_VALUES: dict[str, dict[EffortLevel, Any]] = {
    EffortKeys.N_OPTIMIZER_CANDIDATES.value: {EffortLevel.LOW: 3, EffortLevel.MEDIUM: 5, EffortLevel.HIGH: 6},
    EffortKeys.N_OPTIMIZER_LP_CANDIDATES.value: {EffortLevel.LOW: 4, EffortLevel.MEDIUM: 6, EffortLevel.HIGH: 7},
    EffortKeys.N_GENERATED_TESTS.value: {EffortLevel.LOW: 2, EffortLevel.MEDIUM: 2, EffortLevel.HIGH: 2},
    EffortKeys.MAX_CODE_REPAIRS_PER_TRACE.value: {EffortLevel.LOW: 2, EffortLevel.MEDIUM: 3, EffortLevel.HIGH: 5},
    EffortKeys.REPAIR_UNMATCHED_PERCENTAGE_LIMIT.value: {
        EffortLevel.LOW: 0.2,
        EffortLevel.MEDIUM: 0.3,
        EffortLevel.HIGH: 0.4,
    },
    EffortKeys.TOP_VALID_CANDIDATES_FOR_REFINEMENT.value: {
        EffortLevel.LOW: 2,
        EffortLevel.MEDIUM: 3,
        EffortLevel.HIGH: 4,
    },
    EffortKeys.ADAPTIVE_OPTIMIZATION_THRESHOLD.value: {EffortLevel.LOW: 0, EffortLevel.MEDIUM: 0, EffortLevel.HIGH: 2},
    EffortKeys.MAX_ADAPTIVE_OPTIMIZATIONS_PER_TRACE.value: {
        EffortLevel.LOW: 0,
        EffortLevel.MEDIUM: 0,
        EffortLevel.HIGH: 4,
    },
}


def get_effort_value(key: EffortKeys, effort: EffortLevel | str) -> Any:
    """Look up an effort-dependent parameter value."""
    if isinstance(effort, str):
        effort = EffortLevel(effort)
    return EFFORT_VALUES[key.value][effort]


@dataclass
class TestConfig:
    tests_root: Path
    project_root: Path
    tests_project_rootdir: Path | None = None
    concolic_test_root_dir: Path | None = None
    test_command: str = "pytest"
    test_framework: str = "pytest"
    benchmark_tests_root: Path | None = None
    use_cache: bool = True
    timeout: float = 60.0
    js_project_root: Path | None = None

    def __post_init__(self) -> None:
        self.project_root = Path(self.project_root).resolve()
        if self.tests_project_rootdir is not None:
            self.tests_project_rootdir = Path(self.tests_project_rootdir).resolve()


@dataclass
class TelemetryConfig:
    enabled: bool = True
    posthog_api_key: str = ""
    sentry_dsn: str = ""


@dataclass
class AIConfig:
    base_url: str = "https://app.codeflash.ai"
    api_key: str = ""
    timeout: float = 120.0


CONFIG_FILE_NAMES = ("pyproject.toml", "codeflash.toml")

GLOB_PATTERN_CHARS = frozenset("*?[")


def is_glob_pattern(path_str: str) -> bool:
    """Check if a path string contains glob pattern characters."""
    return any(char in path_str for char in GLOB_PATTERN_CHARS)


def normalize_ignore_paths(paths: list[str], base_path: Path | None = None) -> list[Path]:
    if base_path is None:
        base_path = Path.cwd()

    base_path = base_path.resolve()
    normalized: set[Path] = set()

    for path_str in paths:
        if not path_str:
            continue

        path_str = str(path_str)

        if is_glob_pattern(path_str):
            path_str = path_str.removeprefix("./")
            if path_str.startswith("/"):
                path_str = path_str.lstrip("/")
            for matched_path in base_path.glob(path_str):
                normalized.add(matched_path.resolve())
        else:
            path_obj = Path(path_str)
            if not path_obj.is_absolute():
                path_obj = base_path / path_obj
            if path_obj.exists():
                normalized.add(path_obj.resolve())

    return list(normalized)


@dataclass
class CoreConfig:
    project_root: Path = field(default_factory=Path.cwd)
    module_root: str = ""
    tests_root: str = "tests"
    benchmarks_root: str = ""
    ignore_paths: list[str] = field(default_factory=list)
    formatter_cmds: list[str] = field(default_factory=list)
    disable_telemetry: bool = False
    effort: str = "medium"
    create_pr: bool = False
    pytest_cmd: str = "pytest"
    disable_imports_sorting: bool = False
    git_remote: str = "origin"
    override_fixtures: bool = False

    ai: AIConfig = field(default_factory=AIConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)

    @property
    def resolved_ignore_paths(self) -> list[Path]:
        base = (self.project_root / self.module_root) if self.module_root else self.project_root
        return normalize_ignore_paths(self.ignore_paths, base_path=base)

    @classmethod
    def from_toml(cls, path: Path) -> CoreConfig:
        """Load config from a toml file containing a [tool.codeflash] section.

        Works with both pyproject.toml and codeflash.toml.
        """
        with path.open(encoding="utf-8") as f:
            data = tomlkit.load(f)

        cf: dict[str, Any] = data.get("tool", {}).get("codeflash", {})
        if not cf:
            return cls(project_root=path.parent)

        config = cls(
            project_root=path.parent,
            module_root=cf.get("module-root", ""),
            tests_root=cf.get("tests-root", "tests"),
            benchmarks_root=cf.get("benchmarks-root", ""),
            ignore_paths=cf.get("ignore-paths", []),
            formatter_cmds=cf.get("formatter-cmds", []),
            disable_telemetry=cf.get("disable_telemetry", False),
            effort=cf.get("effort", "medium"),
            create_pr=cf.get("create-pr", False),
            pytest_cmd=cf.get("pytest-cmd", "pytest"),
            disable_imports_sorting=cf.get("disable-imports-sorting", False),
            git_remote=cf.get("git-remote", "origin"),
            override_fixtures=cf.get("override-fixtures", False),
        )

        if config.module_root and not (config.project_root / config.module_root).exists():
            msg = f"module-root '{config.module_root}' does not exist under {config.project_root}"
            raise FileNotFoundError(msg)

        return config

    @classmethod
    def find_and_load(cls, start: Path | None = None) -> CoreConfig:
        """Walk up from start (default cwd) looking for a config file.

        Searches for pyproject.toml and codeflash.toml in each directory
        up to the filesystem root. Returns a default config if nothing is found.
        """
        path = cls.find_config_file(start or Path.cwd())
        if path is None:
            return cls(project_root=start or Path.cwd())
        return cls.from_toml(path)

    @staticmethod
    def find_config_file(start: Path) -> Path | None:
        """Walk up directories looking for a config file with a [tool.codeflash] section."""
        current = start.resolve()
        while True:
            for name in CONFIG_FILE_NAMES:
                candidate = current / name
                if candidate.is_file():
                    try:
                        with candidate.open(encoding="utf-8") as f:
                            data = tomlkit.load(f)
                        if data.get("tool", {}).get("codeflash"):
                            return candidate
                    except Exception:
                        continue
            parent = current.parent
            if parent == current:
                break
            current = parent
        return None

    def resolve_test_config(self) -> TestConfig:
        tests_root = self.project_root / self.tests_root
        if not tests_root.is_dir():
            msg = f"tests-root '{self.tests_root}' does not exist under {self.project_root}"
            raise FileNotFoundError(msg)
        # Compute tests_project_rootdir by walking up from tests_root to find pyproject.toml
        # This is the pytest rootdir used for resolving test module paths
        tests_project_rootdir = self.find_tests_project_rootdir(tests_root)
        return TestConfig(
            tests_root=tests_root,
            project_root=self.project_root,
            tests_project_rootdir=tests_project_rootdir,
            test_command=self.pytest_cmd,
        )

    def find_tests_project_rootdir(self, tests_root: Path) -> Path:
        """Walk up from tests_root looking for a directory containing pyproject.toml or codeflash.toml."""
        current = tests_root.resolve()
        while current != current.parent:
            for name in CONFIG_FILE_NAMES:
                if (current / name).is_file():
                    return current
            current = current.parent
        return self.project_root
