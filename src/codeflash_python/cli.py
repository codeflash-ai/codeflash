"""CLI module for codeflash_python - minimal stub for test support."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from argparse import Namespace


def process_pyproject_config(args: Namespace) -> Namespace:
    """Process pyproject.toml config and populate args.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Updated args namespace with config values.

    """
    try:
        from codeflash_python.code_utils.config_parser import parse_config_file
    except ImportError:
        # Minimal fallback if config_parser not available
        pyproject_config = {}
        if hasattr(args, "config_file") and args.config_file:
            try:
                import tomlkit

                with Path(args.config_file).open() as f:
                    data = tomlkit.load(f)
                    pyproject_config = data.get("tool", {}).get("codeflash", {})
            except Exception:
                pass
    else:
        try:
            pyproject_config, _ = parse_config_file(args.config_file)
        except (ValueError, FileNotFoundError):
            pyproject_config = {}

    supported_keys = [
        "module_root",
        "tests_root",
        "benchmarks_root",
        "ignore_paths",
        "pytest_cmd",
        "formatter_cmds",
        "disable_telemetry",
        "disable_imports_sorting",
        "git_remote",
        "override_fixtures",
    ]

    for key in supported_keys:
        normalized_key = key.replace("-", "_")
        if key in pyproject_config and (
            (hasattr(args, normalized_key) and getattr(args, normalized_key) is None)
            or not hasattr(args, normalized_key)
        ):
            setattr(args, normalized_key, pyproject_config[key])

    # Set defaults
    if not hasattr(args, "module_root") or args.module_root is None:
        args.module_root = str(Path.cwd())

    if not hasattr(args, "tests_root") or args.tests_root is None:
        args.tests_root = str(Path.cwd() / "tests")

    pyproject_file_path = Path(args.config_file) if hasattr(args, "config_file") and args.config_file else None

    if not hasattr(args, "project_root") or args.project_root is None:
        args.project_root = str(project_root_from_module_root(Path(args.module_root), pyproject_file_path))

    if not hasattr(args, "test_project_root") or args.test_project_root is None:
        args.test_project_root = str(project_root_from_module_root(Path(args.tests_root), pyproject_file_path))

    return args


def project_root_from_module_root(module_root: Path, pyproject_file_path: Path | None) -> Path:
    """Find the project root by walking up from module_root."""
    module_root = module_root.resolve()
    if pyproject_file_path is not None and pyproject_file_path.parent.resolve() == module_root:
        return module_root

    current = module_root
    while current != current.parent:
        if (current / "codeflash.toml").exists():
            return current
        current = current.parent

    return module_root.parent.resolve()


def handle_show_config() -> None:
    """Show current or auto-detected Codeflash configuration."""
    from codeflash_python.setup.detector import detect_project, has_existing_config

    project_root = Path.cwd()
    config_exists, _ = has_existing_config(project_root)

    if config_exists:
        from codeflash_python.code_utils.config_parser import parse_config_file

        config, config_file_path = parse_config_file()
        status = "Saved config"

        print()
        print(f"Codeflash Configuration ({status})")
        print(f"Config file: {config_file_path}")
        print()

        print(f"  {'Setting':<20} {'Value'}")
        print(f"  {'-' * 20} {'-' * 40}")
        print(f"  {'Project root':<20} {project_root}")
        print(f"  {'Module root':<20} {config.get('module_root', '(not set)')}")
        print(f"  {'Tests root':<20} {config.get('tests_root', '(not set)')}")
        print(f"  {'Test runner':<20} {config.get('test_framework', config.get('pytest_cmd', '(not set)'))}")
        print(
            f"  {'Formatter':<20} {', '.join(config['formatter_cmds']) if config.get('formatter_cmds') else '(not set)'}"
        )
        ignore_paths = config.get("ignore_paths", [])
        print(f"  {'Ignore paths':<20} {', '.join(str(p) for p in ignore_paths) if ignore_paths else '(none)'}")
    else:
        detected = detect_project(project_root)
        status = "Auto-detected (not saved)"

        print()
        print(f"Codeflash Configuration ({status})")
        print()

        print(f"  {'Setting':<20} {'Value'}")
        print(f"  {'-' * 20} {'-' * 40}")
        print(f"  {'Language':<20} {detected.language}")
        print(f"  {'Project root':<20} {detected.project_root}")
        print(f"  {'Module root':<20} {detected.module_root}")
        print(f"  {'Tests root':<20} {detected.tests_root if detected.tests_root else '(not detected)'}")
        print(f"  {'Test runner':<20} {detected.test_runner or '(not detected)'}")
        print(
            f"  {'Formatter':<20} {', '.join(detected.formatter_cmds) if detected.formatter_cmds else '(not detected)'}"
        )
        print(
            f"  {'Ignore paths':<20} {', '.join(str(p) for p in detected.ignore_paths) if detected.ignore_paths else '(none)'}"
        )
        print(f"  {'Confidence':<20} {detected.confidence:.0%}")

    print()

    if not config_exists:
        print("Run codeflash --file <file> to auto-save this config.")


def handle_reset_config(confirm: bool = True) -> None:
    """Remove Codeflash configuration from project config file."""
    from codeflash_python.setup.config_writer import remove_config
    from codeflash_python.setup.detector import detect_project, has_existing_config

    project_root = Path.cwd()

    config_exists, _ = has_existing_config(project_root)
    if not config_exists:
        print("No Codeflash configuration found to remove.")
        return

    detected = detect_project(project_root)

    if confirm:
        print("This will remove Codeflash configuration from your project.")
        print()

        config_file = "pyproject.toml"
        print(f"  Config file: {project_root / config_file}")
        print()

        try:
            response = input("Are you sure you want to remove the config? [y/N] ")
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled.")
            return

        if response.lower() not in ("y", "yes"):
            print("Cancelled.")
            return

    result = remove_config(project_root)

    if result.is_ok():
        print(f"Done: {result.unwrap()}")
    else:
        print(f"Failed: {result.error}")  # type: ignore[attr-defined]
