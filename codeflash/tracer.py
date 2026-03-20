# Copyright 2024 CodeFlash Inc. All rights reserved.
#
# Licensed under the Business Source License version 1.1.
# License source can be found in the LICENSE file.
#
# This file includes derived work covered by the following copyright and permission notices:
#
#  Copyright Python Software Foundation
#  Licensed under the Apache License, Version 2.0 (the "License").
#  http://www.apache.org/licenses/LICENSE-2.0
#
from __future__ import annotations

import json
import logging
import pickle
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import TYPE_CHECKING

from codeflash.cli_cmds.cli import project_root_from_module_root
from codeflash.cli_cmds.console import console
from codeflash.code_utils.code_utils import get_run_tmp_file
from codeflash.code_utils.compat import SAFE_SYS_EXECUTABLE
from codeflash.code_utils.config_consts import EffortLevel
from codeflash.code_utils.config_parser import parse_config_file
from codeflash.code_utils.shell_utils import make_env_with_project_root
from codeflash.tracing.pytest_parallelization import pytest_split

if TYPE_CHECKING:
    from argparse import Namespace

    from codeflash.languages import Language

logger = logging.getLogger(__name__)


def _detect_non_python_language(args: Namespace | None) -> Language | None:
    """Detect if the project uses a non-Python language from --file or build files.

    Returns a Language enum value if non-Python detected, None otherwise.
    """
    from codeflash.languages import Language

    # Method 1: Check --file argument for non-Python file extension
    file_path_to_check: Path | None = None
    if args is not None and getattr(args, "file", None):
        file_path_to_check = Path(args.file)
    elif args is None and "--file" in sys.argv:
        try:
            file_idx = sys.argv.index("--file")
            if file_idx + 1 < len(sys.argv):
                file_path_to_check = Path(sys.argv[file_idx + 1])
        except (IndexError, ValueError):
            pass

    if file_path_to_check is not None and file_path_to_check.exists():
        try:
            from codeflash.languages import get_language_support

            lang_support = get_language_support(file_path_to_check)
            if lang_support.language != Language.PYTHON:
                return lang_support.language
        except Exception:
            pass

    # Method 2: Detect Java from build files (pom.xml / build.gradle)
    try:
        from codeflash.languages.java.build_tools import BuildTool, detect_build_tool

        cwd = Path.cwd()
        if detect_build_tool(cwd) != BuildTool.UNKNOWN:
            return Language.JAVA
    except Exception:
        pass

    # Method 3: Check config file for language field (JS/TS via package.json)
    try:
        from codeflash.code_utils.config_parser import parse_config_file

        config_file = getattr(args, "config_file_path", None) if args else None
        config, _ = parse_config_file(config_file)
        lang_str = config.get("language", "")
        if lang_str in ("javascript", "typescript"):
            return Language(lang_str)
    except Exception:
        pass

    return None


def main(args: Namespace | None = None) -> ArgumentParser:
    # For non-Python languages, detect early and route to the appropriate handler.
    # Java, JavaScript, and TypeScript use their own test runners (Maven/JUnit, Jest)
    # and should not go through Python tracing.
    #
    # Detection methods (in priority order):
    # 1. --file pointing to a .java/.js/.ts file
    # 2. language field in project config (codeflash.toml or pyproject.toml)
    detected_language = _detect_non_python_language(args)
    if detected_language is not None:
        from codeflash.languages import Language

        if detected_language in (Language.JAVASCRIPT, Language.TYPESCRIPT):
            from codeflash.cli_cmds.cli import parse_args, process_pyproject_config
            from codeflash.optimization import optimizer

            full_args = parse_args()
            full_args = process_pyproject_config(full_args)
            full_args.previous_checkpoint_functions = None
            logger.info("Detected %s project, routing to Optimizer instead of Python tracer", detected_language.value)
            optimizer.run_with_args(full_args)
            return ArgumentParser()

        if detected_language == Language.JAVA:
            return _run_java_tracer(args)

    parser = ArgumentParser(allow_abbrev=False)
    parser.add_argument("-o", "--outfile", dest="outfile", help="Save trace to <outfile>", default="codeflash.trace")
    parser.add_argument("--only-functions", help="Trace only these functions", nargs="+", default=None)
    parser.add_argument(
        "--max-function-count",
        help="Maximum number of inputs for one function to include in the trace.",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--tracer-timeout",
        help="Timeout in seconds for the tracer, if the traced code takes more than this time, then tracing stops and "
        "normal execution continues.",
        type=float,
        default=None,
    )
    parser.add_argument("-m", action="store_true", dest="module", help="Trace a library module", default=False)
    parser.add_argument(
        "--codeflash-config",
        help="Optional path to the project's pyproject.toml file "
        "with the codeflash config. Will be auto-discovered if not specified.",
        default=None,
    )
    parser.add_argument("--trace-only", action="store_true", help="Trace and create replay tests only, don't optimize")
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit the number of test files to process (for -m pytest mode)"
    )

    if args is not None:
        parsed_args = args
        parsed_args.outfile = getattr(args, "output", "codeflash.trace")
        parsed_args.only_functions = getattr(args, "only_functions", None)
        parsed_args.max_function_count = getattr(args, "max_function_count", 100)
        parsed_args.tracer_timeout = getattr(args, "timeout", None)
        parsed_args.codeflash_config = getattr(args, "config_file_path", None)
        parsed_args.trace_only = getattr(args, "trace_only", False)

        temp_parsed, unknown_args = parser.parse_known_args()
        parsed_args.module = temp_parsed.module
        sys.argv[:] = unknown_args

        if getattr(args, "disable", False):
            console.rule("Codeflash: Tracer disabled by --disable option", style="bold red")
            return parser

    else:
        if not sys.argv[1:]:
            parser.print_usage()
            sys.exit(2)

        parsed_args, unknown_args = parser.parse_known_args()
        sys.argv[:] = unknown_args

    # The script that we're profiling may chdir, so capture the absolute path
    # to the output file at startup.
    if parsed_args.outfile is not None:
        parsed_args.outfile = Path(parsed_args.outfile).resolve()
    outfile = parsed_args.outfile
    config, found_config_path = parse_config_file(parsed_args.codeflash_config)
    project_root = project_root_from_module_root(Path(config["module_root"]), found_config_path)
    if len(unknown_args) > 0:
        args_dict = {
            "functions": parsed_args.only_functions,
            "disable": False,
            "project_root": str(project_root),
            "max_function_count": parsed_args.max_function_count,
            "timeout": parsed_args.tracer_timeout,
            "progname": unknown_args[0],
            "config": config,
            "module": parsed_args.module,
        }
        try:
            pytest_splits: list[list[str]] = []
            test_paths: list[str] = []
            replay_test_paths: list[str] = []
            if parsed_args.module and unknown_args[0] == "pytest":
                result_splits, result_paths = pytest_split(unknown_args[1:], limit=parsed_args.limit)
                if result_splits is None or result_paths is None:
                    console.print(f"❌ Could not find test files in the specified paths: {unknown_args[1:]}")
                    console.print(f"Current working directory: {Path.cwd()}")
                    console.print("Please ensure the test directory exists and contains test files.")
                    sys.exit(1)
                pytest_splits = result_splits
                test_paths = result_paths

            if len(pytest_splits) > 1:
                processes = []
                test_paths_set = set(test_paths)
                result_pickle_file_paths = []
                for i, test_split in enumerate(pytest_splits, start=1):
                    result_pickle_file_path = get_run_tmp_file(Path(f"tracer_results_file_{i}.pkl"))
                    result_pickle_file_paths.append(result_pickle_file_path)
                    args_dict["result_pickle_file_path"] = str(result_pickle_file_path)
                    updated_sys_argv = []
                    for elem in sys.argv:
                        if elem in test_paths_set:
                            updated_sys_argv.extend(test_split)
                        else:
                            updated_sys_argv.append(elem)
                    args_dict["command"] = " ".join(updated_sys_argv)
                    env = make_env_with_project_root(project_root)
                    # Disable JIT compilation to ensure tracing captures all function calls
                    env["NUMBA_DISABLE_JIT"] = str(1)
                    env["TORCHDYNAMO_DISABLE"] = str(1)
                    env["PYTORCH_JIT"] = str(0)
                    env["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
                    env["TF_ENABLE_ONEDNN_OPTS"] = str(0)
                    env["JAX_DISABLE_JIT"] = str(1)
                    processes.append(
                        subprocess.Popen(
                            [
                                SAFE_SYS_EXECUTABLE,
                                Path(__file__).parent / "tracing" / "tracing_new_process.py",
                                *updated_sys_argv,
                                json.dumps(args_dict),
                            ],
                            cwd=Path.cwd(),
                            env=env,
                        )
                    )
                for process in processes:
                    process.wait()
                for result_pickle_file_path in result_pickle_file_paths:
                    try:
                        with result_pickle_file_path.open(mode="rb") as f:
                            data = pickle.load(f)
                            replay_test_paths.append(str(data["replay_test_file_path"]))
                    except Exception:
                        console.print("❌ Failed to trace. Exiting...")
                        sys.exit(1)
                    finally:
                        result_pickle_file_path.unlink(missing_ok=True)
            else:
                result_pickle_file_path = get_run_tmp_file(Path("tracer_results_file.pkl"))
                args_dict["result_pickle_file_path"] = str(result_pickle_file_path)
                args_dict["command"] = " ".join(sys.argv)

                env = make_env_with_project_root(project_root)
                # Disable JIT compilation to ensure tracing captures all function calls
                env["NUMBA_DISABLE_JIT"] = str(1)
                env["TORCHDYNAMO_DISABLE"] = str(1)
                env["PYTORCH_JIT"] = str(0)
                env["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
                env["TF_ENABLE_ONEDNN_OPTS"] = str(0)
                env["JAX_DISABLE_JIT"] = str(1)

                subprocess.run(
                    [
                        SAFE_SYS_EXECUTABLE,
                        Path(__file__).parent / "tracing" / "tracing_new_process.py",
                        *sys.argv,
                        json.dumps(args_dict),
                    ],
                    cwd=Path.cwd(),
                    env=env,
                    check=False,
                )
                try:
                    with result_pickle_file_path.open(mode="rb") as f:
                        data = pickle.load(f)
                        replay_test_paths.append(str(data["replay_test_file_path"]))
                except Exception:
                    console.print("❌ Failed to trace. Exiting...")
                    sys.exit(1)
                finally:
                    result_pickle_file_path.unlink(missing_ok=True)
            if not parsed_args.trace_only and replay_test_paths:
                from codeflash.cli_cmds.cli import parse_args, process_pyproject_config
                from codeflash.cli_cmds.console import paneled_text
                from codeflash.cli_cmds.console_constants import CODEFLASH_LOGO
                from codeflash.languages import Language, set_current_language
                from codeflash.telemetry import posthog_cf
                from codeflash.telemetry.sentry import init_sentry

                # Set the language to Python since the tracer is Python-specific
                set_current_language(Language.PYTHON)

                sys.argv = ["codeflash", "--replay-test", *replay_test_paths]
                args = parse_args()
                paneled_text(
                    CODEFLASH_LOGO,
                    panel_args={"title": "https://codeflash.ai", "expand": False},
                    text_args={"style": "bold gold3"},
                )

                args = process_pyproject_config(args)
                args.previous_checkpoint_functions = None
                init_sentry(enabled=not args.disable_telemetry, exclude_errors=True)
                posthog_cf.initialize_posthog(enabled=not args.disable_telemetry)

                from codeflash.optimization import optimizer

                args.effort = EffortLevel.HIGH.value
                optimizer.run_with_args(args)

                # Delete the trace file and the replay test file if they exist
                if outfile:
                    outfile.unlink(missing_ok=True)
                for replay_test_path in replay_test_paths:
                    Path(replay_test_path).unlink(missing_ok=True)

        except BrokenPipeError as exc:
            # Prevent "Exception ignored" during interpreter shutdown.
            sys.stdout = None
            sys.exit(exc.errno)
    else:
        parser.print_usage()
    return parser


def _run_java_tracer(existing_args: Namespace | None = None) -> ArgumentParser:
    """Run the Java two-stage tracer (JFR + argument capture) and optionally optimize."""
    from codeflash.cli_cmds.cli import parse_args, process_pyproject_config

    if existing_args is not None:
        full_args = process_pyproject_config(existing_args)
    else:
        full_args = parse_args()
        full_args = process_pyproject_config(full_args)
    config = full_args

    trace_only = getattr(config, "trace_only", False)
    project_root = Path(getattr(config, "project_root", ".")).resolve()
    module_root = Path(getattr(config, "module_root", project_root)).resolve()
    max_function_count = getattr(config, "max_function_count", 256)
    timeout = int(getattr(config, "timeout", None) or getattr(config, "tracer_timeout", 0) or 0)

    console.print("[bold]Java project detected[/]")
    console.print(f"  Project root: {project_root}")
    console.print(f"  Module root:  {getattr(config, 'module_root', '?')}")
    console.print(f"  Tests root:   {getattr(config, 'tests_root', '?')}")

    from codeflash.code_utils.code_utils import get_run_tmp_file
    from codeflash.languages.java.tracer import JavaTracer, run_java_tracer

    tracer = JavaTracer()
    packages = tracer.detect_packages_from_source(module_root)
    if not packages:
        logger.warning("No Java packages detected in %s, will trace all non-JDK classes", module_root)

    trace_db_path = get_run_tmp_file(Path("java_trace.db"))

    # Place replay tests in the project's test source tree so Maven/Gradle can compile them.
    # Use the config's tests_root (correctly resolved for multi-module projects) not find_test_root().
    tests_root = Path(getattr(config, "tests_root", ""))
    if tests_root.is_dir():
        output_dir = tests_root / "codeflash" / "replay"
    else:
        from codeflash.languages.java.build_tools import find_test_root

        test_root = find_test_root(project_root)
        output_dir = (test_root or project_root / "src" / "test" / "java") / "codeflash" / "replay"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Remaining args after our flags are the Java command
    remaining = sys.argv[sys.argv.index("--file") + 2 :] if "--file" in sys.argv else sys.argv[1:]
    if not remaining:
        console.print("[bold red]Error:[/] No Java command provided.")
        console.print("Usage: codeflash optimize java -jar target/my-app.jar [args...]")
        console.print("       codeflash optimize java -cp target/classes com.example.Main [args...]")
        sys.exit(1)
    java_command = remaining

    trace_db, jfr_file, test_count = run_java_tracer(
        java_command=java_command,
        trace_db_path=trace_db_path,
        packages=packages,
        project_root=project_root,
        output_dir=output_dir,
        max_function_count=max_function_count,
        timeout=timeout,
    )

    console.print(f"[bold green]Java tracing complete:[/] {test_count} replay test files generated")
    if jfr_file.exists():
        console.print(f"  JFR profile: {jfr_file}")
    if trace_db.exists():
        console.print(f"  Trace DB: {trace_db}")

    if not trace_only and test_count > 0:
        from codeflash.code_utils.config_consts import EffortLevel
        from codeflash.languages import Language, set_current_language
        from codeflash.optimization import optimizer

        set_current_language(Language.JAVA)

        replay_test_paths = [p.resolve() for p in output_dir.glob("*.java")]
        config.replay_test = replay_test_paths
        config.previous_checkpoint_functions = None
        config.effort = EffortLevel.HIGH.value
        config.no_pr = True
        config.file = None
        config.function = None
        config.test_project_root = project_root
        optimizer.run_with_args(config)

        # Clean up generated replay tests
        for replay_test_path in replay_test_paths:
            Path(replay_test_path).unlink(missing_ok=True)
        # Clean up codeflash/replay directory if empty
        if output_dir.exists() and not any(output_dir.iterdir()):
            output_dir.rmdir()
            codeflash_dir = output_dir.parent
            if codeflash_dir.exists() and codeflash_dir.name == "codeflash" and not any(codeflash_dir.iterdir()):
                codeflash_dir.rmdir()

    return ArgumentParser()


if __name__ == "__main__":
    main()
