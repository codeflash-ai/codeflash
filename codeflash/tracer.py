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
import os
import pickle
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import TYPE_CHECKING, Any

from codeflash.cli_cmds.cli import project_root_from_module_root
from codeflash.cli_cmds.console import console
from codeflash.code_utils.code_utils import get_run_tmp_file
from codeflash.code_utils.compat import SAFE_SYS_EXECUTABLE
from codeflash.code_utils.config_consts import EffortLevel
from codeflash.code_utils.config_parser import parse_config_file
from codeflash.tracing.pytest_parallelization import pytest_split

if TYPE_CHECKING:
    from argparse import Namespace


def detect_language_from_config(config: dict[str, Any]) -> str:
    """Detect the project language from config or file extensions.

    Args:
        config: Project configuration dictionary.

    Returns:
        Language identifier ('python', 'javascript', or 'typescript').

    """
    # Check explicit language in config
    if "language" in config:
        language: str = config["language"].lower()
        return language

    # Check module root for file types
    module_root = Path(config.get("module_root", "."))
    if module_root.exists():
        js_files = list(module_root.glob("**/*.js")) + list(module_root.glob("**/*.jsx"))
        ts_files = list(module_root.glob("**/*.ts")) + list(module_root.glob("**/*.tsx"))
        py_files = list(module_root.glob("**/*.py"))

        # Filter out node_modules
        js_files = [f for f in js_files if "node_modules" not in str(f)]
        ts_files = [f for f in ts_files if "node_modules" not in str(f)]

        total_js = len(js_files) + len(ts_files)
        total_py = len(py_files)

        if total_js > total_py:
            return "typescript" if len(ts_files) > len(js_files) else "javascript"

    return "python"


def main(args: Namespace | None = None) -> ArgumentParser:
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
    parser.add_argument(
        "--language",
        help="Language to trace (python, javascript, typescript). Auto-detected if not specified.",
        default=None,
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

    # Detect or use specified language
    language = getattr(parsed_args, "language", None) or detect_language_from_config(config)

    # Route to appropriate tracer based on language
    if language in ("javascript", "typescript"):
        if outfile is None:
            outfile = Path("codeflash.trace.sqlite")
        return run_javascript_tracer_main(parsed_args, config, project_root, outfile, unknown_args)

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
                split_result = pytest_split(unknown_args[1:], limit=parsed_args.limit)
                if split_result[0] is None or split_result[1] is None:
                    console.print(f"❌ Could not find test files in the specified paths: {unknown_args[1:]}")
                    console.print(f"Current working directory: {Path.cwd()}")
                    console.print("Please ensure the test directory exists and contains test files.")
                    sys.exit(1)
                pytest_splits, test_paths = split_result

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
                    env = os.environ.copy()
                    pythonpath = env.get("PYTHONPATH", "")
                    project_root_str = str(project_root)
                    if pythonpath:
                        env["PYTHONPATH"] = f"{project_root_str}{os.pathsep}{pythonpath}"
                    else:
                        env["PYTHONPATH"] = project_root_str
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

                env = os.environ.copy()
                # Add project root to PYTHONPATH so imports work correctly
                pythonpath = env.get("PYTHONPATH", "")
                project_root_str = str(project_root)
                if pythonpath:
                    env["PYTHONPATH"] = f"{project_root_str}{os.pathsep}{pythonpath}"
                else:
                    env["PYTHONPATH"] = project_root_str
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
                from codeflash.cli_cmds.cmd_init import CODEFLASH_LOGO
                from codeflash.cli_cmds.console import paneled_text
                from codeflash.languages import set_current_language
                from codeflash.languages.base import Language
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


def run_javascript_tracer_main(
    parsed_args: Namespace, config: dict[str, Any], project_root: Path, outfile: Path, unknown_args: list[str]
) -> ArgumentParser:
    """Run the JavaScript tracer.

    Args:
        parsed_args: Parsed command line arguments.
        config: Project configuration.
        project_root: Project root directory.
        outfile: Output trace file path.
        unknown_args: Remaining command line arguments.

    Returns:
        The argument parser.

    """
    from codeflash.languages.javascript.tracer_runner import (
        check_javascript_tracer_available,
        get_tracer_requirements_message,
        run_javascript_tracer,
    )

    # Check requirements
    if not check_javascript_tracer_available():
        console.print(f"[red]{get_tracer_requirements_message()}[/red]")
        sys.exit(1)

    # Prepare args for the tracer runner
    parsed_args.script_args = unknown_args

    # Run the tracer
    console.print("[bold blue]Running JavaScript tracer...[/bold blue]")
    result = run_javascript_tracer(parsed_args, config, project_root)

    if not result["success"]:
        console.print(f"[red]Tracing failed: {result.get('error', 'Unknown error')}[/red]")
        sys.exit(1)

    console.print(f"[green]Trace saved to: {result['trace_file']}[/green]")

    if result.get("replay_test_file"):
        console.print(f"[green]Replay test generated: {result['replay_test_file']}[/green]")

        # Run optimization if not trace-only mode
        if not parsed_args.trace_only:
            from codeflash.cli_cmds.cli import parse_args as cli_parse_args
            from codeflash.cli_cmds.cli import process_pyproject_config
            from codeflash.cli_cmds.cmd_init import CODEFLASH_LOGO
            from codeflash.cli_cmds.console import paneled_text
            from codeflash.languages import set_current_language
            from codeflash.languages.base import Language
            from codeflash.telemetry import posthog_cf
            from codeflash.telemetry.sentry import init_sentry

            # Set language to JavaScript
            set_current_language(Language.JAVASCRIPT)

            sys.argv = ["codeflash", "--replay-test", result["replay_test_file"]]
            args = cli_parse_args()
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

            # Clean up trace and replay test files
            if outfile:
                outfile.unlink(missing_ok=True)
            Path(result["replay_test_file"]).unlink(missing_ok=True)

    # Return a new parser for API compatibility
    return ArgumentParser(allow_abbrev=False)


if __name__ == "__main__":
    main()
