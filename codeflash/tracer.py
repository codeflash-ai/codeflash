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
from codeflash.code_utils.config_parser import parse_config_file

if TYPE_CHECKING:
    from argparse import Namespace


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

    if args is not None:
        parsed_args = args
        parsed_args.outfile = getattr(args, "output", "codeflash.trace")
        parsed_args.only_functions = getattr(args, "only_functions", None)
        parsed_args.max_function_count = getattr(args, "max_function_count", 100)
        parsed_args.tracer_timeout = getattr(args, "timeout", None)
        parsed_args.codeflash_config = getattr(args, "config_file_path", None)
        parsed_args.trace_only = getattr(args, "trace_only", False)
        parsed_args.module = False

        if getattr(args, "disable", False):
            console.rule("Codeflash: Tracer disabled by --disable option", style="bold red")
            return parser

        unknown_args = []
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
        try:
            result_pickle_file_path = get_run_tmp_file("tracer_results_file.pkl")
            args_dict = {
                "result_pickle_file_path": str(result_pickle_file_path),
                "output": str(parsed_args.outfile),
                "functions": parsed_args.only_functions,
                "disable": False,
                "project_root": str(project_root),
                "max_function_count": parsed_args.max_function_count,
                "timeout": parsed_args.tracer_timeout,
                "command": " ".join(sys.argv),
                "progname": unknown_args[0],
                "config": config,
                "module": parsed_args.module,
            }

            subprocess.run(
                [
                    SAFE_SYS_EXECUTABLE,
                    Path(__file__).parent / "tracing" / "tracing_new_process.py",
                    *sys.argv,
                    json.dumps(args_dict),
                ],
                cwd=Path.cwd(),
                check=False,
            )
            try:
                with result_pickle_file_path.open(mode="rb") as f:
                    data = pickle.load(f)
            except Exception:
                console.info("❌ Failed to trace. Exiting...")
                sys.exit(1)
            finally:
                result_pickle_file_path.unlink(missing_ok=True)

            replay_test_path = data["replay_test_file_path"]
            if not parsed_args.trace_only and replay_test_path is not None:
                from codeflash.cli_cmds.cli import parse_args, process_pyproject_config
                from codeflash.cli_cmds.cmd_init import CODEFLASH_LOGO
                from codeflash.cli_cmds.console import paneled_text
                from codeflash.telemetry import posthog_cf
                from codeflash.telemetry.sentry import init_sentry

                sys.argv = ["codeflash", "--replay-test", str(replay_test_path)]

                args = parse_args()
                paneled_text(
                    CODEFLASH_LOGO,
                    panel_args={"title": "https://codeflash.ai", "expand": False},
                    text_args={"style": "bold gold3"},
                )

                args = process_pyproject_config(args)
                args.previous_checkpoint_functions = None
                init_sentry(not args.disable_telemetry, exclude_errors=True)
                posthog_cf.initialize_posthog(not args.disable_telemetry)

                from codeflash.optimization import optimizer

                optimizer.run_with_args(args)

                # Delete the trace file and the replay test file if they exist
                if outfile:
                    outfile.unlink(missing_ok=True)
                if replay_test_path:
                    replay_test_path.unlink(missing_ok=True)

        except BrokenPipeError as exc:
            # Prevent "Exception ignored" during interpreter shutdown.
            sys.stdout = None
            sys.exit(exc.errno)
    else:
        parser.print_usage()
    return parser


if __name__ == "__main__":
    main()
