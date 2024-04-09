import logging
import os
import pathlib
import pickle
import sqlite3
import sys
import time
from collections import defaultdict
from typing import Any, Optional, List

import isort

from codeflash.cli_cmds.cli import project_root_from_module_root
from codeflash.code_utils.code_utils import module_name_from_file_path
from codeflash.code_utils.config_parser import parse_config_file
from codeflash.discovery.functions_to_optimize import filter_functions, FunctionToOptimize
from codeflash.tracing.replay_test import create_trace_replay_test
from codeflash.tracing.tracing_utils import FunctionModules
from codeflash.verification.verification_utils import get_test_file_path


class Tracer:
    """
    Use this class as a 'with' context manager to trace a function call,
    input arguments, and return value.
    """

    def __init__(
        self,
        output: str = "script.trace",
        functions=None,
        disable: bool = False,
        config_file_path: Optional[str] = None,
    ) -> None:
        if functions is None:
            functions = []
        self.disable = disable
        self.con = None
        self.flag = (
            False  # To ignore the first call to trace_callback due to return from trace_callback
        )
        self.output_file = os.path.abspath(output)
        self.functions = functions
        self.function_modules: List[FunctionModules] = []
        self.function_count = defaultdict(int)
        self.max_function_count = 100
        self.config, found_config_path = parse_config_file(config_file_path)
        self.project_root = project_root_from_module_root(
            self.config["module_root"], found_config_path
        )
        self.ignored_functions = {"<listcomp>", "<genexpr>", "<dictcomp>", "<setcomp>", "<lambda>"}
        self.file_being_called_from: str = str(
            os.path.basename(os.path.realpath(sys._getframe().f_back.f_code.co_filename)).replace(
                ".", "_"
            )
        )

        assert (
            "test_framework" in self.config
        ), "Please specify 'test-framework' in pyproject.toml config file"

    def __enter__(self) -> None:
        if self.disable:
            return

        pathlib.Path(self.output_file).unlink(missing_ok=True)

        self.con = sqlite3.connect(self.output_file)
        cur = self.con.cursor()
        # TODO: Check out if we need to export the function test name as well
        cur.execute(
            "CREATE TABLE events(type TEXT, function TEXT, classname TEXT, filename TEXT, line_number INTEGER, "
            "last_frame_address INTEGER, time_ns INTEGER, arg BLOB, locals BLOB)"
        )
        logging.info("Codeflash: Tracing started!")
        sys.setprofile(self.trace_callback)

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        sys.setprofile(None)
        # TODO: Check if the self.disable condition can be moved before the sys.setprofile call.
        #  Currently, it is below to see if the self.disable check doesn't add extra steps in the tracing
        if self.disable:
            return

        self.con.close()

        # filter any functions where we did not capture the return
        self.function_modules = [
            function
            for function in self.function_modules
            if self.function_count[
                function.file_name
                + ":"
                + (function.class_name + ":" if function.class_name else "")
                + function.function_name
            ]
            > 0
        ]

        replay_test = create_trace_replay_test(
            trace_file=self.output_file,
            functions=self.function_modules,
            test_framework=self.config["test_framework"],
            max_run_count=self.max_function_count,
        )
        if self.functions:
            function_path = "_".join(self.functions)
        else:
            function_path = self.file_being_called_from
        test_file_path = get_test_file_path(
            test_dir=self.config["tests_root"], function_name=function_path, test_type="replay"
        )
        replay_test = isort.code(replay_test)
        with open(test_file_path, "w", encoding="utf8") as file:
            file.write(replay_test)

        logging.info(
            f"Codeflash: Traced successful and replay test created! Path - {test_file_path}"
        )

    def trace_callback(self, frame: Any, event: str, arg: Any) -> None:
        if event not in ["call", "return"]:
            return

        code = frame.f_code
        # TODO : It currently doesn't log the last return call from the first function
        # print(code.co_name, code.co_filename)

        if code.co_name in self.ignored_functions:
            return
        if code.co_name == "__exit__" and code.co_filename == os.path.realpath(__file__):
            return
        file_name = os.path.realpath(code.co_filename)

        if not self.flag:
            # ignore the first call to trace_callback due to return from Tracer __enter__
            self.flag = True
            return
        if self.functions:
            if code.co_name not in self.functions:
                return None

        class_name = None
        if (
            "self" in frame.f_locals
            and hasattr(frame.f_locals["self"], "__class__")
            and hasattr(frame.f_locals["self"].__class__, "__name__")
        ):
            class_name = frame.f_locals["self"].__class__.__name__

        function_qualified_name = (
            file_name + ":" + (class_name + ":" if class_name else "") + code.co_name
        )

        if function_qualified_name not in self.function_count:
            # seeing this function for the first time
            self.function_count[function_qualified_name] = 0
            _, non_filtered_functions_count = filter_functions(
                modified_functions={
                    file_name: [
                        FunctionToOptimize(
                            function_name=code.co_name, file_path=file_name, parents=[]
                        )
                    ]
                },
                tests_root=self.config["tests_root"],
                ignore_paths=self.config["ignore_paths"],
                project_root=self.project_root,
                module_root=self.config["module_root"],
                disable_logs=True,
            )
            if non_filtered_functions_count == 0:
                # we don't want to trace this function because it cannot be optimized
                return
            self.function_modules.append(
                FunctionModules(
                    function_name=code.co_name,
                    file_name=file_name,
                    module_name=module_name_from_file_path(
                        file_name, project_root=self.project_root
                    ),
                    class_name=class_name,
                )
            )

        if self.function_count[function_qualified_name] >= self.max_function_count:
            return

        # TODO: Also check if this function arguments are unique from the values logged earlier

        cur = self.con.cursor()

        t_ns = time.perf_counter_ns()
        try:
            # pickling can be a recursive operator, so we need to increase the recursion limit
            original_recursion_limit = sys.getrecursionlimit()
            sys.setrecursionlimit(20000)
            local_vars = pickle.dumps(frame.f_locals, protocol=pickle.HIGHEST_PROTOCOL)
            arg = pickle.dumps(arg, protocol=pickle.HIGHEST_PROTOCOL)
            sys.setrecursionlimit(original_recursion_limit)
        except (TypeError, pickle.PicklingError, AttributeError) as e:
            # logging.info(f"Error in pickling arguments or local variables - {e}")
            return
        cur.execute(
            "INSERT INTO events VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                event,
                code.co_name,
                class_name,
                file_name,
                frame.f_lineno,
                frame.f_back.__hash__(),
                t_ns,
                arg,
                local_vars,
            ),
        )
        self.con.commit()
        if event == "return":
            self.function_count[function_qualified_name] += 1
