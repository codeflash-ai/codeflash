import logging
import os
import pickle
import sqlite3
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

from codeflash.validation.validation_utils import get_test_file_path

from codeflash.code_utils.code_utils import module_name_from_file_path
from codeflash.code_utils.config_parser import parse_config_file
from codeflash.tracing.replay_test import create_trace_replay_test


class Tracer:
    """
    Use this class as a 'with' context manager to trace a function call,
    input arguments, and return value.
    """

    def __init__(
        self,
        output: str = "script.trace",
        functions: List[str] = [],
        disable: bool = False,
        config_file_path: Optional[str] = None,
    ) -> None:
        self.disable = disable
        self.con = None
        self.flag = (
            False  # To ignore the first call to trace_callback due to return from trace_callback
        )
        self.output_file = os.path.abspath(output)
        self.functions = functions
        self.function_modules: Dict[str, str] = {}
        self.function_filenames: Dict[str, str] = {}
        self.function_count = defaultdict(int)
        self.max_function_count = 50
        self.config = parse_config_file(config_file_path)

        assert (
            "test_framework" in self.config
        ), "Please specify 'test-framework' in pyproject.toml config file"

    def __enter__(self) -> None:
        if self.disable:
            return

        if os.path.exists(self.output_file):
            os.remove(self.output_file)

        self.con = sqlite3.connect(self.output_file)
        cur = self.con.cursor()
        # TODO: Check out if we need to export the function test name as well
        cur.execute(
            "CREATE TABLE events(type TEXT, function TEXT, filename TEXT, line_number INTEGER, "
            "last_frame_address INTEGER, time_ns INTEGER, arg BLOB, locals BLOB)"
        )
        sys.setprofile(self.trace_callback)

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        sys.setprofile(None)
        # TODO: Check if the self.disable condition can be moved before the sys.setprofile call.
        #  Currently, it is below to see if the self.disable check doesn't add extra steps in the tracing
        if self.disable:
            return

        self.con.close()

        module_function = [
            (module, function_name) for function_name, module in self.function_modules.items()
        ]
        replay_test = create_trace_replay_test(
            trace_file=self.output_file,
            functions=module_function,
            test_framework=self.config["test_framework"],
        )
        function_path = "_".join([func for _, func in module_function])
        test_file_path = get_test_file_path(
            self.config["tests_root"], function_path, test_type="replay"
        )
        with open(test_file_path, "w") as file:
            file.write(replay_test)

        logging.info(
            f"CodeFlash: Function Traced successfully and replay test created! Path - {test_file_path}"
        )

    def trace_callback(self, frame: Any, event: str, arg: Any) -> None:
        if event not in ["call", "return"]:
            return None

        code = frame.f_code
        if self.functions:
            if code.co_name not in self.functions:
                return None
            if self.function_count[code.co_name] >= self.max_function_count:
                return
            self.function_count[code.co_name] += 1
            if code.co_name in self.function_filenames:
                assert self.function_filenames[code.co_name] == code.co_filename, (
                    f"Function {code.co_name} is defined in multiple files. "
                    f"Can only trace a unique function name at the moment. Aborting..."
                )
            else:
                self.function_filenames[code.co_name] = code.co_filename
        #         # TODO: Also check if this function arguments are unique from the values logged earlier
        elif not self.flag:
            self.flag = True
            return

        project_root = os.path.realpath(os.path.join(self.config["module_root"], ".."))
        self.function_modules[code.co_name] = module_name_from_file_path(
            code.co_filename, project_root=project_root
        )
        cur = self.con.cursor()

        t_ns = time.perf_counter_ns()
        try:
            local_vars = pickle.dumps(frame.f_locals, protocol=pickle.HIGHEST_PROTOCOL)
            arg = pickle.dumps(arg, protocol=pickle.HIGHEST_PROTOCOL)
        except (TypeError, pickle.PicklingError, AttributeError):
            return
        cur.execute(
            "INSERT INTO events VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
            (
                event,
                code.co_name,
                code.co_filename,
                frame.f_lineno,
                frame.f_back.__hash__(),
                t_ns,
                arg,
                local_vars,
            ),
        )
        self.con.commit()
