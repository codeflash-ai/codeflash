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
import marshal
import os
import pathlib
import pickle
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from typing import Any, List, Optional

import dill
import isort

from codeflash.cli_cmds.cli import project_root_from_module_root
from codeflash.code_utils.code_utils import module_name_from_file_path
from codeflash.code_utils.config_parser import parse_config_file
from codeflash.discovery.functions_to_optimize import filter_files_optimized
from codeflash.tracing.replay_test import create_trace_replay_test
from codeflash.tracing.tracing_utils import FunctionModules
from codeflash.verification.verification_utils import get_test_file_path


class Tracer:
    """Use this class as a 'with' context manager to trace a function call,
    input arguments, and return value.
    """

    def __init__(
        self,
        output: str = "codeflash.trace",
        functions: Optional[List[str]] = None,
        disable: bool = False,
        config_file_path: Optional[str] = None,
        max_function_count: int = 100,
        timeout: Optional[int] = None,  # seconds
    ) -> None:
        """:param output: The path to the output trace file
        :param functions: List of functions to trace. If None, trace all functions
        :param disable: Disable the tracer if True
        :param config_file_path: Path to the pyproject.toml file, if None then it will be auto-discovered
        :param max_function_count: Maximum number of times to trace one function
        :param timeout: Timeout in seconds for the tracer, if the traced code takes more than this time, then tracing
                    stops and normal execution continues. If this is None then no timeout applies
        """
        if functions is None:
            functions = []
        if os.environ.get("CODEFLASH_TRACER_DISABLE", "0") == "1":
            print("Codeflash: Tracer disabled by environment variable CODEFLASH_TRACER_DISABLE")
            disable = True
        self.disable = disable
        if self.disable:
            return
        self.con = None
        self.output_file = os.path.abspath(output)
        self.functions = functions
        self.function_modules: List[FunctionModules] = []
        self.function_count = defaultdict(int)
        self.ignored_qualified_functions = {
            f"{os.path.realpath(__file__)}:Tracer:__exit__",
            f"{os.path.realpath(__file__)}:Tracer:__enter__",
        }
        self.max_function_count = max_function_count
        self.config, found_config_path = parse_config_file(config_file_path)
        self.project_root = project_root_from_module_root(
            self.config["module_root"],
            found_config_path,
        )
        self.ignored_functions = {
            "<listcomp>",
            "<genexpr>",
            "<dictcomp>",
            "<setcomp>",
            "<lambda>",
        }
        self.file_being_called_from: str = str(
            os.path.basename(
                os.path.realpath(sys._getframe().f_back.f_code.co_filename),
            ).replace(
                ".",
                "_",
            ),
        )

        assert timeout is None or timeout > 0, "Timeout should be greater than 0"
        self.timeout = timeout
        self.next_insert = 1000
        self.profiling_info = defaultdict(Counter)

        # Profiler variables
        self.bias = 0  # calibration constant
        self.timings = {}
        self.cur = None
        self.start_time = None
        self.timer = time.process_time_ns
        self.profile_stats_file = "codeflash.stats"
        self.simulate_call("profiler")
        assert (
            "test_framework" in self.config
        ), "Please specify 'test-framework' in pyproject.toml config file"
        self.t = self.timer()

    def __enter__(self) -> None:
        if self.disable:
            return
        if getattr(Tracer, "used_once", False):
            print(
                "Codeflash: Tracer can only be used once per program run. "
                "Please only enable the Tracer once. Skipping tracing this section.",
            )
            self.disable = True
            return
        Tracer.used_once = True

        if pathlib.Path(self.output_file).exists():
            print("Codeflash: Removing existing trace file")
        pathlib.Path(self.output_file).unlink(missing_ok=True)

        self.con = sqlite3.connect(self.output_file)
        cur = self.con.cursor()
        cur.execute("""PRAGMA synchronous = OFF""")
        # TODO: Check out if we need to export the function test name as well
        cur.execute(
            "CREATE TABLE function_calls(type TEXT, function TEXT, classname TEXT, filename TEXT, "
            "line_number INTEGER, last_frame_address INTEGER, time_ns INTEGER, args BLOB)",
        )
        print("Codeflash: Tracing started!")
        frame = sys._getframe(0)  # Get this frame and simulate a call to it
        self.dispatch["call"](self, frame, 0)
        self.start_time = time.time()
        sys.setprofile(self.trace_callback)

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.disable:
            return
        sys.setprofile(None)
        self.con.commit()

        self.print_stats("tottime")

        self.create_stats()
        cur = self.con.cursor()
        cur.execute(
            "CREATE TABLE pstats (filename TEXT, line_number INTEGER, function TEXT, "
            "call_count_nonrecursive INTEGER, num_callers INTEGER, total_time_ns INTEGER, "
            "cumulative_time_ns INTEGER, callers BLOB)",
        )
        for func, (cc, nc, tt, ct, callers) in self.stats.items():
            cur.execute(
                "INSERT INTO pstats VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (func[0], func[1], func[2], cc, nc, tt, ct, pickle.dumps(callers)),
            )
        self.con.commit()
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
        function_path = "_".join(self.functions) if self.functions else self.file_being_called_from
        test_file_path = get_test_file_path(
            test_dir=self.config["tests_root"],
            function_name=function_path,
            test_type="replay",
        )
        replay_test = isort.code(replay_test)
        with open(test_file_path, "w", encoding="utf8") as file:
            file.write(replay_test)

        print(f"Codeflash: Traced successful and replay test created! Path - {test_file_path}")
        for key, value in self.profiling_info.items():
            print(
                f"Profiling info - {key} - count - {value['count']} - time - {value['time']/1e6}ms",
            )

    def trace_callback(self, frame: Any, event: str, arg: Any) -> None:
        timer = self.timer
        t = timer() - self.t - self.bias
        if event == "c_call":
            self.c_func_name = arg.__name__

        if self.dispatch[event](self, frame, t):
            self.t = timer()
        else:
            self.t = timer() - t  # put back unrecorded delta
        t1 = time.perf_counter_ns()
        if event not in ["call", "return"]:
            return
        if self.timeout is not None:
            if (time.time() - self.start_time) > self.timeout:
                sys.setprofile(None)
                print(
                    f"Codeflash: Timeout reached! Stopping tracing at {self.timeout} seconds.",
                )
                return
        t2 = time.perf_counter_ns()
        self.profiling_info["early_return"].update(count=1, time=t2 - t1)
        t3 = time.perf_counter_ns()
        code = frame.f_code
        file_name = code.co_filename
        # TODO : It currently doesn't log the last return call from the first function
        # print(code.co_name, code.co_filename)

        if code.co_name in self.ignored_functions:
            return
        if not os.path.exists(file_name):
            return
        if self.functions:
            if code.co_name not in self.functions:
                return
        class_name = None
        if (
            "self" in frame.f_locals
            and hasattr(frame.f_locals["self"], "__class__")
            and hasattr(frame.f_locals["self"].__class__, "__name__")
        ):
            class_name = frame.f_locals["self"].__class__.__name__

        function_qualified_name = file_name + ":" + (class_name + ":" if class_name else "") + code.co_name
        if function_qualified_name in self.ignored_qualified_functions:
            # print(function_qualified_name)
            return
        t4 = time.perf_counter_ns()
        self.profiling_info["ignored_functions"].update(count=1, time=t4 - t3)
        t9 = time.perf_counter_ns()
        if event == "return":
            self.function_count[function_qualified_name] += 1
            if self.function_count[function_qualified_name] >= self.max_function_count:
                self.ignored_qualified_functions.add(function_qualified_name)
                del self.function_count[function_qualified_name]  # save memory
            return

        if function_qualified_name not in self.function_count:
            # seeing this function for the first time
            self.function_count[function_qualified_name] = 0
            file_valid = filter_files_optimized(
                file_path=file_name,
                tests_root=self.config["tests_root"],
                ignore_paths=self.config["ignore_paths"],
                module_root=self.config["module_root"],
            )
            if not file_valid:
                # we don't want to trace this function because it cannot be optimized
                self.ignored_qualified_functions.add(function_qualified_name)
                t10 = time.perf_counter_ns()
                self.profiling_info["filter_functions"].update(count=1, time=t10 - t9)
                return
            self.function_modules.append(
                FunctionModules(
                    function_name=code.co_name,
                    file_name=file_name,
                    module_name=module_name_from_file_path(
                        file_name,
                        project_root=self.project_root,
                    ),
                    class_name=class_name,
                ),
            )
        t10 = time.perf_counter_ns()
        self.profiling_info["filter_functions"].update(count=1, time=t10 - t9)
        t11 = time.perf_counter_ns()

        # TODO: Also check if this function arguments are unique from the values logged earlier

        cur = self.con.cursor()

        t_ns = time.perf_counter_ns()
        self.profiling_info["con.cursor"].update(count=1, time=t_ns - t11)
        t12 = time.perf_counter_ns()
        original_recursion_limit = sys.getrecursionlimit()
        try:
            # pickling can be a recursive operator, so we need to increase the recursion limit
            sys.setrecursionlimit(10000)
            local_vars = pickle.dumps(
                frame.f_locals,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
            sys.setrecursionlimit(original_recursion_limit)
        except (TypeError, pickle.PicklingError, AttributeError, RecursionError):
            # we retry with dill if pickle fails. It's slower but more comprehensive
            try:
                local_vars = dill.dumps(
                    frame.f_locals,
                    protocol=dill.HIGHEST_PROTOCOL,
                )
                sys.setrecursionlimit(original_recursion_limit)

            except (TypeError, dill.PicklingError, AttributeError, RecursionError):
                # give up
                # TODO: If this branch hits then its possible there are no paired arg, return values in the replay test.
                #  Filter them out
                return
        t13 = time.perf_counter_ns()
        self.profiling_info["pickle.dumps"].update(count=1, time=t13 - t12)
        t14 = time.perf_counter_ns()
        cur.execute(
            "INSERT INTO function_calls VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
            (
                event,
                code.co_name,
                class_name,
                file_name,
                frame.f_lineno,
                frame.f_back.__hash__(),
                t_ns,
                local_vars,
            ),
        )
        self.next_insert -= 1
        if self.next_insert == 0:
            self.next_insert = 1000
            self.con.commit()

        t15 = time.perf_counter_ns()
        self.profiling_info["cur.execute"].update(count=1, time=t15 - t14)

    def trace_dispatch_call(self, frame, t):
        # print("trace_dispatch_call", self.cur)
        if self.cur and frame.f_back is not self.cur[-2]:
            rpt, rit, ret, rfn, rframe, rcur = self.cur
            if not isinstance(rframe, Tracer.fake_frame):
                assert rframe.f_back is frame.f_back, (
                    "Bad call",
                    rfn,
                    rframe,
                    rframe.f_back,
                    frame,
                    frame.f_back,
                )
                self.trace_dispatch_return(rframe, 0)
                assert self.cur is None or frame.f_back is self.cur[-2], ("Bad call", self.cur[-3])
        fcode = frame.f_code
        fn = (fcode.co_filename, fcode.co_firstlineno, fcode.co_name)
        self.cur = (t, 0, 0, fn, frame, self.cur)
        timings = self.timings
        if fn in timings:
            cc, ns, tt, ct, callers = timings[fn]
            timings[fn] = cc, ns + 1, tt, ct, callers
        else:
            timings[fn] = 0, 0, 0, 0, {}
        # print("trace_dispatch_callreturn", self.cur)
        return 1

    def trace_dispatch_exception(self, frame, t):
        # print("trace_dispatch_exception", self.cur)
        rpt, rit, ret, rfn, rframe, rcur = self.cur
        if (rframe is not frame) and rcur:
            return self.trace_dispatch_return(rframe, t)
        self.cur = rpt, rit + t, ret, rfn, rframe, rcur
        return 1

    def trace_dispatch_c_call(self, frame, t):
        # print("trace_dispatch_c_call", self.cur)
        fn = ("", 0, self.c_func_name)
        self.cur = (t, 0, 0, fn, frame, self.cur)
        timings = self.timings
        if fn in timings:
            cc, ns, tt, ct, callers = timings[fn]
            timings[fn] = cc, ns + 1, tt, ct, callers
        else:
            timings[fn] = 0, 0, 0, 0, {}
        return 1

    def trace_dispatch_return(self, frame, t):
        # print("trace_dispatch_return", self.cur)
        if frame is not self.cur[-2]:
            print(self.cur[-2], self.cur[-2].f_back)
            assert frame is self.cur[-2].f_back, ("Bad return", self.cur[-3])
            self.trace_dispatch_return(self.cur[-2], 0)

        # Prefix "r" means part of the Returning or exiting frame.
        # Prefix "p" means part of the Previous or Parent or older frame.

        rpt, rit, ret, rfn, frame, rcur = self.cur
        rit = rit + t
        frame_total = rit + ret

        ppt, pit, pet, pfn, pframe, pcur = rcur
        self.cur = ppt, pit + rpt, pet + frame_total, pfn, pframe, pcur

        timings = self.timings
        cc, ns, tt, ct, callers = timings[rfn]
        if not ns:
            # This is the only occurrence of the function on the stack.
            # Else this is a (directly or indirectly) recursive call, and
            # its cumulative time will get updated when the topmost call to
            # it returns.
            ct = ct + frame_total
            cc = cc + 1

        if pfn in callers:
            callers[pfn] = callers[pfn] + 1  # hack: gather more
            # stats such as the amount of time added to ct courtesy
            # of this specific call, and the contribution to cc
            # courtesy of this call.
        else:
            callers[pfn] = 1

        timings[rfn] = cc, ns - 1, tt + rit, ct, callers

        return 1

    dispatch = {
        "call": trace_dispatch_call,
        "exception": trace_dispatch_exception,
        "return": trace_dispatch_return,
        "c_call": trace_dispatch_c_call,
        "c_exception": trace_dispatch_return,  # the C function returned
        "c_return": trace_dispatch_return,
    }

    class fake_code:
        def __init__(self, filename, line, name):
            self.co_filename = filename
            self.co_line = line
            self.co_name = name
            self.co_firstlineno = 0

        def __repr__(self):
            return repr((self.co_filename, self.co_line, self.co_name))

    class fake_frame:
        def __init__(self, code, prior):
            self.f_code = code
            self.f_back = prior

    def simulate_call(self, name):
        code = self.fake_code("profiler", 0, name)
        if self.cur:
            pframe = self.cur[-2]
        else:
            pframe = None
        frame = self.fake_frame(code, pframe)
        self.dispatch["call"](self, frame, 0)

    def simulate_cmd_complete(self):
        get_time = self.timer
        t = get_time() - self.t
        while self.cur[-1]:
            # We *can* cause assertion errors here if
            # dispatch_trace_return checks for a frame match!
            self.dispatch["return"](self, self.cur[-2], t)
            t = 0
        self.t = get_time() - t

    def print_stats(self, sort=-1):
        import pstats

        if not isinstance(sort, tuple):
            sort = (sort,)
        pstats.Stats(self).strip_dirs().sort_stats(*sort).print_stats()

    def dump_stats(self, file):
        with open(file, "wb") as f:
            self.create_stats()
            marshal.dump(self.stats, f)

    def create_stats(self):
        self.simulate_cmd_complete()
        self.snapshot_stats()

    def snapshot_stats(self):
        self.stats = {}
        for func, (cc, ns, tt, ct, callers) in self.timings.items():
            callers = callers.copy()
            nc = 0
            for callcnt in callers.values():
                nc += callcnt
            self.stats[func] = cc, nc, tt, ct, callers
