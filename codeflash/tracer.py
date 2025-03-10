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

import importlib.machinery
import io
import json
import marshal
import os
import pickle
import shutil
import sqlite3
import sys
import tempfile
import threading
import time
import uuid
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, ClassVar

import dill
import isort
from rich.align import Align
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from codeflash.cli_cmds.cli import project_root_from_module_root
from codeflash.cli_cmds.console import console
from codeflash.code_utils.code_utils import cleanup_paths, module_name_from_file_path
from codeflash.code_utils.config_parser import parse_config_file
from codeflash.discovery.functions_to_optimize import filter_files_optimized
from codeflash.tracing.replay_test import create_trace_replay_test
from codeflash.tracing.tracing_utils import FunctionModules
from codeflash.verification.verification_utils import get_test_file_path

if TYPE_CHECKING:
    from types import FrameType, TracebackType


class fake_code:  # noqa: N801
    def __init__(self, filename: str, line: int, name: str) -> None:
        self.co_filename = filename
        self.co_line = line
        self.co_name = name
        self.co_firstlineno = 0

    def __repr__(self) -> str:
        return repr((self.co_filename, self.co_line, self.co_name, None))


class fake_frame:  # noqa: N801
    def __init__(self, code: fake_code, prior: fake_frame | None) -> None:
        self.f_code = code
        self.f_back = prior
        self.f_locals = {}


# Debug this file by simply adding print statements. This file is not meant to be debugged by the debugger.
class Tracer:
    """Use this class as a 'with' context manager to trace a function call.

    Traces function calls, input arguments, and profiling info.
    """

    used_once: ClassVar[bool] = False  # Class variable to track if Tracer has been used

    def __init__(
        self,
        output: str = "codeflash.trace",
        functions: list[str] | None = None,
        disable: bool = False,
        config_file_path: Path | None = None,
        max_function_count: int = 256,
        timeout: int | None = None,  # seconds
    ) -> None:
        """Initialize Tracer."""
        if functions is None:
            functions = []
        if os.environ.get("CODEFLASH_TRACER_DISABLE", "0") == "1":
            console.print("Codeflash: Tracer disabled by environment variable CODEFLASH_TRACER_DISABLE")
            disable = True
        self.disable = disable
        if self.disable:
            return
        if sys.getprofile() is not None or sys.gettrace() is not None:
            console.print(
                "WARNING - Codeflash: Another profiler, debugger or coverage tool is already running. "
                "Please disable it before starting the Codeflash Tracer, both can't run. Codeflash Tracer is DISABLED."
            )
            self.disable = True
            return

        # Setup output paths
        self.output_file = Path(output).resolve()
        self.output_dir = self.output_file.parent
        self.output_base = self.output_file.stem
        self.output_ext = self.output_file.suffix
        self.thread_db_files: dict[int, Path] = {}  # Thread ID to DB file path

        self.functions = functions
        self.function_modules: list[FunctionModules] = []
        self.function_count = defaultdict(int)
        self.current_file_path = Path(__file__).resolve()
        self.ignored_qualified_functions = {
            f"{self.current_file_path}:Tracer:__exit__",
            f"{self.current_file_path}:Tracer:__enter__",
        }
        self.max_function_count = max_function_count
        self.config, found_config_path = parse_config_file(config_file_path)
        self.project_root = project_root_from_module_root(Path(self.config["module_root"]), found_config_path)
        console.print("project_root", self.project_root)
        self.ignored_functions = {"<listcomp>", "<genexpr>", "<dictcomp>", "<setcomp>", "<lambda>", "<module>"}

        self.file_being_called_from: str = str(Path(sys._getframe().f_back.f_code.co_filename).name).replace(".", "_")  # noqa: SLF001

        assert timeout is None or timeout > 0, "Timeout should be greater than 0"
        self.timeout = timeout
        self.next_insert = 1000
        self.trace_count = 0

        self.db_lock = threading.RLock()
        self.thread_local = threading.local()

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Profiler variables
        self.bias = 0  # calibration constant
        self.timings: dict[Any, Any] = {}
        self.cur: Any = None
        self.start_time: float | None = None
        self.timer = time.process_time_ns
        self.total_tt = 0
        self.simulate_call("profiler")
        assert "test_framework" in self.config, "Please specify 'test-framework' in pyproject.toml config file"
        self.t = self.timer()
        self.main_db_created = False  # Flag to track main DB creation

    def get_thread_db_path(self) -> Path:
        """Get the database path for the current thread."""
        thread_id = threading.get_ident()
        if thread_id not in self.thread_db_files:
            # Create a unique filename for this thread
            unique_id = uuid.uuid4().hex[:8]
            db_path = self.output_dir / f"{self.output_base}_{thread_id}_{unique_id}{self.output_ext}"
            self.thread_db_files[thread_id] = db_path
        return self.thread_db_files[thread_id]

    def get_connection(self) -> sqlite3.Connection:
        """Get a dedicated connection for the current thread."""
        if not hasattr(self.thread_local, "con"):
            db_path = self.get_thread_db_path()
            self.thread_local.con = sqlite3.connect(db_path)
            # Create the necessary tables if they don't exist
            cur = self.thread_local.con.cursor()
            cur.execute("""PRAGMA synchronous = OFF""")
            cur.execute(
                "CREATE TABLE IF NOT EXISTS function_calls(type TEXT, function TEXT, classname TEXT, "
                "filename TEXT, line_number INTEGER, last_frame_address INTEGER, time_ns INTEGER, args BLOB)"
            )
            self.thread_local.con.commit()
        return self.thread_local.con

    def _create_main_db_if_not_exists(self) -> None:
        """Create the main output database if it doesn't exist."""
        if not self.main_db_created:  # Use flag to prevent redundant checks
            if not self.output_file.exists():
                try:
                    main_con = sqlite3.connect(self.output_file)
                    main_cur = main_con.cursor()
                    main_cur.execute("""PRAGMA synchronous = OFF""")  # Added pragma for main db too
                    main_cur.execute(
                        "CREATE TABLE IF NOT EXISTS function_calls(type TEXT, function TEXT, classname TEXT, "
                        "filename TEXT, line_number INTEGER, last_frame_address INTEGER, time_ns INTEGER, args BLOB)"
                    )
                    main_con.commit()
                    main_con.close()
                    self.main_db_created = True  # Set flag after successful creation
                except Exception as e:  # noqa: BLE001
                    console.print(f"Error creating main database: {e}")
            else:
                self.main_db_created = True  # Main DB already exists

    def __enter__(self) -> None:
        if self.disable:
            return
        if Tracer.used_once:
            console.print(
                "Codeflash: Tracer can only be used once per program run. "
                "Please only enable the Tracer once. Skipping tracing this section."
            )
            self.disable = True
            return
        Tracer.used_once = True  # Mark Tracer as used at the start of __enter__

        # Clean up any existing trace files
        if self.output_file.exists():
            console.print("Codeflash: Removing existing trace file")
            cleanup_paths([self.output_file])

        self._create_main_db_if_not_exists()
        self.con = sqlite3.connect(self.output_file)  # Keep connection open during tracing
        console.print("Codeflash: Tracing started!")
        console.rule("Program Output Begin", style="bold blue")
        frame = sys._getframe(0)  # Get this frame and simulate a call to it  # noqa: SLF001
        self.dispatch["call"](self, frame, 0)
        self.start_time = time.time()
        sys.setprofile(self.trace_callback)
        threading.setprofile(self.trace_callback)

    def _close_thread_connection(self) -> None:
        """Close thread-local connection and handle potential errors."""
        if hasattr(self.thread_local, "con") and self.thread_local.con:
            try:
                self.thread_local.con.commit()
                self.thread_local.con.close()
                del self.thread_local.con
            except Exception as e:  # noqa: BLE001
                console.print(f"Error closing current thread's connection: {e}")

    def _merge_thread_dbs(self) -> int:
        total_rows_copied = 0
        processed_files: list[Path] = []

        for thread_id, db_path in self.thread_db_files.items():
            if not db_path.exists():
                console.print(f"Thread database for thread {thread_id} not found, skipping.")
                continue

            rows_copied = self._process_thread_db(thread_id, db_path)
            if rows_copied >= 0:  # _process_thread_db returns -1 on failure
                total_rows_copied += rows_copied
                processed_files.append(db_path)
            else:
                console.print(f"Failed to merge from thread database {thread_id}")

        for thread_id, db_path in self.thread_db_files.items():
            if db_path in processed_files or not db_path.exists():
                continue

            rows_copied = self._process_thread_db_with_copy(thread_id, db_path)
            if rows_copied >= 0:
                total_rows_copied += rows_copied
                processed_files.append(db_path)
            else:
                console.print(f"Failed to merge from thread database {thread_id} even with copy approach.")

        return total_rows_copied

    def _process_thread_db(self, thread_id: int, db_path: Path) -> int:
        try:
            thread_con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=2.0)
            thread_cur = thread_con.cursor()

            thread_cur.execute("SELECT * FROM function_calls")
            main_cur = self.con.cursor()

            self.con.execute("BEGIN TRANSACTION")

            batch_size = 100
            batch = thread_cur.fetchmany(batch_size)
            rows_processed = 0

            while batch:
                for row in batch:
                    try:
                        main_cur.execute("INSERT INTO function_calls VALUES(?, ?, ?, ?, ?, ?, ?, ?)", row)
                        rows_processed += 1
                    except sqlite3.Error as e:  # noqa: PERF203
                        console.print(f"Error inserting row {rows_processed} from thread {thread_id}: {e}")
                batch = thread_cur.fetchmany(batch_size)

            self.con.commit()
            thread_con.close()

        except sqlite3.Error as e:
            console.print(f"Could not open thread database {thread_id} directly: {e}")
            return -1
        else:
            return rows_processed

    def _process_thread_db_with_copy(self, thread_id: int, db_path: Path) -> int:
        console.print(f"Attempting file copy approach for thread {thread_id}...")

        temp_dir = tempfile.gettempdir()
        temp_db_path = Path(temp_dir) / f"codeflash_temp_{uuid.uuid4().hex}.trace"
        rows_processed = 0

        try:
            shutil.copy2(db_path, temp_db_path)

            temp_con = sqlite3.connect(temp_db_path)
            temp_cur = temp_con.cursor()

            temp_cur.execute("SELECT COUNT(*) FROM function_calls")
            row_count = temp_cur.fetchone()[0]

            if row_count > 0:
                temp_cur.execute("SELECT * FROM function_calls")
                main_cur = self.con.cursor()

                self.con.execute("BEGIN TRANSACTION")
                batch_size = 100
                batch = temp_cur.fetchmany(batch_size)

                while batch:
                    for row in batch:
                        try:
                            main_cur.execute("INSERT INTO function_calls VALUES(?, ?, ?, ?, ?, ?, ?, ?)", row)
                            rows_processed += 1
                        except sqlite3.Error as e:
                            console.print(f"Error inserting row from thread {thread_id} copy: {e}")

                        batch = temp_cur.fetchmany(batch_size)

                self.con.commit()

            temp_con.close()
            cleanup_paths([temp_db_path])
            console.print(f"Successfully merged {rows_processed} rows from thread {thread_id} (via copy)")
        except Exception as e:  # noqa: BLE001
            console.print(f"Error with file copy approach for thread {thread_id}: {e}")
            cleanup_paths([temp_db_path])
            return -1

        else:
            return rows_processed

    def _generate_stats_and_replay_test(self) -> None:
        """Generate statistics, pstats compatible data, print stats and create replay test."""
        try:
            self.create_stats()

            try:
                main_cur = self.con.cursor()
                main_cur.execute(
                    "CREATE TABLE pstats (filename TEXT, line_number INTEGER, function TEXT, class_name TEXT, "
                    "call_count_nonrecursive INTEGER, num_callers INTEGER, total_time_ns INTEGER, "
                    "cumulative_time_ns INTEGER, callers BLOB)"
                )

                for func, (cc, nc, tt, ct, callers) in self.stats.items():
                    remapped_callers = [{"key": k, "value": v} for k, v in callers.items()]
                    main_cur.execute(
                        "INSERT INTO pstats VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            str(Path(func[0]).resolve()),
                            func[1],
                            func[2],
                            func[3],
                            cc,
                            nc,
                            tt,
                            ct,
                            json.dumps(remapped_callers),
                        ),
                    )

                self.con.commit()  # Use main DB connection

                self.make_pstats_compatible()
                self.print_stats("tottime")

                main_cur.execute("CREATE TABLE total_time (time_ns INTEGER)")
                main_cur.execute("INSERT INTO total_time VALUES (?)", (self.total_tt,))
                self.con.commit()  # Use main DB connection

            except Exception as e:  # noqa: BLE001
                console.print(f"Error generating stats tables: {e}")
                import traceback

                traceback.print_exc()

        except Exception as e:  # noqa: BLE001
            console.print(f"Error during stats generation: {e}")
            console.print_exception()

        # Generate the replay test
        try:
            replay_test = create_trace_replay_test(
                trace_file=self.output_file,
                functions=self.function_modules,
                test_framework=self.config["test_framework"],
                max_run_count=self.max_function_count,
            )
            function_path = "_".join(self.functions) if self.functions else self.file_being_called_from
            test_file_path = get_test_file_path(
                test_dir=Path(self.config["tests_root"]), function_name=function_path, test_type="replay"
            )
            replay_test = isort.code(replay_test)

            with test_file_path.open("w", encoding="utf8") as file:
                file.write(replay_test)

            console.print(
                f"Codeflash: Traced {self.trace_count} function calls successfully and replay test created at - {test_file_path}",
                crop=False,
                soft_wrap=False,
                overflow="ignore",
            )
        except Exception as e:  # noqa: BLE001
            console.print(f"Error creating replay test: {e}")
            console.print_exception()

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None
    ) -> None:
        if self.disable:
            return
        console.rule("Program Output End", style="bold blue")
        sys.setprofile(None)
        threading.setprofile(None)

        self._close_thread_connection()

        # Give threads time to complete their database operations
        time.sleep(1)

        self._merge_thread_dbs()
        self._generate_stats_and_replay_test()

        all_db_paths = list(self.thread_db_files.values())
        cleanup_paths(all_db_paths)

        # Filter any functions where we did not capture the return - moved to replay test generation for clarity
        self.function_modules = [
            function
            for function in self.function_modules
            if self.function_count[
                str(function.file_name)
                + ":"
                + (function.class_name + ":" if function.class_name else "")
                + function.function_name
            ]
            > 0
        ]

        if self.con:
            self.con.close()
            self.con = None

    def tracer_logic(self, frame: FrameType, event: str) -> None:
        if event != "call":
            return
        if self.timeout is not None and (time.time() - self.start_time) > self.timeout:
            sys.setprofile(None)
            console.print(f"Codeflash: Timeout reached! Stopping tracing at {self.timeout} seconds.")
            return
        code = frame.f_code
        file_name = Path(code.co_filename).resolve()
        # TODO : It currently doesn't log the last return call from the first function

        if code.co_name in self.ignored_functions:
            return
        if not file_name.exists():
            return
        if self.functions and code.co_name not in self.functions:
            return
        class_name = None
        arguments = frame.f_locals
        try:
            if (
                "self" in arguments
                and hasattr(arguments["self"], "__class__")
                and hasattr(arguments["self"].__class__, "__name__")
            ):
                class_name = arguments["self"].__class__.__name__
            elif "cls" in arguments and hasattr(arguments["cls"], "__name__"):
                class_name = arguments["cls"].__name__
        except:  # noqa: E722
            # someone can override the getattr method and raise an exception. I'm looking at you wrapt
            return

        function_qualified_name = f"{file_name}:{(class_name + ':' if class_name else '')}{code.co_name}"
        if function_qualified_name in self.ignored_qualified_functions:
            return
        if function_qualified_name not in self.function_count:
            # seeing this function for the first time
            self.function_count[function_qualified_name] = 0
            file_valid = filter_files_optimized(
                file_path=file_name,
                tests_root=Path(self.config["tests_root"]),
                ignore_paths=[Path(p) for p in self.config["ignore_paths"]],
                module_root=Path(self.config["module_root"]),
            )
            if not file_valid:
                # we don't want to trace this function because it cannot be optimized
                self.ignored_qualified_functions.add(function_qualified_name)
                return
            self.function_modules.append(
                FunctionModules(
                    function_name=code.co_name,
                    file_name=file_name,
                    module_name=module_name_from_file_path(file_name, project_root_path=self.project_root),
                    class_name=class_name,
                    line_no=code.co_firstlineno,
                )
            )
        else:
            self.function_count[function_qualified_name] += 1
            if self.function_count[function_qualified_name] >= self.max_function_count:
                self.ignored_qualified_functions.add(function_qualified_name)
                return

        # Get thread-specific connection
        conn = self.get_connection()
        cur = conn.cursor()

        t_ns = time.perf_counter_ns()
        original_recursion_limit = sys.getrecursionlimit()
        try:
            # pickling can be a recursive operator, so we need to increase the recursion limit
            sys.setrecursionlimit(10000)
            # We do not pickle self for __init__ to avoid recursion errors, and instead instantiate its class
            # directly with the rest of the arguments in the replay tests. We copy the arguments to avoid memory
            # leaks, bad references or side effects when unpickling.
            arguments = dict(arguments.items())
            if class_name and code.co_name == "__init__":
                del arguments["self"]
            local_vars = pickle.dumps(arguments, protocol=pickle.HIGHEST_PROTOCOL)
            sys.setrecursionlimit(original_recursion_limit)
        except (TypeError, pickle.PicklingError, AttributeError, RecursionError, OSError):
            # we retry with dill if pickle fails. It's slower but more comprehensive
            try:
                local_vars = dill.dumps(arguments, protocol=dill.HIGHEST_PROTOCOL)
                sys.setrecursionlimit(original_recursion_limit)

            except (TypeError, dill.PicklingError, AttributeError, RecursionError, OSError):
                # give up
                self.function_count[function_qualified_name] -= 1
                return
        try:
            cur.execute(
                "INSERT INTO function_calls VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    event,
                    code.co_name,
                    class_name,
                    str(file_name),
                    frame.f_lineno,
                    frame.f_back.__hash__(),
                    t_ns,
                    local_vars,
                ),
            )

            # Add thread-safe counter increment for trace_count
            with self.db_lock:
                self.trace_count += 1

            self.next_insert -= 1
            if self.next_insert == 0:
                self.next_insert = 1000
                conn.commit()
        except sqlite3.Error as e:
            thread_id = threading.get_ident()
            console.print(f"SQLite error in tracer (thread {thread_id}): {e}")

    def trace_callback(self, frame: FrameType, event: str, arg: str | None) -> None:
        # profiler section
        timer = self.timer
        t = timer() - self.t - self.bias
        if event == "c_call":
            self.c_func_name = arg.__name__

        prof_success = bool(self.dispatch[event](self, frame, t))
        # tracer section
        self.tracer_logic(frame, event)
        # measure the time as the last thing before return
        if prof_success:
            self.t = timer()
        else:
            self.t = timer() - t  # put back unrecorded delta

    def trace_dispatch_call(self, frame: FrameType, t: int) -> int:
        """Handle call events in the profiler."""
        try:
            # In multi-threaded contexts, we need to be more careful about frame comparisons
            if self.cur and frame.f_back is not self.cur[-2]:
                # This happens when we're in a different thread
                rpt, rit, ret, rfn, rframe, rcur = self.cur

                # Only attempt to handle the frame mismatch if we have a valid rframe
                if (
                    not isinstance(rframe, Tracer.fake_frame)
                    and hasattr(rframe, "f_back")
                    and hasattr(frame, "f_back")
                    and rframe.f_back is frame.f_back
                ):
                    self.trace_dispatch_return(rframe, 0)

            # Get function information
            fcode = frame.f_code
            arguments = frame.f_locals
            class_name = None
            try:
                if (
                    "self" in arguments
                    and hasattr(arguments["self"], "__class__")
                    and hasattr(arguments["self"].__class__, "__name__")
                ):
                    class_name = arguments["self"].__class__.__name__
                elif "cls" in arguments and hasattr(arguments["cls"], "__name__"):
                    class_name = arguments["cls"].__name__
            except Exception:  # noqa: BLE001, S110
                pass

            fn = (fcode.co_filename, fcode.co_firstlineno, fcode.co_name, class_name)
            self.cur = (t, 0, 0, fn, frame, self.cur)
            timings = self.timings
            if fn in timings:
                cc, ns, tt, ct, callers = timings[fn]
                timings[fn] = cc, ns + 1, tt, ct, callers
            else:
                timings[fn] = 0, 0, 0, 0, {}
            return 1  # noqa: TRY300
        except Exception:  # noqa: BLE001
            # Handle any errors gracefully
            return 0

    def trace_dispatch_exception(self, frame: FrameType, t: int) -> int:
        rpt, rit, ret, rfn, rframe, rcur = self.cur
        if (rframe is not frame) and rcur:
            return self.trace_dispatch_return(rframe, t)
        self.cur = rpt, rit + t, ret, rfn, rframe, rcur
        return 1

    def trace_dispatch_c_call(self, frame: FrameType, t: int) -> int:
        fn = ("", 0, self.c_func_name, None)
        self.cur = (t, 0, 0, fn, frame, self.cur)
        timings = self.timings
        if fn in timings:
            cc, ns, tt, ct, callers = timings[fn]
            timings[fn] = cc, ns + 1, tt, ct, callers
        else:
            timings[fn] = 0, 0, 0, 0, {}
        return 1

    def trace_dispatch_return(self, frame: FrameType, t: int) -> int:
        """Handle return events in the profiler."""
        try:
            # Check if we have a valid current frame
            if not self.cur or not self.cur[-2]:
                return 0

            # In multi-threaded environments, frames can get mismatched
            if frame is not self.cur[-2]:
                # Don't assert in threaded environments - frames can legitimately differ
                if hasattr(frame, "f_back") and hasattr(self.cur[-2], "f_back") and frame is self.cur[-2].f_back:
                    self.trace_dispatch_return(self.cur[-2], 0)
                else:
                    # We're in a different thread or context, can't continue with this frame
                    return 0

            rpt, rit, ret, rfn, frame, rcur = self.cur
            rit = rit + t
            frame_total = rit + ret

            # Guard against invalid rcur (w threading)
            if not rcur:
                return 0

            ppt, pit, pet, pfn, pframe, pcur = rcur
            self.cur = ppt, pit + rpt, pet + frame_total, pfn, pframe, pcur

            timings = self.timings
            if rfn not in timings:
                # w threading, rfn can be missing
                timings[rfn] = 0, 0, 0, 0, {}

            cc, ns, tt, ct, callers = timings[rfn]
            if not ns:
                # This is the only occurrence of the function on the stack.
                ct = ct + frame_total
                cc = cc + 1

            if pfn in callers:
                callers[pfn] = callers[pfn] + 1
            else:
                callers[pfn] = 1

            timings[rfn] = cc, ns - 1, tt + rit, ct, callers

            return 1
        except Exception:
            # Handle errors gracefully
            return 0

    dispatch: ClassVar[dict[str, Callable[[Tracer, FrameType, int], int]]] = {
        "call": trace_dispatch_call,
        "exception": trace_dispatch_exception,
        "return": trace_dispatch_return,
        "c_call": trace_dispatch_c_call,
        "c_exception": trace_dispatch_return,  # the C function returned
        "c_return": trace_dispatch_return,
    }

    def simulate_call(self, name) -> None:
        code = fake_code("profiler", 0, name)
        pframe = self.cur[-2] if self.cur else None
        frame = fake_frame(code, pframe)
        self.dispatch["call"](self, frame, 0)

    def simulate_cmd_complete(self) -> None:
        get_time = self.timer
        t = get_time() - self.t
        while self.cur[-1]:
            # We *can* cause assertion errors here if
            # dispatch_trace_return checks for a frame match!
            self.dispatch["return"](self, self.cur[-2], t)
            t = 0
        self.t = get_time() - t

    def print_stats(self, sort: str | int | tuple = -1) -> None:
        if not self.stats:
            console.print("Codeflash: No stats available to print")
            self.total_tt = 0
            return

        if not isinstance(sort, tuple):
            sort = (sort,)

        # First, convert stats to make them pstats-compatible
        try:
            # Initialize empty collections for pstats
            self.files = []
            self.top_level = []

            # Create entirely new dictionaries instead of modifying existing ones
            new_stats = {}
            new_timings = {}

            # Convert stats dictionary
            stats_items = list(self.stats.items())
            for func, stats_data in stats_items:
                try:
                    # Make sure we have 5 elements in stats_data
                    if len(stats_data) != 5:
                        console.print(f"Skipping malformed stats data for {func}: {stats_data}")
                        continue

                    cc, nc, tt, ct, callers = stats_data

                    if len(func) == 4:
                        file_name, line_num, func_name, class_name = func
                        new_func_name = f"{class_name}.{func_name}" if class_name else func_name
                        new_func = (file_name, line_num, new_func_name)
                    else:
                        new_func = func  # Keep as is if already in correct format

                    new_callers = {}
                    callers_items = list(callers.items())
                    for caller_func, count in callers_items:
                        if isinstance(caller_func, tuple):
                            if len(caller_func) == 4:
                                caller_file, caller_line, caller_name, caller_class = caller_func
                                caller_new_name = f"{caller_class}.{caller_name}" if caller_class else caller_name
                                new_caller_func = (caller_file, caller_line, caller_new_name)
                            else:
                                new_caller_func = caller_func
                        else:
                            console.print(f"Unexpected caller format: {caller_func}")
                            new_caller_func = str(caller_func)

                        new_callers[new_caller_func] = count

                    # Store with new format
                    new_stats[new_func] = (cc, nc, tt, ct, new_callers)
                except Exception as e:  # noqa: BLE001
                    console.print(f"Error converting stats for {func}: {e}")
                    continue

            timings_items = list(self.timings.items())
            for func, timing_data in timings_items:
                try:
                    if len(timing_data) != 5:
                        console.print(f"Skipping malformed timing data for {func}: {timing_data}")
                        continue

                    cc, ns, tt, ct, callers = timing_data

                    if len(func) == 4:
                        file_name, line_num, func_name, class_name = func
                        new_func_name = f"{class_name}.{func_name}" if class_name else func_name
                        new_func = (file_name, line_num, new_func_name)
                    else:
                        new_func = func

                    new_callers = {}
                    callers_items = list(callers.items())
                    for caller_func, count in callers_items:
                        if isinstance(caller_func, tuple):
                            if len(caller_func) == 4:
                                caller_file, caller_line, caller_name, caller_class = caller_func
                                caller_new_name = f"{caller_class}.{caller_name}" if caller_class else caller_name
                                new_caller_func = (caller_file, caller_line, caller_new_name)
                            else:
                                new_caller_func = caller_func
                        else:
                            console.print(f"Unexpected caller format: {caller_func}")
                            new_caller_func = str(caller_func)

                        new_callers[new_caller_func] = count

                    new_timings[new_func] = (cc, ns, tt, ct, new_callers)
                except Exception as e:  # noqa: BLE001
                    console.print(f"Error converting timings for {func}: {e}")
                    continue

            self.stats = new_stats
            self.timings = new_timings

            self.total_tt = sum(tt for _, _, tt, _, _ in self.stats.values())

            total_calls = sum(cc for cc, _, _, _, _ in self.stats.values())
            total_primitive = sum(nc for _, nc, _, _, _ in self.stats.values())

            summary = Text.assemble(
                f"{total_calls:,} function calls ",
                ("(" + f"{total_primitive:,} primitive calls" + ")", "dim"),
                f" in {self.total_tt / 1e6:.3f}milliseconds",
            )

            console.print(Align.center(Panel(summary, border_style="blue", width=80, padding=(0, 2), expand=False)))

            table = Table(
                show_header=True,
                header_style="bold magenta",
                border_style="blue",
                title="[bold]Function Profile[/bold] (ordered by internal time)",
                title_style="cyan",
                caption=f"Showing top 25 of {len(self.stats)} functions",
            )

            table.add_column("Calls", justify="right", style="green", width=10)
            table.add_column("Time (ms)", justify="right", style="cyan", width=10)
            table.add_column("Per Call", justify="right", style="cyan", width=10)
            table.add_column("Cum (ms)", justify="right", style="yellow", width=10)
            table.add_column("Cum/Call", justify="right", style="yellow", width=10)
            table.add_column("Function", style="blue")

            sorted_stats = sorted(
                ((func, stats) for func, stats in self.stats.items() if isinstance(func, tuple) and len(func) == 3),
                key=lambda x: x[1][2],  # Sort by tt (internal time)
                reverse=True,
            )[:25]  # Limit to top 25

            # Format and add each row to the table
            for func, (cc, nc, tt, ct, _) in sorted_stats:
                filename, lineno, funcname = func

                # Format calls - show recursive format if different
                calls_str = f"{cc}/{nc}" if cc != nc else f"{cc:,}"

                # Convert to milliseconds
                tt_ms = tt / 1e6
                ct_ms = ct / 1e6

                # Calculate per-call times
                per_call = tt_ms / cc if cc > 0 else 0
                cum_per_call = ct_ms / nc if nc > 0 else 0
                base_filename = Path(filename).name
                file_link = f"[link=file://{filename}]{base_filename}[/link]"

                table.add_row(
                    calls_str,
                    f"{tt_ms:.3f}",
                    f"{per_call:.3f}",
                    f"{ct_ms:.3f}",
                    f"{cum_per_call:.3f}",
                    f"{funcname} [dim]({file_link}:{lineno})[/dim]",
                )

            console.print(Align.center(table))

        except Exception as e:  # noqa: BLE001
            console.print(f"[bold red]Error in stats processing:[/bold red] {e}")
            console.print(f"Traced {self.trace_count:,} function calls")
            self.total_tt = 0

    def make_pstats_compatible(self) -> None:
        # delete the extra class_name item from the function tuple
        self.files = []
        self.top_level = []
        new_stats = {}
        for func, (cc, ns, tt, ct, callers) in self.stats.items():
            new_callers = {(k[0], k[1], k[2]): v for k, v in callers.items()}
            new_stats[(func[0], func[1], func[2])] = (cc, ns, tt, ct, new_callers)
        new_timings = {}
        for func, (cc, ns, tt, ct, callers) in self.timings.items():
            new_callers = {(k[0], k[1], k[2]): v for k, v in callers.items()}
            new_timings[(func[0], func[1], func[2])] = (cc, ns, tt, ct, new_callers)
        self.stats = new_stats
        self.timings = new_timings

    def dump_stats(self, file: str) -> None:
        with Path(file).open("wb") as f:
            self.create_stats()
            marshal.dump(self.stats, f)

    def create_stats(self) -> None:
        self.simulate_cmd_complete()
        self.snapshot_stats()

    def snapshot_stats(self) -> None:
        self.stats = {}
        for func, (cc, _ns, tt, ct, caller_dict) in self.timings.items():
            callers = caller_dict.copy()
            nc = 0
            for callcnt in callers.values():
                nc += callcnt
            self.stats[func] = cc, nc, tt, ct, callers

    def runctx(self, cmd: str, global_vars: dict[str, Any], local_vars: dict[str, Any]) -> Tracer | None:
        self.__enter__()
        try:
            exec(cmd, global_vars, local_vars)  # noqa: S102
        finally:
            self.__exit__(None, None, None)
        return self


def main() -> ArgumentParser:
    parser = ArgumentParser(allow_abbrev=False)
    parser.add_argument("-o", "--outfile", dest="outfile", help="Save trace to <outfile>", required=True)
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

    if not sys.argv[1:]:
        parser.print_usage()
        sys.exit(2)

    args, unknown_args = parser.parse_known_args()
    sys.argv[:] = unknown_args

    # The script that we're profiling may chdir, so capture the absolute path
    # to the output file at startup.
    if args.outfile is not None:
        args.outfile = Path(args.outfile).resolve()

    if len(unknown_args) > 0:
        if args.module:
            import runpy

            code = "run_module(modname, run_name='__main__')"
            globs = {"run_module": runpy.run_module, "modname": unknown_args[0]}
        else:
            progname = unknown_args[0]
            sys.path.insert(0, str(Path(progname).resolve().parent))
            with io.open_code(progname) as fp:
                code = compile(fp.read(), progname, "exec")
            spec = importlib.machinery.ModuleSpec(name="__main__", loader=None, origin=progname)
            globs = {
                "__spec__": spec,
                "__file__": spec.origin,
                "__name__": spec.name,
                "__package__": None,
                "__cached__": None,
            }
        try:
            Tracer(
                output=args.outfile,
                functions=args.only_functions,
                max_function_count=args.max_function_count,
                timeout=args.tracer_timeout,
                config_file_path=args.codeflash_config,
            ).runctx(code, globs, None)

        except BrokenPipeError as exc:
            # Prevent "Exception ignored" during interpreter shutdown.
            sys.stdout = None
            sys.exit(exc.errno)
    else:
        parser.print_usage()
    return parser


if __name__ == "__main__":
    main()
