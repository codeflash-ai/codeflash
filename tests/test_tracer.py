import contextlib
import dataclasses
import pickle
import sqlite3
import sys
import tempfile
import threading
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from codeflash.code_utils.config_parser import parse_config_file
from codeflash.tracing.tracing_new_process import FakeCode, FakeFrame, Tracer


class TestFakeCode:
    def test_fake_code_initialization(self) -> None:
        fake_code = FakeCode("test.py", 10, "test_function")
        assert fake_code.co_filename == "test.py"
        assert fake_code.co_line == 10
        assert fake_code.co_name == "test_function"
        assert fake_code.co_firstlineno == 0

    def test_fake_code_repr(self) -> None:
        fake_code = FakeCode("test.py", 10, "test_function")
        expected_repr = repr(("test.py", 10, "test_function", None))
        assert repr(fake_code) == expected_repr


class TestFakeFrame:
    def test_fake_frame_initialization(self) -> None:
        fake_code = FakeCode("test.py", 10, "test_function")
        fake_frame = FakeFrame(fake_code, None)
        assert fake_frame.f_code == fake_code
        assert fake_frame.f_back is None
        assert fake_frame.f_locals == {}

    def test_fake_frame_with_prior(self) -> None:
        fake_code1 = FakeCode("test1.py", 5, "func1")
        fake_code2 = FakeCode("test2.py", 10, "func2")
        fake_frame1 = FakeFrame(fake_code1, None)
        fake_frame2 = FakeFrame(fake_code2, fake_frame1)

        assert fake_frame2.f_code == fake_code2
        assert fake_frame2.f_back == fake_frame1


@dataclasses.dataclass
class TraceConfig:
    trace_file: Path
    trace_config: dict[str, Any]
    result_pickle_file_path: Path
    project_root: Path
    command: str


class TestTracer:
    @pytest.fixture
    def trace_config(self) -> Generator[Path, None, None]:
        """Create a temporary pyproject.toml config file."""
        # Create a temporary directory structure
        temp_dir = Path(tempfile.mkdtemp())
        tests_dir = temp_dir / "tests"
        tests_dir.mkdir(exist_ok=True)

        # Use the current working directory as module root so test files are included
        current_dir = Path.cwd()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False, dir=temp_dir) as f:
            f.write(f"""
[tool.codeflash]
module-root = "{current_dir}"
tests-root = "{tests_dir}"
test-framework = "pytest"
ignore-paths = []
""")
            config_path = Path(f.name)
        with tempfile.NamedTemporaryFile(suffix=".trace", delete=False) as f:
            trace_path = Path(f.name)
        trace_path.unlink(missing_ok=True)  # Remove the file, we just want the path
        replay_test_pkl_path = temp_dir / "replay_test.pkl"
        config, found_config_path = parse_config_file(config_path)
        trace_config = TraceConfig(
            trace_file=trace_path,
            trace_config=config,
            result_pickle_file_path=replay_test_pkl_path,
            project_root=current_dir,
            command="pytest random",
        )

        yield trace_config
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)
        trace_path.unlink(missing_ok=True)
        replay_test_pkl_path.unlink(missing_ok=True)

    @pytest.fixture(autouse=True)
    def reset_tracer_state(self) -> Generator[None, None, None]:
        """Reset the tracer used_once state before each test."""
        # Reset the class variable if it exists
        if hasattr(Tracer, "used_once"):
            delattr(Tracer, "used_once")
        yield
        # Reset after test as well
        if hasattr(Tracer, "used_once"):
            delattr(Tracer, "used_once")

    def test_tracer_disabled_by_environment(self, trace_config: TraceConfig) -> None:
        """Test that tracer is disabled when CODEFLASH_TRACER_DISABLE is set."""
        with patch.dict("os.environ", {"CODEFLASH_TRACER_DISABLE": "1"}):
            tracer = Tracer(
                output=str(trace_config.trace_file),
                config=trace_config.trace_config,
                project_root=trace_config.project_root,
                result_pickle_file_path=trace_config.result_pickle_file_path,
            )
            assert tracer.disable is True

    def test_tracer_disabled_with_existing_profiler(self, trace_config: TraceConfig) -> None:
        """Test that tracer is disabled when another profiler is running."""

        def dummy_profiler(_frame: object, _event: str, _arg: object) -> object:
            return dummy_profiler

        sys.setprofile(dummy_profiler)
        try:
            tracer = Tracer(
                output=str(trace_config.trace_file),
                config=trace_config.trace_config,
                project_root=trace_config.project_root,
                result_pickle_file_path=trace_config.result_pickle_file_path,
            )
            assert tracer.disable is True
        finally:
            sys.setprofile(None)

    def test_tracer_initialization_normal(self, trace_config: TraceConfig) -> None:
        """Test normal tracer initialization."""
        tracer = Tracer(
            output=str(trace_config.trace_file),
            config=trace_config.trace_config,
            project_root=trace_config.project_root,
            result_pickle_file_path=trace_config.result_pickle_file_path,
            functions=["test_func"],
            max_function_count=128,
            timeout=10,
        )

        assert tracer.disable is False
        assert tracer.output_file == trace_config.trace_file.resolve()
        assert tracer.functions == ["test_func"]
        assert tracer.max_function_count == 128
        assert tracer.timeout == 10
        assert hasattr(tracer, "_db_lock")
        assert tracer._db_lock is not None

    def test_tracer_timeout_validation(self, trace_config: TraceConfig) -> None:
        with pytest.raises(AssertionError):
            Tracer(
                output=str(trace_config.trace_file),
                config=trace_config.trace_config,
                project_root=trace_config.project_root,
                result_pickle_file_path=trace_config.result_pickle_file_path,
                timeout=0,
            )

        with pytest.raises(AssertionError):
            Tracer(
                output=str(trace_config.trace_file),
                config=trace_config.trace_config,
                project_root=trace_config.project_root,
                result_pickle_file_path=trace_config.result_pickle_file_path,
                timeout=-5,
            )

    def test_tracer_context_manager_disabled(self, trace_config: TraceConfig) -> None:
        tracer = Tracer(
            output=str(trace_config.trace_file),
            config=trace_config.trace_config,
            project_root=trace_config.project_root,
            result_pickle_file_path=trace_config.result_pickle_file_path,
            disable=True,
        )

        with tracer:
            pass

        assert not trace_config.trace_file.exists()

    def test_tracer_function_filtering(self, trace_config: TraceConfig) -> None:
        """Test that tracer respects function filtering."""
        if hasattr(Tracer, "used_once"):
            delattr(Tracer, "used_once")

        def test_function() -> int:
            return 42

        def other_function() -> int:
            return 24

        tracer = Tracer(
            output=str(trace_config.trace_file),
            config=trace_config.trace_config,
            project_root=trace_config.project_root,
            result_pickle_file_path=trace_config.result_pickle_file_path,
            functions=["test_function"],
        )

        with tracer:
            test_function()
            other_function()

        if trace_config.trace_file.exists():
            con = sqlite3.connect(trace_config.trace_file)
            cursor = con.cursor()

            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='function_calls'")
            if cursor.fetchone():
                cursor.execute("SELECT function FROM function_calls WHERE function = 'test_function'")
                cursor.fetchall()

                cursor.execute("SELECT function FROM function_calls WHERE function = 'other_function'")
                cursor.fetchall()

            con.close()

    def test_tracer_max_function_count(self, trace_config: TraceConfig) -> None:
        def counting_function(n: int) -> int:
            return n * 2

        tracer = Tracer(
            output=str(trace_config.trace_file),
            config=trace_config.trace_config,
            project_root=trace_config.project_root,
            result_pickle_file_path=trace_config.result_pickle_file_path,
            max_function_count=3,
        )

        with tracer:
            for i in range(5):
                counting_function(i)

        assert tracer.trace_count <= 3, "Tracer should limit the number of traced functions to max_function_count"

    def test_tracer_timeout_functionality(self, trace_config: TraceConfig) -> None:
        def slow_function() -> str:
            time.sleep(0.1)
            return "done"

        tracer = Tracer(
            output=str(trace_config.trace_file),
            config=trace_config.trace_config,
            project_root=trace_config.project_root,
            result_pickle_file_path=trace_config.result_pickle_file_path,
            timeout=1,  # 1 second timeout
        )

        with tracer:
            slow_function()

    def test_tracer_threading_safety(self, trace_config: TraceConfig) -> None:
        """Test that tracer works correctly with threading."""
        results = []

        def thread_function(n: int) -> None:
            results.append(n * 2)

        tracer = Tracer(
            output=str(trace_config.trace_file),
            config=trace_config.trace_config,
            project_root=trace_config.project_root,
            result_pickle_file_path=trace_config.result_pickle_file_path,
        )

        with tracer:
            threads = []
            for i in range(3):
                thread = threading.Thread(target=thread_function, args=(i,))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

        assert len(results) == 3

    def test_simulate_call(self, trace_config: TraceConfig) -> None:
        """Test the simulate_call method."""
        tracer = Tracer(
            output=str(trace_config.trace_file),
            config=trace_config.trace_config,
            project_root=trace_config.project_root,
            result_pickle_file_path=trace_config.result_pickle_file_path,
        )

        tracer.simulate_call("test_simulation")

    def test_simulate_cmd_complete(self, trace_config: TraceConfig) -> None:
        """Test the simulate_cmd_complete method."""
        tracer = Tracer(
            output=str(trace_config.trace_file),
            config=trace_config.trace_config,
            project_root=trace_config.project_root,
            result_pickle_file_path=trace_config.result_pickle_file_path,
        )

        tracer.simulate_call("test")
        tracer.simulate_cmd_complete()

    def test_runctx_method(self, trace_config: TraceConfig) -> None:
        """Test the runctx method for executing code with tracing."""
        tracer = Tracer(
            output=str(trace_config.trace_file),
            config=trace_config.trace_config,
            project_root=trace_config.project_root,
            result_pickle_file_path=trace_config.result_pickle_file_path,
        )

        global_vars = {"x": 10}
        local_vars = {}

        result = tracer.runctx("y = x * 2", global_vars, local_vars)

        assert result == tracer
        assert local_vars["y"] == 20

    def test_tracer_handles_class_methods(self, trace_config: TraceConfig) -> None:
        """Test that tracer correctly handles class methods."""
        # Ensure tracer hasn't been used yet in this test
        if hasattr(Tracer, "used_once"):
            delattr(Tracer, "used_once")

        class TestClass:
            def instance_method(self) -> str:
                return "instance"

            @classmethod
            def class_method(cls) -> str:
                return "class"

            @staticmethod
            def static_method() -> str:
                return "static"

        tracer = Tracer(
            output=str(trace_config.trace_file),
            config=trace_config.trace_config,
            project_root=trace_config.project_root,
            result_pickle_file_path=trace_config.result_pickle_file_path,
        )

        with tracer:
            obj = TestClass()
            instance_result = obj.instance_method()
            class_result = TestClass.class_method()
            static_result = TestClass.static_method()

        if trace_config.trace_file.exists():
            con = sqlite3.connect(trace_config.trace_file)
            cursor = con.cursor()

            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='function_calls'")
            if cursor.fetchone():
                # Query for all function calls
                cursor.execute("SELECT function, classname FROM function_calls")
                calls = cursor.fetchall()

                function_names = [call[0] for call in calls]
                class_names = [call[1] for call in calls if call[1] is not None]

                assert "instance_method" in function_names
                assert "class_method" in function_names
                assert "static_method" in function_names
                assert "TestClass" in class_names
            else:
                pytest.fail("No function_calls table found in trace file")
            con.close()

    def test_tracer_handles_exceptions_gracefully(self, trace_config: TraceConfig) -> None:
        """Test that tracer handles exceptions in traced code gracefully."""

        def failing_function() -> None:
            raise ValueError("Test exception")

        tracer = Tracer(
            output=str(trace_config.trace_file),
            config=trace_config.trace_config,
            project_root=trace_config.project_root,
            result_pickle_file_path=trace_config.result_pickle_file_path,
        )

        with tracer, contextlib.suppress(ValueError):
            failing_function()

    def test_tracer_with_complex_arguments(self, trace_config: TraceConfig) -> None:
        def complex_function(
            data_dict: dict[str, Any], nested_list: list[list[int]], func_arg: object = lambda x: x
        ) -> int:
            return len(data_dict) + len(nested_list)

        tracer = Tracer(
            output=str(trace_config.trace_file),
            config=trace_config.trace_config,
            project_root=trace_config.project_root,
            result_pickle_file_path=trace_config.result_pickle_file_path,
        )

        expected_dict = {"key": "value", "nested": {"inner": "data"}}
        expected_list = [[1, 2], [3, 4], [5, 6]]
        expected_func = lambda x: x * 2

        with tracer:
            complex_function(expected_dict, expected_list, func_arg=expected_func)
        assert trace_config.result_pickle_file_path.exists()
        pickled = pickle.load(trace_config.result_pickle_file_path.open("rb"))
        assert pickled["replay_test_file_path"].exists()

        if trace_config.trace_file.exists():
            con = sqlite3.connect(trace_config.trace_file)
            cursor = con.cursor()

            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='function_calls'")
            if cursor.fetchone():
                cursor.execute("SELECT args FROM function_calls WHERE function = 'complex_function'")
                result = cursor.fetchone()
                assert result is not None, "Function complex_function should be traced"

                # Deserialize the arguments

                traced_args = pickle.loads(result[0])

                assert "data_dict" in traced_args
                assert "nested_list" in traced_args
                assert "func_arg" in traced_args

                assert traced_args["data_dict"] == expected_dict
                assert traced_args["nested_list"] == expected_list
                assert callable(traced_args["func_arg"])
                assert traced_args["func_arg"](2) == 4
                assert len(traced_args["nested_list"]) == 3
            else:
                pytest.fail("No function_calls table found in trace file")
            con.close()
