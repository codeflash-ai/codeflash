from __future__ import annotations

import asyncio
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

import pytest
import dill as pickle

from codeflash.code_utils.codeflash_wrap_decorator import (
    codeflash_behavior_async,
    codeflash_performance_async,
)
from codeflash.verification.codeflash_capture import VerificationType


@pytest.mark.skipif(sys.platform == "win32", reason="pending support for asyncio on windows")
class TestAsyncWrapperSQLiteValidation:

    @pytest.fixture
    def test_env_setup(self, request):
        original_env = {}
        test_env = {
            "CODEFLASH_LOOP_INDEX": "1",
            "CODEFLASH_TEST_ITERATION": "0",
            "CODEFLASH_TEST_MODULE": __name__,
            "CODEFLASH_TEST_CLASS": "TestAsyncWrapperSQLiteValidation",
            "CODEFLASH_TEST_FUNCTION": request.node.name,
            "CODEFLASH_CURRENT_LINE_ID": "test_unit",
        }
        
        for key, value in test_env.items():
            original_env[key] = os.environ.get(key)
            os.environ[key] = value
        
        yield test_env
        
        for key, original_value in original_env.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value

    @pytest.fixture
    def temp_db_path(self, test_env_setup):
        iteration = test_env_setup["CODEFLASH_TEST_ITERATION"]
        from codeflash.code_utils.codeflash_wrap_decorator import get_run_tmp_file
        db_path = get_run_tmp_file(Path(f"test_return_values_{iteration}.sqlite"))
        
        yield db_path
        
        if db_path.exists():
            db_path.unlink()

    @pytest.mark.asyncio
    async def test_behavior_async_basic_function(self, test_env_setup, temp_db_path):
        
        @codeflash_behavior_async
        async def simple_async_add(a: int, b: int) -> int:
            await asyncio.sleep(0.001)
            return a + b

        os.environ['CODEFLASH_CURRENT_LINE_ID'] = 'simple_async_add_59'
        result = await simple_async_add(5, 3)
        
        assert result == 8
        
        assert temp_db_path.exists()
        
        con = sqlite3.connect(temp_db_path)
        cur = con.cursor()
        
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='test_results'")
        assert cur.fetchone() is not None
        
        cur.execute("SELECT * FROM test_results")
        rows = cur.fetchall()
        
        assert len(rows) == 1
        row = rows[0]
        
        (test_module_path, test_class_name, test_function_name, function_getting_tested,
         loop_index, iteration_id, runtime, return_value_blob, verification_type) = row
        
        assert test_module_path == __name__
        assert test_class_name == "TestAsyncWrapperSQLiteValidation" 
        assert test_function_name == "test_behavior_async_basic_function"
        assert function_getting_tested == "simple_async_add"
        assert loop_index == 1
        # Line ID will be the actual line number from the source code, not a simple counter
        assert iteration_id.startswith("simple_async_add_") and iteration_id.endswith("_0")
        assert runtime > 0
        assert verification_type == VerificationType.FUNCTION_CALL.value
        
        unpickled_data = pickle.loads(return_value_blob)
        args, kwargs, return_val = unpickled_data
        
        assert args == (5, 3)
        assert kwargs == {}
        assert return_val == 8
        
        con.close()

    @pytest.mark.asyncio
    async def test_behavior_async_exception_handling(self, test_env_setup, temp_db_path):
        
        @codeflash_behavior_async
        async def async_divide(a: int, b: int) -> float:
            await asyncio.sleep(0.001)
            if b == 0:
                raise ValueError("Cannot divide by zero")
            return a / b

        result = await async_divide(10, 2)
        assert result == 5.0
        
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            await async_divide(10, 0)
        
        con = sqlite3.connect(temp_db_path)
        cur = con.cursor()
        cur.execute("SELECT * FROM test_results ORDER BY iteration_id")
        rows = cur.fetchall()
        
        assert len(rows) == 2
        
        success_row = rows[0]
        success_data = pickle.loads(success_row[7])  # return_value_blob
        args, kwargs, return_val = success_data
        assert args == (10, 2)
        assert return_val == 5.0
        
        # Check exception record
        exception_row = rows[1]
        exception_data = pickle.loads(exception_row[7])  # return_value_blob
        assert isinstance(exception_data, ValueError)
        assert str(exception_data) == "Cannot divide by zero"
        
        con.close()

    @pytest.mark.asyncio
    async def test_performance_async_no_database_storage(self, test_env_setup, temp_db_path, capsys):
        """Test performance async decorator doesn't store to database."""
        
        @codeflash_performance_async
        async def async_multiply(a: int, b: int) -> int:
            """Async function for performance testing."""
            await asyncio.sleep(0.002)
            return a * b

        result = await async_multiply(4, 7)
        
        assert result == 28
        
        assert not temp_db_path.exists()
        
        captured = capsys.readouterr()
        output_lines = captured.out.strip().split('\n')
        
        assert len([line for line in output_lines if "!$######" in line]) == 1
        assert len([line for line in output_lines if "!######" in line and "######!" in line]) == 1
        
        closing_tag = [line for line in output_lines if "!######" in line and "######!" in line][0]
        assert "async_multiply" in closing_tag
        
        timing_part = closing_tag.split(":")[-1].replace("######!", "")
        timing_value = int(timing_part)
        assert timing_value > 0  # Should have positive timing

    @pytest.mark.asyncio
    async def test_multiple_calls_indexing(self, test_env_setup, temp_db_path):
        
        @codeflash_behavior_async
        async def async_increment(value: int) -> int:
            await asyncio.sleep(0.001)
            return value + 1

        # Call the function multiple times
        results = []
        for i in range(3):
            result = await async_increment(i)
            results.append(result)
        
        assert results == [1, 2, 3]
        
        con = sqlite3.connect(temp_db_path)
        cur = con.cursor()
        cur.execute("SELECT iteration_id, return_value FROM test_results ORDER BY iteration_id")
        rows = cur.fetchall()
        
        assert len(rows) == 3
        
        actual_ids = [row[0] for row in rows]
        assert len(actual_ids) == 3
        
        base_pattern = actual_ids[0].rsplit('_', 1)[0]  # e.g., "async_increment_199"
        expected_pattern = [f"{base_pattern}_{i}" for i in range(3)]
        assert actual_ids == expected_pattern
        
        for i, (_, return_value_blob) in enumerate(rows):
            args, kwargs, return_val = pickle.loads(return_value_blob)
            assert args == (i,)
            assert return_val == i + 1
        
        con.close()

    @pytest.mark.asyncio
    async def test_complex_async_function_with_kwargs(self, test_env_setup, temp_db_path):
        
        @codeflash_behavior_async
        async def complex_async_func(
            pos_arg: str,
            *args: int,
            keyword_arg: str = "default",
            **kwargs: str
        ) -> dict:
            await asyncio.sleep(0.001)
            return {
                "pos_arg": pos_arg,
                "args": args,
                "keyword_arg": keyword_arg,
                "kwargs": kwargs,
            }

        result = await complex_async_func(
            "hello",
            1, 2, 3,
            keyword_arg="custom",
            extra1="value1",
            extra2="value2"
        )
        
        expected_result = {
            "pos_arg": "hello",
            "args": (1, 2, 3),
            "keyword_arg": "custom",
            "kwargs": {"extra1": "value1", "extra2": "value2"}
        }
        
        assert result == expected_result
        
        con = sqlite3.connect(temp_db_path)
        cur = con.cursor()
        cur.execute("SELECT return_value FROM test_results")
        row = cur.fetchone()
        
        stored_args, stored_kwargs, stored_result = pickle.loads(row[0])
        
        assert stored_args == ("hello", 1, 2, 3)
        assert stored_kwargs == {"keyword_arg": "custom", "extra1": "value1", "extra2": "value2"}
        assert stored_result == expected_result
        
        con.close()

    @pytest.mark.asyncio
    async def test_database_schema_validation(self, test_env_setup, temp_db_path):
        
        @codeflash_behavior_async
        async def schema_test_func() -> str:
            return "schema_test"
        
        await schema_test_func()
        
        con = sqlite3.connect(temp_db_path)
        cur = con.cursor()
        
        cur.execute("PRAGMA table_info(test_results)")
        columns = cur.fetchall()
        
        expected_columns = [
            (0, 'test_module_path', 'TEXT', 0, None, 0),
            (1, 'test_class_name', 'TEXT', 0, None, 0),
            (2, 'test_function_name', 'TEXT', 0, None, 0),
            (3, 'function_getting_tested', 'TEXT', 0, None, 0),
            (4, 'loop_index', 'INTEGER', 0, None, 0),
            (5, 'iteration_id', 'TEXT', 0, None, 0),
            (6, 'runtime', 'INTEGER', 0, None, 0),
            (7, 'return_value', 'BLOB', 0, None, 0),
            (8, 'verification_type', 'TEXT', 0, None, 0)
        ]
        
        assert columns == expected_columns
        con.close()

