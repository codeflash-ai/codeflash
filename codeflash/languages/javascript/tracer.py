"""Function tracing instrumentation for JavaScript.

This module provides functionality to parse JavaScript function traces and generate
replay tests. Tracing is performed via Babel AST transformation using the
babel-tracer-plugin.js and trace-runner.js in the npm package.

The tracer uses Babel plugin for AST transformation which:
- Works with both CommonJS and ESM
- Handles async functions, arrow functions, methods correctly
- Preserves source maps and formatting

Database Schema (matches Python tracer):
- function_calls: Main trace data (type, function, classname, filename, line_number, time_ns, args)
- metadata: Key-value metadata about the trace session
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import textwrap
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from pathlib import Path

_NON_ALNUM_RE: re.Pattern[str] = re.compile(r"[^a-zA-Z0-9]")

logger = logging.getLogger(__name__)


@dataclass
class JavaScriptFunctionInfo:
    """Information about a traced JavaScript function."""

    function_name: str
    file_name: str
    module_path: str
    class_name: Optional[str] = None
    line_number: Optional[int] = None


class JavaScriptTracer:
    """Parses JavaScript function traces and generates replay tests.

    Tracing is performed via Babel AST transformation (trace-runner.js).
    This class handles:
    - Parsing trace results from SQLite database
    - Extracting traced function information
    - Generating replay test files for Jest/Vitest
    """

    SCHEMA_VERSION = "1.0.0"

    def __init__(self, output_db: Path) -> None:
        """Initialize the tracer.

        Args:
            output_db: Path to SQLite database for storing traces.

        """
        self.output_db = output_db

    @staticmethod
    def parse_results(trace_file: Path) -> list[dict[str, Any]]:
        """Parse tracing results from output file.

        Supports both the new function_calls schema and legacy traces schema.

        Args:
            trace_file: Path to traces file (SQLite or JSON).

        Returns:
            List of trace records.

        """
        json_file = trace_file.with_suffix(".json")

        if json_file.exists():
            try:
                with json_file.open("r") as f:
                    return json.load(f)
            except Exception as e:
                logger.exception("Failed to parse trace JSON: %s", e)
                return []

        if not trace_file.exists():
            return []

        try:
            conn = sqlite3.connect(trace_file)
            cursor = conn.cursor()

            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = {row[0] for row in cursor.fetchall()}

            traces = []

            if "function_calls" in tables:
                cursor.execute(
                    "SELECT type, function, classname, filename, line_number, "
                    "last_frame_address, time_ns, args FROM function_calls ORDER BY time_ns"
                )
                for row in cursor.fetchall():
                    traces.append(
                        {
                            "type": row[0],
                            "function": row[1],
                            "classname": row[2],
                            "filename": row[3],
                            "line_number": row[4],
                            "last_frame_address": row[5],
                            "time_ns": row[6],
                            "args": json.loads(row[7]) if row[7] else [],
                        }
                    )
            elif "traces" in tables:
                # Legacy schema
                cursor.execute("SELECT * FROM traces ORDER BY id")
                for row in cursor.fetchall():
                    traces.append(
                        {
                            "id": row[0],
                            "call_id": row[1],
                            "function": row[2],
                            "file": row[3],
                            "args": json.loads(row[4]) if row[4] else [],
                            "result": json.loads(row[5]) if row[5] else None,
                            "error": json.loads(row[6]) if row[6] and row[6] != "null" else None,
                            "runtime_ns": int(row[7]) if row[7] else 0,
                            "timestamp": row[8] if len(row) > 8 else None,
                        }
                    )

            conn.close()
            return traces

        except Exception as e:
            logger.exception("Failed to parse trace database: %s", e)
            return []

    @staticmethod
    def get_traced_functions(trace_file: Path) -> list[JavaScriptFunctionInfo]:
        """Get list of functions that were traced.

        Args:
            trace_file: Path to trace database.

        Returns:
            List of traced function information.

        """
        if not trace_file.exists():
            return []

        try:
            conn = sqlite3.connect(trace_file)
            cursor = conn.cursor()

            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = {row[0] for row in cursor.fetchall()}

            functions = []

            if "function_calls" in tables:
                cursor.execute(
                    "SELECT DISTINCT function, filename, classname, line_number FROM function_calls WHERE type = 'call'"
                )
                for row in cursor.fetchall():
                    func_name = row[0]
                    file_name = row[1]
                    class_name = row[2]
                    line_number = row[3]

                    module_path = file_name.replace("\\", "/").replace(".js", "").replace(".ts", "")
                    module_path = module_path.removeprefix("./")

                    functions.append(
                        JavaScriptFunctionInfo(
                            function_name=func_name,
                            file_name=file_name,
                            module_path=module_path,
                            class_name=class_name,
                            line_number=line_number,
                        )
                    )

            conn.close()
            return functions

        except Exception as e:
            logger.exception("Failed to get traced functions: %s", e)
            return []

    def create_replay_test(
        self,
        trace_file: Path,
        output_path: Path,
        framework: str = "jest",
        max_run_count: int = 100,
        project_root: Optional[Path] = None,
    ) -> Optional[str]:
        """Generate a replay test file from traced function calls.

        Args:
            trace_file: Path to the trace database.
            output_path: Path to write the test file.
            framework: Test framework ('jest' or 'vitest').
            max_run_count: Maximum number of test cases per function.
            project_root: Project root for calculating relative imports.

        Returns:
            Path to generated test file, or None if generation failed.

        """
        functions = self.get_traced_functions(trace_file)
        if not functions:
            logger.warning("No traced functions found in %s", trace_file)
            return None

        is_vitest = framework.lower() == "vitest"

        imports = []
        if is_vitest:
            imports.append("import { describe, test } from 'vitest';")

        imports.append("const { getNextArg } = require('codeflash/replay');")
        imports.append("")

        for func in functions:
            alias = self._get_function_alias(func.module_path, func.function_name, func.class_name)
            if func.class_name:
                imports.append(f"const {{ {func.class_name}: {alias}_class }} = require('./{func.module_path}');")
            else:
                imports.append(f"const {{ {func.function_name}: {alias} }} = require('./{func.module_path}');")

        imports.append("")

        trace_path = trace_file.as_posix()
        metadata = [
            f"const traceFilePath = '{trace_path}';",
            f"const functions = {json.dumps([f.function_name for f in functions])};",
            "",
        ]

        test_cases = []
        for func in functions:
            alias = self._get_function_alias(func.module_path, func.function_name, func.class_name)
            test_name = f"{func.class_name}.{func.function_name}" if func.class_name else func.function_name
            class_arg = f"'{func.class_name}'" if func.class_name else "null"

            if func.class_name:
                test_cases.append(
                    textwrap.dedent(f"""
describe('Replay: {test_name}', () => {{
    const traces = getNextArg(traceFilePath, '{func.function_name}', '{func.file_name}', {max_run_count}, {class_arg});

    test.each(traces.map((args, i) => [i, args]))('call %i', (index, args) => {{
        const instance = new {alias}_class();
        instance.{func.function_name}(...args);
    }});
}});
""")
                )
            else:
                test_cases.append(
                    textwrap.dedent(f"""
describe('Replay: {test_name}', () => {{
    const traces = getNextArg(traceFilePath, '{func.function_name}', '{func.file_name}', {max_run_count});

    test.each(traces.map((args, i) => [i, args]))('call %i', (index, args) => {{
        {alias}(...args);
    }});
}});
""")
                )

        content = "\n".join(
            [
                "// Auto-generated replay test by Codeflash",
                "// Do not edit this file directly",
                "",
                *imports,
                *metadata,
                *test_cases,
            ]
        )

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(content)
            logger.info("Generated replay test: %s", output_path)
            return str(output_path)
        except Exception as e:
            logger.exception("Failed to write replay test: %s", e)
            return None

    @staticmethod
    def _get_function_alias(module_path: str, function_name: str, class_name: Optional[str] = None) -> str:
        """Create a function alias for imports."""
        module_alias = _NON_ALNUM_RE.sub("_", module_path).strip("_")

        if class_name:
            return f"{module_alias}_{class_name}_{function_name}"
        return f"{module_alias}_{function_name}"
