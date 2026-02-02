"""Function tracing instrumentation for JavaScript.

This module provides functionality to wrap JavaScript functions to capture their
inputs, outputs, and execution behavior. This is used for generating replay tests
and verifying optimization correctness.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from codeflash.discovery.functions_to_optimize import FunctionToOptimize

logger = logging.getLogger(__name__)


class JavaScriptTracer:
    """Instruments JavaScript code to capture function inputs and outputs.

    Similar to Python's tracing system, this wraps functions to record:
    - Input arguments
    - Return values
    - Exceptions thrown
    - Execution time
    """

    def __init__(self, output_db: Path) -> None:
        """Initialize the tracer.

        Args:
            output_db: Path to SQLite database for storing traces.

        """
        self.output_db = output_db
        self.tracer_var = "__codeflash_tracer__"

    def instrument_source(self, source: str, file_path: Path, functions: list[FunctionToOptimize]) -> str:
        """Instrument JavaScript source code with function tracing.

        Wraps specified functions to capture their inputs and outputs.

        Args:
            source: Original JavaScript source code.
            file_path: Path to the source file.
            functions: List of functions to instrument.

        Returns:
            Instrumented source code with tracing.

        """
        if not functions:
            return source

        # Add tracer initialization at the top
        tracer_init = self._generate_tracer_init()

        # Add instrumentation to each function
        lines = source.splitlines(keepends=True)

        # Process functions in reverse order to preserve line numbers
        for func in sorted(functions, key=lambda f: f.start_line, reverse=True):
            instrumented = self._instrument_function(func, lines, file_path)
            start_idx = func.start_line - 1
            end_idx = func.end_line
            lines = lines[:start_idx] + instrumented + lines[end_idx:]

        instrumented_source = "".join(lines)

        # Add tracer save at the end
        tracer_save = self._generate_tracer_save()

        return tracer_init + "\n" + instrumented_source + "\n" + tracer_save

    def _generate_tracer_init(self) -> str:
        """Generate JavaScript code for tracer initialization."""
        return f"""
// Codeflash function tracer initialization
const {self.tracer_var} = {{
    traces: [],
    callId: 0,

    serialize: function(value) {{
        try {{
            // Handle special cases
            if (value === undefined) return {{ __type__: 'undefined' }};
            if (value === null) return null;
            if (typeof value === 'function') return {{ __type__: 'function', name: value.name }};
            if (typeof value === 'symbol') return {{ __type__: 'symbol', value: value.toString() }};
            if (value instanceof Error) return {{
                __type__: 'error',
                name: value.name,
                message: value.message,
                stack: value.stack
            }};
            if (typeof value === 'bigint') return {{ __type__: 'bigint', value: value.toString() }};
            if (value instanceof Date) return {{ __type__: 'date', value: value.toISOString() }};
            if (value instanceof RegExp) return {{ __type__: 'regexp', value: value.toString() }};
            if (value instanceof Map) return {{
                __type__: 'map',
                value: Array.from(value.entries()).map(([k, v]) => [this.serialize(k), this.serialize(v)])
            }};
            if (value instanceof Set) return {{
                __type__: 'set',
                value: Array.from(value).map(v => this.serialize(v))
            }};

            // Handle circular references with a simple check
            return JSON.parse(JSON.stringify(value));
        }} catch (e) {{
            return {{ __type__: 'unserializable', error: e.message }};
        }}
    }},

    wrap: function(originalFunc, funcName, filePath) {{
        const self = this;

        if (originalFunc.constructor.name === 'AsyncFunction') {{
            return async function(...args) {{
                const callId = self.callId++;
                const start = process.hrtime.bigint();
                let result, error;

                try {{
                    result = await originalFunc.apply(this, args);
                }} catch (e) {{
                    error = e;
                }}

                const end = process.hrtime.bigint();

                self.traces.push({{
                    call_id: callId,
                    function: funcName,
                    file: filePath,
                    args: args.map(a => self.serialize(a)),
                    result: error ? null : self.serialize(result),
                    error: error ? self.serialize(error) : null,
                    runtime_ns: (end - start).toString(),
                    timestamp: Date.now()
                }});

                if (error) throw error;
                return result;
            }};
        }}

        return function(...args) {{
            const callId = self.callId++;
            const start = process.hrtime.bigint();
            let result, error;

            try {{
                result = originalFunc.apply(this, args);
            }} catch (e) {{
                error = e;
            }}

            const end = process.hrtime.bigint();

            self.traces.push({{
                call_id: callId,
                function: funcName,
                file: filePath,
                args: args.map(a => self.serialize(a)),
                result: error ? null : self.serialize(result),
                error: error ? self.serialize(error) : null,
                runtime_ns: (end - start).toString(),
                timestamp: Date.now()
            }});

            if (error) throw error;
            return result;
        }};
    }},

    saveToDb: function() {{
        const sqlite3 = require('sqlite3').verbose();
        const fs = require('fs');
        const path = require('path');

        const dbPath = '{self.output_db.as_posix()}';
        const dbDir = path.dirname(dbPath);

        if (!fs.existsSync(dbDir)) {{
            fs.mkdirSync(dbDir, {{ recursive: true }});
        }}

        const db = new sqlite3.Database(dbPath);

        db.serialize(() => {{
            // Create table
            db.run(`
                CREATE TABLE IF NOT EXISTS traces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    call_id INTEGER,
                    function TEXT,
                    file TEXT,
                    args TEXT,
                    result TEXT,
                    error TEXT,
                    runtime_ns TEXT,
                    timestamp INTEGER
                )
            `);

            // Insert traces
            const stmt = db.prepare(`
                INSERT INTO traces (call_id, function, file, args, result, error, runtime_ns, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            `);

            for (const trace of this.traces) {{
                stmt.run(
                    trace.call_id,
                    trace.function,
                    trace.file,
                    JSON.stringify(trace.args),
                    JSON.stringify(trace.result),
                    JSON.stringify(trace.error),
                    trace.runtime_ns,
                    trace.timestamp
                );
            }}

            stmt.finalize();
        }});

        db.close();
    }},

    saveToJson: function() {{
        const fs = require('fs');
        const path = require('path');

        const jsonPath = '{self.output_db.with_suffix(".json").as_posix()}';
        const jsonDir = path.dirname(jsonPath);

        if (!fs.existsSync(jsonDir)) {{
            fs.mkdirSync(jsonDir, {{ recursive: true }});
        }}

        fs.writeFileSync(jsonPath, JSON.stringify(this.traces, null, 2));
    }}
}};
"""

    def _generate_tracer_save(self) -> str:
        """Generate JavaScript code to save tracer results."""
        return f"""
// Save tracer results on process exit
process.on('exit', () => {{
    try {{
        {self.tracer_var}.saveToJson();
        // Try SQLite, but don't fail if sqlite3 is not installed
        try {{
            {self.tracer_var}.saveToDb();
        }} catch (e) {{
            // SQLite not available, JSON is sufficient
        }}
    }} catch (e) {{
        console.error('Failed to save traces:', e);
    }}
}});
"""

    def _instrument_function(self, func: FunctionToOptimize, lines: list[str], file_path: Path) -> list[str]:
        """Instrument a single function with tracing.

        Args:
            func: Function to instrument.
            lines: Source lines.
            file_path: Path to source file.

        Returns:
            Instrumented function lines.

        """
        func_lines = lines[func.start_line - 1 : func.end_line]
        func_text = "".join(func_lines)

        # Detect function pattern
        func_name = func.name
        is_arrow = "=>" in func_text.split("\n")[0]
        is_method = func.is_method
        is_async = func.is_async

        # Generate wrapper code based on function type
        if is_arrow:
            # For arrow functions: const foo = (a, b) => { ... }
            # Replace with: const foo = __codeflash_tracer__.wrap((a, b) => { ... }, 'foo', 'file.js')
            return self._wrap_arrow_function(func_lines, func_name, file_path)
        if is_method:
            # For methods: methodName(a, b) { ... }
            # Wrap the method body
            return self._wrap_method(func_lines, func_name, file_path, is_async)
        # For regular functions: function foo(a, b) { ... }
        # Wrap the entire function
        return self._wrap_regular_function(func_lines, func_name, file_path, is_async)

    def _wrap_arrow_function(self, func_lines: list[str], func_name: str, file_path: Path) -> list[str]:
        """Wrap an arrow function with tracing."""
        # Find the assignment line
        first_line = func_lines[0]
        indent = len(first_line) - len(first_line.lstrip())
        indent_str = " " * indent

        # Insert wrapper call
        func_text = "".join(func_lines).rstrip()

        # Find the '=' and wrap everything after it
        if "=" in func_text:
            parts = func_text.split("=", 1)
            wrapped = f"{parts[0]}= {self.tracer_var}.wrap({parts[1]}, '{func_name}', '{file_path.as_posix()}');\n"
            return [wrapped]

        return func_lines

    def _wrap_method(self, func_lines: list[str], func_name: str, file_path: Path, is_async: bool) -> list[str]:
        """Wrap a class method with tracing."""
        # For methods, we wrap by reassigning them after definition
        # This is complex, so for now we'll return unwrapped
        # TODO: Implement method wrapping
        logger.warning("Method wrapping not fully implemented for %s", func_name)
        return func_lines

    def _wrap_regular_function(
        self, func_lines: list[str], func_name: str, file_path: Path, is_async: bool
    ) -> list[str]:
        """Wrap a regular function declaration with tracing."""
        # Replace: function foo(a, b) { ... }
        # With: const __original_foo = function foo(a, b) { ... }; const foo = __codeflash_tracer__.wrap(__original_foo, 'foo', 'file.js');

        func_text = "".join(func_lines).rstrip()
        first_line = func_lines[0]
        indent = len(first_line) - len(first_line.lstrip())
        indent_str = " " * indent

        wrapped = (
            f"{indent_str}const __original_{func_name}__ = {func_text};\n"
            f"{indent_str}const {func_name} = {self.tracer_var}.wrap(__original_{func_name}__, '{func_name}', '{file_path.as_posix()}');\n"
        )

        return [wrapped]

    @staticmethod
    def parse_results(trace_file: Path) -> list[dict[str, Any]]:
        """Parse tracing results from output file.

        Args:
            trace_file: Path to traces JSON file.

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

        # Try SQLite database
        if not trace_file.exists():
            return []

        try:
            conn = sqlite3.connect(trace_file)
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM traces ORDER BY id")

            traces = []
            for row in cursor.fetchall():
                traces.append(
                    {
                        "id": row[0],
                        "call_id": row[1],
                        "function": row[2],
                        "file": row[3],
                        "args": json.loads(row[4]),
                        "result": json.loads(row[5]),
                        "error": json.loads(row[6]) if row[6] != "null" else None,
                        "runtime_ns": int(row[7]),
                        "timestamp": row[8],
                    }
                )

            conn.close()
            return traces

        except Exception as e:
            logger.exception("Failed to parse trace database: %s", e)
            return []
