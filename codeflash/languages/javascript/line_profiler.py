"""Line profiler instrumentation for JavaScript.

This module provides functionality to instrument JavaScript code with line-level
profiling similar to Python's line_profiler. It tracks execution counts and timing
for each line in instrumented functions.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from codeflash.languages.treesitter_utils import get_analyzer_for_file

if TYPE_CHECKING:
    from pathlib import Path

    from codeflash.languages.base import FunctionInfo

logger = logging.getLogger(__name__)


class JavaScriptLineProfiler:
    """Instruments JavaScript code for line-level profiling.

    This class adds profiling code to JavaScript functions to track:
    - How many times each line executes
    - How much time is spent on each line
    - Total execution time per function
    """

    def __init__(self, output_file: Path) -> None:
        """Initialize the line profiler.

        Args:
            output_file: Path where profiling results will be written.

        """
        self.output_file = output_file
        self.profiler_var = "__codeflash_line_profiler__"

    def instrument_source(self, source: str, file_path: Path, functions: list[FunctionInfo]) -> str:
        """Instrument JavaScript source code with line profiling.

        Adds profiling instrumentation to track line-level execution for the
        specified functions.

        Args:
            source: Original JavaScript source code.
            file_path: Path to the source file.
            functions: List of functions to instrument.

        Returns:
            Instrumented source code with profiling.

        """
        if not functions:
            return source

        # Initialize line contents map to collect source content during instrumentation
        self.line_contents: dict[str, str] = {}

        # Add instrumentation to each function
        lines = source.splitlines(keepends=True)

        # Process functions in reverse order to preserve line numbers
        for func in sorted(functions, key=lambda f: f.start_line, reverse=True):
            func_lines = self._instrument_function(func, lines, file_path)
            start_idx = func.start_line - 1
            end_idx = func.end_line
            lines = lines[:start_idx] + func_lines + lines[end_idx:]

        instrumented_source = "".join(lines)

        # Add profiler initialization at the top (after collecting line contents)
        profiler_init = self._generate_profiler_init()

        # Add profiler save at the end
        profiler_save = self._generate_profiler_save()

        return profiler_init + "\n" + instrumented_source + "\n" + profiler_save

    def _generate_profiler_init(self) -> str:
        """Generate JavaScript code for profiler initialization."""
        # Serialize line contents map for embedding in JavaScript
        line_contents_json = json.dumps(getattr(self, "line_contents", {}))

        return f"""
// Codeflash line profiler initialization
// @ts-nocheck
const {self.profiler_var} = {{
    stats: {{}},
    lineContents: {line_contents_json},
    lastLineTime: null,
    lastKey: null,

    totalHits: 0,

    // Called at the start of each function to reset timing state
    // This prevents "between function calls" time from being attributed to the last line
    enterFunction: function() {{
        this.lastKey = null;
        this.lastLineTime = null;
    }},

    hit: function(file, line) {{
        const now = performance.now();  // microsecond precision

        // Attribute elapsed time to the PREVIOUS line (the one that was executing)
        if (this.lastKey !== null && this.lastLineTime !== null) {{
            this.stats[this.lastKey].time += (now - this.lastLineTime);
        }}

        const key = file + ':' + line;
        if (!this.stats[key]) {{
            this.stats[key] = {{ hits: 0, time: 0, file: file, line: line }};
        }}
        this.stats[key].hits++;

        // Record current line as the one now executing
        this.lastKey = key;
        this.lastLineTime = now;

        this.totalHits++;
        // Save every 100 hits to ensure we capture results even with --forceExit
        if (this.totalHits % 100 === 0) {{
            this.save();
        }}
    }},

    save: function() {{
        const fs = require('fs');
        const pathModule = require('path');
        const outputDir = pathModule.dirname('{self.output_file.as_posix()}');
        try {{
            if (!fs.existsSync(outputDir)) {{
                fs.mkdirSync(outputDir, {{ recursive: true }});
            }}
            // Merge line contents into stats before saving
            const statsWithContent = {{}};
            for (const key of Object.keys(this.stats)) {{
                statsWithContent[key] = {{
                    ...this.stats[key],
                    content: this.lineContents[key] || ''
                }};
            }}
            fs.writeFileSync(
                '{self.output_file.as_posix()}',
                JSON.stringify(statsWithContent, null, 2)
            );
        }} catch (e) {{
            console.error('Failed to save line profile results:', e);
        }}
    }}
}};
"""

    def _generate_profiler_save(self) -> str:
        """Generate JavaScript code to save profiler results."""
        return f"""
// Save profiler results on process exit and periodically
// Use beforeExit for graceful shutdowns
process.on('beforeExit', () => {self.profiler_var}.save());
process.on('exit', () => {self.profiler_var}.save());
process.on('SIGINT', () => {{ {self.profiler_var}.save(); process.exit(); }});
process.on('SIGTERM', () => {{ {self.profiler_var}.save(); process.exit(); }});

// For Jest --forceExit compatibility, save periodically (every 500ms)
const __codeflash_save_interval__ = setInterval(() => {self.profiler_var}.save(), 500);
if (__codeflash_save_interval__.unref) __codeflash_save_interval__.unref(); // Don't keep process alive
"""

    def _instrument_function(self, func: FunctionInfo, lines: list[str], file_path: Path) -> list[str]:
        """Instrument a single function with line profiling.

        Args:
            func: Function to instrument.
            lines: Source lines.
            file_path: Path to source file.

        Returns:
            Instrumented function lines.

        """
        func_lines = lines[func.start_line - 1 : func.end_line]
        instrumented_lines = []

        # Parse the function to find executable lines
        analyzer = get_analyzer_for_file(file_path)
        source = "".join(func_lines)

        try:
            tree = analyzer.parse(source.encode("utf8"))
            executable_lines = self._find_executable_lines(tree.root_node, source.encode("utf8"))
        except Exception as e:
            logger.warning(f"Failed to parse function {func.name}: {e}")
            return func_lines

        # Add profiling to each executable line
        # executable_lines contains 1-indexed line numbers within the function snippet
        function_entry_added = False

        for local_idx, line in enumerate(func_lines):
            local_line_num = local_idx + 1  # 1-indexed within function
            global_line_num = func.start_line + local_idx  # Global line number in original file
            stripped = line.strip()

            # Add enterFunction() call after the opening brace of the function
            if not function_entry_added and "{" in line:
                # Find indentation for the function body (use next line's indentation or default)
                body_indent = "    "  # Default 4 spaces
                if local_idx + 1 < len(func_lines):
                    next_line = func_lines[local_idx + 1]
                    if next_line.strip():
                        body_indent = " " * (len(next_line) - len(next_line.lstrip()))

                # Add the line with enterFunction() call after it
                instrumented_lines.append(line)
                instrumented_lines.append(f"{body_indent}{self.profiler_var}.enterFunction();\n")
                function_entry_added = True
                continue

            # Skip empty lines, comments, and closing braces
            if local_line_num in executable_lines and stripped and not stripped.startswith("//") and stripped != "}":
                # Get indentation
                indent = len(line) - len(line.lstrip())
                indent_str = " " * indent

                # Store line content for the profiler output
                content_key = f"{file_path.as_posix()}:{global_line_num}"
                self.line_contents[content_key] = stripped

                # Add hit() call before the line
                profiled_line = (
                    f"{indent_str}{self.profiler_var}.hit('{file_path.as_posix()}', {global_line_num});\n{line}"
                )
                instrumented_lines.append(profiled_line)
            else:
                instrumented_lines.append(line)

        return instrumented_lines

    def _find_executable_lines(self, node, source_bytes: bytes) -> set[int]:
        """Find lines that contain executable statements.

        Args:
            node: Tree-sitter AST node.
            source_bytes: Source code as bytes.

        Returns:
            Set of line numbers with executable statements.

        """
        executable_lines = set()

        # Node types that represent executable statements
        executable_types = {
            "expression_statement",
            "return_statement",
            "if_statement",
            "for_statement",
            "while_statement",
            "do_statement",
            "switch_statement",
            "throw_statement",
            "try_statement",
            "variable_declaration",
            "lexical_declaration",
            "assignment_expression",
            "call_expression",
            "await_expression",
        }

        def walk(n) -> None:
            if n.type in executable_types:
                # Add the starting line (1-indexed)
                executable_lines.add(n.start_point[0] + 1)

            for child in n.children:
                walk(child)

        walk(node)
        return executable_lines

    @staticmethod
    def parse_results(profile_file: Path) -> dict:
        """Parse line profiling results from output file.

        Args:
            profile_file: Path to profiling results JSON file.

        Returns:
            Dictionary with profiling statistics.

        """
        if not profile_file.exists():
            return {"timings": {}, "unit": 1e-9, "functions": {}}

        try:
            with profile_file.open("r") as f:
                data = json.load(f)

            # Group by file and function
            timings = {}
            for key, stats in data.items():
                file_path, line_num = key.rsplit(":", 1)
                line_num = int(line_num)
                # performance.now() returns milliseconds, convert to nanoseconds
                time_ms = float(stats["time"])
                time_ns = int(time_ms * 1e6)
                hits = stats["hits"]

                if file_path not in timings:
                    timings[file_path] = {}

                content = stats.get("content", "")
                timings[file_path][line_num] = {
                    "hits": hits,
                    "time_ns": time_ns,
                    "time_ms": time_ms,
                    "content": content,
                }

            return {
                "timings": timings,
                "unit": 1e-9,  # nanoseconds
                "raw_data": data,
            }

        except Exception as e:
            logger.exception(f"Failed to parse line profile results: {e}")
            return {"timings": {}, "unit": 1e-9, "functions": {}}
