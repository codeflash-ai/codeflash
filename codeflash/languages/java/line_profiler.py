"""Line profiler instrumentation for Java.

This module provides functionality to instrument Java code with line-level
profiling similar to Python's line_profiler and JavaScript's profiler.
It tracks execution counts and timing for each line in instrumented functions.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tree_sitter import Node

    from codeflash.languages.base import FunctionInfo

logger = logging.getLogger(__name__)


class JavaLineProfiler:
    """Instruments Java code for line-level profiling.

    This class adds profiling code to Java functions to track:
    - How many times each line executes
    - How much time is spent on each line (in nanoseconds)
    - Total execution time per function

    Example:
        profiler = JavaLineProfiler(output_file=Path("profile.json"))
        instrumented = profiler.instrument_source(source, file_path, functions)
        # Run instrumented code
        results = JavaLineProfiler.parse_results(Path("profile.json"))
    """

    def __init__(self, output_file: Path) -> None:
        """Initialize the line profiler.

        Args:
            output_file: Path where profiling results will be written (JSON format).

        """
        self.output_file = output_file
        self.profiler_class = "CodeflashLineProfiler"
        self.profiler_var = "__codeflashProfiler__"
        self.line_contents: dict[str, str] = {}

    def instrument_source(
        self,
        source: str,
        file_path: Path,
        functions: list[FunctionInfo],
        analyzer=None,
    ) -> str:
        """Instrument Java source code with line profiling.

        Adds profiling instrumentation to track line-level execution for the
        specified functions.

        Args:
            source: Original Java source code.
            file_path: Path to the source file.
            functions: List of functions to instrument.
            analyzer: Optional JavaAnalyzer instance.

        Returns:
            Instrumented source code with profiling.

        """
        if not functions:
            return source

        if analyzer is None:
            from codeflash.languages.java.parser import get_java_analyzer

            analyzer = get_java_analyzer()

        # Initialize line contents map
        self.line_contents = {}

        lines = source.splitlines(keepends=True)

        # Process functions in reverse order to preserve line numbers
        for func in sorted(functions, key=lambda f: f.starting_line, reverse=True):
            func_lines = self._instrument_function(func, lines, file_path, analyzer)
            start_idx = func.starting_line - 1
            end_idx = func.ending_line
            lines = lines[:start_idx] + func_lines + lines[end_idx:]

        # Add profiler class and initialization
        profiler_class_code = self._generate_profiler_class()

        # Insert profiler class before the package's first class
        # Find the first class/interface/enum/record declaration
        # Must handle any combination of modifiers: public final class, abstract class, etc.
        class_pattern = re.compile(
            r"^(?:(?:public|private|protected|final|abstract|static|sealed|non-sealed)\s+)*"
            r"(?:class|interface|enum|record)\s+"
        )
        import_end_idx = 0
        for i, line in enumerate(lines):
            if class_pattern.match(line.strip()):
                import_end_idx = i
                break

        lines_with_profiler = (
            lines[:import_end_idx] + [profiler_class_code + "\n"] + lines[import_end_idx:]
        )

        return "".join(lines_with_profiler)

    def _generate_profiler_class(self) -> str:
        """Generate Java code for profiler class."""
        # Store line contents as a simple map (embedded directly in code)
        line_contents_code = self._generate_line_contents_map()

        return f'''
/**
 * Codeflash line profiler - tracks per-line execution statistics.
 * Auto-generated - do not modify.
 */
class {self.profiler_class} {{
    private static final java.util.Map<String, LineStats> stats = new java.util.concurrent.ConcurrentHashMap<>();
    private static final java.util.Map<String, String> lineContents = initLineContents();
    private static final ThreadLocal<Long> lastLineTime = new ThreadLocal<>();
    private static final ThreadLocal<String> lastKey = new ThreadLocal<>();
    private static final java.util.concurrent.atomic.AtomicInteger totalHits = new java.util.concurrent.atomic.AtomicInteger(0);
    private static final String OUTPUT_FILE = "{str(self.output_file)}";

    static class LineStats {{
        public final java.util.concurrent.atomic.AtomicLong hits = new java.util.concurrent.atomic.AtomicLong(0);
        public final java.util.concurrent.atomic.AtomicLong timeNs = new java.util.concurrent.atomic.AtomicLong(0);
        public String file;
        public int line;

        public LineStats(String file, int line) {{
            this.file = file;
            this.line = line;
        }}
    }}

    private static java.util.Map<String, String> initLineContents() {{
        java.util.Map<String, String> map = new java.util.HashMap<>();
{line_contents_code}
        return map;
    }}

    /**
     * Called at the start of each instrumented function to reset timing state.
     */
    public static void enterFunction() {{
        lastKey.set(null);
        lastLineTime.set(null);
    }}

    /**
     * Record a hit on a specific line.
     *
     * @param file The source file path
     * @param line The line number
     */
    public static void hit(String file, int line) {{
        long now = System.nanoTime();

        // Attribute elapsed time to the PREVIOUS line (the one that was executing)
        String prevKey = lastKey.get();
        Long prevTime = lastLineTime.get();

        if (prevKey != null && prevTime != null) {{
            LineStats prevStats = stats.get(prevKey);
            if (prevStats != null) {{
                prevStats.timeNs.addAndGet(now - prevTime);
            }}
        }}

        String key = file + ":" + line;
        stats.computeIfAbsent(key, k -> new LineStats(file, line)).hits.incrementAndGet();

        // Record current line as the one now executing
        lastKey.set(key);
        lastLineTime.set(now);

        int hits = totalHits.incrementAndGet();

        // Save every 100 hits to ensure we capture results even if JVM exits abruptly
        if (hits % 100 == 0) {{
            save();
        }}
    }}

    /**
     * Save profiling results to output file.
     */
    public static synchronized void save() {{
        try {{
            java.io.File outputFile = new java.io.File(OUTPUT_FILE);
            java.io.File parentDir = outputFile.getParentFile();
            if (parentDir != null && !parentDir.exists()) {{
                parentDir.mkdirs();
            }}

            // Build JSON with stats
            StringBuilder json = new StringBuilder();
            json.append("{{\\n");

            boolean first = true;
            for (java.util.Map.Entry<String, LineStats> entry : stats.entrySet()) {{
                if (!first) json.append(",\\n");
                first = false;

                String key = entry.getKey();
                LineStats st = entry.getValue();
                String content = lineContents.getOrDefault(key, "");

                // Escape quotes in content
                content = content.replace("\\"", "\\\\\\"");

                json.append("  \\"").append(key).append("\\": {{\\n");
                json.append("    \\"hits\\": ").append(st.hits.get()).append(",\\n");
                json.append("    \\"time\\": ").append(st.timeNs.get()).append(",\\n");
                json.append("    \\"file\\": \\"").append(st.file).append("\\",\\n");
                json.append("    \\"line\\": ").append(st.line).append(",\\n");
                json.append("    \\"content\\": \\"").append(content).append("\\"\\n");
                json.append("  }}");
            }}

            json.append("\\n}}");

            java.nio.file.Files.write(
                outputFile.toPath(),
                json.toString().getBytes(java.nio.charset.StandardCharsets.UTF_8)
            );
        }} catch (Exception e) {{
            System.err.println("Failed to save line profile results: " + e.getMessage());
        }}
    }}

    // Register shutdown hook to save results on JVM exit
    static {{
        Runtime.getRuntime().addShutdownHook(new Thread(() -> save()));
    }}
}}
'''

    def _instrument_function(
        self,
        func: FunctionInfo,
        lines: list[str],
        file_path: Path,
        analyzer,
    ) -> list[str]:
        """Instrument a single function with line profiling.

        Args:
            func: Function to instrument.
            lines: Source lines.
            file_path: Path to source file.
            analyzer: JavaAnalyzer instance.

        Returns:
            Instrumented function lines.

        """
        func_lines = lines[func.starting_line - 1 : func.ending_line]
        instrumented_lines = []

        # Parse the function to find executable lines
        source = "".join(func_lines)

        try:
            tree = analyzer.parse(source.encode("utf8"))
            executable_lines = self._find_executable_lines(tree.root_node)
        except Exception as e:
            logger.warning("Failed to parse function %s: %s", func.function_name, e)
            return func_lines

        # Add profiling to each executable line
        function_entry_added = False

        
        # Precompute constants used in loop
        profiler_class = self.profiler_class
        file_path_posix = file_path.as_posix()
        enter_function_call = f"{profiler_class}.enterFunction();\n"

        for local_idx, line in enumerate(func_lines):
            local_line_num = local_idx + 1  # 1-indexed within function
            global_line_num = func.starting_line + local_idx  # Global line number
            stripped = line.strip()

            # Add enterFunction() call after the method's opening brace
            if not function_entry_added and "{" in line:
                # Find indentation for the function body
                body_indent = "        "  # Default 8 spaces (class + method indent)
                if local_idx + 1 < len(func_lines):
                    next_line = func_lines[local_idx + 1]
                    if next_line.strip():
                        body_indent = " " * (len(next_line) - len(next_line.lstrip()))

                # Add the line with enterFunction() call after it
                instrumented_lines.append(line)
                instrumented_lines.append(f"{body_indent}{enter_function_call}")
                function_entry_added = True
                continue

            # Skip empty lines, comments, closing braces
            # Match Python's line_profiler behavior:
            # - Track: if, elif (else if), try, catch (except), while, for
            # - Skip: else (standalone), finally

            # Check if this is an else-if statement (like Python's elif)
            # Tree-sitter marks "} else if" lines as executable (contains if_statement)
            # We should track these, not skip them (matches Python's elif tracking)
            is_else_if = "else if" in stripped

            # Skip standalone else and finally (matches Python behavior)
            # But NOT else-if, which should be tracked
            is_standalone_else_or_finally = False
            if not is_else_if:  # Only check if it's not else-if
                # Check for standalone else
                if stripped.startswith(("else{", "else {", "} else{", "} else {")) or stripped == "else":
                    is_standalone_else_or_finally = True
                # Check for finally
                elif stripped.startswith(("finally{", "finally {", "} finally{", "} finally {")) or stripped == "finally":
                    is_standalone_else_or_finally = True

            # For catch: Cannot instrument before it (syntax error), but should track it
            # Python tracks "except" lines, so we should track "catch" lines
            # Solution: Detect catch and handle it specially (see below)
            is_catch = stripped.startswith(("catch ", "catch(", "} catch ", "} catch("))

            # Determine if we should skip instrumenting BEFORE this line
            # Skip standalone else/finally (matches Python)
            # Also skip else-if - it's tracked by tree-sitter marking it as executable,
            # but we can't insert code before it (syntax error)

            # Determine if we should skip instrumenting BEFORE this line
            # Skip standalone else/finally (matches Python)
            # Also skip else-if - it's tracked by tree-sitter marking it as executable,
            # but we can't insert code before it (syntax error)
            should_skip = is_standalone_else_or_finally or is_else_if

            if (
                local_line_num in executable_lines
                and stripped
                and not stripped.startswith(("//", "/*", "*"))
                and stripped not in ("}", "};")
                and not should_skip
            ):
                # Get indentation
                indent = len(line) - len(line.lstrip())
                indent_str = " " * indent

                # Store line content for profiler output
                content_key = f"{file_path_posix}:{global_line_num}"
                self.line_contents[content_key] = stripped

                # Special handling for catch and else-if blocks
                # Cannot instrument before these (syntax error), so instrument inside
                # - catch: Like Python's except
                # - else if: Like Python's elif
                if is_catch or is_else_if:
                    # Add the keyword line as-is (catch or else-if)
                    instrumented_lines.append(line)
                    # Find indentation for catch block body
                    # Look ahead to find proper indentation, or default to indent + 4 spaces
                    body_indent = indent_str + "    "
                    if local_idx + 1 < len(func_lines):
                        next_line = func_lines[local_idx + 1]
                        if next_line.strip() and not next_line.strip().startswith("}"):
                            # Use next line's indentation if it's a statement
                            body_indent = " " * (len(next_line) - len(next_line.lstrip()))

                    # Add hit() as first line inside block with keyword line number
                    # This matches Python's behavior:
                    # - catch → except (Python tracks except lines)
                    # - else if → elif (Python tracks elif lines)
                    instrumented_lines.append(
                        f'{body_indent}{profiler_class}.hit("{file_path_posix}", {global_line_num});\n'
                    )
                else:
                    # Normal case: Add hit() call before the line
                    profiled_line = (
                        f'{indent_str}{profiler_class}.hit("{file_path_posix}", {global_line_num});\n{line}'
                    )
                    instrumented_lines.append(profiled_line)
            else:
                instrumented_lines.append(line)

        return instrumented_lines

    def _generate_line_contents_map(self) -> str:
        """Generate Java code to initialize line contents map."""
        lines = []
        for key, content in self.line_contents.items():
            # Escape special characters for Java string
            escaped = content.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
            lines.append(f'        map.put("{key}", "{escaped}");')
        return "\n".join(lines)

    def _find_executable_lines(self, node: Node) -> set[int]:
        """Find lines that contain executable statements.

        Args:
            node: Tree-sitter AST node.

        Returns:
            Set of line numbers with executable statements.

        """
        executable_lines = set()

        # Java executable statement types
        executable_types = {
            "expression_statement",
            "return_statement",
            "if_statement",
            "for_statement",
            "enhanced_for_statement",  # for-each loop
            "while_statement",
            "do_statement",
            "switch_expression",
            "switch_statement",
            "throw_statement",
            "try_statement",
            "try_with_resources_statement",
            "local_variable_declaration",
            "assert_statement",
            "break_statement",
            "continue_statement",
            "method_invocation",
            "object_creation_expression",
            "assignment_expression",
        }

        def walk(n: Node) -> None:
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
            Dictionary with profiling statistics:
                {
                    "timings": {
                        "file_path": {
                            line_num: {
                                "hits": int,
                                "time_ns": int,
                                "time_ms": float,
                                "content": str
                            }
                        }
                    },
                    "unit": 1e-9,
                    "raw_data": {...}
                }

        """
        if not profile_file.exists():
            return {"timings": {}, "unit": 1e-9, "raw_data": {}, "str_out": ""}

        try:
            with profile_file.open("r") as f:
                data = json.load(f)

            # Group by file
            timings = {}
            for key, stats in data.items():
                file_path, line_num_str = key.rsplit(":", 1)
                line_num = int(line_num_str)
                time_ns = int(stats["time"])  # nanoseconds
                time_ms = time_ns / 1e6  # convert to milliseconds
                hits = stats["hits"]
                content = stats.get("content", "")

                if file_path not in timings:
                    timings[file_path] = {}

                timings[file_path][line_num] = {
                    "hits": hits,
                    "time_ns": time_ns,
                    "time_ms": time_ms,
                    "content": content,
                }

            result = {
                "timings": timings,
                "unit": 1e-9,  # nanoseconds
                "raw_data": data,
            }
            result["str_out"] = format_line_profile_results(result)
            return result

        except Exception as e:
            logger.error("Failed to parse line profile results: %s", e)
            return {"timings": {}, "unit": 1e-9, "raw_data": {}, "str_out": ""}


def format_line_profile_results(results: dict, file_path: Path | None = None) -> str:
    """Format line profiling results for display.

    Args:
        results: Results from parse_results().
        file_path: Optional file path to filter results.

    Returns:
        Formatted string showing per-line statistics.

    """
    if not results or not results.get("timings"):
        return "No profiling data available"

    output = []
    output.append("Line Profiling Results")
    output.append("=" * 80)

    timings = results["timings"]

    # Filter to specific file if requested
    if file_path:
        file_key = str(file_path)
        timings = {file_key: timings.get(file_key, {})}

    for file, lines in sorted(timings.items()):
        if not lines:
            continue

        output.append(f"\nFile: {file}")
        output.append("-" * 80)
        output.append(f"{'Line':>6} | {'Hits':>10} | {'Time (ms)':>12} | {'Avg (ms)':>12} | Code")
        output.append("-" * 80)

        # Sort by line number
        for line_num in sorted(lines.keys()):
            stats = lines[line_num]
            hits = stats["hits"]
            time_ms = stats["time_ms"]
            avg_ms = time_ms / hits if hits > 0 else 0
            content = stats.get("content", "")[:50]  # Truncate long lines

            output.append(
                f"{line_num:6d} | {hits:10d} | {time_ms:12.3f} | {avg_ms:12.6f} | {content}"
            )

    return "\n".join(output)
