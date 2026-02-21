"""Line profiler for Java via bytecode instrumentation agent.

This module generates configuration for the CodeFlash profiler Java agent, which
instruments bytecode at class-load time using ASM. The agent uses zero-allocation
thread-local arrays for hit counting and a per-thread call stack for accurate
self-time attribution.

No source code modification is needed — the agent intercepts class loading via
-javaagent and injects probes at each LineNumber table entry.
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

AGENT_JAR_NAME = "codeflash-runtime-1.0.0.jar"
DEFAULT_WARMUP_ITERATIONS = 100


class JavaLineProfiler:
    """Configures the Java profiler agent for line-level profiling.

    Example:
        profiler = JavaLineProfiler(output_file=Path("profile.json"))
        config_path = profiler.generate_agent_config(source, file_path, functions, config_path)
        jvm_arg = profiler.build_javaagent_arg(config_path)
        # Run Java with: java <jvm_arg> -cp ... ClassName
        results = JavaLineProfiler.parse_results(Path("profile.json"))

    """

    def __init__(self, output_file: Path, warmup_iterations: int = DEFAULT_WARMUP_ITERATIONS) -> None:
        self.output_file = output_file
        self.warmup_iterations = warmup_iterations
        self.profiler_class = "CodeflashLineProfiler"

        self.executable_types = frozenset(
            {
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
        )

    # === Agent-based profiling (bytecode instrumentation) ===

    def generate_agent_config(
        self, source: str, file_path: Path, functions: list[FunctionInfo], config_output_path: Path
    ) -> Path:
        """Generate config JSON for the profiler agent.

        Reads the source to extract line contents and resolves the JVM internal
        class name, then writes a config JSON that the agent uses to know which
        classes/methods to instrument at class-load time.

        Args:
            source: Java source code of the file.
            file_path: Absolute path to the source file.
            functions: Functions to profile.
            config_output_path: Where to write the config JSON.

        Returns:
            Path to the written config file.

        """
        class_name = resolve_internal_class_name(file_path, source)
        lines = source.splitlines()
        line_contents: dict[str, str] = {}
        method_targets = []

        for func in functions:
            for line_num in range(func.starting_line, func.ending_line + 1):
                if 1 <= line_num <= len(lines):
                    content = lines[line_num - 1].strip()
                    if (
                        content
                        and not content.startswith("//")
                        and not content.startswith("/*")
                        and not content.startswith("*")
                    ):
                        key = f"{file_path.as_posix()}:{line_num}"
                        line_contents[key] = content

            method_targets.append(
                {
                    "name": func.function_name,
                    "startLine": func.starting_line,
                    "endLine": func.ending_line,
                    "sourceFile": file_path.as_posix(),
                }
            )

        config = {
            "outputFile": str(self.output_file),
            "warmupIterations": self.warmup_iterations,
            "targets": [{"className": class_name, "methods": method_targets}],
            "lineContents": line_contents,
        }

        config_output_path.parent.mkdir(parents=True, exist_ok=True)
        config_output_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        return config_output_path

    def build_javaagent_arg(self, config_path: Path) -> str:
        """Return the -javaagent JVM argument string."""
        agent_jar = find_agent_jar()
        if agent_jar is None:
            msg = f"{AGENT_JAR_NAME} not found in resources or dev build directory"
            raise FileNotFoundError(msg)
        return f"-javaagent:{agent_jar}=config={config_path}"

    # === Source-level instrumentation ===

    def instrument_source(self, source: str, file_path: Path, functions: list[FunctionInfo], analyzer=None) -> str:
        """Instrument Java source code with line profiling.

        Injects a profiler class and per-line hit() calls directly into the source.

        Args:
            source: Java source code of the file.
            file_path: Absolute path to the source file.
            functions: Functions to instrument.
            analyzer: JavaAnalyzer instance for parsing/validation.

        Returns:
            Instrumented source code, or original source if instrumentation fails.

        """
        # Initialize line contents map
        self.line_contents = {}

        lines = source.splitlines(keepends=True)

        # Process functions in reverse order to preserve line numbers
        for func in sorted(functions, key=lambda f: f.starting_line, reverse=True):
            func_lines = self.instrument_function(func, lines, file_path, analyzer)
            start_idx = func.starting_line - 1
            end_idx = func.ending_line
            lines = lines[:start_idx] + func_lines + lines[end_idx:]

        # Add profiler class and initialization
        profiler_class_code = self.generate_profiler_class()

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

        lines_with_profiler = [*lines[:import_end_idx], profiler_class_code + "\n", *lines[import_end_idx:]]

        result = "".join(lines_with_profiler)
        if not analyzer.validate_syntax(result):
            logger.warning("Line profiler instrumentation produced invalid Java, returning original source")
            return source
        return result

    def generate_profiler_class(self) -> str:
        """Generate Java code for profiler class."""
        # Store line contents as a simple map (embedded directly in code)
        line_contents_code = self.generate_line_contents_map()

        return f"""
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
    private static final String OUTPUT_FILE = "{self.output_file!s}";

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
"""

    def instrument_function(self, func: FunctionInfo, lines: list[str], file_path: Path, analyzer) -> list[str]:
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
            executable_lines = self.find_executable_lines(tree.root_node)
        except Exception as e:
            logger.warning("Failed to parse function %s: %s", func.function_name, e)
            return func_lines

        # Add profiling to each executable line
        function_entry_added = False

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
                instrumented_lines.append(f"{body_indent}{self.profiler_class}.enterFunction();\n")
                function_entry_added = True
                continue

            # Skip empty lines, comments, closing braces
            if (
                local_line_num in executable_lines
                and stripped
                and not stripped.startswith("//")
                and not stripped.startswith("/*")
                and not stripped.startswith("*")
                and stripped not in ("}", "};")
            ):
                # Get indentation
                indent = len(line) - len(line.lstrip())
                indent_str = " " * indent

                # Store line content for profiler output
                content_key = f"{file_path.as_posix()}:{global_line_num}"
                self.line_contents[content_key] = stripped

                # Add hit() call before the line
                profiled_line = (
                    f'{indent_str}{self.profiler_class}.hit("{file_path.as_posix()}", {global_line_num});\n{line}'
                )
                instrumented_lines.append(profiled_line)
            else:
                instrumented_lines.append(line)

        return instrumented_lines

    def generate_line_contents_map(self) -> str:
        """Generate Java code to initialize line contents map."""
        lines = []
        for key, content in self.line_contents.items():
            # Escape special characters for Java string
            escaped = content.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
            lines.append(f'        map.put("{key}", "{escaped}");')
        return "\n".join(lines)

    def find_executable_lines(self, node: Node) -> set[int]:
        """Find lines that contain executable statements.

        Args:
            node: Tree-sitter AST node.

        Returns:
            Set of line numbers with executable statements.

        """
        executable_lines: set[int] = set()

        # Use an explicit stack to avoid recursion overhead on deep ASTs.
        stack = [node]
        types = self.executable_types
        add_line = executable_lines.add

        while stack:
            n = stack.pop()
            if n.type in types:
                # Add the starting line (1-indexed)
                add_line(n.start_point[0] + 1)

            # Push children onto the stack for further traversal
            # Access children once per node to minimize attribute lookups.
            children = n.children
            if children:
                stack.extend(children)

        return executable_lines

    # === Result parsing (shared by both approaches) ===

    @staticmethod
    def parse_results(profile_file: Path) -> dict:
        """Parse line profiling results from the agent's JSON output.

        Returns the same format as parse_line_profile_test_output.parse_line_profile_results()
        for non-Python languages:
            {
                "timings": {(filename, start_lineno, func_name): [(lineno, hits, time_ns), ...]},
                "unit": 1e-9,
                "str_out": "<tabulate pipe formatted output>"
            }

        """
        if not profile_file.exists():
            return {"timings": {}, "unit": 1e-9, "str_out": ""}

        try:
            with profile_file.open("r") as f:
                data = json.load(f)

            # Load method ranges and line contents from config file
            method_ranges, config_line_contents = load_method_ranges(profile_file)

            line_contents: dict[tuple[str, int], str] = {}

            if method_ranges:
                # Group lines by method using config ranges
                grouped_timings: dict[tuple[str, int, str], list[tuple[int, int, int]]] = {}
                for key, stats in data.items():
                    fp = stats.get("file")
                    line_num = stats.get("line")
                    if fp is None or line_num is None:
                        fp, line_str = key.rsplit(":", 1)
                        line_num = int(line_str)
                    line_num = int(line_num)

                    line_contents[(fp, line_num)] = stats.get("content", "")
                    entry = (line_num, int(stats.get("hits", 0)), int(stats.get("time", 0)))

                    method_name, method_start = find_method_for_line(fp, line_num, method_ranges)
                    group_key = (fp, method_start, method_name)
                    grouped_timings.setdefault(group_key, []).append(entry)

                # Fill in missing lines from config (closing braces, etc.)
                for config_key, content in config_line_contents.items():
                    fp, line_str = config_key.rsplit(":", 1)
                    line_num = int(line_str)
                    if (fp, line_num) not in line_contents:
                        line_contents[(fp, line_num)] = content
                        method_name, method_start = find_method_for_line(fp, line_num, method_ranges)
                        group_key = (fp, method_start, method_name)
                        grouped_timings.setdefault(group_key, []).append((line_num, 0, 0))

                for group_key in grouped_timings:
                    grouped_timings[group_key].sort(key=lambda t: t[0])
            else:
                # No config — fall back to grouping all lines by file
                lines_by_file: dict[str, list[tuple[int, int, int]]] = {}
                for key, stats in data.items():
                    fp = stats.get("file")
                    line_num = stats.get("line")
                    if fp is None or line_num is None:
                        fp, line_str = key.rsplit(":", 1)
                        line_num = int(line_str)
                    line_num = int(line_num)

                    lines_by_file.setdefault(fp, []).append(
                        (line_num, int(stats.get("hits", 0)), int(stats.get("time", 0)))
                    )
                    line_contents[(fp, line_num)] = stats.get("content", "")

                grouped_timings = {}
                for fp, line_stats in lines_by_file.items():
                    sorted_stats = sorted(line_stats, key=lambda t: t[0])
                    if sorted_stats:
                        grouped_timings[(fp, sorted_stats[0][0], Path(fp).name)] = sorted_stats

            result: dict = {"timings": grouped_timings, "unit": 1e-9, "line_contents": line_contents}
            result["str_out"] = format_line_profile_results(result, line_contents)
            return result

        except Exception:
            logger.exception("Failed to parse line profile results")
            return {"timings": {}, "unit": 1e-9, "str_out": ""}


def load_method_ranges(profile_file: Path) -> tuple[list[tuple[str, str, int, int]], dict[str, str]]:
    """Load method ranges and line contents from the agent config file.

    Returns:
        (method_ranges, config_line_contents) where method_ranges is a list of
        (source_file, method_name, start_line, end_line) and config_line_contents
        is the lineContents dict from the config (key: "file:line", value: source text).

    """
    config_path = profile_file.with_suffix(".config.json")
    if not config_path.exists():
        return [], {}
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        ranges = []
        for target in config.get("targets", []):
            for method in target.get("methods", []):
                ranges.append((method.get("sourceFile", ""), method["name"], method["startLine"], method["endLine"]))
        return ranges, config.get("lineContents", {})
    except Exception:
        return [], {}


def find_method_for_line(
    file_path: str, line_num: int, method_ranges: list[tuple[str, str, int, int]]
) -> tuple[str, int]:
    """Find which method a line belongs to based on config ranges.

    Returns (method_name, method_start_line). Falls back to (basename, line_num)
    if no matching method range is found.
    """
    for source_file, method_name, start_line, end_line in method_ranges:
        if file_path == source_file and start_line <= line_num <= end_line:
            return method_name, start_line
    return Path(file_path).name, line_num


def find_agent_jar() -> Path | None:
    """Locate the profiler agent JAR file (now bundled in codeflash-runtime).

    Checks local Maven repo, package resources, and development build directory.
    """
    # Check local Maven repository first (fastest)
    m2_jar = Path.home() / ".m2" / "repository" / "com" / "codeflash" / "codeflash-runtime" / "1.0.0" / AGENT_JAR_NAME
    if m2_jar.exists():
        return m2_jar

    # Check bundled JAR in package resources
    resources_jar = Path(__file__).parent / "resources" / AGENT_JAR_NAME
    if resources_jar.exists():
        return resources_jar

    # Check development build directory
    dev_jar = Path(__file__).parent.parent.parent.parent / "codeflash-java-runtime" / "target" / AGENT_JAR_NAME
    if dev_jar.exists():
        return dev_jar

    return None


def resolve_internal_class_name(file_path: Path, source: str) -> str:
    """Resolve the JVM internal class name (slash-separated) from source.

    Parses the package statement and combines with the filename stem.
    e.g. "package com.example;" + "Calculator.java" → "com/example/Calculator"
    """
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("package "):
            package = stripped[8:].rstrip(";").strip()
            return f"{package.replace('.', '/')}/{file_path.stem}"
    # No package — default package
    return file_path.stem


def format_line_profile_results(results: dict, line_contents: dict[tuple[str, int], str] | None = None) -> str:
    """Format line profiling results using the same tabulate pipe format as Python.

    Args:
        results: Parsed results with timings in grouped format:
            {(filename, start_lineno, func_name): [(lineno, hits, time_ns), ...]}
        line_contents: Mapping of (filename, lineno) to source line content.

    Returns:
        Formatted string matching the Python line_profiler output format.

    """
    if not results or not results.get("timings"):
        return ""

    if line_contents is None:
        line_contents = results.get("line_contents", {})

    from codeflash.verification.parse_line_profile_test_output import show_text_non_python

    return show_text_non_python(results, line_contents)
