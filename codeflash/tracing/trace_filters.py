"""Advanced filtering system for codeflash tracing to reduce noise and focus on optimization targets."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


class TraceFilter:
    """Intelligent filtering system to reduce noise in trace output."""

    def __init__(self, config: dict[str, Any], project_root: Path) -> None:
        self.config = config
        self.project_root = project_root

        # Built-in functions that are rarely optimization targets - expanded set
        self.builtin_noise_functions = {
            # Core Python built-ins
            "isinstance", "getattr", "hasattr", "len", "iter", "next", "list", "dict", "set", "tuple",
            "str", "repr", "hash", "id", "type", "bool", "int", "float", "abs", "min", "max",
            "sum", "all", "any", "enumerate", "zip", "range", "sorted", "reversed", "filter", "map",
            
            # Container methods (very common, low optimization value)
            "append", "extend", "update", "pop", "remove", "clear", "copy", "keys", "values", "items",
            "get", "setdefault", "popitem", "insert", "count", "reverse", "sort", "index",
            
            # String methods (very frequent, rarely worth optimizing)
            "join", "split", "strip", "replace", "format", "startswith", "endswith", "find", "index",
            "rstrip", "lstrip", "removeprefix", "removesuffix", "upper", "lower", "capitalize",
            "title", "swapcase", "isdigit", "isalpha", "isalnum", "isspace", "islower", "isupper",
            
            # File and path operations
            "intern", "fspath", "stat", "open", "close", "read", "write", "seek", "tell", "flush",
            
            # Regular expressions and parsing
            "compile", "loads", "dumps", "sub", "repl", "match", "search", "findall", "finditer",
            
            # AST and inspection
            "walk", "collect", "iter_child_nodes", "iter_fields", "dump", "parse",
            
            # Threading and synchronization
            "acquire", "release", "locked", "wait", "notify", "notify_all",
            
            # Import and module system
            "find_spec", "create_dynamic", "import_module", "reload", "invalidate_caches",
            
            # Memory and garbage collection
            "collect", "get_count", "get_threshold", "set_threshold", "get_stats",
            
            # Internal Python functions (very noisy)
            "_type_repr", "_type_check", "_is_dunder", "_parse_path", "_path_join", "_get_sep",
            "_str_normcase", "_joinrealpath", "_parse", "_compile", "get_statement_startend2",
            "_getframe", "_current_frames", "_getdefaultencoding", "_getfilesystemencoding",
            
            # Common property and descriptor methods
            "__get__", "__set__", "__delete__", "__set_name__",
            
            # Common iterator protocol methods
            "__iter__", "__next__", "__reversed__",
            
            # Common container protocol methods  
            "__len__", "__getitem__", "__setitem__", "__delitem__", "__contains__",
            
            # Attribute access methods (very frequent)
            "__getattribute__", "__getattr__", "__setattr__", "__delattr__", "__dir__",
        }

        # Standard library modules that are usually not optimization targets - expanded
        self.stdlib_noise_patterns = [
            r"^.*[\\/]ast\.py:",
            r"^.*[\\/]typing\.py:",
            r"^.*[\\/]inspect\.py:",
            r"^.*[\\/]collections[\\/]",
            r"^.*[\\/]pathlib\.py:",
            r"^.*[\\/]re\.py:",
            r"^.*[\\/]json[\\/]",
            r"^.*[\\/]pickle\.py:",
            r"^.*[\\/]sqlite3[\\/]",
            r"^.*[\\/]threading\.py:",
            r"^.*[\\/]contextlib\.py:",
            r"^.*[\\/]functools\.py:",
            r"^.*[\\/]importlib[\\/]",
            r"^.*[\\/]runpy\.py:",
            r"^.*[\\/]io\.py:",
            r"^.*[\\/]tempfile\.py:",
            
            # Additional stdlib modules that create noise
            r"^.*[\\/]os\.py:",
            r"^.*[\\/]sys\.py:",
            r"^.*[\\/]abc\.py:",
            r"^.*[\\/]types\.py:",
            r"^.*[\\/]weakref\.py:",
            r"^.*[\\/]gc\.py:",
            r"^.*[\\/]warnings\.py:",
            r"^.*[\\/]traceback\.py:",
            r"^.*[\\/]linecache\.py:",
            r"^.*[\\/]sre_",  # regex internals
            r"^.*[\\/]enum\.py:",
            r"^.*[\\/]operator\.py:",
            r"^.*[\\/]itertools\.py:",
            r"^.*[\\/]copy\.py:",
            r"^.*[\\/]copyreg\.py:",
            r"^.*[\\/]dataclasses\.py:",
            r"^.*[\\/]decimal\.py:",
            r"^.*[\\/]fractions\.py:",
            r"^.*[\\/]logging[\\/]",
            r"^.*[\\/]urllib[\\/]",
            r"^.*[\\/]http[\\/]",
            r"^.*[\\/]email[\\/]",
            r"^.*[\\/]xml[\\/]",
            r"^.*[\\/]html[\\/]",
            r"^.*[\\/]zipfile\.py:",
            r"^.*[\\/]tarfile\.py:",
            r"^.*[\\/]gzip\.py:",
            r"^.*[\\/]shutil\.py:",
            r"^.*[\\/]glob\.py:",
            r"^.*[\\/]fnmatch\.py:",
            r"^.*[\\/]posixpath\.py:",
            r"^.*[\\/]ntpath\.py:",
            r"^.*[\\/]genericpath\.py:",
        ]

        # Test framework noise patterns
        self.test_framework_patterns = [
            r"^.*[\\/]pytest[\\/]",
            r"^.*[\\/]_pytest[\\/]",
            r"^.*[\\/]unittest[\\/]",
            r"^.*[\\/]nose[\\/]",
            r"^.*[\\/]coverage[\\/]",
            r"^.*test.*\.py:.*collect",
            r"^.*test.*\.py:.*setup",
            r"^.*test.*\.py:.*teardown",
        ]

        # Functions that are implementation details, not optimization targets
        self.implementation_detail_functions = {
            "__init__", "__new__", "__del__", "__enter__", "__exit__",
            "__getattr__", "__setattr__", "__delattr__", "__getattribute__",
            "__getitem__", "__setitem__", "__delitem__", "__contains__",
            "__str__", "__repr__", "__format__", "__bytes__",
            "__hash__", "__bool__", "__len__", "__iter__", "__next__",
            "__eq__", "__ne__", "__lt__", "__le__", "__gt__", "__ge__",
            "__add__", "__sub__", "__mul__", "__truediv__", "__floordiv__",
            "__mod__", "__pow__", "__and__", "__or__", "__xor__", "__lshift__", "__rshift__",
        }

        # Compile regex patterns for performance
        self.compiled_stdlib_patterns = [re.compile(pattern) for pattern in self.stdlib_noise_patterns]
        self.compiled_test_patterns = [re.compile(pattern) for pattern in self.test_framework_patterns]

        # Configuration-based thresholds - make them more reasonable for real use
        self.min_call_threshold = config.get("trace_min_calls", 1)  # Show single calls too
        self.min_time_threshold_ms = config.get("trace_min_time_ms", 0.01)  # Very low threshold - 0.01ms
        self.max_functions_display = config.get("trace_max_display", 50)

    def should_trace_function(self, filename: str, function_name: str, class_name: str | None = None) -> bool:
        """Determine if a function should be traced based on its characteristics."""

        # Skip built-in noise functions
        if function_name in self.builtin_noise_functions:
            return False

        # Skip implementation detail methods unless specifically targeted
        if function_name in self.implementation_detail_functions:
            return False

        # Skip standard library noise
        for pattern in self.compiled_stdlib_patterns:
            if pattern.match(filename):
                return False

        # Skip test framework noise
        for pattern in self.compiled_test_patterns:
            if pattern.match(filename):
                return False

        # Skip functions with noise-indicating names
        if self._is_noise_function_name(function_name):
            return False

        # Additional aggressive filtering for very common internal functions
        if self._is_builtin_file(filename):
            return False

        return True

    def _is_builtin_file(self, filename: str) -> bool:
        """Check if this is a built-in Python file that we should never trace."""
        builtin_indicators = [
            ":0)",  # Built-in functions like isinstance (:0)
            "<frozen",  # Frozen modules like <frozen posixpath>
            "site-packages",  # Third-party packages
            "__pycache__",  # Compiled bytecode
            "<built-in>",  # Built-in functions
            "<string>",  # eval/exec strings
            "importlib._bootstrap",  # Bootstrap imports
            "zipimport.py",  # Zip imports
            ".pyc",  # Compiled Python files
            "/lib/python",  # System Python installation
            "/usr/lib/python",  # System Python installation
            "/opt/python",  # Optional Python installation
            "\\lib\\python",  # Windows system Python
            "\\Lib\\",  # Windows Python lib directory
        ]

        for indicator in builtin_indicators:
            if indicator in filename:
                return True

        return False

    def should_display_in_stats(self, func_stats: tuple[int, int, int, int, dict]) -> bool:
        """Determine if function stats should be displayed based on performance impact."""
        call_count, _, total_time_ns, _, _ = func_stats

        # Apply minimum thresholds
        if call_count < self.min_call_threshold:
            return False

        total_time_ms = total_time_ns / 1e6
        if total_time_ms < self.min_time_threshold_ms:
            return False

        return True

    def should_display_function(self, filename: str, function_name: str) -> bool:
        """Additional display-level filtering for functions that made it through tracing."""

        # Skip built-in noise functions at display time too
        if function_name in self.builtin_noise_functions:
            return False

        # Skip built-in files
        if self._is_builtin_file(filename):
            return False

        # For user functions that made it through tracing, be more lenient
        # Only filter out obvious noise functions, not based on project path
        # since the tracer already handles project scope filtering
        
        return True

    def categorize_function_impact(self, func_stats: tuple[int, int, int, int, dict]) -> str:
        """Categorize function performance impact for better user understanding."""
        call_count, _, total_time_ns, cumulative_time_ns, _ = func_stats

        total_time_ms = total_time_ns / 1e6
        cumulative_time_ms = cumulative_time_ns / 1e6

        # High impact: significant time or many calls
        if total_time_ms > 100 or call_count > 1000:
            return "🔥 High Impact"
        # Medium impact: moderate time and calls
        elif total_time_ms > 10 or call_count > 100:
            return "⚡ Medium Impact"
        # Low impact: but still worth showing
        else:
            return "💡 Low Impact"

    def get_optimization_suggestion(self, filename: str, function_name: str, func_stats: tuple[int, int, int, int, dict]) -> str:
        """Provide specific optimization suggestions based on function characteristics."""
        call_count, _, total_time_ns, _, _ = func_stats
        total_time_ms = total_time_ns / 1e6

        suggestions = []

        # High call count suggestions
        if call_count > 1000:
            suggestions.append("Consider caching or memoization")

        # High time per call suggestions
        avg_time_ms = total_time_ms / call_count if call_count > 0 else 0
        if avg_time_ms > 10:
            suggestions.append("Optimize algorithm or use faster data structures")

        # I/O related suggestions
        if any(keyword in function_name.lower() for keyword in ["read", "write", "load", "save", "fetch"]):
            suggestions.append("Consider async I/O or batching")

        # Loop-related suggestions
        if "loop" in function_name.lower() or call_count > 10000:
            suggestions.append("Check for nested loops or vectorization opportunities")

        return " | ".join(suggestions) if suggestions else "Profile deeper to identify bottlenecks"

    def _is_noise_function_name(self, function_name: str) -> bool:
        """Check if function name indicates it's likely noise."""
        noise_patterns = [
            r"^_.*",  # Private functions (often internal)
            r".*wrapper.*",  # Wrapper functions
            r".*decorator.*",  # Decorator functions
            r".*<.*>.*",  # Comprehensions and lambda
        ]

        for pattern in noise_patterns:
            if re.match(pattern, function_name):
                return True

        return False
