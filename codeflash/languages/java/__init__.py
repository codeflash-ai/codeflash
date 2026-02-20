"""Java language support for codeflash.

This module provides Java-specific functionality for code analysis,
test execution, and optimization using tree-sitter for parsing and
Maven/Gradle for build operations.
"""

from codeflash.languages.java.build_tools import (
    BuildTool,
    JavaProjectInfo,
    MavenTestResult,
    add_codeflash_dependency_to_pom,
    compile_maven_project,
    detect_build_tool,
    find_gradle_executable,
    find_maven_executable,
    find_source_root,
    find_test_root,
    get_classpath,
    get_project_info,
    install_codeflash_runtime,
    run_maven_tests,
)
from codeflash.languages.java.comparator import compare_invocations_directly, compare_test_results
from codeflash.languages.java.config import (
    JavaProjectConfig,
    detect_java_project,
    get_test_class_pattern,
    get_test_file_pattern,
    is_java_project,
)
from codeflash.languages.java.context import (
    extract_class_context,
    extract_code_context,
    extract_function_source,
    extract_read_only_context,
    find_helper_functions,
)
from codeflash.languages.java.discovery import (
    discover_functions,
    discover_functions_from_source,
    discover_test_methods,
    get_class_methods,
    get_method_by_name,
)
from codeflash.languages.java.formatter import JavaFormatter, format_java_code, format_java_file, normalize_java_code
from codeflash.languages.java.import_resolver import (
    JavaImportResolver,
    ResolvedImport,
    find_helper_files,
    resolve_imports_for_file,
)
from codeflash.languages.java.instrumentation import (
    create_benchmark_test,
    instrument_existing_test,
    instrument_for_behavior,
    instrument_for_benchmarking,
    instrument_generated_java_test,
    remove_instrumentation,
)
from codeflash.languages.java.parser import (
    JavaAnalyzer,
    JavaClassNode,
    JavaFieldInfo,
    JavaImportInfo,
    JavaMethodNode,
    get_java_analyzer,
)
from codeflash.languages.java.remove_asserts import (
    JavaAssertTransformer,
    remove_assertions_from_test,
    transform_java_assertions,
)
from codeflash.languages.java.replacement import (
    add_runtime_comments,
    insert_method,
    remove_method,
    remove_test_functions,
    replace_function,
    replace_method_body,
)
from codeflash.languages.java.support import JavaSupport, get_java_support
from codeflash.languages.java.test_discovery import (
    build_test_mapping_for_project,
    discover_all_tests,
    discover_tests,
    find_tests_for_function,
    get_test_class_for_source_class,
    get_test_file_suffix,
    get_test_methods_for_class,
    is_test_file,
)
from codeflash.languages.java.test_runner import (
    JavaTestRunResult,
    get_test_run_command,
    parse_surefire_results,
    parse_test_results,
    run_behavioral_tests,
    run_benchmarking_tests,
    run_tests,
)

__all__ = [
    # Build tools
    "BuildTool",
    # Parser
    "JavaAnalyzer",
    # Assertion removal
    "JavaAssertTransformer",
    "JavaClassNode",
    "JavaFieldInfo",
    # Formatter
    "JavaFormatter",
    "JavaImportInfo",
    # Import resolver
    "JavaImportResolver",
    "JavaMethodNode",
    # Config
    "JavaProjectConfig",
    "JavaProjectInfo",
    # Support
    "JavaSupport",
    # Test runner
    "JavaTestRunResult",
    "MavenTestResult",
    "ResolvedImport",
    "add_codeflash_dependency_to_pom",
    # Replacement
    "add_runtime_comments",
    # Test discovery
    "build_test_mapping_for_project",
    # Comparator
    "compare_invocations_directly",
    "compare_test_results",
    "compile_maven_project",
    # Instrumentation
    "create_benchmark_test",
    "detect_build_tool",
    "detect_java_project",
    "discover_all_tests",
    # Discovery
    "discover_functions",
    "discover_functions_from_source",
    "discover_test_methods",
    "discover_tests",
    # Context
    "extract_class_context",
    "extract_code_context",
    "extract_function_source",
    "extract_read_only_context",
    "find_gradle_executable",
    "find_helper_files",
    "find_helper_functions",
    "find_maven_executable",
    "find_source_root",
    "find_test_root",
    "find_tests_for_function",
    "format_java_code",
    "format_java_file",
    "get_class_methods",
    "get_classpath",
    "get_java_analyzer",
    "get_java_support",
    "get_method_by_name",
    "get_project_info",
    "get_test_class_for_source_class",
    "get_test_class_pattern",
    "get_test_file_pattern",
    "get_test_file_suffix",
    "get_test_methods_for_class",
    "get_test_run_command",
    "insert_method",
    "install_codeflash_runtime",
    "instrument_existing_test",
    "instrument_for_behavior",
    "instrument_for_benchmarking",
    "instrument_generated_java_test",
    "is_java_project",
    "is_test_file",
    "normalize_java_code",
    "parse_surefire_results",
    "parse_test_results",
    "remove_assertions_from_test",
    "remove_instrumentation",
    "remove_method",
    "remove_test_functions",
    "replace_function",
    "replace_method_body",
    "resolve_imports_for_file",
    "run_behavioral_tests",
    "run_benchmarking_tests",
    "run_maven_tests",
    "run_tests",
    "transform_java_assertions",
]
