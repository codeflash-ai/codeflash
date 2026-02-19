from __future__ import annotations

import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from codeflash.cli_cmds.console import logger
from codeflash.languages.registry import get_language_support
from codeflash.models.models import GeneratedTests, GeneratedTestsList

if TYPE_CHECKING:
    from codeflash.models.models import InvocationId


# TODO:{self} Needs cleanup for jest logic in else block
def unique_inv_id(inv_id_runtimes: dict[InvocationId, list[int]], tests_project_rootdir: Path) -> dict[str, int]:
    unique_inv_ids: dict[str, int] = {}
    logger.debug(f"[unique_inv_id] Processing {len(inv_id_runtimes)} invocation IDs")
    for inv_id, runtimes in inv_id_runtimes.items():
        test_qualified_name = (
            inv_id.test_class_name + "." + inv_id.test_function_name  # type: ignore[operator]
            if inv_id.test_class_name
            else inv_id.test_function_name
        )

        # Detect if test_module_path is a file path (like in js tests) or a Python module name
        # File paths contain slashes, module names use dots
        test_module_path = inv_id.test_module_path
        if "/" in test_module_path or "\\" in test_module_path:
            # Already a file path - use directly
            abs_path = tests_project_rootdir / Path(test_module_path)
        else:
            # Check for Jest test file extensions (e.g., tests.fibonacci.test.ts)
            # These need special handling to avoid converting .test.ts -> /test/ts
            jest_test_extensions = (
                ".test.ts",
                ".test.js",
                ".test.tsx",
                ".test.jsx",
                ".spec.ts",
                ".spec.js",
                ".spec.tsx",
                ".spec.jsx",
                ".ts",
                ".js",
                ".tsx",
                ".jsx",
                ".mjs",
                ".mts",
            )
            matched_ext = None
            for ext in jest_test_extensions:
                if test_module_path.endswith(ext):
                    matched_ext = ext
                    break

            if matched_ext:
                # JavaScript/TypeScript: convert module-style path to file path
                # "tests.fibonacci__perfonlyinstrumented.test.ts" -> "tests/fibonacci__perfonlyinstrumented.test.ts"
                base_path = test_module_path[: -len(matched_ext)]
                file_path = base_path.replace(".", os.sep) + matched_ext
                # Check if the module path includes the tests directory name
                tests_dir_name = tests_project_rootdir.name
                if file_path.startswith((tests_dir_name + os.sep, tests_dir_name + "/")):
                    # Module path includes "tests." - use parent directory
                    abs_path = tests_project_rootdir.parent / Path(file_path)
                else:
                    # Module path doesn't include tests dir - use tests root directly
                    abs_path = tests_project_rootdir / Path(file_path)
            else:
                # Python module name - convert dots to path separators and add .py
                abs_path = tests_project_rootdir / Path(test_module_path.replace(".", os.sep)).with_suffix(".py")

        abs_path_str = str(abs_path.resolve().with_suffix(""))
        # Include both unit test and perf test paths for runtime annotations
        # (performance test runtimes are used for annotations)
        if ("__unit_test_" not in abs_path_str and "__perf_test_" not in abs_path_str) or not test_qualified_name:
            logger.debug(f"[unique_inv_id] Skipping: path={abs_path_str}, test_qualified_name={test_qualified_name}")
            continue
        key = test_qualified_name + "#" + abs_path_str
        parts = inv_id.iteration_id.split("_").__len__()  # type: ignore[union-attr]
        cur_invid = inv_id.iteration_id.split("_")[0] if parts < 3 else "_".join(inv_id.iteration_id.split("_")[:-1])  # type: ignore[union-attr]
        match_key = key + "#" + cur_invid
        logger.debug(f"[unique_inv_id] Adding key: {match_key} with runtime {min(runtimes)}")
        if match_key not in unique_inv_ids:
            unique_inv_ids[match_key] = 0
        unique_inv_ids[match_key] += min(runtimes)
    logger.debug(f"[unique_inv_id] Result has {len(unique_inv_ids)} entries")
    return unique_inv_ids


def add_runtime_comments_to_generated_tests(
    generated_tests: GeneratedTestsList,
    original_runtimes: dict[InvocationId, list[int]],
    optimized_runtimes: dict[InvocationId, list[int]],
    tests_project_rootdir: Optional[Path] = None,
) -> GeneratedTestsList:
    """Add runtime performance comments to function calls in generated tests."""
    original_runtimes_dict = unique_inv_id(original_runtimes, tests_project_rootdir or Path())
    optimized_runtimes_dict = unique_inv_id(optimized_runtimes, tests_project_rootdir or Path())
    # Process each generated test
    modified_tests = []
    for test in generated_tests.generated_tests:
        try:
            language_support = get_language_support(test.behavior_file_path)
            modified_source = language_support.add_runtime_comments(
                test.generated_original_test_source, original_runtimes_dict, optimized_runtimes_dict
            )
            modified_test = GeneratedTests(
                generated_original_test_source=modified_source,
                instrumented_behavior_test_source=test.instrumented_behavior_test_source,
                instrumented_perf_test_source=test.instrumented_perf_test_source,
                behavior_file_path=test.behavior_file_path,
                perf_file_path=test.perf_file_path,
            )
            modified_tests.append(modified_test)
        except Exception as e:
            logger.debug(f"Failed to add runtime comments to test: {e}")
            modified_tests.append(test)

    return GeneratedTestsList(generated_tests=modified_tests)


def remove_functions_from_generated_tests(
    generated_tests: GeneratedTestsList, test_functions_to_remove: list[str]
) -> GeneratedTestsList:
    # Pre-compile patterns for all function names to remove
    function_patterns = _compile_function_patterns(test_functions_to_remove)
    new_generated_tests = []

    for generated_test in generated_tests.generated_tests:
        source = generated_test.generated_original_test_source

        # Apply all patterns without redundant searches
        for pattern in function_patterns:
            # Use finditer and sub only if necessary to avoid unnecessary .search()/.sub() calls
            for match in pattern.finditer(source):
                # Skip if "@pytest.mark.parametrize" present
                # Only the matched function's code is targeted
                if "@pytest.mark.parametrize" in match.group(0):
                    continue
                # Remove function from source
                # If match, remove the function by substitution in the source
                # Replace using start/end indices for efficiency
                start, end = match.span()
                source = source[:start] + source[end:]
                # After removal, break since .finditer() is from left to right, and only one match expected per function in source
                break

        generated_test.generated_original_test_source = source
        new_generated_tests.append(generated_test)

    return GeneratedTestsList(generated_tests=new_generated_tests)


# Pre-compile all function removal regexes upfront for efficiency.
def _compile_function_patterns(test_functions_to_remove: list[str]) -> list[re.Pattern[str]]:
    return [
        re.compile(
            rf"(@pytest\.mark\.parametrize\(.*?\)\s*)?(async\s+)?def\s+{re.escape(func)}\(.*?\):.*?(?=\n(async\s+)?def\s|$)",
            re.DOTALL,
        )
        for func in test_functions_to_remove
    ]
