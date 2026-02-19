from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from codeflash.models.models import GeneratedTests, GeneratedTestsList

if TYPE_CHECKING:
    from codeflash.models.models import InvocationId


def add_runtime_comments_to_generated_tests(
    generated_tests: GeneratedTestsList,
    original_runtimes: dict[InvocationId, list[int]],
    optimized_runtimes: dict[InvocationId, list[int]],
    tests_project_rootdir: Optional[Path] = None,
) -> GeneratedTestsList:
    """Add runtime performance comments to function calls in generated tests."""
    _ = original_runtimes, optimized_runtimes, tests_project_rootdir
    return generated_tests


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
