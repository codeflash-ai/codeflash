from pathlib import Path

from codeflash.discovery.functions_to_optimize import FunctionToOptimize
from codeflash.models.models import CodeOptimizationContext, CodeStringsMarkdown, FunctionParent
from codeflash.optimization.function_optimizer import FunctionOptimizer
from codeflash.verification.verification_utils import TestConfig


class Args:
    disable_imports_sorting = True
    formatter_cmds = ["disabled"]


def test_multi_file_replcement01() -> None:
    root_dir = Path(__file__).parent.parent.resolve()
    helper_file = (root_dir / "code_to_optimize/temp_helper.py").resolve()

    helper_file.write_text(
        """import re
from collections.abc import Sequence

from pydantic_ai_slim.pydantic_ai.messages import BinaryContent, UserContent

_TOKEN_SPLIT_RE = re.compile(r'[\\s",.:]+')

def _estimate_string_tokens(content: str | Sequence[UserContent]) -> int:
    if not content:
        return 0

    if isinstance(content, str):
        return len(_TOKEN_SPLIT_RE.split(content.strip()))

    tokens = 0
    for part in content:
        if isinstance(part, str):
            tokens += len(_TOKEN_SPLIT_RE.split(part.strip()))
        elif isinstance(part, BinaryContent):
            tokens += len(part.data)
        # TODO(Marcelo): We need to study how we can estimate the tokens for AudioUrl or ImageUrl.

    return tokens
""",
        encoding="utf-8",
    )

    main_file = (root_dir / "code_to_optimize/temp_main.py").resolve()

    original_main = """from temp_helper import _estimate_string_tokens
from pydantic_ai_slim.pydantic_ai.usage import Usage

def _get_string_usage(text: str) -> Usage:
    response_tokens = _estimate_string_tokens(text)
    return Usage(response_tokens=response_tokens, total_tokens=response_tokens)
"""
    main_file.write_text(original_main, encoding="utf-8")

    optimized_code = f"""```python:{helper_file.relative_to(root_dir)}
import re
from collections.abc import Sequence

from pydantic_ai_slim.pydantic_ai.messages import BinaryContent, UserContent

_TOKEN_SPLIT_RE = re.compile(r'[\\s",.:]+')
_translate_table = {{ord(c): ord(' ') for c in ' \\t\\n\\r\\x0b\\x0c",.:'}}

def _estimate_string_tokens(content: str | Sequence[UserContent]) -> int:
    if not content:
        return 0

    if isinstance(content, str):
        # Fast path using translate and split instead of regex when separat
        s = content.strip()
        if s:
            s = s.translate(_translate_table)
            # Split on whitespace (default). This handles multiple consecut
            return len(s.split())
        return 0

    tokens = 0
    for part in content:
        if isinstance(part, str):
            s = part.strip()
            if s:
                s = s.translate(_translate_table)
                tokens += len(s.split())
        elif isinstance(part, BinaryContent):
            tokens += len(part.data)

    return tokens
```
```python:{main_file.relative_to(root_dir)}
from temp_helper import _estimate_string_tokens
from pydantic_ai_slim.pydantic_ai.usage import Usage

def _get_string_usage(text: str) -> Usage:
    response_tokens = _estimate_string_tokens(text)
    return Usage(response_tokens=response_tokens, total_tokens=response_tokens)
```
"""

    func = FunctionToOptimize(function_name="_get_string_usage", parents=[], file_path=main_file)
    test_config = TestConfig(
        tests_root=root_dir / "tests/pytest",
        tests_project_rootdir=root_dir,
        project_root_path=root_dir,
        test_framework="pytest",
        pytest_cmd="pytest",
    )
    func_optimizer = FunctionOptimizer(function_to_optimize=func, test_cfg=test_config)
    code_context: CodeOptimizationContext = func_optimizer.get_code_optimization_context().unwrap()

    original_helper_code: dict[Path, str] = {}
    helper_function_paths = {hf.file_path for hf in code_context.helper_functions}
    for helper_function_path in helper_function_paths:
        with helper_function_path.open(encoding="utf8") as f:
            helper_code = f.read()
            original_helper_code[helper_function_path] = helper_code

    func_optimizer.args = Args()
    func_optimizer.replace_function_and_helpers_with_optimized_code(
        code_context=code_context,
        optimized_code=CodeStringsMarkdown.parse_markdown_code(optimized_code),
        original_helper_code=original_helper_code,
    )
    new_code = main_file.read_text(encoding="utf-8")
    new_helper_code = helper_file.read_text(encoding="utf-8")

    helper_file.unlink(missing_ok=True)
    main_file.unlink(missing_ok=True)

    expected_helper = """import re
from collections.abc import Sequence

from pydantic_ai_slim.pydantic_ai.messages import BinaryContent, UserContent

_translate_table = {ord(c): ord(' ') for c in ' \\t\\n\\r\\x0b\\x0c",.:'}

_TOKEN_SPLIT_RE = re.compile(r'[\\s",.:]+')

def _estimate_string_tokens(content: str | Sequence[UserContent]) -> int:
    if not content:
        return 0

    if isinstance(content, str):
        # Fast path using translate and split instead of regex when separat
        s = content.strip()
        if s:
            s = s.translate(_translate_table)
            # Split on whitespace (default). This handles multiple consecut
            return len(s.split())
        return 0

    tokens = 0
    for part in content:
        if isinstance(part, str):
            s = part.strip()
            if s:
                s = s.translate(_translate_table)
                tokens += len(s.split())
        elif isinstance(part, BinaryContent):
            tokens += len(part.data)

    return tokens
"""

    assert new_code.rstrip() == original_main.rstrip()  # No Change
    assert new_helper_code.rstrip() == expected_helper.rstrip()


def test_optimized_code_for_different_file_not_applied_to_current_file() -> None:
    """Test that optimized code for one file is not incorrectly applied to a different file.

    This reproduces the bug from PR #1309 where optimized code for `formatter.py`
    was incorrectly applied to `support.py`, causing `normalize_java_code` to be
    duplicated. The bug was in `get_optimized_code_for_module` which had a fallback
    that applied a single code block to ANY file being processed.

    The scenario:
    1. `support.py` imports `normalize_java_code` from `formatter.py`
    2. AI returns optimized code with a single code block for `formatter.py`
    3. BUG: When processing `support.py`, the fallback applies `formatter.py`'s code
    4. EXPECTED: No code should be applied to `support.py` since the paths don't match
    """
    from codeflash.code_utils.code_extractor import find_preexisting_objects
    from codeflash.code_utils.code_replacer import replace_function_definitions_in_module
    from codeflash.models.models import CodeStringsMarkdown

    root_dir = Path(__file__).parent.parent.resolve()

    # Create support.py - the file that imports the helper
    support_file = (root_dir / "code_to_optimize/temp_pr1309_support.py").resolve()
    original_support = '''from temp_pr1309_formatter import normalize_java_code


class JavaSupport:
    """Support class for Java operations."""

    def normalize_code(self, source: str) -> str:
        """Normalize code for deduplication."""
        return normalize_java_code(source)
'''
    support_file.write_text(original_support, encoding="utf-8")

    # AI returns optimized code for formatter.py ONLY (with explicit path)
    # This simulates what happens when the AI optimizes the helper function
    optimized_markdown = '''```python:code_to_optimize/temp_pr1309_formatter.py
def normalize_java_code(source: str) -> str:
    """Optimized version with fast-path."""
    if not source:
        return ""
    return "\\n".join(line.strip() for line in source.splitlines() if line.strip())
```
'''

    preexisting_objects = find_preexisting_objects(original_support)

    # Process support.py with the optimized code that's meant for formatter.py
    replace_function_definitions_in_module(
        function_names=["JavaSupport.normalize_code"],
        optimized_code=CodeStringsMarkdown.parse_markdown_code(optimized_markdown),
        module_abspath=support_file,
        preexisting_objects=preexisting_objects,
        project_root_path=root_dir,
    )

    new_support_code = support_file.read_text(encoding="utf-8")

    # Cleanup
    support_file.unlink(missing_ok=True)

    # CRITICAL: support.py should NOT have normalize_java_code defined!
    # The optimized code was for formatter.py, not support.py.
    def_count = new_support_code.count("def normalize_java_code")
    assert def_count == 0, (
        f"Bug: normalize_java_code was incorrectly added to support.py!\n"
        f"Found {def_count} definition(s) when there should be 0.\n"
        f"The optimized code was for formatter.py, not support.py.\n"
        f"Resulting code:\n{new_support_code}"
    )

    # The file should remain unchanged since no code matched its path
    assert new_support_code.strip() == original_support.strip(), (
        f"support.py was modified when it shouldn't have been.\n"
        f"Original:\n{original_support}\n"
        f"New:\n{new_support_code}"
    )
