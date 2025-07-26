from pathlib import Path
from codeflash.discovery.functions_to_optimize import FunctionToOptimize
from codeflash.models.models import CodeOptimizationContext, get_code_block_splitter
from codeflash.optimization.function_optimizer import FunctionOptimizer
from codeflash.verification.verification_utils import TestConfig


class Args:
    disable_imports_sorting = True
    formatter_cmds = ["disabled"]

def test_multi_file_replcement01() -> None:
    root_dir = Path(__file__).parent.parent.resolve()
    helper_file = (root_dir / "code_to_optimize/temp_helper.py").resolve()
    
    helper_file.write_text("""import re
from collections.abc import Sequence

from pydantic_ai_slim.pydantic_ai.messages import BinaryContent, UserContent

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


_TOKEN_SPLIT_RE = re.compile(r'[\\s",.:]+')
""", encoding="utf-8")

    main_file = (root_dir / "code_to_optimize/temp_main.py").resolve()

    original_main = """from temp_helper import _estimate_string_tokens
from pydantic_ai_slim.pydantic_ai.usage import Usage

def _get_string_usage(text: str) -> Usage:
    response_tokens = _estimate_string_tokens(text)
    return Usage(response_tokens=response_tokens, total_tokens=response_tokens)
"""
    main_file.write_text(original_main, encoding="utf-8")

    optimized_code = f"""{get_code_block_splitter(helper_file.relative_to(root_dir))}
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

{get_code_block_splitter(main_file.relative_to(root_dir))}
from temp_helper import _estimate_string_tokens
from pydantic_ai_slim.pydantic_ai.usage import Usage

def _get_string_usage(text: str) -> Usage:
    response_tokens = _estimate_string_tokens(text)
    return Usage(response_tokens=response_tokens, total_tokens=response_tokens)
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
        code_context=code_context, optimized_code=optimized_code, original_helper_code=original_helper_code
    )
    new_code = main_file.read_text(encoding="utf-8")
    new_helper_code = helper_file.read_text(encoding="utf-8")
    
    helper_file.unlink(missing_ok=True)
    main_file.unlink(missing_ok=True)
    
    expected_helper = """import re
from collections.abc import Sequence

from pydantic_ai_slim.pydantic_ai.messages import BinaryContent, UserContent

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


_TOKEN_SPLIT_RE = re.compile(r'[\\s",.:]+')

_translate_table = {ord(c): ord(' ') for c in ' \\t\\n\\r\\x0b\\x0c",.:'}
"""

    assert new_code.rstrip() == original_main.rstrip() # No Change
    assert new_helper_code.rstrip() == expected_helper.rstrip()