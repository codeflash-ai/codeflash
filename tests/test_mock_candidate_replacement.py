"""Test replace_function_and_helpers_with_optimized_code with mock candidate from mock_candidate.txt."""

import tempfile
from pathlib import Path

import pytest

from codeflash.discovery.functions_to_optimize import FunctionToOptimize
from codeflash.languages.python.context.unused_definition_remover import detect_unused_helper_functions
from codeflash.models.function_types import FunctionParent
from codeflash.models.models import CodeStringsMarkdown
from codeflash.optimization.function_optimizer import FunctionOptimizer
from codeflash.verification.verification_utils import TestConfig

ORIGINAL_SOURCE = '''\
import contextlib
from typing import BinaryIO, TypeVar, Union

_SymbolT = TypeVar("_SymbolT", PSLiteral, PSKeyword)


PSLiteralTable = PSSymbolTable(PSLiteral)
PSKeywordTable = PSSymbolTable(PSKeyword)
LIT = PSLiteralTable.intern
KWD = PSKeywordTable.intern
KEYWORD_DICT_BEGIN = KWD(b"<<")
KEYWORD_DICT_END = KWD(b">>")


PSBaseParserToken = Union[float, bool, PSLiteral, PSKeyword, bytes]


class PSBaseParser:

    def __init__(self, fp: BinaryIO) -> None:
        self.fp = fp
        self.eof = False
        self.seek(0)

    def _parse_main(self, s: bytes, i: int) -> int:
        m = NONSPC.search(s, i)
        if not m:
            return len(s)
        j = m.start(0)
        c = s[j : j + 1]
        self._curtokenpos = self.bufpos + j
        if c == b"%":
            self._curtoken = b"%"
            self._parse1 = self._parse_comment
            return j + 1
        elif c == b"/":
            self._curtoken = b""
            self._parse1 = self._parse_literal
            return j + 1
        elif c in b"-+" or c.isdigit():
            self._curtoken = c
            self._parse1 = self._parse_number
            return j + 1
        elif c == b".":
            self._curtoken = c
            self._parse1 = self._parse_float
            return j + 1
        elif c.isalpha():
            self._curtoken = c
            self._parse1 = self._parse_keyword
            return j + 1
        elif c == b"(":
            self._curtoken = b""
            self.paren = 1
            self._parse1 = self._parse_string
            return j + 1
        elif c == b"<":
            self._curtoken = b""
            self._parse1 = self._parse_wopen
            return j + 1
        elif c == b">":
            self._curtoken = b""
            self._parse1 = self._parse_wclose
            return j + 1
        elif c == b"\\x00":
            return j + 1
        else:
            self._add_token(KWD(c))
            return j + 1

    def _add_token(self, obj: PSBaseParserToken) -> None:
        self._tokens.append((self._curtokenpos, obj))

    def _parse_comment(self, s: bytes, i: int) -> int:
        m = EOL.search(s, i)
        if not m:
            self._curtoken += s[i:]
            return len(s)
        j = m.start(0)
        self._curtoken += s[i:j]
        self._parse1 = self._parse_main
        return j

    def _parse_literal(self, s: bytes, i: int) -> int:
        m = END_LITERAL.search(s, i)
        if not m:
            self._curtoken += s[i:]
            return len(s)
        j = m.start(0)
        self._curtoken += s[i:j]
        c = s[j : j + 1]
        if c == b"#":
            self.hex = b""
            self._parse1 = self._parse_literal_hex
            return j + 1
        try:
            name: str | bytes = str(self._curtoken, "utf-8")
        except Exception:
            name = self._curtoken
        self._add_token(LIT(name))
        self._parse1 = self._parse_main
        return j

    def _parse_number(self, s: bytes, i: int) -> int:
        m = END_NUMBER.search(s, i)
        if not m:
            self._curtoken += s[i:]
            return len(s)
        j = m.start(0)
        self._curtoken += s[i:j]
        c = s[j : j + 1]
        if c == b".":
            self._curtoken += b"."
            self._parse1 = self._parse_float
            return j + 1
        with contextlib.suppress(ValueError):
            self._add_token(int(self._curtoken))
        self._parse1 = self._parse_main
        return j

    def _parse_float(self, s: bytes, i: int) -> int:
        m = END_NUMBER.search(s, i)
        if not m:
            self._curtoken += s[i:]
            return len(s)
        j = m.start(0)
        self._curtoken += s[i:j]
        with contextlib.suppress(ValueError):
            self._add_token(float(self._curtoken))
        self._parse1 = self._parse_main
        return j

    def _parse_keyword(self, s: bytes, i: int) -> int:
        m = END_KEYWORD.search(s, i)
        if m:
            j = m.start(0)
            self._curtoken += s[i:j]
        else:
            self._curtoken += s[i:]
            return len(s)
        if self._curtoken == b"true":
            token: bool | PSKeyword = True
        elif self._curtoken == b"false":
            token = False
        else:
            token = KWD(self._curtoken)
        self._add_token(token)
        self._parse1 = self._parse_main
        return j

    def _parse_string(self, s: bytes, i: int) -> int:
        m = END_STRING.search(s, i)
        if not m:
            self._curtoken += s[i:]
            return len(s)
        j = m.start(0)
        self._curtoken += s[i:j]
        c = s[j : j + 1]
        if c == b"\\\\":
            self.oct = b""
            self._parse1 = self._parse_string_1
            return j + 1
        if c == b"(":
            self.paren += 1
            self._curtoken += c
            return j + 1
        if c == b")":
            self.paren -= 1
            if self.paren:
                self._curtoken += c
                return j + 1
        self._add_token(self._curtoken)
        self._parse1 = self._parse_main
        return j + 1

    def _parse_wopen(self, s: bytes, i: int) -> int:
        c = s[i : i + 1]
        if c == b"<":
            self._add_token(KEYWORD_DICT_BEGIN)
            self._parse1 = self._parse_main
            i += 1
        else:
            self._parse1 = self._parse_hexstring
        return i

    def _parse_wclose(self, s: bytes, i: int) -> int:
        c = s[i : i + 1]
        if c == b">":
            self._add_token(KEYWORD_DICT_END)
            i += 1
        self._parse1 = self._parse_main
        return i
'''

MOCK_CANDIDATE_MARKDOWN = '''\
```python
#!/usr/bin/env python3


import contextlib
from typing import BinaryIO, TypeVar, Union

_SymbolT = TypeVar("_SymbolT", PSLiteral, PSKeyword)


PSLiteralTable = PSSymbolTable(PSLiteral)
PSKeywordTable = PSSymbolTable(PSKeyword)
LIT = PSLiteralTable.intern
KWD = PSKeywordTable.intern
KEYWORD_DICT_BEGIN = KWD(b"<<")
KEYWORD_DICT_END = KWD(b">>")


PSBaseParserToken = Union[float, bool, PSLiteral, PSKeyword, bytes]


class PSBaseParser:

    def __init__(self, fp: BinaryIO) -> None:
        self.fp = fp
        self.eof = False
        self.seek(0)

    def _parse_main(self, s: bytes, i: int) -> int:
        m = NONSPC.search(s, i)
        if not m:
            return len(s)
        j = m.start(0)
        # Use integer byte access to avoid creating a new one-byte bytes object.
        c_int = s[j]
        c_byte = bytes((c_int,))
        self._curtokenpos = self.bufpos + j
        if c_int == 37:  # b"%"
            self._curtoken = b"%"
            self._parse1 = self._parse_comment
            return j + 1
        elif c_int == 47:  # b"/"
            self._curtoken = b""
            self._parse1 = self._parse_literal
            return j + 1
        # b"-" is 45, b"+" is 43
        elif c_int == 45 or c_int == 43 or (48 <= c_int <= 57):
            self._curtoken = c_byte
            self._parse1 = self._parse_number
            return j + 1
        elif c_int == 46:  # b"."
            self._curtoken = c_byte
            self._parse1 = self._parse_float
            return j + 1
        # ASCII alphabetic check
        elif (65 <= c_int <= 90) or (97 <= c_int <= 122):
            self._curtoken = c_byte
            self._parse1 = self._parse_keyword
            return j + 1
        elif c_int == 40:  # b"("
            self._curtoken = b""
            self.paren = 1
            self._parse1 = self._parse_string
            return j + 1
        elif c_int == 60:  # b"<"
            self._curtoken = b""
            self._parse1 = self._parse_wopen
            return j + 1
        elif c_int == 62:  # b">"
            self._curtoken = b""
            self._parse1 = self._parse_wclose
            return j + 1
        elif c_int == 0:  # b"\\x00"
            return j + 1
        else:
            self._add_token(KWD(c_byte))
            return j + 1

    def _add_token(self, obj: PSBaseParserToken) -> None:
        self._tokens.append((self._curtokenpos, obj))

    def _parse_comment(self, s: bytes, i: int) -> int:
        m = EOL.search(s, i)
        if not m:
            self._curtoken += s[i:]
            return len(s)
        j = m.start(0)
        self._curtoken += s[i:j]
        self._parse1 = self._parse_main
        # We ignore comments.
        # self._tokens.append(self._curtoken)
        return j

    def _parse_literal(self, s: bytes, i: int) -> int:
        m = END_LITERAL.search(s, i)
        if not m:
            self._curtoken += s[i:]
            return len(s)
        j = m.start(0)
        self._curtoken += s[i:j]
        c_int = s[j]
        if c_int == 35:  # b"#"
            self.hex = b""
            self._parse1 = self._parse_literal_hex
            return j + 1
        try:
            name: str | bytes = str(self._curtoken, "utf-8")
        except Exception:
            name = self._curtoken
        self._add_token(LIT(name))
        self._parse1 = self._parse_main
        return j

    def _parse_number(self, s: bytes, i: int) -> int:
        m = END_NUMBER.search(s, i)
        if not m:
            self._curtoken += s[i:]
            return len(s)
        j = m.start(0)
        self._curtoken += s[i:j]
        c_int = s[j]
        if c_int == 46:  # b"."
            self._curtoken += b"."
            self._parse1 = self._parse_float
            return j + 1
        with contextlib.suppress(ValueError):
            self._add_token(int(self._curtoken))
        self._parse1 = self._parse_main
        return j

    def _parse_float(self, s: bytes, i: int) -> int:
        m = END_NUMBER.search(s, i)
        if not m:
            self._curtoken += s[i:]
            return len(s)
        j = m.start(0)
        self._curtoken += s[i:j]
        with contextlib.suppress(ValueError):
            self._add_token(float(self._curtoken))
        self._parse1 = self._parse_main
        return j

    def _parse_keyword(self, s: bytes, i: int) -> int:
        m = END_KEYWORD.search(s, i)
        if m:
            j = m.start(0)
            self._curtoken += s[i:j]
        else:
            self._curtoken += s[i:]
            return len(s)
        if self._curtoken == b"true":
            token: bool | PSKeyword = True
        elif self._curtoken == b"false":
            token = False
        else:
            token = KWD(self._curtoken)
        self._add_token(token)
        self._parse1 = self._parse_main
        return j

    def _parse_string(self, s: bytes, i: int) -> int:
        m = END_STRING.search(s, i)
        if not m:
            self._curtoken += s[i:]
            return len(s)
        j = m.start(0)
        self._curtoken += s[i:j]
        c_int = s[j]
        if c_int == 92:  # b"\\\\"
            self.oct = b""
            self._parse1 = self._parse_string_1
            return j + 1
        if c_int == 40:  # b"("
            self.paren += 1
            # append the literal "(" byte
            self._curtoken += b"("
            return j + 1
        if c_int == 41:  # b")"
            self.paren -= 1
            if self.paren:
                # WTF, they said balanced parens need no special treatment.
                self._curtoken += b")"
                return j + 1
        self._add_token(self._curtoken)
        self._parse1 = self._parse_main
        return j + 1

    def _parse_wopen(self, s: bytes, i: int) -> int:
        c_int = s[i]
        if c_int == 60:  # b"<"
            self._add_token(KEYWORD_DICT_BEGIN)
            self._parse1 = self._parse_main
            i += 1
        else:
            self._parse1 = self._parse_hexstring
        return i

    def _parse_wclose(self, s: bytes, i: int) -> int:
        c_int = s[i]
        if c_int == 62:  # b">"
            self._add_token(KEYWORD_DICT_END)
            i += 1
        self._parse1 = self._parse_main
        return i
```
'''

EXPECTED_OUTPUT = '''\
import contextlib
from typing import BinaryIO, TypeVar, Union

_SymbolT = TypeVar("_SymbolT", PSLiteral, PSKeyword)


PSLiteralTable = PSSymbolTable(PSLiteral)
PSKeywordTable = PSSymbolTable(PSKeyword)
LIT = PSLiteralTable.intern
KWD = PSKeywordTable.intern
KEYWORD_DICT_BEGIN = KWD(b"<<")
KEYWORD_DICT_END = KWD(b">>")


PSBaseParserToken = Union[float, bool, PSLiteral, PSKeyword, bytes]


class PSBaseParser:

    def __init__(self, fp: BinaryIO) -> None:
        self.fp = fp
        self.eof = False
        self.seek(0)

    def _parse_main(self, s: bytes, i: int) -> int:
        m = NONSPC.search(s, i)
        if not m:
            return len(s)
        j = m.start(0)
        # Use integer byte access to avoid creating a new one-byte bytes object.
        c_int = s[j]
        c_byte = bytes((c_int,))
        self._curtokenpos = self.bufpos + j
        if c_int == 37:  # b"%"
            self._curtoken = b"%"
            self._parse1 = self._parse_comment
            return j + 1
        elif c_int == 47:  # b"/"
            self._curtoken = b""
            self._parse1 = self._parse_literal
            return j + 1
        # b"-" is 45, b"+" is 43
        elif c_int == 45 or c_int == 43 or (48 <= c_int <= 57):
            self._curtoken = c_byte
            self._parse1 = self._parse_number
            return j + 1
        elif c_int == 46:  # b"."
            self._curtoken = c_byte
            self._parse1 = self._parse_float
            return j + 1
        # ASCII alphabetic check
        elif (65 <= c_int <= 90) or (97 <= c_int <= 122):
            self._curtoken = c_byte
            self._parse1 = self._parse_keyword
            return j + 1
        elif c_int == 40:  # b"("
            self._curtoken = b""
            self.paren = 1
            self._parse1 = self._parse_string
            return j + 1
        elif c_int == 60:  # b"<"
            self._curtoken = b""
            self._parse1 = self._parse_wopen
            return j + 1
        elif c_int == 62:  # b">"
            self._curtoken = b""
            self._parse1 = self._parse_wclose
            return j + 1
        elif c_int == 0:  # b"\\x00"
            return j + 1
        else:
            self._add_token(KWD(c_byte))
            return j + 1

    def _add_token(self, obj: PSBaseParserToken) -> None:
        self._tokens.append((self._curtokenpos, obj))

    def _parse_comment(self, s: bytes, i: int) -> int:
        m = EOL.search(s, i)
        if not m:
            self._curtoken += s[i:]
            return len(s)
        j = m.start(0)
        self._curtoken += s[i:j]
        self._parse1 = self._parse_main
        # We ignore comments.
        # self._tokens.append(self._curtoken)
        return j

    def _parse_literal(self, s: bytes, i: int) -> int:
        m = END_LITERAL.search(s, i)
        if not m:
            self._curtoken += s[i:]
            return len(s)
        j = m.start(0)
        self._curtoken += s[i:j]
        c_int = s[j]
        if c_int == 35:  # b"#"
            self.hex = b""
            self._parse1 = self._parse_literal_hex
            return j + 1
        try:
            name: str | bytes = str(self._curtoken, "utf-8")
        except Exception:
            name = self._curtoken
        self._add_token(LIT(name))
        self._parse1 = self._parse_main
        return j

    def _parse_number(self, s: bytes, i: int) -> int:
        m = END_NUMBER.search(s, i)
        if not m:
            self._curtoken += s[i:]
            return len(s)
        j = m.start(0)
        self._curtoken += s[i:j]
        c_int = s[j]
        if c_int == 46:  # b"."
            self._curtoken += b"."
            self._parse1 = self._parse_float
            return j + 1
        with contextlib.suppress(ValueError):
            self._add_token(int(self._curtoken))
        self._parse1 = self._parse_main
        return j

    def _parse_float(self, s: bytes, i: int) -> int:
        m = END_NUMBER.search(s, i)
        if not m:
            self._curtoken += s[i:]
            return len(s)
        j = m.start(0)
        self._curtoken += s[i:j]
        with contextlib.suppress(ValueError):
            self._add_token(float(self._curtoken))
        self._parse1 = self._parse_main
        return j

    def _parse_keyword(self, s: bytes, i: int) -> int:
        m = END_KEYWORD.search(s, i)
        if m:
            j = m.start(0)
            self._curtoken += s[i:j]
        else:
            self._curtoken += s[i:]
            return len(s)
        if self._curtoken == b"true":
            token: bool | PSKeyword = True
        elif self._curtoken == b"false":
            token = False
        else:
            token = KWD(self._curtoken)
        self._add_token(token)
        self._parse1 = self._parse_main
        return j

    def _parse_string(self, s: bytes, i: int) -> int:
        m = END_STRING.search(s, i)
        if not m:
            self._curtoken += s[i:]
            return len(s)
        j = m.start(0)
        self._curtoken += s[i:j]
        c_int = s[j]
        if c_int == 92:  # b"\\\\"
            self.oct = b""
            self._parse1 = self._parse_string_1
            return j + 1
        if c_int == 40:  # b"("
            self.paren += 1
            # append the literal "(" byte
            self._curtoken += b"("
            return j + 1
        if c_int == 41:  # b")"
            self.paren -= 1
            if self.paren:
                # WTF, they said balanced parens need no special treatment.
                self._curtoken += b")"
                return j + 1
        self._add_token(self._curtoken)
        self._parse1 = self._parse_main
        return j + 1

    def _parse_wopen(self, s: bytes, i: int) -> int:
        c_int = s[i]
        if c_int == 60:  # b"<"
            self._add_token(KEYWORD_DICT_BEGIN)
            self._parse1 = self._parse_main
            i += 1
        else:
            self._parse1 = self._parse_hexstring
        return i

    def _parse_wclose(self, s: bytes, i: int) -> int:
        c_int = s[i]
        if c_int == 62:  # b">"
            self._add_token(KEYWORD_DICT_END)
            i += 1
        self._parse1 = self._parse_main
        return i
'''


@pytest.fixture
def temp_project():
    temp_dir = Path(tempfile.mkdtemp())
    source_file = temp_dir / "psparser.py"
    source_file.write_text(ORIGINAL_SOURCE, encoding="utf-8")

    test_cfg = TestConfig(
        tests_root=temp_dir / "tests",
        tests_project_rootdir=temp_dir,
        project_root_path=temp_dir,
        test_framework="pytest",
        pytest_cmd="pytest",
    )

    yield temp_dir, source_file, test_cfg

    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


def run_replacement(temp_project):
    """Helper: run the full replacement pipeline and return (optimizer, code_context, final_content)."""
    temp_dir, source_file, test_cfg = temp_project

    function_to_optimize = FunctionToOptimize(
        file_path=source_file,
        function_name="_parse_main",
        parents=[FunctionParent(name="PSBaseParser", type="ClassDef")],
    )

    optimizer = FunctionOptimizer(
        function_to_optimize=function_to_optimize,
        test_cfg=test_cfg,
        function_to_optimize_source_code=source_file.read_text(encoding="utf-8"),
    )

    ctx_result = optimizer.get_code_optimization_context()
    assert ctx_result.is_successful(), f"Failed to get context: {ctx_result.failure()}"
    code_context = ctx_result.unwrap()

    original_content = source_file.read_text(encoding="utf-8")
    original_helper_code = {source_file: original_content}
    optimized_code = CodeStringsMarkdown.parse_markdown_code(MOCK_CANDIDATE_MARKDOWN)

    did_update = optimizer.replace_function_and_helpers_with_optimized_code(
        code_context, optimized_code, original_helper_code
    )
    assert did_update, "Expected the code to be updated"

    final_content = source_file.read_text(encoding="utf-8")
    return optimizer, code_context, final_content


def test_replace_with_mock_candidate(temp_project):
    """Verify replace_function_and_helpers_with_optimized_code produces the exact expected output.

    The code context detects ALL sibling methods as helpers of _parse_main.
    replace_function_definitions_in_module replaces ALL method bodies.
    detect_unused_helper_functions correctly recognizes methods referenced via attribute
    assignment (self._parse1 = self._parse_literal) as used, so they are NOT reverted.
    """
    _, code_context, final_content = run_replacement(temp_project)

    # Code context correctly detects ALL methods as helpers
    helper_names = {h.qualified_name for h in code_context.helper_functions}
    assert helper_names == {
        "PSBaseParser._parse_comment",
        "PSBaseParser._parse_literal",
        "PSBaseParser._parse_number",
        "PSBaseParser._parse_float",
        "PSBaseParser._parse_keyword",
        "PSBaseParser._parse_string",
        "PSBaseParser._parse_wopen",
        "PSBaseParser._parse_wclose",
        "PSBaseParser._add_token",
        "KWD",
    }

    # The final content should match the expected output exactly
    assert final_content == EXPECTED_OUTPUT


def test_detect_unused_helpers_handles_attribute_refs(temp_project):
    """Verify detect_unused_helper_functions recognizes methods referenced via attribute assignment.

    When _parse_main does `self._parse1 = self._parse_literal`, the method is referenced as
    an ast.Attribute value (not an ast.Call). The detection should recognize these as used.
    """
    temp_dir, source_file, test_cfg = temp_project

    function_to_optimize = FunctionToOptimize(
        file_path=source_file,
        function_name="_parse_main",
        parents=[FunctionParent(name="PSBaseParser", type="ClassDef")],
    )

    optimizer = FunctionOptimizer(
        function_to_optimize=function_to_optimize,
        test_cfg=test_cfg,
        function_to_optimize_source_code=source_file.read_text(encoding="utf-8"),
    )

    ctx_result = optimizer.get_code_optimization_context()
    assert ctx_result.is_successful()
    code_context = ctx_result.unwrap()

    optimized_code = CodeStringsMarkdown.parse_markdown_code(MOCK_CANDIDATE_MARKDOWN)

    unused_helpers = detect_unused_helper_functions(
        optimizer.function_to_optimize, code_context, optimized_code
    )
    unused_names = {h.qualified_name for h in unused_helpers}

    # No helpers should be detected as unused — all are either directly called or
    # referenced via attribute assignment (self._parse1 = self._parse_X)
    assert unused_names == set(), f"Expected no unused helpers, got: {unused_names}"


def test_replace_produces_valid_python(temp_project):
    """Verify the final output is valid, parseable Python."""
    _, _, final_content = run_replacement(temp_project)

    import ast
    ast.parse(final_content)
