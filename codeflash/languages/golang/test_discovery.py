from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from codeflash.languages.base import TestInfo

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from codeflash.discovery.functions_to_optimize import FunctionToOptimize

logger = logging.getLogger(__name__)

GO_TEST_FUNC_RE = re.compile(r"^func\s+(Test\w+)\s*\(", re.MULTILINE)


def discover_tests(test_root: Path, source_functions: Sequence[FunctionToOptimize]) -> dict[str, list[TestInfo]]:
    func_name_to_qn: dict[str, list[str]] = {}
    for func in source_functions:
        func_name_to_qn.setdefault(func.function_name, []).append(func.qualified_name)

    test_files = list(test_root.rglob("*_test.go"))
    result: dict[str, list[TestInfo]] = {}

    for test_file in test_files:
        try:
            content = test_file.read_text(encoding="utf-8")
        except Exception:
            logger.debug("Could not read test file %s", test_file)
            continue

        test_func_names = GO_TEST_FUNC_RE.findall(content)
        for test_func_name in test_func_names:
            matched_qns = _match_test_to_functions(test_func_name, content, func_name_to_qn)
            for qn in matched_qns:
                info = TestInfo(test_name=test_func_name, test_file=test_file)
                result.setdefault(qn, []).append(info)

    return result


def _match_test_to_functions(test_func_name: str, test_source: str, func_name_to_qn: dict[str, list[str]]) -> list[str]:
    matched: list[str] = []

    target_name = _extract_target_name(test_func_name)
    if target_name and target_name in func_name_to_qn:
        matched.extend(func_name_to_qn[target_name])
        return matched

    for func_name, qns in func_name_to_qn.items():
        if _test_calls_function(test_source, test_func_name, func_name):
            matched.extend(qns)

    return matched


def _extract_target_name(test_func_name: str) -> str | None:
    if not test_func_name.startswith("Test"):
        return None
    remainder = test_func_name[4:]
    if not remainder:
        return None
    name = remainder.split("_")[0]
    if not name:
        return None
    return name


def _test_calls_function(test_source: str, test_func_name: str, func_name: str) -> bool:
    func_body = _extract_test_body(test_source, test_func_name)
    if func_body is None:
        return False
    call_pattern = re.compile(rf"\b{re.escape(func_name)}\s*\(")
    return call_pattern.search(func_body) is not None


def _extract_test_body(test_source: str, test_func_name: str) -> str | None:
    pattern = re.compile(rf"func\s+{re.escape(test_func_name)}\s*\([^)]*\)\s*\{{")
    match = pattern.search(test_source)
    if match is None:
        return None

    start = match.end()
    depth = 1
    pos = start
    while pos < len(test_source) and depth > 0:
        ch = test_source[pos]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        pos += 1

    return test_source[start : pos - 1] if depth == 0 else None
