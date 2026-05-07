from __future__ import annotations

import re

_FUNC_BODY_RE = re.compile(r"^(func\s+)(Test\w+)(\s*\(\s*)(\w+)(\s+\*testing\.T\s*\)\s*\{)", re.MULTILINE)
_PARALLEL_RE = re.compile(r"^\s*\w+\.Parallel\(\)\s*\n?", re.MULTILINE)
_HELPER_RE = re.compile(r"^\s*\w+\.Helper\(\)\s*\n?", re.MULTILINE)


def _test_matches_target(test_name: str, target_function_name: str) -> bool:
    remainder = test_name[len("Test") :]
    segments = remainder.split("_")
    return target_function_name in segments


def convert_tests_to_benchmarks(test_source: str, target_function_name: str = "") -> str:
    if not test_source.strip():
        return test_source

    if not _FUNC_BODY_RE.search(test_source):
        return test_source

    result = test_source

    for match in reversed(list(_FUNC_BODY_RE.finditer(result))):
        func_prefix = match.group(1)
        test_name = match.group(2)
        paren_open = match.group(3)
        param_name = match.group(4)

        body_start = match.end()
        brace_depth = 1
        pos = body_start
        while pos < len(result) and brace_depth > 0:
            if result[pos] == "{":
                brace_depth += 1
            elif result[pos] == "}":
                brace_depth -= 1
            pos += 1

        if target_function_name and not _test_matches_target(test_name, target_function_name):
            result = result[: match.start()] + result[pos:]
            continue

        body = result[body_start : pos - 1]
        bench_name = "Benchmark" + test_name[len("Test") :]

        new_sig = f"{func_prefix}{bench_name}{paren_open}{param_name} *testing.B) {{\n\tfor i := 0; i < {param_name}.N; i++ {{"
        new_func = f"{new_sig}{body}\t}}\n}}"
        result = result[: match.start()] + new_func + result[pos:]

    result = result.replace("*testing.T", "*testing.B")
    result = _PARALLEL_RE.sub("", result)
    return _HELPER_RE.sub("", result)
