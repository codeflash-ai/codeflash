from __future__ import annotations

import ast
import tempfile
from pathlib import Path
from unittest.mock import patch

from codeflash.discovery.functions_to_optimize import (
    discovery_cache,
    find_all_functions_in_file,
    inspect_top_level_functions_or_methods,
    parse_ast_cached,
    read_file_cached,
)


def test_read_file_cached_without_context_manager(tmp_path: Path) -> None:
    f = tmp_path / "sample.py"
    f.write_text("x = 1\n", encoding="utf-8")
    assert read_file_cached(f) == "x = 1\n"


def test_read_file_cached_returns_same_object_within_context(tmp_path: Path) -> None:
    f = tmp_path / "sample.py"
    f.write_text("x = 1\n", encoding="utf-8")
    with discovery_cache():
        result1 = read_file_cached(f)
        result2 = read_file_cached(f)
    assert result1 is result2


def test_read_file_cached_does_not_persist_across_contexts(tmp_path: Path) -> None:
    f = tmp_path / "sample.py"
    f.write_text("x = 1\n", encoding="utf-8")
    with discovery_cache():
        result1 = read_file_cached(f)
    f.write_text("x = 2\n", encoding="utf-8")
    with discovery_cache():
        result2 = read_file_cached(f)
    assert result1 != result2


def test_parse_ast_cached_returns_same_object_within_context(tmp_path: Path) -> None:
    f = tmp_path / "sample.py"
    f.write_text("def foo():\n    return 1\n", encoding="utf-8")
    with discovery_cache():
        tree1 = parse_ast_cached(f)
        tree2 = parse_ast_cached(f)
    assert tree1 is tree2
    assert isinstance(tree1, ast.Module)


def test_parse_ast_cached_uses_provided_source(tmp_path: Path) -> None:
    f = tmp_path / "sample.py"
    f.write_text("x = 1\n", encoding="utf-8")
    source = "y = 2\n"
    with discovery_cache():
        tree = parse_ast_cached(f, source=source)
    assert any(
        isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id == "y"
        for n in ast.walk(tree)
    )


def test_discovery_cache_avoids_redundant_reads(tmp_path: Path) -> None:
    f = tmp_path / "module.py"
    f.write_text("def bar():\n    return 42\n", encoding="utf-8")
    with discovery_cache():
        with patch.object(Path, "read_text", wraps=f.read_text) as mock_read:
            read_file_cached(f)
            read_file_cached(f)
            read_file_cached(f)
            assert mock_read.call_count == 1


def test_find_all_functions_in_file_uses_cache(tmp_path: Path) -> None:
    f = tmp_path / "module.py"
    f.write_text("def compute(x):\n    return x * 2\n", encoding="utf-8")
    with discovery_cache():
        result = find_all_functions_in_file(f)
        assert f in result
        assert result[f][0].function_name == "compute"


def test_inspect_top_level_functions_uses_cache(tmp_path: Path) -> None:
    f = tmp_path / "module.py"
    f.write_text("def top_func(a, b):\n    return a + b\n", encoding="utf-8")
    with discovery_cache():
        props = inspect_top_level_functions_or_methods(f, "top_func")
    assert props is not None
    assert props.is_top_level
    assert props.has_args


def test_find_and_inspect_share_cached_content(tmp_path: Path) -> None:
    f = tmp_path / "module.py"
    f.write_text(
        "class MyClass:\n    def method(self):\n        return 1\n\ndef standalone():\n    return 2\n",
        encoding="utf-8",
    )
    with discovery_cache():
        with patch.object(Path, "read_text", wraps=f.read_text) as mock_read:
            find_all_functions_in_file(f)
            props = inspect_top_level_functions_or_methods(f, "method", class_name="MyClass")
            assert mock_read.call_count == 1
    assert props is not None
    assert props.is_top_level


def test_discovery_results_correct_with_multiple_files(tmp_path: Path) -> None:
    f1 = tmp_path / "a.py"
    f1.write_text("def alpha():\n    return 'a'\n", encoding="utf-8")
    f2 = tmp_path / "b.py"
    f2.write_text("def beta(x):\n    return x + 1\n", encoding="utf-8")

    with discovery_cache():
        r1 = find_all_functions_in_file(f1)
        r2 = find_all_functions_in_file(f2)

    assert r1[f1][0].function_name == "alpha"
    assert r2[f2][0].function_name == "beta"


def test_cache_handles_invalid_syntax_gracefully(tmp_path: Path) -> None:
    f = tmp_path / "broken.py"
    f.write_text("def incomplete(:\n", encoding="utf-8")
    with discovery_cache():
        result = find_all_functions_in_file(f)
    assert result == {}


def test_cache_handles_nonexistent_file_in_parse_ast(tmp_path: Path) -> None:
    f = tmp_path / "nonexistent.py"
    with discovery_cache():
        try:
            parse_ast_cached(f)
            assert False, "Should have raised"
        except (FileNotFoundError, OSError):
            pass
