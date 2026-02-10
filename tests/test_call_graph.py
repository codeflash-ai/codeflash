from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from codeflash.context.call_graph import CallGraph, IndexResult


@pytest.fixture
def project(tmp_path: Path) -> Path:
    project_root = tmp_path / "project"
    project_root.mkdir()
    return project_root


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "cache.db"


def write_file(project: Path, name: str, content: str) -> Path:
    fp = project / name
    fp.write_text(content, encoding="utf-8")
    return fp


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def test_simple_function_call(project: Path, db_path: Path) -> None:
    write_file(
        project,
        "mod.py",
        """\
def helper():
    return 1

def caller():
    return helper()
""",
    )
    cg = CallGraph(project, db_path=db_path)
    try:
        _, result_list = cg.get_callees({project / "mod.py": {"caller"}})
        callee_qns = {fs.qualified_name for fs in result_list}
        assert "helper" in callee_qns
    finally:
        cg.close()


def test_cross_file_call(project: Path, db_path: Path) -> None:
    write_file(
        project,
        "utils.py",
        """\
def utility():
    return 42
""",
    )
    write_file(
        project,
        "main.py",
        """\
from utils import utility

def caller():
    return utility()
""",
    )
    cg = CallGraph(project, db_path=db_path)
    try:
        _, result_list = cg.get_callees({project / "main.py": {"caller"}})
        callee_qns = {fs.qualified_name for fs in result_list}
        assert "utility" in callee_qns
        # Should be in the utils.py file
        callee_files = {fs.file_path.resolve() for fs in result_list if fs.qualified_name == "utility"}
        assert (project / "utils.py").resolve() in callee_files
    finally:
        cg.close()


def test_class_instantiation(project: Path, db_path: Path) -> None:
    write_file(
        project,
        "mod.py",
        """\
class MyClass:
    def __init__(self):
        pass

def caller():
    obj = MyClass()
    return obj
""",
    )
    cg = CallGraph(project, db_path=db_path)
    try:
        _, result_list = cg.get_callees({project / "mod.py": {"caller"}})
        callee_types = {fs.definition_type for fs in result_list}
        assert "class" in callee_types
    finally:
        cg.close()


def test_nested_function_excluded(project: Path, db_path: Path) -> None:
    write_file(
        project,
        "mod.py",
        """\
def caller():
    def inner():
        return 1
    return inner()
""",
    )
    cg = CallGraph(project, db_path=db_path)
    try:
        _, result_list = cg.get_callees({project / "mod.py": {"caller"}})
        assert len(result_list) == 0
    finally:
        cg.close()


def test_module_level_not_tracked(project: Path, db_path: Path) -> None:
    write_file(
        project,
        "mod.py",
        """\
def helper():
    return 1

x = helper()
""",
    )
    cg = CallGraph(project, db_path=db_path)
    try:
        # Module level calls have no enclosing function, so no edges
        _, result_list = cg.get_callees({project / "mod.py": {"helper"}})
        # helper itself doesn't call anything
        assert len(result_list) == 0
    finally:
        cg.close()


def test_site_packages_excluded(project: Path, db_path: Path) -> None:
    write_file(
        project,
        "mod.py",
        """\
import os

def caller():
    return os.path.join("a", "b")
""",
    )
    cg = CallGraph(project, db_path=db_path)
    try:
        _, result_list = cg.get_callees({project / "mod.py": {"caller"}})
        # os.path.join is stdlib, should not appear
        assert len(result_list) == 0
    finally:
        cg.close()


def test_empty_file(project: Path, db_path: Path) -> None:
    write_file(project, "mod.py", "")
    cg = CallGraph(project, db_path=db_path)
    try:
        _, result_list = cg.get_callees({project / "mod.py": set()})
        assert len(result_list) == 0
    finally:
        cg.close()


def test_syntax_error_file(project: Path, db_path: Path) -> None:
    write_file(project, "mod.py", "def broken(\n")
    cg = CallGraph(project, db_path=db_path)
    try:
        _, result_list = cg.get_callees({project / "mod.py": {"broken"}})
        assert len(result_list) == 0
    finally:
        cg.close()


# ---------------------------------------------------------------------------
# Caching tests
# ---------------------------------------------------------------------------


def test_caching_no_reindex(project: Path, db_path: Path) -> None:
    write_file(
        project,
        "mod.py",
        """\
def helper():
    return 1

def caller():
    return helper()
""",
    )
    cg = CallGraph(project, db_path=db_path)
    try:
        cg.get_callees({project / "mod.py": {"caller"}})
        # Second call should use in-memory cache (hash unchanged)
        resolved = str((project / "mod.py").resolve())
        assert resolved in cg.indexed_file_hashes
        old_hash = cg.indexed_file_hashes[resolved]
        cg.get_callees({project / "mod.py": {"caller"}})
        assert cg.indexed_file_hashes[resolved] == old_hash
    finally:
        cg.close()


def test_incremental_update_on_change(project: Path, db_path: Path) -> None:
    fp = write_file(
        project,
        "mod.py",
        """\
def helper():
    return 1

def caller():
    return helper()
""",
    )
    cg = CallGraph(project, db_path=db_path)
    try:
        _, result_list = cg.get_callees({project / "mod.py": {"caller"}})
        assert any(fs.qualified_name == "helper" for fs in result_list)

        # Modify the file — caller no longer calls helper
        fp.write_text(
            """\
def helper():
    return 1

def new_helper():
    return 2

def caller():
    return new_helper()
""",
            encoding="utf-8",
        )
        _, result_list = cg.get_callees({project / "mod.py": {"caller"}})
        callee_qns = {fs.qualified_name for fs in result_list}
        assert "new_helper" in callee_qns
    finally:
        cg.close()


def test_persistence_across_sessions(project: Path, db_path: Path) -> None:
    write_file(
        project,
        "mod.py",
        """\
def helper():
    return 1

def caller():
    return helper()
""",
    )
    # First session: index the file
    cg1 = CallGraph(project, db_path=db_path)
    try:
        _, result_list = cg1.get_callees({project / "mod.py": {"caller"}})
        assert any(fs.qualified_name == "helper" for fs in result_list)
    finally:
        cg1.close()

    # Second session: should read from DB without re-indexing
    cg2 = CallGraph(project, db_path=db_path)
    try:
        assert len(cg2.indexed_file_hashes) == 0  # in-memory cache is empty
        _, result_list = cg2.get_callees({project / "mod.py": {"caller"}})
        assert any(fs.qualified_name == "helper" for fs in result_list)
    finally:
        cg2.close()


def test_build_index_with_progress(project: Path, db_path: Path) -> None:
    write_file(
        project,
        "a.py",
        """\
def helper_a():
    return 1

def caller_a():
    return helper_a()
""",
    )
    write_file(
        project,
        "b.py",
        """\
from a import helper_a

def caller_b():
    return helper_a()
""",
    )

    cg = CallGraph(project, db_path=db_path)
    try:
        progress_calls: list[IndexResult] = []
        files = [project / "a.py", project / "b.py"]
        cg.build_index(files, on_progress=progress_calls.append)

        # Callback fired once per file
        assert len(progress_calls) == 2

        # Verify IndexResult fields for freshly indexed files
        for result in progress_calls:
            assert isinstance(result, IndexResult)
            assert not result.error
            assert not result.cached
            assert result.num_edges > 0
            assert len(result.edges) == result.num_edges
            assert result.cross_file_edges >= 0

        # Files are now indexed — get_callees should return correct results
        _, result_list = cg.get_callees({project / "a.py": {"caller_a"}})
        callee_qns = {fs.qualified_name for fs in result_list}
        assert "helper_a" in callee_qns
    finally:
        cg.close()


def test_build_index_cached_results(project: Path, db_path: Path) -> None:
    write_file(
        project,
        "a.py",
        """\
def helper_a():
    return 1

def caller_a():
    return helper_a()
""",
    )
    write_file(
        project,
        "b.py",
        """\
from a import helper_a

def caller_b():
    return helper_a()
""",
    )

    cg = CallGraph(project, db_path=db_path)
    try:
        files = [project / "a.py", project / "b.py"]
        # First pass — fresh indexing
        cg.build_index(files)

        # Second pass — should all be cached
        cached_results: list[IndexResult] = []
        cg.build_index(files, on_progress=cached_results.append)

        assert len(cached_results) == 2
        for result in cached_results:
            assert result.cached
            assert not result.error
            assert result.num_edges == 0
            assert result.edges == ()
            assert result.cross_file_edges == 0
    finally:
        cg.close()


def test_cross_file_edges_tracked(project: Path, db_path: Path) -> None:
    write_file(
        project,
        "utils.py",
        """\
def utility():
    return 42
""",
    )
    write_file(
        project,
        "main.py",
        """\
from utils import utility

def caller():
    return utility()
""",
    )

    cg = CallGraph(project, db_path=db_path)
    try:
        progress_calls: list[IndexResult] = []
        cg.build_index([project / "utils.py", project / "main.py"], on_progress=progress_calls.append)

        # main.py should have cross-file edges (calls into utils.py)
        main_result = next(r for r in progress_calls if r.file_path.name == "main.py")
        assert main_result.cross_file_edges > 0
        # At least one edge tuple should have is_cross_file=True
        assert any(is_cross_file for _, _, is_cross_file in main_result.edges)
    finally:
        cg.close()


def test_same_file_edges_not_cross_file(project: Path, db_path: Path) -> None:
    write_file(
        project,
        "mod.py",
        """\
def helper():
    return 1

def caller():
    return helper()
""",
    )

    cg = CallGraph(project, db_path=db_path)
    try:
        progress_calls: list[IndexResult] = []
        cg.build_index([project / "mod.py"], on_progress=progress_calls.append)

        assert len(progress_calls) == 1
        result = progress_calls[0]
        assert result.cross_file_edges == 0
        # All edges should have is_cross_file=False
        assert all(not is_cross_file for _, _, is_cross_file in result.edges)
    finally:
        cg.close()
