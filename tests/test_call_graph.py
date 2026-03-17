from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from codeflash.languages.python.reference_graph import ReferenceGraph
from codeflash.models.call_graph import (
    CallEdge,
    CalleeMetadata,
    CallGraph,
    FunctionNode,
    augment_with_trace,
    callees_from_graph,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def node(name: str, file: str = "mod.py") -> FunctionNode:
    return FunctionNode(file_path=__import__("pathlib").Path(file), qualified_name=name)


def edge(caller: str, callee: str, *, cross: bool = False, file: str = "mod.py") -> CallEdge:
    return CallEdge(caller=node(caller, file), callee=node(callee, file), is_cross_file=cross)


def make_graph(edges: list[CallEdge]) -> CallGraph:
    return CallGraph(edges=edges)


# ---------------------------------------------------------------------------
# CallGraph unit tests
# ---------------------------------------------------------------------------


class TestCalleesOf:
    def test_returns_direct_callees(self) -> None:
        g = make_graph([edge("a", "b"), edge("a", "c"), edge("b", "c")])
        callees = g.callees_of(node("a"))
        callee_names = {e.callee.qualified_name for e in callees}
        assert callee_names == {"b", "c"}

    def test_returns_empty_for_leaf(self) -> None:
        g = make_graph([edge("a", "b")])
        assert g.callees_of(node("b")) == []

    def test_returns_empty_for_unknown_node(self) -> None:
        g = make_graph([edge("a", "b")])
        assert g.callees_of(node("z")) == []


class TestCallersOf:
    def test_returns_direct_callers(self) -> None:
        g = make_graph([edge("a", "c"), edge("b", "c")])
        callers = g.callers_of(node("c"))
        caller_names = {e.caller.qualified_name for e in callers}
        assert caller_names == {"a", "b"}

    def test_returns_empty_for_root(self) -> None:
        g = make_graph([edge("a", "b")])
        assert g.callers_of(node("a")) == []


class TestDescendants:
    def test_transitive_descendants(self) -> None:
        g = make_graph([edge("a", "b"), edge("b", "c"), edge("c", "d")])
        desc = g.descendants(node("a"))
        assert {n.qualified_name for n in desc} == {"b", "c", "d"}

    def test_max_depth_limits_traversal(self) -> None:
        g = make_graph([edge("a", "b"), edge("b", "c"), edge("c", "d")])
        desc = g.descendants(node("a"), max_depth=1)
        assert {n.qualified_name for n in desc} == {"b"}

    def test_max_depth_two(self) -> None:
        g = make_graph([edge("a", "b"), edge("b", "c"), edge("c", "d")])
        desc = g.descendants(node("a"), max_depth=2)
        assert {n.qualified_name for n in desc} == {"b", "c"}

    def test_handles_cycle(self) -> None:
        g = make_graph([edge("a", "b"), edge("b", "a")])
        desc = g.descendants(node("a"))
        assert {n.qualified_name for n in desc} == {"b", "a"}

    def test_empty_for_leaf(self) -> None:
        g = make_graph([edge("a", "b")])
        assert g.descendants(node("b")) == set()


class TestAncestors:
    def test_transitive_ancestors(self) -> None:
        g = make_graph([edge("a", "b"), edge("b", "c"), edge("c", "d")])
        anc = g.ancestors(node("d"))
        assert {n.qualified_name for n in anc} == {"a", "b", "c"}

    def test_max_depth_limits_traversal(self) -> None:
        g = make_graph([edge("a", "b"), edge("b", "c"), edge("c", "d")])
        anc = g.ancestors(node("d"), max_depth=1)
        assert {n.qualified_name for n in anc} == {"c"}

    def test_empty_for_root(self) -> None:
        g = make_graph([edge("a", "b")])
        assert g.ancestors(node("a")) == set()


class TestLeafAndRootFunctions:
    def test_leaf_functions(self) -> None:
        g = make_graph([edge("a", "b"), edge("a", "c"), edge("b", "d")])
        leaves = g.leaf_functions()
        assert {n.qualified_name for n in leaves} == {"c", "d"}

    def test_root_functions(self) -> None:
        g = make_graph([edge("a", "b"), edge("a", "c"), edge("b", "d")])
        roots = g.root_functions()
        assert {n.qualified_name for n in roots} == {"a"}

    def test_single_edge(self) -> None:
        g = make_graph([edge("a", "b")])
        assert {n.qualified_name for n in g.leaf_functions()} == {"b"}
        assert {n.qualified_name for n in g.root_functions()} == {"a"}


class TestSubgraph:
    def test_filters_to_selected_nodes(self) -> None:
        g = make_graph([edge("a", "b"), edge("b", "c"), edge("c", "d")])
        sub = g.subgraph({node("a"), node("b"), node("c")})
        assert len(sub.edges) == 2
        callee_names = {e.callee.qualified_name for e in sub.edges}
        assert "d" not in callee_names

    def test_empty_subgraph(self) -> None:
        g = make_graph([edge("a", "b")])
        sub = g.subgraph(set())
        assert sub.edges == []


class TestTopologicalOrder:
    def test_linear_chain(self) -> None:
        # a -> b -> c -> d
        g = make_graph([edge("a", "b"), edge("b", "c"), edge("c", "d")])
        order = g.topological_order()
        names = [n.qualified_name for n in order]
        # Leaves-first: d before c before b before a
        assert names.index("d") < names.index("c")
        assert names.index("c") < names.index("b")
        assert names.index("b") < names.index("a")

    def test_diamond(self) -> None:
        # a -> b, a -> c, b -> d, c -> d
        g = make_graph([edge("a", "b"), edge("a", "c"), edge("b", "d"), edge("c", "d")])
        order = g.topological_order()
        names = [n.qualified_name for n in order]
        assert names.index("d") < names.index("b")
        assert names.index("d") < names.index("c")
        assert names.index("b") < names.index("a")
        assert names.index("c") < names.index("a")

    def test_empty_graph(self) -> None:
        g = make_graph([])
        assert g.topological_order() == []


class TestNodes:
    def test_collects_all_nodes(self) -> None:
        g = make_graph([edge("a", "b"), edge("c", "d")])
        names = {n.qualified_name for n in g.nodes}
        assert names == {"a", "b", "c", "d"}

    def test_empty_graph(self) -> None:
        g = make_graph([])
        assert g.nodes == set()


# ---------------------------------------------------------------------------
# augment_with_trace tests
# ---------------------------------------------------------------------------


class TestAugmentWithTrace:
    def test_overlays_runtime_data(self, tmp_path: Path) -> None:
        db_path = tmp_path / "trace.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            """
            CREATE TABLE pstats (
                filename TEXT, line_number INTEGER, function TEXT, class_name TEXT,
                call_count_nonrecursive INTEGER, num_callers INTEGER,
                total_time_ns INTEGER, cumulative_time_ns INTEGER, callers BLOB
            )
            """
        )
        conn.execute(
            "INSERT INTO pstats VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("mod.py", 1, "helper", None, 10, 1, 5000, 5000, b"[]"),
        )
        conn.commit()
        conn.close()

        g = make_graph([edge("caller", "helper")])
        augmented = augment_with_trace(g, db_path)

        assert len(augmented.edges) == 1
        e = augmented.edges[0]
        assert e.call_count == 10
        assert e.total_time_ns == 5000

    def test_unmatched_edges_preserved(self, tmp_path: Path) -> None:
        db_path = tmp_path / "trace.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            """
            CREATE TABLE pstats (
                filename TEXT, line_number INTEGER, function TEXT, class_name TEXT,
                call_count_nonrecursive INTEGER, num_callers INTEGER,
                total_time_ns INTEGER, cumulative_time_ns INTEGER, callers BLOB
            )
            """
        )
        conn.commit()
        conn.close()

        g = make_graph([edge("caller", "helper")])
        augmented = augment_with_trace(g, db_path)

        assert len(augmented.edges) == 1
        e = augmented.edges[0]
        assert e.call_count is None
        assert e.total_time_ns is None

    def test_missing_pstats_table(self, tmp_path: Path) -> None:
        db_path = tmp_path / "trace.db"
        conn = sqlite3.connect(str(db_path))
        conn.close()

        g = make_graph([edge("caller", "helper")])
        result = augment_with_trace(g, db_path)
        assert result.edges == g.edges

    def test_class_method_matching(self, tmp_path: Path) -> None:
        db_path = tmp_path / "trace.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            """
            CREATE TABLE pstats (
                filename TEXT, line_number INTEGER, function TEXT, class_name TEXT,
                call_count_nonrecursive INTEGER, num_callers INTEGER,
                total_time_ns INTEGER, cumulative_time_ns INTEGER, callers BLOB
            )
            """
        )
        conn.execute(
            "INSERT INTO pstats VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("mod.py", 5, "process", "MyClass", 3, 2, 9000, 12000, b"[]"),
        )
        conn.commit()
        conn.close()

        callee = FunctionNode(file_path=__import__("pathlib").Path("mod.py"), qualified_name="MyClass.process")
        caller = FunctionNode(file_path=__import__("pathlib").Path("mod.py"), qualified_name="main")
        g = CallGraph(edges=[CallEdge(caller=caller, callee=callee, is_cross_file=False)])

        augmented = augment_with_trace(g, db_path)
        assert augmented.edges[0].call_count == 3
        assert augmented.edges[0].total_time_ns == 9000


# ---------------------------------------------------------------------------
# ReferenceGraph.get_call_graph integration tests
# ---------------------------------------------------------------------------


def write_file(project: Path, name: str, content: str) -> Path:
    fp = project / name
    fp.write_text(content, encoding="utf-8")
    return fp


@pytest.fixture
def project(tmp_path: Path) -> Path:
    project_root = tmp_path / "project"
    project_root.mkdir()
    return project_root


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "cache.db"


class TestReferenceGraphGetCallGraph:
    def test_simple_call_graph(self, project: Path, db_path: Path) -> None:
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
        cg = ReferenceGraph(project, db_path=db_path)
        try:
            graph = cg.get_call_graph({project / "mod.py": {"caller"}})
            assert len(graph.edges) == 1
            assert graph.edges[0].caller.qualified_name == "caller"
            assert graph.edges[0].callee.qualified_name == "helper"
            assert not graph.edges[0].is_cross_file
        finally:
            cg.close()

    def test_cross_file_call_graph(self, project: Path, db_path: Path) -> None:
        write_file(project, "utils.py", "def utility():\n    return 42\n")
        write_file(
            project,
            "main.py",
            """\
from utils import utility

def caller():
    return utility()
""",
        )
        cg = ReferenceGraph(project, db_path=db_path)
        try:
            graph = cg.get_call_graph({project / "main.py": {"caller"}})
            assert len(graph.edges) == 1
            assert graph.edges[0].is_cross_file
            assert graph.edges[0].callee.qualified_name == "utility"
        finally:
            cg.close()

    def test_multiple_callees(self, project: Path, db_path: Path) -> None:
        write_file(
            project,
            "mod.py",
            """\
def a():
    return 1

def b():
    return 2

def caller():
    return a() + b()
""",
        )
        cg = ReferenceGraph(project, db_path=db_path)
        try:
            graph = cg.get_call_graph({project / "mod.py": {"caller"}})
            callee_names = {e.callee.qualified_name for e in graph.edges}
            assert callee_names == {"a", "b"}
        finally:
            cg.close()

    def test_empty_input(self, project: Path, db_path: Path) -> None:
        cg = ReferenceGraph(project, db_path=db_path)
        try:
            graph = cg.get_call_graph({})
            assert graph.edges == []
        finally:
            cg.close()

    def test_leaf_has_no_callees(self, project: Path, db_path: Path) -> None:
        write_file(
            project,
            "mod.py",
            """\
def leaf():
    return 42
""",
        )
        cg = ReferenceGraph(project, db_path=db_path)
        try:
            graph = cg.get_call_graph({project / "mod.py": {"leaf"}})
            assert graph.edges == []
        finally:
            cg.close()

    def test_include_metadata(self, project: Path, db_path: Path) -> None:
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
        cg = ReferenceGraph(project, db_path=db_path)
        try:
            graph = cg.get_call_graph({project / "mod.py": {"caller"}}, include_metadata=True)
            assert len(graph.edges) == 1
            e = graph.edges[0]
            assert e.callee_metadata is not None
            assert e.callee_metadata.only_function_name == "helper"
            assert e.callee_metadata.definition_type == "function"
            assert e.callee_metadata.fully_qualified_name != ""
            assert e.callee_metadata.source_line != ""
        finally:
            cg.close()

    def test_include_metadata_false_has_no_metadata(self, project: Path, db_path: Path) -> None:
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
        cg = ReferenceGraph(project, db_path=db_path)
        try:
            graph = cg.get_call_graph({project / "mod.py": {"caller"}})
            assert len(graph.edges) == 1
            assert graph.edges[0].callee_metadata is None
        finally:
            cg.close()

    def test_get_callees_includes_statement_dependencies(self, project: Path, db_path: Path) -> None:
        write_file(
            project,
            "mod.py",
            """\
X = 1

def caller():
    return X + 1
""",
        )
        cg = ReferenceGraph(project, db_path=db_path)
        try:
            _, function_sources = cg.get_callees({project / "mod.py": {"caller"}})
            assert [(source.qualified_name, source.definition_type) for source in function_sources] == [
                ("X", "statement")
            ]
        finally:
            cg.close()


# ---------------------------------------------------------------------------
# CalleeMetadata unit tests
# ---------------------------------------------------------------------------


def test_edge_with_metadata() -> None:
    meta = CalleeMetadata(
        fully_qualified_name="mod.helper",
        only_function_name="helper",
        definition_type="function",
        source_line="def helper(): ...",
    )
    e = CallEdge(caller=node("caller"), callee=node("helper"), is_cross_file=False, callee_metadata=meta)
    assert e.callee_metadata is meta
    assert e.callee_metadata.only_function_name == "helper"


def test_edge_without_metadata() -> None:
    e = CallEdge(caller=node("caller"), callee=node("helper"), is_cross_file=False)
    assert e.callee_metadata is None


# ---------------------------------------------------------------------------
# callees_from_graph unit tests
# ---------------------------------------------------------------------------


def test_callees_from_graph_extracts_function_sources() -> None:
    meta = CalleeMetadata(
        fully_qualified_name="mod.helper",
        only_function_name="helper",
        definition_type="function",
        source_line="def helper(): ...",
    )
    e = CallEdge(caller=node("caller"), callee=node("helper"), is_cross_file=False, callee_metadata=meta)
    g = CallGraph(edges=[e])

    file_map, source_list = callees_from_graph(g)
    assert len(source_list) == 1
    fs = source_list[0]
    assert fs.qualified_name == "helper"
    assert fs.fully_qualified_name == "mod.helper"
    assert fs.only_function_name == "helper"
    assert fs.source_code == "def helper(): ..."
    assert fs.definition_type == "function"

    from pathlib import Path

    assert Path("mod.py") in file_map
    assert fs in file_map[Path("mod.py")]


def test_callees_from_graph_skips_edges_without_metadata() -> None:
    e1 = CallEdge(caller=node("a"), callee=node("b"), is_cross_file=False)
    meta = CalleeMetadata(
        fully_qualified_name="mod.c", only_function_name="c", definition_type="function", source_line="def c(): ..."
    )
    e2 = CallEdge(caller=node("a"), callee=node("c"), is_cross_file=False, callee_metadata=meta)
    g = CallGraph(edges=[e1, e2])

    _, source_list = callees_from_graph(g)
    assert len(source_list) == 1
    assert source_list[0].qualified_name == "c"


def test_callees_from_graph_empty() -> None:
    g = CallGraph(edges=[])
    file_map, source_list = callees_from_graph(g)
    assert file_map == {}
    assert source_list == []
